import argparse
import random
import numpy as np
import os
from functools import partial
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, enable_wrap, wrap
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from transformers.models.phi.modeling_phi import PhiDecoderLayer
# from torch.distributed.fsdp import state_dict_type, FullStateDictConfig

import wandb


def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def calculate_DPO_loss(model_preferred_logprob, model_dispreferred_logprob,
                       ref_preferred_logprob, ref_dispreferred_logprob,
                       beta=0.5):

    preferred_relative_logprob = model_preferred_logprob - ref_preferred_logprob
    dispreferred_relative_logprob = model_dispreferred_logprob - ref_dispreferred_logprob

    reward_accuracies = (preferred_relative_logprob > dispreferred_relative_logprob).float().mean()
    reward_margins = (preferred_relative_logprob - dispreferred_relative_logprob).mean()

    loss = -F.logsigmoid(beta * (preferred_relative_logprob - dispreferred_relative_logprob)).mean()

    return loss, preferred_relative_logprob.mean(), dispreferred_relative_logprob.mean(), reward_accuracies, reward_margins


# def get_log_prob(logits, labels, prompt_lengths):
#     log_probs = F.log_softmax(logits, dim=-1)
#     token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

#     batch_size, seq_len = labels.shape
#     response_mask = torch.arange(seq_len, device=labels.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
#     response_mask = response_mask.float()

#     response_log_probs = (token_log_probs * response_mask).sum(dim=-1)
#     response_lengths = response_mask.sum(dim=-1).clamp(min=1)
#     return response_log_probs / response_lengths

def compute_logprobs(logits, labels, selection_mask):
    shifted_labels = labels[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    logprobs = F.log_softmax(shifted_logits, dim=-1)
    selected_log_probs = torch.gather(
      input=logprobs,
      dim=-1,
      index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)
    shifted_mask = selection_mask[:, 1:].clone()
    masked_selected_log_probs = shifted_mask * selected_log_probs
    return masked_selected_log_probs.sum(-1) / (shifted_mask.sum(-1)+1e-6)


def wrap_with_fsdp(model):
    mixed_precision = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={PhiDecoderLayer})

    with enable_wrap(auto_wrap_policy=auto_wrap_policy, wrapper_cls=FSDP, mixed_precision=mixed_precision):
        for name, module in model.named_children():
            model.add_module(name, wrap(module))

    return model.cuda()


# def collate_fn(batch, tokenizer, max_length):
#     prompt_texts = ['<|system|>\n' + item['system']+ '<|user|>\n' +item['prompt'] + '\n' for item in batch]
#     chosen_texts = ['<|assistant|>\n' + item['chosen'] for item in batch]
#     rejected_texts = ['<|assistant|>\n' + item['rejected'] for item in batch]

#     chosen_inputs = tokenizer(
#         [p + c for p, c in zip(prompt_texts, chosen_texts)],
#         padding='max_length', truncation=True, max_length=max_length, return_tensors='pt'
#     )
#     rejected_inputs = tokenizer(
#         [p + r for p, r in zip(prompt_texts, rejected_texts)],
#         padding='max_length', truncation=True, max_length=max_length, return_tensors='pt'
#     )

#     # Get where prompt ends
#     prompt_only_inputs = tokenizer(
#         prompt_texts,
#         padding='max_length', truncation=True, max_length=max_length, return_tensors='pt'
#     )
#     prompt_lengths = prompt_only_inputs.attention_mask.sum(dim=-1)  # [B]

#     return {
#         'prompt_preferred_ids': chosen_inputs.input_ids,
#         'prompt_dispreferred_ids': rejected_inputs.input_ids,
#         'prompt_preferred_mask': chosen_inputs.attention_mask,
#         'prompt_dispreferred_mask': rejected_inputs.attention_mask,
#         'prompt_lengths': prompt_lengths
#     }

def dpo_collate_fn(batch, tokenizer, max_length):
    pad_token_id = tokenizer.pad_token_id
    prompts = [x['prompt'] for x in batch]
    chosens = [x['chosen'] for x in batch]
    rejecteds = [x['rejected'] for x in batch]
    
    chosen_prompts = [f'{q}\n{a}' for q, a in zip(prompts, chosens)]
    rejecteds_prompts = [f'{q}\n{a}' for q, a in zip(prompts, rejecteds)]
    
    
    
    prompts_tokenized = tokenizer(prompts, truncation=True, max_length=max_length)['input_ids']
    chosen_prompts_tokenized = tokenizer(chosen_prompts, truncation=True, max_length=max_length)['input_ids']
    rejected_prompts_tokenized = tokenizer(rejecteds_prompts, truncation=True, max_length=max_length)['input_ids']
    
    max_seq_length = max(max([len(x) for x in chosen_prompts_tokenized]), max([len(x) for x in rejected_prompts_tokenized]))
    
    batch_tokenized = {
      # 'prompt': [],
      # 'prompt_mask': [],
      'chosen': [],
      'rejected': [],
      'chosen_mask': [],
      'chosen_padding_mask': [],
      'rejected_mask': [],
      'rejected_padding_mask': [],
    }
    
    for question, chosen, rejected in zip(prompts_tokenized, chosen_prompts_tokenized , rejected_prompts_tokenized):
        question_len = len(question)
        # question_padded = [pad_token_id] *  (max_seq_length - question_len) + question
        # question_mask = [1] * len(question_padded)
        # question_mask[:(max_seq_length - question_len)] = [0] * (max_seq_length - question_len)
        # batch_tokenized['prompt'].append(question_padded)
        # batch_tokenized['prompt_mask'].append(question_mask)
        
        chosen_pad_length = max_seq_length - len(chosen)
        chosen_padded = [pad_token_id] *  chosen_pad_length + chosen
        chosen_mask = [1] * len(chosen_padded)
        chosen_mask[:(chosen_pad_length+question_len)] = [0] * (chosen_pad_length+question_len)
        chosen_padding_mask = [1] * len(chosen_padded)
        chosen_padding_mask[:(chosen_pad_length)] = [0] * (chosen_pad_length)
        batch_tokenized['chosen'].append(chosen_padded)
        batch_tokenized['chosen_mask'].append(chosen_mask)
        batch_tokenized['chosen_padding_mask'].append(chosen_padding_mask)
        
        
        rejected_pad_length = max_seq_length - len(rejected)
        rejected_padded = [pad_token_id] *  rejected_pad_length + rejected
        rejected_mask = [1] * len(rejected_padded)
        rejected_mask[:(rejected_pad_length+question_len)] = [0] * (rejected_pad_length+question_len)
        rejected_padding_mask = [1] * len(rejected_padded)
        rejected_padding_mask[:(rejected_pad_length)] = [0] * (rejected_pad_length)
        batch_tokenized['rejected'].append(rejected_padded)
        batch_tokenized['rejected_mask'].append(rejected_mask)
        batch_tokenized['rejected_padding_mask'].append(rejected_padding_mask)
    
    # for k in batch_tokenized.keys():
    #   print(f"{k}: {[len(x) for x in batch_tokenized[k]]}")
    
    # return {k: torch.tensor(v) if '_mask' not in k else torch.tensor(v, dtype=bool) for k, v in batch_tokenized.items()}
    return {k: torch.tensor(v) for k, v in batch_tokenized.items()}



def evaluate(model, ref_model, dataloader, local_rank, global_rank, beta, args):
    model.eval()
    # print(f"[EVAL ENTRY] global_rank={global_rank}, local_rank={local_rank}, has {len(dataloader)} eval batches", flush=True)
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_preferred_logits = model(batch['chosen'], attention_mask=batch['chosen_padding_mask']).logits
                
                model_dispreferred_logits = model(batch['rejected'], attention_mask=batch['rejected_padding_mask']).logits

            # model_preferred_logprob = get_log_prob(model_preferred_logits, batch['prompt_preferred_ids'], batch['prompt_lengths'])
            model_preferred_logprob = compute_logprobs(model_preferred_logits, batch['chosen'], batch['chosen_mask'])
            # model_dispreferred_logprob = get_log_prob(model_dispreferred_logits, batch['prompt_dispreferred_ids'], batch['prompt_lengths'])
            model_dispreferred_logprob = compute_logprobs(model_dispreferred_logits, batch['rejected'], batch['rejected_mask'])

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                ref_preferred_logits = ref_model(batch['chosen'], attention_mask=batch['chosen_padding_mask']).logits
                ref_dispreferred_logits = ref_model(batch['rejected'], attention_mask=batch['rejected_padding_mask']).logits

            # ref_preferred_logprob = get_log_prob(ref_preferred_logits, batch['prompt_preferred_ids'], batch['prompt_lengths'])
            ref_preferred_logprob = compute_logprobs(ref_preferred_logits, batch['chosen'], batch['chosen_mask'])
            # ref_dispreferred_logprob = get_log_prob(ref_dispreferred_logits, batch['prompt_dispreferred_ids'], batch['prompt_lengths'])
            ref_dispreferred_logprob = compute_logprobs(ref_dispreferred_logits, batch['rejected'], batch['rejected_mask'])

            loss, preferred_relative_logprob, dispreferred_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(
                model_preferred_logprob, model_dispreferred_logprob,
                ref_preferred_logprob, ref_dispreferred_logprob,
                beta=beta)

            for i in range(batch['chosen'].size(0)):
                results.append([
                    loss.item(),
                    preferred_relative_logprob.item(),
                    dispreferred_relative_logprob.item(),
                    reward_accuracies.item(),
                    reward_margins.item(),
                ])

    # Gather from all ranks
    if results:
        local_results = torch.tensor(results, dtype=torch.float32, device=torch.cuda.current_device())
    else:
        local_results = torch.zeros((1, 5), dtype=torch.float32, device=torch.cuda.current_device())  # dummy row
    
    results_obj = results
    gathered_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_results, results_obj)
    
    flat = sum(gathered_results, [])
    all_results = torch.tensor(flat, dtype=torch.float32, device=torch.cuda.current_device())
    
    # (OPTIONAL) Remove dummy rows if needed
    all_results = all_results[~torch.all(all_results == 0, dim=1)]

    if global_rank == 0:
        # print("[global_rank 0] inside wandb logging block", flush=True)
        try:
            avg = all_results.mean(dim=0).tolist()
            log_D = {
                'eval/loss': avg[0],
                'eval/preferred_relative_logprob': avg[1],
                'eval/dispreferred_relative_logprob': avg[2],
                'eval/reward_accuracy': avg[3],
                'eval/reward_margin': avg[4],
            }
            # print("[global_rank 0] log dict:", log_D, flush=True)
            if args.wandb_enable:
                wandb.log(log_D)
            else:
                print(f"Eval:\n {log_D}")
            # print("[global_rank 0] finished wandb.log", flush=True)
        except Exception as e:
            print(f"[global_rank 0] wandb.log failed: {e}", flush=True)

    model.train()




def train(model, ref_model, tokenizer, optimizer, train_loader, eval_loader, local_rank, global_rank, args, epochs=1, beta=0.1):
    model.train()
    model.gradient_checkpointing_enable()
    ref_model.eval()

    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            optimizer.zero_grad()
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_preferred_logits = model(batch['chosen'], attention_mask=batch['chosen_padding_mask']).logits
                model_dispreferred_logits = model(batch['rejected'], attention_mask=batch['rejected_padding_mask']).logits

            # model_preferred_logprob = get_log_prob(model_preferred_logits, batch['prompt_preferred_ids'], batch['prompt_lengths'])
            model_preferred_logprob = compute_logprobs(model_preferred_logits, batch['chosen'], batch['chosen_mask'])
            # model_dispreferred_logprob = get_log_prob(model_dispreferred_logits, batch['prompt_dispreferred_ids'], batch['prompt_lengths'])
            model_dispreferred_logprob = compute_logprobs(model_dispreferred_logits, batch['rejected'], batch['rejected_mask'])

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16): 
                    ref_preferred_logits = ref_model(batch['chosen'], attention_mask=batch['chosen_padding_mask']).logits
                    ref_dispreferred_logits = ref_model(batch['rejected'], attention_mask=batch['rejected_padding_mask']).logits
                    
                # ref_preferred_logprob = get_log_prob(ref_preferred_logits, batch['prompt_preferred_ids'], batch['prompt_lengths'])
                ref_preferred_logprob = compute_logprobs(ref_preferred_logits, batch['chosen'], batch['chosen_mask'])
                # ref_dispreferred_logprob = get_log_prob(ref_dispreferred_logits, batch['prompt_dispreferred_ids'], batch['prompt_lengths'])
                ref_dispreferred_logprob = compute_logprobs(ref_dispreferred_logits, batch['rejected'], batch['rejected_mask'])

            loss, preferred_relative_logprob, dispreferred_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(
                model_preferred_logprob, model_dispreferred_logprob,
                ref_preferred_logprob, ref_dispreferred_logprob,
                beta=beta)

            loss.backward()
            optimizer.step()

            if step % 20 == 0 and global_rank == 0:
                log_D = {
                        'loss': loss.item(),
                        'preferred_relative_logprob': preferred_relative_logprob.item(),
                        'dispreferred_relative_logprob': dispreferred_relative_logprob.item(),
                        'reward_accuracy': reward_accuracies.item(),
                        'reward_margin': reward_margins.item()
                    }
                if args.wandb_enable:
                    wandb.log(log_D)
                else:
                    print(f"train epoch: {epoch}, step: {step}\n {log_D}")
                

        evaluate(model, ref_model, eval_loader, local_rank, global_rank, beta, args)
        # print(f"[rank {global_rank}] finished evaluate()", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dataset_name", type=str, default="jondurbin/truthy-dpo-v0.1")
    parser.add_argument("--wandb_project", type=str, default="truthy-dpo")
    parser.add_argument("--wandb_enable", type=bool, default=False)
    args = parser.parse_args()

    seed_everything(args.seed)
    init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    if global_rank == 0 and args.wandb_enable:
        wandb.login()
        wandb.init(project=args.wandb_project, config=args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    model = wrap_with_fsdp(base_model)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).cuda()
    ref_model.eval()

    dataset = load_dataset(args.dataset_name, split="train")
    if global_rank == 0:
        dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
        dataset.save_to_disk("cached_split")
    dist.barrier()
    dataset = DatasetDict.load_from_disk("cached_split")

    train_dataset = dataset["train"].shard(num_shards=dist.get_world_size(), index=local_rank)
    eval_dataset = dataset["test"].shard(num_shards=dist.get_world_size(), index=local_rank)
    


    collate = partial(dpo_collate_fn, tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    if len(eval_dataset) == 0:
        eval_loader = DataLoader([], batch_size=8)
    else:
        eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=collate)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    train(model, ref_model, tokenizer, optimizer, train_loader, eval_loader, local_rank, global_rank, args, epochs=args.epochs, beta=args.beta)

    if dist.get_rank() == 0:
        print("[rank 0] preparing to save model...", flush=True)

    # Gather the full state dict across all ranks
    state_dict = model.state_dict()
    dist.barrier()
    
    # Save only on rank 0
    if dist.get_rank() == 0:
        torch.save(state_dict, "fsdp_model_checkpoint.pt")
        print("[rank 0] checkpoint saved!", flush=True)
    
    dist.barrier()  # optional: sync all ranks before continuing

    # if global_rank == 0:
    #     print("Saving FSDP model...", flush=True)
    
    # # Gather full state dict on rank 0 only
    # with FSDP.state_dict_type(
    #     model,
    #     state_dict_type="full",
    #     state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    # ):
    #     full_state = model.state_dict()
    
    # # Now save only on rank 0
    # if global_rank == 0:
    #     torch.save(full_state, "fsdp_model_checkpoint.pt")
    #     print("Checkpoint saved.", flush=True)


if __name__ == "__main__":
    main()
