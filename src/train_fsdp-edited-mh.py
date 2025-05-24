import argparse
import random
import numpy as np
import os
from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
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
import torch.distributed.tensor.parallel as tp
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from functools import partial

# from torch.distributed.fsdp import state_dict_type, FullStateDictConfig

import wandb


import torch.distributed as dist
from collections import defaultdict
from collections.abc import Iterable



def gather_and_aggregate_results(local_results, metric_keys):
    """
    Gathers and aggregates per-example results (list of dicts) across all ranks.
    Supports both scalar and list/array values for each metric key.

    Args:
        local_results (List[Dict[str, Union[float, List[float]]]]): Local list of metric dicts.
        metric_keys (List[str]): Metric keys to aggregate.

    Returns:
        Dict[str, List[float]]: Aggregated values across all ranks per key.
    """
    if not local_results:
        local_results = [{k: 0.0 for k in metric_keys}]

    gathered_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_results, local_results)

    # Flatten gathered list of lists
    all_results = sum(gathered_results, [])

    # Filter out dummy rows (all 0.0s)
    def is_dummy(d):
        return all(
            (isinstance(v, Iterable) and not isinstance(v, str) and all(x == 0.0 for x in v))
            or (not isinstance(v, Iterable) and v == 0.0)
            for v in d.values()
        )
    all_results = [r for r in all_results if not is_dummy(r)]

    # Aggregate
    aggregated = defaultdict(list)
    for res in all_results:
        for k in metric_keys:
            v = res[k]
            if isinstance(v, Iterable) and not isinstance(v, str):
                aggregated[k].extend(v)
            else:
                aggregated[k].append(v)

    return dict(aggregated)


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

    reward = beta * (preferred_relative_logprob - dispreferred_relative_logprob)

    return loss, preferred_relative_logprob.mean(), dispreferred_relative_logprob.mean(), reward_accuracies, reward_margins, reward


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


# def wrap_with_fsdp(model):
#     mixed_precision = MixedPrecision(
#         param_dtype=torch.float32,
#         reduce_dtype=torch.bfloat16,
#         buffer_dtype=torch.bfloat16,
#     )

#     auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={PhiDecoderLayer})

#     with enable_wrap(auto_wrap_policy=auto_wrap_policy, wrapper_cls=FSDP, mixed_precision=mixed_precision):
#         for name, module in model.named_children():
#             model.add_module(name, wrap(module))

#     return model.cuda()

def is_fully_sharded(model):
    return any(isinstance(m, FSDP) for m in model.modules())

def wrap_with_fsdp(model, local_rank=0):
    if is_fully_sharded(model):
        print("[FSDP] Model already wrapped — skipping rewrapping.")
        return model.cuda()

    print("[FSDP] Wrapping model with FSDP...")

    mixed_precision = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # ✅ No auto_wrap_policy because some children are already wrapped
    wrapped_model = FSDP(model,
                         mixed_precision=mixed_precision,
                         device_id=local_rank)

    return wrapped_model


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



def evaluate(model, ref_model, dataloader, local_rank, global_rank, betas, args):
    model.eval()
    # print(f"[EVAL ENTRY] global_rank={global_rank}, local_rank={local_rank}, has {len(dataloader)} eval batches", flush=True)
    results = []
    

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_preferred_logits = model(batch['chosen'], attention_mask=batch['chosen_padding_mask'])
                model_dispreferred_logits = model(batch['rejected'], attention_mask=batch['rejected_padding_mask'])

            model_preferred_logprob = []
            model_dispreferred_logprob = []
            for i, beta in enumerate(betas):
                # print(f"model_preferred_logits[i]: {model_preferred_logits[i].shape}")
                model_preferred_logprob.append(compute_logprobs(model_preferred_logits[i], batch['chosen'], batch['chosen_mask']))
                model_dispreferred_logprob.append(compute_logprobs(model_dispreferred_logits[i], batch['rejected'], batch['rejected_mask']))

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                ref_preferred_logits = ref_model(batch['chosen'], attention_mask=batch['chosen_padding_mask']).logits
                ref_dispreferred_logits = ref_model(batch['rejected'], attention_mask=batch['rejected_padding_mask']).logits

            
            ref_preferred_logprob = compute_logprobs(ref_preferred_logits, batch['chosen'], batch['chosen_mask'])
            ref_dispreferred_logprob = compute_logprobs(ref_dispreferred_logits, batch['rejected'], batch['rejected_mask'])


            losses = torch.zeros(len(betas)).cuda()
            rewards = []
            preferred_relative_logprobs = torch.zeros(len(betas)).cuda()
            dispreferred_relative_logprobs = torch.zeros(len(betas)).cuda()
            reward_accuracies = torch.zeros(len(betas)).cuda()
            reward_margins = torch.zeros(len(betas)).cuda()
            rewards = []
            
            for i, beta in enumerate(betas):
                losses[i], preferred_relative_logprobs[i], dispreferred_relative_logprobs[i], reward_accuracies[i], reward_margins[i], reward = calculate_DPO_loss(
                model_preferred_logprob[i], model_dispreferred_logprob[i],
                ref_preferred_logprob, ref_dispreferred_logprob,
                beta=beta)
                rewards.append(reward)
        
            total_loss = losses.mean()
            rewards_tensor = torch.stack(rewards, dim=0)  # shape: (n, B)
            
            with torch.no_grad():
                mean_rewards = rewards_tensor.mean(dim=0)  # shape: (B,)
            regularizer = ((rewards_tensor - mean_rewards)**2).mean()
            
            total_loss += 1.0 * regularizer

            for i in range(batch['chosen'].size(0)):
                log_D = {
                        'eval/total loss': total_loss.item(),
                        'eval/regularizer': regularizer.item()
                    }
                for i, beta in enumerate(betas):
                    log_D[f"eval/loss[{beta}]"] = losses[i].item()
                    log_D[f"eval/preferred_relative_logprob[{beta}]"] = preferred_relative_logprobs[i].item()
                    log_D[f"eval/dispreferred_relative_logprob[{beta}]"] = preferred_relative_logprobs[i].item()
                    log_D[f"eval/reward_accuracy[{beta}]"] = reward_accuracies[i].item()
                    log_D[f"eval/reward_margin[{beta}]"] = reward_margins[i].item()
                results.append(log_D)


    metrics = ['eval/total loss', 'eval/regularizer']
    for beta in betas:
        metrics = metrics + [f"eval/loss[{beta}]", f"eval/preferred_relative_logprob[{beta}]", f"eval/dispreferred_relative_logprob[{beta}]", f"eval/reward_accuracy[{beta}]", f"eval/reward_margin[{beta}]"]

    aggregated_results = gather_and_aggregate_results(results, metrics)

    for key in aggregated_results.keys():
        aggregated_results[key] = np.mean(aggregated_results[key]).item()

    if global_rank == 0:
        try:
            if args.wandb_enable:
                wandb.log(aggregated_results)
            else:
                print(f"Eval:\n {aggregated_results}")
            # print("[global_rank 0] finished wandb.log", flush=True)
        except Exception as e:
            print(f"[global_rank 0] wandb.log failed: {e}", flush=True)

    model.train()




def train(model, ref_model, tokenizer, optimizer, train_loader, eval_loader, local_rank, global_rank, args, epochs=1, betas=[0.1]):
    model.train()
    model.gradient_checkpointing_enable()
    ref_model.eval()

    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            optimizer.zero_grad()
            batch = {k: v.cuda() for k, v in batch.items()}

            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_preferred_logits = model(batch['chosen'], attention_mask=batch['chosen_padding_mask'])
                model_dispreferred_logits = model(batch['rejected'], attention_mask=batch['rejected_padding_mask'])
                    
            model_preferred_logprob = []
            model_dispreferred_logprob = []
            for i, beta in enumerate(betas):
                # print(f"model_preferred_logits[i]: {model_preferred_logits[i].shape}")
                model_preferred_logprob.append(compute_logprobs(model_preferred_logits[i], batch['chosen'], batch['chosen_mask']))
                model_dispreferred_logprob.append(compute_logprobs(model_dispreferred_logits[i], batch['rejected'], batch['rejected_mask']))

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16): 
                    ref_preferred_logits = ref_model(batch['chosen'], attention_mask=batch['chosen_padding_mask']).logits
                    ref_dispreferred_logits = ref_model(batch['rejected'], attention_mask=batch['rejected_padding_mask']).logits
                    
                ref_preferred_logprob = compute_logprobs(ref_preferred_logits, batch['chosen'], batch['chosen_mask'])
                ref_dispreferred_logprob = compute_logprobs(ref_dispreferred_logits, batch['rejected'], batch['rejected_mask'])
                
            
            losses = torch.zeros(len(betas)).cuda()
            rewards = []
            preferred_relative_logprobs = torch.zeros(len(betas)).cuda()
            dispreferred_relative_logprobs = torch.zeros(len(betas)).cuda()
            reward_accuracies = torch.zeros(len(betas)).cuda()
            reward_margins = torch.zeros(len(betas)).cuda()
            rewards = []
            for i, beta in enumerate(betas):
                losses[i], preferred_relative_logprobs[i], dispreferred_relative_logprobs[i], reward_accuracies[i], reward_margins[i], reward = calculate_DPO_loss(
                model_preferred_logprob[i], model_dispreferred_logprob[i],
                ref_preferred_logprob, ref_dispreferred_logprob,
                beta=beta)
                rewards.append(reward)
                

                
            total_loss = losses.mean()


            # Stack into a tensor of shape (n, B)
            rewards_tensor = torch.stack(rewards, dim=0)  # shape: (n, B)
            
            # Compute mean across n
            with torch.no_grad():
                mean_rewards = rewards_tensor.mean(dim=0)  # shape: (B,)
            
            # Compute squared differences and average
            regularizer = ((rewards_tensor - mean_rewards)**2).mean()
            
            total_loss += 1.0 * regularizer
            
            total_loss.backward()
            optimizer.step()

            if step % 20 == 0 and global_rank == 0:
                log_D = {
                        'total loss': total_loss.item(),
                        'regularizer': regularizer.item()
                    }
                for i, beta in enumerate(betas):
                    log_D[f"loss[{beta}]"] = losses[i].item()
                    log_D[f"preferred_relative_logprob[{beta}]"] = preferred_relative_logprobs[i].item()
                    log_D[f"dispreferred_relative_logprob[{beta}]"] = preferred_relative_logprobs[i].item()
                    log_D[f"reward_accuracy[{beta}]"] = reward_accuracies[i].item()
                    log_D[f"reward_margin[{beta}]"] = reward_margins[i].item()
                if args.wandb_enable:
                    wandb.log(log_D)
                else:
                    print(f"train epoch: {epoch}, step: {step}\n {log_D}")
                

        evaluate(model, ref_model, eval_loader, local_rank, global_rank, betas, args)
        # print(f"[rank {global_rank}] finished evaluate()", flush=True)


class MultiHeadCausalLM(nn.Module):
    def __init__(self, model_name: str, num_heads: int = 2, dtype=torch.bfloat16):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self.base_transformer = self.base_model.model

        self.hidden_dim = self.base_model.config.hidden_size
        self.vocab_size = self.base_model.config.vocab_size
        self.num_heads = num_heads

        # ✅ Define heads first
        self.heads = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.vocab_size).to(dtype)
            for _ in range(self.num_heads)
        ])

        # Now copy from base lm_head
        weight = self.base_model.lm_head.weight
        bias = self.base_model.lm_head.bias

        if isinstance(weight, torch.distributed.tensor.DTensor):
            weight = weight.to_local()
        if bias is not None and isinstance(bias, torch.distributed.tensor.DTensor):
            bias = bias.to_local()

        base_weight = weight.detach().clone()
        base_bias = bias.detach().clone() if bias is not None else None

        for head in self.heads:
            with torch.no_grad():
                head.weight.copy_(base_weight)
                if head.bias is not None and base_bias is not None:
                    head.bias.copy_(base_bias)


    def gradient_checkpointing_enable(self):
        self.base_model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask=None):
        # Get hidden states (B, T, H)
        outputs = self.base_transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # shape: (B, T, H)

        # Compute logits for each head: shape (H, B, T, V)
        all_logits = torch.stack([head(hidden_states) for head in self.heads], dim=0)
        return all_logits  # shape: (num_heads, B, T, vocab_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--betas", type=float, nargs='+', default=[0.1, 0.01], help="List of beta values")
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

    num_heads = len(args.betas)
    # base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    base_model = MultiHeadCausalLM(args.model_name, num_heads = num_heads, dtype=torch.bfloat16)
    model = wrap_with_fsdp(base_model)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).cuda()
    ref_model.eval()

    dataset = load_dataset(args.dataset_name, split="train")
    if global_rank == 0:
        dataset = dataset.train_test_split(test_size=0.8, seed=args.seed)
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
    train(model, ref_model, tokenizer, optimizer, train_loader, eval_loader, local_rank, global_rank, args, epochs=args.epochs, betas=args.betas)

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
