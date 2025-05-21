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


def get_log_prob(logits, labels, prompt_lengths):
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

    batch_size, seq_len = labels.shape
    response_mask = torch.arange(seq_len, device=labels.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
    response_mask = response_mask.float()

    response_log_probs = (token_log_probs * response_mask).sum(dim=-1)
    response_lengths = response_mask.sum(dim=-1).clamp(min=1)
    return response_log_probs / response_lengths


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


def collate_fn(batch, tokenizer, max_length):
    prompt_encodings = tokenizer(
        ['Instruct: ' + item['prompt'] + '\n' for item in batch],
        padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    chosen_encodings = tokenizer(
        ['Output: ' + item['chosen'] for item in batch],
        padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    rejected_encodings = tokenizer(
        ['Output: ' + item['rejected'] for item in batch],
        padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

    prompt_preferred_ids = torch.cat([prompt_encodings.input_ids, chosen_encodings.input_ids], dim=-1)
    prompt_dispreferred_ids = torch.cat([prompt_encodings.input_ids, rejected_encodings.input_ids], dim=-1)
    prompt_preferred_mask = torch.cat([prompt_encodings.attention_mask, chosen_encodings.attention_mask], dim=-1)
    prompt_dispreferred_mask = torch.cat([prompt_encodings.attention_mask, rejected_encodings.attention_mask], dim=-1)
    prompt_lengths = prompt_encodings.attention_mask.sum(dim=-1)

    return {
        'prompt_preferred_ids': prompt_preferred_ids,
        'prompt_dispreferred_ids': prompt_dispreferred_ids,
        'prompt_preferred_mask': prompt_preferred_mask,
        'prompt_dispreferred_mask': prompt_dispreferred_mask,
        'prompt_lengths': prompt_lengths
    }


def evaluate(model, ref_model, dataloader, local_rank, global_rank, beta):
    model.eval()
    print(f"[EVAL ENTRY] global_rank={global_rank}, local_rank={local_rank}, has {len(dataloader)} eval batches", flush=True)
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_preferred_logits = model(
                    input_ids=batch['prompt_preferred_ids'], attention_mask=batch['prompt_preferred_mask']).logits
                model_dispreferred_logits = model(
                    input_ids=batch['prompt_dispreferred_ids'], attention_mask=batch['prompt_dispreferred_mask']).logits

            model_preferred_logprob = get_log_prob(model_preferred_logits, batch['prompt_preferred_ids'], batch['prompt_lengths'])
            model_dispreferred_logprob = get_log_prob(model_dispreferred_logits, batch['prompt_dispreferred_ids'], batch['prompt_lengths'])

            ref_preferred_logits = ref_model(
                input_ids=batch['prompt_preferred_ids'], attention_mask=batch['prompt_preferred_mask']).logits
            ref_dispreferred_logits = ref_model(
                input_ids=batch['prompt_dispreferred_ids'], attention_mask=batch['prompt_dispreferred_mask']).logits

            ref_preferred_logprob = get_log_prob(ref_preferred_logits, batch['prompt_preferred_ids'], batch['prompt_lengths'])
            ref_dispreferred_logprob = get_log_prob(ref_dispreferred_logits, batch['prompt_dispreferred_ids'], batch['prompt_lengths'])

            loss, preferred_relative_logprob, dispreferred_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(
                model_preferred_logprob, model_dispreferred_logprob,
                ref_preferred_logprob, ref_dispreferred_logprob,
                beta=beta)

            for i in range(batch['prompt_preferred_ids'].size(0)):
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
            wandb.log(log_D)
            # print("[global_rank 0] finished wandb.log", flush=True)
        except Exception as e:
            print(f"[global_rank 0] wandb.log failed: {e}", flush=True)

    model.train()




def train(model, ref_model, tokenizer, optimizer, train_loader, eval_loader, local_rank, global_rank, epochs=1, beta=0.1):
    model.train()
    model.gradient_checkpointing_enable()
    ref_model.eval()

    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            optimizer.zero_grad()
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_preferred_logits = model(input_ids=batch['prompt_preferred_ids'], attention_mask=batch['prompt_preferred_mask']).logits
                model_dispreferred_logits = model(input_ids=batch['prompt_dispreferred_ids'], attention_mask=batch['prompt_dispreferred_mask']).logits

            model_preferred_logprob = get_log_prob(model_preferred_logits, batch['prompt_preferred_ids'], batch['prompt_lengths'])
            model_dispreferred_logprob = get_log_prob(model_dispreferred_logits, batch['prompt_dispreferred_ids'], batch['prompt_lengths'])

            with torch.no_grad():
                ref_preferred_logits = ref_model(input_ids=batch['prompt_preferred_ids'], attention_mask=batch['prompt_preferred_mask']).logits
                ref_dispreferred_logits = ref_model(input_ids=batch['prompt_dispreferred_ids'], attention_mask=batch['prompt_dispreferred_mask']).logits
                ref_preferred_logprob = get_log_prob(ref_preferred_logits, batch['prompt_preferred_ids'], batch['prompt_lengths'])
                ref_dispreferred_logprob = get_log_prob(ref_dispreferred_logits, batch['prompt_dispreferred_ids'], batch['prompt_lengths'])

            loss, preferred_relative_logprob, dispreferred_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(
                model_preferred_logprob, model_dispreferred_logprob,
                ref_preferred_logprob, ref_dispreferred_logprob,
                beta=beta)

            loss.backward()
            optimizer.step()

            if step % 20 == 0 and global_rank == 0:
                wandb.log({
                    'loss': loss.item(),
                    'preferred_relative_logprob': preferred_relative_logprob.item(),
                    'dispreferred_relative_logprob': dispreferred_relative_logprob.item(),
                    'reward_accuracy': reward_accuracies.item(),
                    'reward_margin': reward_margins.item()
                })

        evaluate(model, ref_model, eval_loader, local_rank, global_rank, beta)
        print(f"[rank {global_rank}] finished evaluate()", flush=True)


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
    args = parser.parse_args()

    seed_everything(args.seed)
    init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    if global_rank == 0:
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
    


    collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    if len(eval_dataset) == 0:
        eval_loader = DataLoader([], batch_size=args.batch_size)
    else:
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    train(model, ref_model, tokenizer, optimizer, train_loader, eval_loader, local_rank, global_rank, epochs=args.epochs, beta=args.beta)

    if global_rank == 0:
        print("rank 0 is saving the checkpoint")
        torch.save(model.state_dict(), "fsdp_model_checkpoint.pt")


if __name__ == "__main__":
    main()
