import argparse
import random
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
# from torch.distributed.fsdp.sharding_strategy import ShardingStrategy
import torch.distributed as dist

import wandb
from tqdm import tqdm

import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from transformers.models.phi.modeling_phi import PhiDecoderLayer
from torch.distributed.fsdp.wrap import enable_wrap, wrap


def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


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

def collate_fn(batch, tokenizer, max_length, device):
    prompt_encodings = tokenizer(
        ['Instruct: ' + item['prompt'] + '\n' for item in batch],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    chosen_encodings = tokenizer(
        ['Output: ' + item['chosen'] for item in batch],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    rejected_encodings = tokenizer(
        ['Output: ' + item['rejected'] for item in batch],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    prompt_preferred_ids = torch.cat([
        prompt_encodings.input_ids,
        chosen_encodings.input_ids
    ], dim=-1).to(device)
    
    prompt_dispreferred_ids = torch.cat([
        prompt_encodings.input_ids,
        rejected_encodings.input_ids
    ], dim=-1).to(device)

    prompt_preferred_mask = torch.cat([
        prompt_encodings.attention_mask,
        chosen_encodings.attention_mask
    ], dim=-1).to(device)
    
    prompt_dispreferred_mask = torch.cat([
        prompt_encodings.attention_mask,
        rejected_encodings.attention_mask
    ], dim=-1).to(device)

    prompt_lengths = prompt_encodings.attention_mask.sum(dim=-1)

    return {
        'prompt_preferred_ids': prompt_preferred_ids,
        'prompt_dispreferred_ids': prompt_dispreferred_ids,
        'prompt_preferred_mask': prompt_preferred_mask,
        'prompt_dispreferred_mask': prompt_dispreferred_mask,
        'prompt_lengths': prompt_lengths
    }




# def wrap_with_fsdp(model):
#     mixed_precision = MixedPrecision(
#         param_dtype=torch.float32,
#         reduce_dtype=torch.bfloat16,
#         buffer_dtype=torch.bfloat16,
#     )

#     auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={PhiDecoderLayer})

#     return FSDP(
#         model,
#         auto_wrap_policy=auto_wrap_policy,
#         mixed_precision=mixed_precision,
#         device_id=torch.cuda.current_device()
#     )

def wrap_with_fsdp(model):
    mixed_precision = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={PhiDecoderLayer})

    # Enable FSDP wrapping + checkpointing
    with enable_wrap(auto_wrap_policy=auto_wrap_policy, wrapper_cls=FSDP, mixed_precision=mixed_precision):
        for name, module in model.named_children():
            model.add_module(name, wrap(module))  # recursively wrap

    return model.cuda()



def train(model, ref_model, tokenizer, optimizer, train_dataloader, local_rank, epochs=1, beta=0.1):
    model.train()
    model.gradient_checkpointing_enable()
    ref_model.eval()

    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            optimizer.zero_grad()
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model_preferred_logits = model(
                    input_ids=batch['prompt_preferred_ids'],
                    attention_mask=batch['prompt_preferred_mask']
                ).logits
                model_dispreferred_logits = model(
                    input_ids=batch['prompt_dispreferred_ids'],
                    attention_mask=batch['prompt_dispreferred_mask']
                ).logits

            model_preferred_logprob = get_log_prob(
                model_preferred_logits,
                batch['prompt_preferred_ids'],
                batch['prompt_lengths']
            )

            model_dispreferred_logprob = get_log_prob(
                model_dispreferred_logits,
                batch['prompt_dispreferred_ids'],
                batch['prompt_lengths']
            )

            with torch.no_grad():
                ref_preferred_logits = ref_model(
                    input_ids=batch['prompt_preferred_ids'],
                    attention_mask=batch['prompt_preferred_mask']
                ).logits
                ref_dispreferred_logits = ref_model(
                    input_ids=batch['prompt_dispreferred_ids'],
                    attention_mask=batch['prompt_dispreferred_mask']
                ).logits

                ref_preferred_logprob = get_log_prob(
                    ref_preferred_logits,
                    batch['prompt_preferred_ids'],
                    batch['prompt_lengths']
                )
                ref_dispreferred_logprob = get_log_prob(
                    ref_dispreferred_logits,
                    batch['prompt_dispreferred_ids'],
                    batch['prompt_lengths']
                )

            loss, preferred_relative_logprob, dispreferred_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(
                model_preferred_logprob,
                model_dispreferred_logprob,
                ref_preferred_logprob,
                ref_dispreferred_logprob,
                beta=beta
            )

            loss.backward()
            optimizer.step()
            if step%10 == 0 and local_rank == 0:
                log_D = {
                    'loss': loss.item(),
                    'preferred_relative_logprob': preferred_relative_logprob.item(),
                    'dispreferred_relative_logprob': dispreferred_relative_logprob.item(),
                    'reward_accuracy': reward_accuracies.item(),
                    'reward_margin': reward_margins.item(),
                    'step': step,
                }
                wandb.log(log_D)
                
                


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

    wandb.login()
    wandb.init(project=args.wandb_project, config=args)
    
    init_distributed()
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    base_model.gradient_checkpointing_enable()
    model = wrap_with_fsdp(base_model)

    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).cuda()
    ref_model.eval()

    dataset = load_dataset(args.dataset_name, split="train")

    def collate_fn(examples):
        chosen = [ex["chosen"] for ex in examples]
        rejected = [ex["rejected"] for ex in examples]
        chosen_enc = tokenizer(chosen, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
        rejected_enc = tokenizer(rejected, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
        return {
            "prompt_preferred_ids": chosen_enc["input_ids"],
            "prompt_preferred_mask": chosen_enc["attention_mask"],
            "prompt_dispreferred_ids": rejected_enc["input_ids"],
            "prompt_dispreferred_mask": rejected_enc["attention_mask"],
            "prompt_lengths": torch.sum(chosen_enc["attention_mask"], dim=1),
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    train(model, ref_model, tokenizer, optimizer, dataloader, local_rank, epochs=args.epochs, beta=args.beta)

    if local_rank == 0:
        save_path = "fsdp_model_checkpoint.pt"
        state_dict = model.state_dict()
        torch.save(state_dict, save_path)
        print(f"Saved FSDP model checkpoint to {save_path}")


if __name__ == "__main__":
    main()
