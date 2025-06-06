#!/usr/bin/env sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=1 src/train_fsdp-edited-mh.py\
    --epochs 10 \
    --batch_size 2 \
    --max_length 512 \
    --lr 1e-5 \
    --betas 0.1 0.01 \
    --seed 2003 \
    --model_name "microsoft/phi-1_5" \
    --dataset_name "jondurbin/truthy-dpo-v0.1" \
    --wandb_project "truthy-dpo" \
    --wandb_enable True
