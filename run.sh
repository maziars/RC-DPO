#!/usr/bin/env sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=1 src/train_fsdp.py\
    --epochs 1 \
    --batch_size 1 \
    --max_length 512 \
    --lr 1e-6 \
    --beta 0.1 \
    --seed 2003 \
    --model_name "microsoft/phi-2" \
    --dataset_name "jondurbin/truthy-dpo-v0.1" \
    --wandb_project "truthy-dpo"
