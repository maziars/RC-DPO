#!/usr/bin/env sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=8 src/train_fsdp.py\
    --epochs 10 \
    --batch_size 4 \
    --max_length 512 \
    --lr 1e-6 \
    --beta 0.01 \
    --seed 2003 \
    --model_name "microsoft/phi-2" \
    --dataset_name "jondurbin/truthy-dpo-v0.1" \
    --wandb_project "truthy-dpo"
