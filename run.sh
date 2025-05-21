#!/usr/bin/env sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=8 src/train_fsdp.py\
    --epochs 3 \
    --batch_size 8 \
    --max_length 512 \
    --lr 1e-5 \
    --beta 0.1 \
    --seed 2003 \
    --model_name "microsoft/phi-2" \
    --dataset_name "jondurbin/truthy-dpo-v0.1" \
    --wandb_project "truthy-dpo"
