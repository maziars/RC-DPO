#!/usr/bin/env sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=1 src/train-fsdp-mh-fused.py\
    --epochs 10 \
    --batch_size 16 \
    --max_length 1024 \
    --lr 1e-5 \
    --betas 0.5 0.55 0.45 0.4 0.6 \
    --seed 2003 \
    --model_name "microsoft/phi-1_5" \
    --dataset_name "jondurbin/truthy-dpo-v0.1" \
    --wandb_project "truthy-dpo" \
    --reg_weight 1e-3 \
    --wandb_enable True \
    --eval_ratio 0.9
