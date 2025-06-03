cat args.txt | parallel --colsep ' ' --jobs 1 --results logs_1/ '
  export CUDA_VISIBLE_DEVICES={%};
  lr={1}; margin={2}; reg={3};
  outdir="checkpoints/lr=${lr}_margin=${margin}_reg=${reg}";
  mkdir -p "$outdir";
  echo "Saving to $outdir";
  torchrun --nproc-per-node=1 src/train-fsdp-mh-fused-SV.py \
    --epochs 10 \
    --batch_size 16 \
    --max_length 1024 \
    --lr $lr \
    --betas 0.5 0.55 0.45 0.4 0.6 \
    --seed 2003 \
    --model_name "microsoft/phi-1_5" \
    --dataset_name "jondurbin/truthy-dpo-v0.1" \
    --wandb_project "truthy-dpo" \
    --reg_weight $reg \
    --wandb_enable True \
    --eval_ratio 0.9 \
    --margin $margin \
    --output_dir $outdir'
