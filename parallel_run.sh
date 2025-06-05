cat args.txt | parallel --colsep ' ' --jobs 4 --results logs/ '
  export CUDA_VISIBLE_DEVICES={%};
  lr={1}; margin={2}; reg={3};
  port=$((29500 + {%}));
  outdir="checkpoints/lr=${lr}_margin=${margin}_reg=${reg}";
  mkdir -p "$outdir";
  echo "GPU {%} using port $port and saving to $outdir";

  torchrun --nproc-per-node=1 --master-port $port src/train-fsdp-mh-fused-SV-v2.py \
    --epochs 10 \
    --batch_size 16 \
    --max_length 1024 \
    --lr $lr \
    --betas 0.5 0.55 0.45 0.4 0.6 \
    --seed 2003 \
    --model_name "microsoft/phi-1_5" \
    --dataset_name "jondurbin/truthy-dpo-v0.1" \
    --wandb_project "truthy-dpo-parallel-test" \
    --reg_weight $reg \
    --wandb_enable True \
    --eval_ratio 0.9 \
    --margin $margin \
    --output_dir $outdir'
