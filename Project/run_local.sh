@echo off
mkdir checkpoints logs submissions 2>nul

python src/train.py ^
    --data_dir         data ^
    --ckpt_dir         checkpoints ^
    --epochs           30 ^
    --batch_size       8 ^
    --lr               1e-3 ^
    --num_workers      2 ^
    --augment_prob     0.5 ^
    --include_dynamics ^
    --wandb_mode       disabled