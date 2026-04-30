#!/bin/bash
#SBATCH --job-name=dlav-m1
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

set -e
module load gcc python cuda
export PATH="/home/garate/miniconda3/envs/nanofm/bin:$PATH"
cd /home/garate/Project

mkdir -p logs checkpoints submissions

/home/garate/miniconda3/envs/nanofm/bin/python src/train.py \
    --data_dir         data \
    --ckpt_dir         checkpoints \
    --epochs           300 \
    --batch_size       32 \
    --lr               1e-3 \
    --num_workers      2 \
    --augment_prob     0.5 \
    --include_dynamics \
    --wandb_mode       online