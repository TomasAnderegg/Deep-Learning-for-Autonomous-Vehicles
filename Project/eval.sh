#!/bin/bash
#SBATCH --job-name=dlav-eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

export PATH="/home/garate/miniconda3/envs/nanofm/bin:$PATH"
cd /home/garate/Project

mkdir -p logs submissions

/home/garate/miniconda3/envs/nanofm/bin/python src/eval.py \
    --data_dir  data \
    --ckpt_dir  checkpoints \
    --output_dir submissions
