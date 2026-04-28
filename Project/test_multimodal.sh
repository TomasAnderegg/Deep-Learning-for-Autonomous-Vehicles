#!/bin/bash
#SBATCH --job-name=dlav-test-multimodal
#SBATCH --output=logs/test_multimodal_%j.log
#SBATCH --error=logs/test_multimodal_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

set -e
module load gcc python cuda
export PATH="/home/garate/miniconda3/envs/nanofm/bin:$PATH"
cd /home/garate/Project

mkdir -p logs

/home/garate/miniconda3/envs/nanofm/bin/python - <<'EOF'
import torch
from src.model import MultiModalDrivingPlanner
from src.multimodal_loss import multimodal_loss, get_best_trajectory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

B = 4
model = MultiModalDrivingPlanner(history_input_dim=9).to(device)  # include_dynamics=True
model.eval()

image   = torch.randn(B, 3, 224, 224).to(device)
command = torch.randint(0, 3, (B,)).to(device)
history = torch.randn(B, 21, 9).to(device)
gt      = torch.randn(B, 60, 3).to(device)

with torch.no_grad():
    trajs, probs = model(image, command, history)

print(f"trajectories : {trajs.shape}")           # attendu : (4, 5, 60, 3)
print(f"probs        : {probs.shape}")           # attendu : (4, 5)
print(f"probs sum    : {probs.sum(dim=-1)}")     # doit etre ~1.0

loss, best_idx = multimodal_loss(trajs, probs, gt)
print(f"loss         : {loss.item():.4f}")
print(f"best_idx     : {best_idx}")              # (4,) indices 0-4

best = get_best_trajectory(trajs, probs)
print(f"best_traj    : {best.shape}")            # attendu : (4, 60, 3)

print("\nTout OK !")
EOF
