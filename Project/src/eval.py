import os
import sys
import csv
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from data_loader import AugmentedNuPlanDataset, get_data_paths
from model import DrivingPlanner


def generate_submission(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Charger le checkpoint
    ckpt = torch.load(os.path.join(args.ckpt_dir, 'best.pth'), map_location=device)
    saved_args = ckpt.get('args', {})
    include_dynamics = saved_args.get('include_dynamics', False)

    history_dim = 8 if include_dynamics else 3
    model = DrivingPlanner(history_input_dim=history_dim).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val ADE={ckpt['val_ade']:.4f})")

    # Test set
    paths = get_data_paths(args.data_dir)
    test_ds = AugmentedNuPlanDataset(
        paths['test'], test=True,
        include_dynamics=include_dynamics,
        augment_prob=0.0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True
    )
    print(f"Test samples: {len(test_ds)}")

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            image   = batch['camera'].to(device)
            command = batch['driving_command'].to(device)
            history = batch['history'].to(device)

            pred = model(image, command, history)   # (B, 60, 3)
            all_preds.append(pred.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 60, 3)
    print(f"Predictions shape: {all_preds.shape}")

    # ── Format CSV pour Kaggle ─────────────────────────────────────────
    # Regarde le format attendu sur la page Kaggle avant de submit !
    # Format typique : id, x_0, y_0, x_1, y_1, ..., x_59, y_59
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'submission.csv')

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['id'] + [f'{coord}_{t}' for t in range(60) for coord in ['x', 'y']]
        writer.writerow(header)
        # Rows
        for i, traj in enumerate(all_preds):
            row = [i] + [val for t in range(60) for val in [traj[t, 0], traj[t, 1]]]
            writer.writerow(row)

    print(f"Submission saved → {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str, default='../data')
    parser.add_argument('--ckpt_dir',   type=str, default='../checkpoints')
    parser.add_argument('--output_dir', type=str, default='../submissions')
    args = parser.parse_args()
    generate_submission(args)