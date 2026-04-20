import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

sys.path.append(os.path.dirname(__file__))
from data_loader import AugmentedNuPlanDataset, get_data_paths
from model import DrivingPlanner


# ── Metric ────────────────────────────────────────────────────────────────────
def compute_ade(pred, target):
    """pred, target: (B, 60, 3) — ADE sur x, y uniquement"""
    dist = torch.norm(pred[..., :2] - target[..., :2], dim=-1)  # (B, 60)
    return dist.mean(dim=1).mean()                               # scalar


# ── Val loop ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_ade, n = 0.0, 0
    for batch in loader:
        image   = batch['camera'].to(device)
        command = batch['driving_command'].to(device)
        history = batch['history'].to(device)
        future  = batch['future'].to(device)

        pred = model(image, command, history)
        ade  = compute_ade(pred, future)
        total_ade += ade.item() * image.size(0)
        n += image.size(0)
    return total_ade / n


# ── Train ─────────────────────────────────────────────────────────────────────
def train(args):
    wandb.init(
        project="dlav-m1",
        entity="tjga-98-epfl", #USer Wandb
        config=vars(args),
        mode=args.wandb_mode,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    paths = get_data_paths(args.data_dir)
    train_ds = AugmentedNuPlanDataset(
        paths['train'],
        test=False,
        include_dynamics=args.include_dynamics,
        augment_prob=args.augment_prob,
    )
    val_ds = AugmentedNuPlanDataset(
        paths['val'],
        test=False,
        include_dynamics=args.include_dynamics,
        augment_prob=0.0,   # pas d'augmentation en val
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    history_dim = 9 if args.include_dynamics else 4
    model = DrivingPlanner(history_input_dim=history_dim).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer + Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.L1Loss()

    best_val_ade = float('inf')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            image   = batch['camera'].to(device)
            command = batch['driving_command'].to(device)
            history = batch['history'].to(device)
            future  = batch['future'].to(device)

            optimizer.zero_grad()
            pred = model(image, command, history)   # (B, 60, 3)
            weights = torch.linspace(1.0, 2.0, 60).to(device)  # timesteps lointains pèsent plus
            loss = (criterion(pred[..., :2], future[..., :2]) * weights[None, :, None]).mean()

            # loss = criterion(pred, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        train_loss /= len(train_loader)
        val_ade = validate(model, val_loader, device)

        print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val ADE: {val_ade:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        wandb.log({'train_loss': train_loss, 'val_ade': val_ade,
                   'lr': scheduler.get_last_lr()[0], 'epoch': epoch})

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_ade': val_ade,
                'args': vars(args),
            }, os.path.join(args.ckpt_dir, 'best.pth'))
            print(f"  ✓ Saved best (ADE={best_val_ade:.4f})")

    print(f"\nBest Val ADE: {best_val_ade:.4f}")
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',         type=str,   default='../data')
    parser.add_argument('--ckpt_dir',         type=str,   default='../checkpoints')
    parser.add_argument('--epochs',           type=int,   default=50)
    parser.add_argument('--batch_size',       type=int,   default=32)
    parser.add_argument('--lr',               type=float, default=1e-3)
    parser.add_argument('--num_workers',      type=int,   default=4)
    parser.add_argument('--augment_prob',     type=float, default=0.5)
    parser.add_argument('--include_dynamics', action='store_true')
    parser.add_argument('--wandb_mode',       type=str,   default='online',
                        choices=['online', 'offline', 'disabled'])
    args = parser.parse_args()
    train(args)

    # python src/train.py --data_dir data --ckpt_dir checkpoints --epochs 30 --batch_size 8 --lr 1e-3 --num_workers 2 --augment_prob 0.5 --include_dynamics --wandb_mode disabled
