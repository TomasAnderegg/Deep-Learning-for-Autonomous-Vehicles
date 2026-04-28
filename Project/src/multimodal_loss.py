import torch
import torch.nn as nn
import torch.nn.functional as F


def multimodal_loss(pred_trajs, pred_probs, gt_traj, reg_weight=1.0, cls_weight=0.1):
    """
    Best-of-K loss for multimodal trajectory prediction.

    Args:
        pred_trajs : (B, K, 60, 3)  — K predicted trajectories
        pred_probs : (B, K)          — predicted mode probabilities (after softmax)
        gt_traj    : (B, 60, 3)      — ground truth trajectory
        reg_weight : weight for regression loss
        cls_weight : weight for classification loss

    Returns:
        loss       : scalar
        best_idx   : (B,) index of the best mode per sample (useful for logging)
    """
    B, K, T, _ = pred_trajs.shape

    # ── 1. Compute ADE per mode ───────────────────────────────────────────────
    # On étend le ground truth pour comparer avec chaque mode
    gt_expanded = gt_traj.unsqueeze(1).expand(B, K, T, 3)          # (B, K, 60, 3)

    # Distance euclidienne sur x, y uniquement (comme la loss originale)
    dist = torch.norm(pred_trajs[..., :2] - gt_expanded[..., :2], dim=-1)  # (B, K, 60)
    ade_per_mode = dist.mean(dim=-1)                                         # (B, K)

    # ── 2. Best-of-K : sélectionner la meilleure tête ────────────────────────
    best_idx = ade_per_mode.argmin(dim=1)          # (B,) — indice du meilleur mode
    best_ade = ade_per_mode[range(B), best_idx]    # (B,) — ADE du meilleur mode
    reg_loss = best_ade.mean()                      # scalar

    # ── 3. Classification loss : apprendre à identifier le meilleur mode ─────
    # pred_probs est déjà après softmax donc on utilise nll_loss avec log
    log_probs = torch.log(pred_probs + 1e-8)       # (B, K) — stabilité numérique
    cls_loss = F.nll_loss(log_probs, best_idx)     # scalar

    # ── 4. Loss finale ────────────────────────────────────────────────────────
    loss = reg_weight * reg_loss + cls_weight * cls_loss

    return loss, best_idx


def get_best_trajectory(pred_trajs, pred_probs):
    """
    Inference helper : retourne la trajectoire la plus probable.

    Args:
        pred_trajs : (B, K, 60, 3)
        pred_probs : (B, K)

    Returns:
        best_traj  : (B, 60, 3)
    """
    best_idx = pred_probs.argmax(dim=1)            # (B,)
    best_traj = pred_trajs[range(pred_trajs.size(0)), best_idx]  # (B, 60, 3)
    return best_traj
