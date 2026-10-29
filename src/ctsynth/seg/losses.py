"""Cross-entropy + soft Dice loss and class-wise segmentation metrics (categorical targets, never
regression on label IDs)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_ce_loss(
    logits: torch.Tensor, target: torch.Tensor, w_dice: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """logits (B,K,D,H,W), target (B,D,H,W) int64. Mean soft Dice over classes present in the batch
    + cross-entropy."""
    ce = F.cross_entropy(logits, target)
    k = logits.shape[1]
    prob = torch.softmax(logits.float(), dim=1)
    onehot = F.one_hot(target, k).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = (prob * onehot).sum(dims)
    denom = prob.sum(dims) + onehot.sum(dims)
    present = onehot.sum(dims) > 0
    dice = (2 * inter + eps) / (denom + eps)
    return ce + w_dice * (1 - dice[present].mean())


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, n_classes: int) -> torch.Tensor:
    """(K,K) counts, rows = truth, columns = prediction."""
    idx = target.flatten().long() * n_classes + pred.flatten().long()
    return torch.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def dice_iou_from_confusion(cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class Dice and IoU (NaN for classes absent from both truth and prediction)."""
    tp = cm.diag().double()
    fp = cm.sum(0).double() - tp
    fn = cm.sum(1).double() - tp
    nan = torch.full_like(tp, float("nan"))
    dice = torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn).clamp(min=1), nan)
    iou = torch.where(tp + fp + fn > 0, tp / (tp + fp + fn).clamp(min=1), nan)
    return dice, iou
