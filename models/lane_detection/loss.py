import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss — critical for lane detection because lanes occupy
    very few pixels (heavy class imbalance).
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs  = torch.sigmoid(logits)
        flat_p = probs.view(-1)
        flat_t = targets.view(-1)
        inter  = (flat_p * flat_t).sum()
        dice   = (2.0 * inter + self.smooth) / (flat_p.sum() + flat_t.sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal Loss — down-weights easy background pixels."""
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt    = torch.where(targets == 1, probs, 1 - probs)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    """
    BCE + Dice — best combination for lane segmentation:
      BCE : standard pixel-wise classification
      Dice: handles class imbalance from thin lane pixels
    pos_weight > 1 further penalizes missing lanes.
    """
    def __init__(self, bce_w: float = 0.5, dice_w: float = 0.5,
                 pos_weight: float = 5.0):
        super().__init__()
        self.bce_w  = bce_w
        self.dice_w = dice_w
        pw = torch.tensor([pos_weight])
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        return self.bce_w * self.bce(logits, targets) + \
               self.dice_w * self.dice(logits, targets)


class LaneMetrics:
    """Accumulates batch metrics and returns epoch averages."""

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps       = eps
        self.reset()

    def reset(self):
        self.total_iou  = 0.0
        self.total_dice = 0.0
        self.total_f1   = 0.0
        self.total_acc  = 0.0
        self.n_batches  = 0

    @torch.no_grad()
    def update(self, logits, targets):
        preds   = (torch.sigmoid(logits) > self.threshold).float()
        targets = targets.float()
        tp = (preds * targets).sum(dim=(2, 3))
        fp = (preds * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - preds) * targets).sum(dim=(2, 3))
        tn = ((1 - preds) * (1 - targets)).sum(dim=(2, 3))
        iou  = (tp / (tp + fp + fn + self.eps)).mean().item()
        dice = (2 * tp / (2 * tp + fp + fn + self.eps)).mean().item()
        prec = (tp / (tp + fp + self.eps)).mean().item()
        rec  = (tp / (tp + fn + self.eps)).mean().item()
        f1   = 2 * prec * rec / (prec + rec + self.eps)
        acc  = ((tp + tn) / (tp + fp + fn + tn + self.eps)).mean().item()
        self.total_iou  += iou
        self.total_dice += dice
        self.total_f1   += f1
        self.total_acc  += acc
        self.n_batches  += 1

    def compute(self) -> dict:
        n = max(self.n_batches, 1)
        return {"iou":  self.total_iou  / n,
                "dice": self.total_dice / n,
                "f1":   self.total_f1   / n,
                "acc":  self.total_acc  / n}

print("✓ Losses & Metrics defined")