"""CenterNet losses: penalty-reduced focal loss + L1 on size/offset."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def heatmap_focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Penalty-reduced pixel-wise focal loss (CornerNet/CenterNet).

    `target` is the gaussian-splatted heatmap in [0, 1]; cells equal to 1 are
    positives, everything else is a penalty-reduced negative.
    """
    pred = torch.sigmoid(pred_logits).clamp(1e-6, 1 - 1e-6)
    pos_mask = target.eq(1).float()
    neg_mask = 1.0 - pos_mask

    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
    neg_loss = -((1 - target) ** beta) * (pred**alpha) * torch.log(1 - pred) * neg_mask

    num_pos = pos_mask.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 over positive cells only. pred/target: (B, 2, H, W); mask: (B, 1, H, W)."""
    num_pos = mask.sum().clamp(min=1.0)
    return (F.l1_loss(pred * mask, target * mask, reduction="sum")) / num_pos


def detection_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    wh_weight: float = 0.1,
    offset_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    mask = targets["mask"]
    heat = heatmap_focal_loss(outputs["heatmap"], targets["heatmap"])
    wh = masked_l1_loss(outputs["wh"], targets["wh"], mask)
    offset = masked_l1_loss(outputs["offset"], targets["offset"], mask)
    total = heat + wh_weight * wh + offset_weight * offset
    return {"total": total, "heatmap": heat, "wh": wh, "offset": offset}
