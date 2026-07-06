"""Evaluate the from-scratch detector: AP50 for the player class.

    uv run python -m scratch_detector.eval --data data/<dataset>/data.yaml \
        --weights scratch_detector/runs/best.pt

AP is computed as area under the precision-recall curve with greedy IoU@0.5
matching — implemented here by hand (that is the point of this chapter).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hoopvision.detect import default_device  # noqa: E402

from .data import PlayerDataset  # noqa: E402
from .model import CenterNetLite, decode  # noqa: E402


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (n, 4) and (m, 4) xyxy boxes → (n, m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(
    detections: list[np.ndarray], ground_truths: list[np.ndarray], iou_threshold: float = 0.5
) -> float:
    """detections: per-image (n, 5) xyxy+score; ground_truths: per-image (m, 4)."""
    rows = []  # (score, is_true_positive)
    total_gt = sum(len(g) for g in ground_truths)
    for dets, gts in zip(detections, ground_truths, strict=True):
        order = np.argsort(-dets[:, 4]) if len(dets) else []
        matched = np.zeros(len(gts), dtype=bool)
        ious = iou_matrix(dets[:, :4], gts)
        for i in order:
            best_j, best_iou = -1, iou_threshold
            for j in range(len(gts)):
                if not matched[j] and ious[i, j] >= best_iou:
                    best_j, best_iou = j, ious[i, j]
            if best_j >= 0:
                matched[best_j] = True
                rows.append((dets[i, 4], 1))
            else:
                rows.append((dets[i, 4], 0))
    if total_gt == 0 or not rows:
        return 0.0
    rows.sort(key=lambda r: -r[0])
    tps = np.cumsum([r[1] for r in rows])
    fps = np.cumsum([1 - r[1] for r in rows])
    recall = tps / total_gt
    precision = tps / (tps + fps)
    # Area under the PR curve with monotone-decreasing precision envelope,
    # anchored at recall 0 so the first segment is counted.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0]], precision])
    return float(np.trapezoid(precision, recall))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default="scratch_detector/runs/best.pt")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.05, help="low: AP sweeps scores")
    args = parser.parse_args()

    device = default_device()
    model = CenterNetLite(pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    dataset = PlayerDataset(args.data, "val", args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch)

    all_dets: list[np.ndarray] = []
    all_gts: list[np.ndarray] = []
    grid = args.img_size
    cursor = 0
    for batch in loader:
        outputs = model(batch["image"].to(device))
        for boxes in decode(outputs, conf_threshold=args.conf):
            all_dets.append(boxes.cpu().numpy())
        for _ in range(len(batch["image"])):
            norm = dataset.load_boxes(dataset.images[cursor])  # cx, cy, w, h in [0,1]
            if len(norm):
                cx, cy, w, h = (
                    norm[:, 0] * grid,
                    norm[:, 1] * grid,
                    norm[:, 2] * grid,
                    norm[:, 3] * grid,
                )
                gts = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            else:
                gts = np.zeros((0, 4))
            all_gts.append(gts)
            cursor += 1

    ap50 = average_precision(all_dets, all_gts, iou_threshold=0.5)
    print(f"player AP50 = {ap50:.3f}  ({len(dataset)} val images)")


if __name__ == "__main__":
    main()
