"""YOLO-format dataset loader + CenterNet target encoding (player class only)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .model import OUTPUT_STRIDE

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def player_class_id(data_yaml: str | Path) -> int:
    """Find the `player` class id in a Roboflow data.yaml."""
    spec = yaml.safe_load(Path(data_yaml).read_text())
    names = spec["names"]
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for idx, name in items:
        if str(name).lower() == "player":
            return int(idx)
    raise ValueError(f"No 'player' class in {data_yaml}: {names}")


def split_dir(data_yaml: str | Path, split: str) -> Path:
    """Resolve the image directory for a split ('train'/'val'/'test')."""
    root = Path(data_yaml).parent
    spec = yaml.safe_load(Path(data_yaml).read_text())
    rel = spec.get(split) or spec.get({"val": "valid"}.get(split, split))
    if rel is None:
        raise ValueError(f"Split '{split}' missing from {data_yaml}")
    return (root / rel).resolve()


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """CornerNet radius: gaussians whose peak boxes keep IoU >= min_overlap."""
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(b1**2 - 4 * a1 * c1)) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 - math.sqrt(b2**2 - 4 * a2 * c2)) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return max(1.0, min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> None:
    diameter = 2 * radius + 1
    sigma = diameter / 6
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    gaussian = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    if right + left <= 0 or top + bottom <= 0:
        return
    patch = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    g_patch = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(patch, g_patch, out=patch)


class PlayerDataset(Dataset):
    """Images + YOLO txt labels → (image tensor, CenterNet target tensors)."""

    def __init__(self, data_yaml: str | Path, split: str = "train", img_size: int = 512):
        self.img_size = img_size
        self.class_id = player_class_id(data_yaml)
        image_dir = split_dir(data_yaml, split)
        self.images = sorted(
            p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.images:
            raise ValueError(f"No images under {image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def _label_path(self, image_path: Path) -> Path:
        return Path(str(image_path.parent).replace("images", "labels")) / (image_path.stem + ".txt")

    def load_boxes(self, image_path: Path) -> np.ndarray:
        """Player boxes as (n, 4) normalized cx, cy, w, h."""
        label_path = self._label_path(image_path)
        boxes = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5 and int(float(parts[0])) == self.class_id:
                    boxes.append([float(v) for v in parts[1:5]])
        return np.array(boxes, dtype=np.float32).reshape(-1, 4)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path = self.images[index]
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, (self.img_size, self.img_size))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))

        grid = self.img_size // OUTPUT_STRIDE
        heatmap = np.zeros((grid, grid), dtype=np.float32)
        wh = np.zeros((2, grid, grid), dtype=np.float32)
        offset = np.zeros((2, grid, grid), dtype=np.float32)
        mask = np.zeros((1, grid, grid), dtype=np.float32)

        for cx_n, cy_n, w_n, h_n in self.load_boxes(image_path):
            w, h = w_n * grid, h_n * grid
            cx, cy = cx_n * grid, cy_n * grid
            ix, iy = min(int(cx), grid - 1), min(int(cy), grid - 1)
            radius = int(gaussian_radius(h, w))
            draw_gaussian(heatmap, ix, iy, radius)
            heatmap[iy, ix] = 1.0
            wh[:, iy, ix] = (w, h)
            offset[:, iy, ix] = (cx - ix, cy - iy)
            mask[0, iy, ix] = 1.0

        return {
            "image": tensor,
            "heatmap": torch.from_numpy(heatmap).unsqueeze(0),
            "wh": torch.from_numpy(wh),
            "offset": torch.from_numpy(offset),
            "mask": torch.from_numpy(mask),
        }
