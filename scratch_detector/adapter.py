"""Adapter: plug the from-scratch detector into the pipeline's Detector protocol.

from scratch_detector.adapter import ScratchDetector
analysis = hoopvision.pipeline.analyze(video, detector=ScratchDetector("best.pt"))
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hoopvision.detect import PLAYER, Detection, default_device  # noqa: E402

from .data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from .model import CenterNetLite, decode  # noqa: E402


class ScratchDetector:
    """Single-class (player) detector satisfying hoopvision.detect.Detector."""

    def __init__(self, weights: str | Path, img_size: int = 512, conf: float = 0.30):
        self.device = default_device()
        self.img_size = img_size
        self.conf = conf
        self.model = CenterNetLite(pretrained_backbone=False).to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        self.model.eval()

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (self.img_size, self.img_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = (
            torch.from_numpy(((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )
        boxes = decode(self.model(tensor), conf_threshold=self.conf)[0].cpu().numpy()
        sx, sy = width / self.img_size, height / self.img_size
        return [
            Detection(
                (float(x1 * sx), float(y1 * sy), float(x2 * sx), float(y2 * sy)),
                PLAYER,
                float(score),
            )
            for x1, y1, x2, y2, score in boxes
        ]
