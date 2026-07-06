"""Detector protocol and the Ultralytics YOLO implementation.

The pipeline only depends on the `Detector` protocol, so the fine-tuned YOLO
baseline and the from-scratch detector (scratch_detector/) are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

PLAYER = "player"
BALL = "ball"
RIM = "rim"

# COCO class ids used when running a pretrained (non fine-tuned) checkpoint.
# There is no rim class in COCO, so rim-dependent features are unavailable
# until the fine-tuned weights from scripts/finetune_yolo.py are used.
_COCO_TO_HOOP = {0: PLAYER, 32: BALL}  # person, sports ball


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    class_name: str
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-center of the box — where a player touches the court."""
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, y2)


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def default_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class YoloDetector:
    """Ultralytics YOLO detector.

    Accepts either a pretrained COCO checkpoint (person/sports-ball mapped to
    player/ball) or a checkpoint fine-tuned on player/ball/rim classes; the
    class mapping is derived from the checkpoint's own class names.
    """

    def __init__(
        self,
        weights: str | Path = "yolo11n.pt",
        conf: float = 0.25,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device or default_device()
        names: dict[int, str] = self.model.names
        if {PLAYER, BALL} <= set(names.values()):
            self.class_map = {i: n for i, n in names.items() if n in (PLAYER, BALL, RIM)}
        else:
            self.class_map = dict(_COCO_TO_HOOP)

    @property
    def has_rim_class(self) -> bool:
        return RIM in self.class_map.values()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model(
            frame,
            conf=self.conf,
            device=self.device,
            classes=list(self.class_map),
            verbose=False,
        )[0]
        detections = []
        boxes = result.boxes
        for xyxy, conf, cls in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
        ):
            name = self.class_map.get(int(cls))
            if name is not None:
                detections.append(Detection(tuple(xyxy), name, float(conf)))
        return detections
