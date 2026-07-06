"""Team assignment: jersey-crop color features + k-means (k=2).

Each tracked player contributes one color feature per frame (mean LAB color of
the torso region of its box). After the clip is scanned, all observations are
clustered into two teams and each track is assigned by majority vote over its
own observations — this is the "smoothed over track history" step.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

# Torso region within a player bbox (fractions of box width/height):
# skip the head (top 20%) and legs (bottom 45%), keep the central 60% width.
_TORSO = (0.2, 0.2, 0.8, 0.55)  # x1, y1, x2, y2 fractions


def torso_crop(frame: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(np.clip(x1 + _TORSO[0] * bw, 0, w - 1))
    cy1 = int(np.clip(y1 + _TORSO[1] * bh, 0, h - 1))
    cx2 = int(np.clip(x1 + _TORSO[2] * bw, 0, w))
    cy2 = int(np.clip(y1 + _TORSO[3] * bh, 0, h))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    return frame[cy1:cy2, cx1:cx2]


def color_feature(crop_bgr: np.ndarray) -> np.ndarray:
    """Mean LAB color of a torso crop. LAB distances track perceived color."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)


def kmeans_two(features: np.ndarray, seed: int = 0) -> np.ndarray:
    """Cluster (N, D) float32 features into 2 groups; returns labels (N,).

    Cluster ids are normalized so that cluster 0 is the one containing the
    feature closest to the overall darkest L value — deterministic across runs.
    """
    features = np.asarray(features, dtype=np.float32)
    if len(features) < 2:
        return np.zeros(len(features), dtype=int)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    rng = cv2.setRNGSeed(seed)  # noqa: F841 — makes KMEANS_PP deterministic
    _, labels, centers = cv2.kmeans(
        features, 2, None, criteria, attempts=5, flags=cv2.KMEANS_PP_CENTERS
    )
    labels = labels.ravel().astype(int)
    if centers[0][0] > centers[1][0]:  # normalize: darker jerseys → team 0
        labels = 1 - labels
    return labels


@dataclass
class TeamAssigner:
    """Collect per-frame jersey features during pass 1, then fit and query."""

    _features: list[np.ndarray] = field(default_factory=list)
    _track_ids: list[int] = field(default_factory=list)
    _assignment: dict[int, int] = field(default_factory=dict)

    def observe(
        self, track_id: int, frame: np.ndarray, xyxy: tuple[float, float, float, float]
    ) -> None:
        crop = torso_crop(frame, xyxy)
        if crop is None:
            return
        self._features.append(color_feature(crop))
        self._track_ids.append(track_id)

    def observe_feature(self, track_id: int, feature: np.ndarray) -> None:
        """Direct feature injection (used by unit tests)."""
        self._features.append(np.asarray(feature, dtype=np.float32))
        self._track_ids.append(track_id)

    def fit(self) -> dict[int, int]:
        if not self._features:
            self._assignment = {}
            return self._assignment
        labels = kmeans_two(np.stack(self._features))
        votes: dict[int, list[int]] = defaultdict(list)
        for tid, label in zip(self._track_ids, labels, strict=True):
            votes[tid].append(int(label))
        self._assignment = {tid: int(round(np.mean(v))) for tid, v in votes.items()}
        return self._assignment

    def team_of(self, track_id: int) -> int | None:
        return self._assignment.get(track_id)
