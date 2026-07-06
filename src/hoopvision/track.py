"""ByteTrack wrapper: persistent player IDs across frames.

Only PLAYER detections are tracked. The ball is too small/blurred for IoU
association to help (its track is assembled in events.py via interpolation),
and the rim is static (pipeline uses the median rim box per clip).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import supervision as sv

from .detect import PLAYER, Detection


@dataclass(frozen=True)
class TrackedPlayer:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    team: int | None = None

    @property
    def foot(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, y2)

    def with_team(self, team: int | None) -> TrackedPlayer:
        return replace(self, team=team)


class PlayerTracker:
    # sv.ByteTrack is deprecated in supervision 0.28+ (pinned <0.30 in
    # pyproject). Migrate to roboflow/trackers' ByteTrackTracker once its
    # wheel actually ships the package (2.5.0 on PyPI is empty).
    def __init__(self, frame_rate: float = 30.0):
        self._tracker = sv.ByteTrack(frame_rate=int(round(frame_rate)))

    def update(self, detections: list[Detection]) -> list[TrackedPlayer]:
        players = [d for d in detections if d.class_name == PLAYER]
        if players:
            dets = sv.Detections(
                xyxy=np.array([d.xyxy for d in players], dtype=np.float32),
                confidence=np.array([d.confidence for d in players], dtype=np.float32),
                class_id=np.zeros(len(players), dtype=int),
            )
        else:
            dets = sv.Detections.empty()
        tracked = self._tracker.update_with_detections(dets)
        return [
            TrackedPlayer(int(tid), tuple(map(float, box)), float(conf))
            for box, conf, tid in zip(
                tracked.xyxy, tracked.confidence, tracked.tracker_id, strict=True
            )
        ]
