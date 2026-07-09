"""Court homography: image pixels → NBA halfcourt coordinates in feet.

Court model (all feet). Origin = left corner of the baseline, looking at the
basket. x runs along the baseline (0..50), y runs toward halfcourt (0..47).

    (0,47) ──────────── (50,47)   halfcourt line
      │                    │
      │     ( paint )      │
    (0,0) ──────────────(50,0)    baseline

Key landmarks are provided in LANDMARKS for the calibration tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

COURT_WIDTH_FT = 50.0  # baseline (x axis)
COURT_LENGTH_FT = 47.0  # halfcourt depth (y axis)
RIM_CENTER = (25.0, 5.25)
RIM_RADIUS_FT = 0.75
BACKBOARD_Y = 4.0
PAINT_HALF_WIDTH = 8.0  # lane is 16 ft wide
FT_LINE_Y = 19.0
FT_CIRCLE_RADIUS = 6.0
THREE_PT_RADIUS = 23.75
CORNER_THREE_X = 3.0  # 3 ft from sideline
CORNER_THREE_Y = 14.0  # corner three extends 14 ft from baseline

# Named landmarks a user can click during calibration (name → court x/y feet).
LANDMARKS: dict[str, tuple[float, float]] = {
    "baseline-left-corner": (0.0, 0.0),
    "baseline-right-corner": (COURT_WIDTH_FT, 0.0),
    "halfcourt-left-corner": (0.0, COURT_LENGTH_FT),
    "halfcourt-right-corner": (COURT_WIDTH_FT, COURT_LENGTH_FT),
    "paint-left-baseline": (25.0 - PAINT_HALF_WIDTH, 0.0),
    "paint-right-baseline": (25.0 + PAINT_HALF_WIDTH, 0.0),
    "ft-line-left": (25.0 - PAINT_HALF_WIDTH, FT_LINE_Y),
    "ft-line-right": (25.0 + PAINT_HALF_WIDTH, FT_LINE_Y),
}

# Ordered keypoint schema for the v2 court-registration model. The index is a
# permanent contract: it is the heatmap channel a keypoint model predicts and
# the column order in every pseudo-labeled annotation, so APPEND ONLY — never
# reorder or delete an entry once a dataset or checkpoint depends on it.
#
# Points are geometrically unambiguous court features a human (or ICP-refined
# homography) can locate: line intersections, arc/circle extrema, corners.
COURT_KEYPOINTS: list[tuple[str, float, float]] = [
    ("baseline-left-corner", 0.0, 0.0),
    ("baseline-right-corner", COURT_WIDTH_FT, 0.0),
    ("paint-left-baseline", 25.0 - PAINT_HALF_WIDTH, 0.0),
    ("paint-right-baseline", 25.0 + PAINT_HALF_WIDTH, 0.0),
    ("ft-line-left", 25.0 - PAINT_HALF_WIDTH, FT_LINE_Y),
    ("ft-line-right", 25.0 + PAINT_HALF_WIDTH, FT_LINE_Y),
    ("ft-line-center", 25.0, FT_LINE_Y),
    ("ft-circle-top", 25.0, FT_LINE_Y + FT_CIRCLE_RADIUS),
    ("three-pt-arc-top", 25.0, RIM_CENTER[1] + THREE_PT_RADIUS),
    ("corner-three-left", CORNER_THREE_X, CORNER_THREE_Y),
    ("corner-three-right", COURT_WIDTH_FT - CORNER_THREE_X, CORNER_THREE_Y),
    ("rim-center", RIM_CENTER[0], RIM_CENTER[1]),
    ("halfcourt-left-corner", 0.0, COURT_LENGTH_FT),
    ("halfcourt-right-corner", COURT_WIDTH_FT, COURT_LENGTH_FT),
    ("halfcourt-center", 25.0, COURT_LENGTH_FT),
    ("center-circle-near", 25.0, COURT_LENGTH_FT - FT_CIRCLE_RADIUS),
]

KEYPOINT_NAMES: list[str] = [name for name, _x, _y in COURT_KEYPOINTS]
KEYPOINT_COURT_FT: np.ndarray = np.array(
    [(x, y) for _name, x, y in COURT_KEYPOINTS], dtype=np.float64
)


@dataclass
class CourtCalibration:
    homography: np.ndarray  # 3x3, image px → court feet
    image_points: np.ndarray  # (N, 2) px
    court_points: np.ndarray  # (N, 2) feet

    @classmethod
    def from_points(
        cls,
        image_points: np.ndarray | list[tuple[float, float]],
        court_points: np.ndarray | list[tuple[float, float]],
    ) -> CourtCalibration:
        image_points = np.asarray(image_points, dtype=np.float64)
        court_points = np.asarray(court_points, dtype=np.float64)
        if len(image_points) < 4 or len(image_points) != len(court_points):
            raise ValueError(
                "Calibration needs >= 4 matched (image, court) point pairs, "
                f"got {len(image_points)} image / {len(court_points)} court"
            )
        h, status = cv2.findHomography(image_points, court_points, method=0)
        if h is None:
            raise ValueError("findHomography failed — are the points collinear?")
        return cls(h, image_points, court_points)

    def to_court(self, points_px: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
        """Project (N, 2) image pixels to (N, 2) court feet."""
        return self._apply(self.homography, points_px)

    def to_image(self, points_ft: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
        return self._apply(np.linalg.inv(self.homography), points_ft)

    @staticmethod
    def _apply(h: np.ndarray, points: np.ndarray | list) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2))
        return cv2.perspectiveTransform(pts, h).reshape(-1, 2)

    def reprojection_error_ft(self) -> float:
        """Mean distance (feet) between projected calibration points and truth."""
        projected = self.to_court(self.image_points)
        return float(np.mean(np.linalg.norm(projected - self.court_points, axis=1)))

    def in_bounds(self, points_ft: np.ndarray, margin_ft: float = 3.0) -> np.ndarray:
        """Boolean mask for court points within the halfcourt (+margin)."""
        pts = np.asarray(points_ft)
        return (
            (pts[:, 0] >= -margin_ft)
            & (pts[:, 0] <= COURT_WIDTH_FT + margin_ft)
            & (pts[:, 1] >= -margin_ft)
            & (pts[:, 1] <= COURT_LENGTH_FT + margin_ft)
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "homography": self.homography.tolist(),
            "image_points": self.image_points.tolist(),
            "court_points": self.court_points.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CourtCalibration:
        payload = json.loads(Path(path).read_text())
        return cls(
            np.array(payload["homography"]),
            np.array(payload["image_points"]),
            np.array(payload["court_points"]),
        )
