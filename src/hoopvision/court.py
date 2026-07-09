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


# Court dimensions that vary by level. The universal ones (court 50x47 ft, FT
# line 19 ft from baseline, FT circle 6 ft, rim center 5.25 ft) are the module
# constants above and hold for NBA/NCAA/HS/FIBA; only the lane width, 3-pt
# radius, and corner-three geometry change. This lets the same keypoint schema
# produce metrically correct pseudo-labels on any level's court (see v2 §4.1).
@dataclass(frozen=True)
class CourtProfile:
    name: str
    lane_half_width_ft: float  # half the painted lane (NBA 8, NCAA/HS 6)
    three_pt_radius_ft: float  # arc radius from rim center
    # The straight corner-3 segment (parallel to the sideline) only exists on
    # courts whose corner distance is shorter than the arc radius — NBA. On a
    # pure-arc court (HS, and treated so for NCAA here) there is no such point,
    # so the corner-three landmarks are undefined (None → absent in labels).
    corner_three_x_ft: float | None
    corner_three_y_ft: float | None


NBA = CourtProfile("nba", PAINT_HALF_WIDTH, THREE_PT_RADIUS, CORNER_THREE_X, CORNER_THREE_Y)
NCAA = CourtProfile("ncaa", 6.0, 22.146, None, None)  # men's; women's/HS use 19.75
HIGH_SCHOOL = CourtProfile("hs", 6.0, 19.75, None, None)
PROFILES: dict[str, CourtProfile] = {p.name: p for p in (NBA, NCAA, HIGH_SCHOOL)}

# Ordered keypoint schema for the v2 court-registration model. The index is a
# permanent contract: it is the heatmap channel a keypoint model predicts and
# the column order in every pseudo-labeled annotation, so APPEND ONLY — never
# reorder or delete an entry once a dataset or checkpoint depends on it.
#
# Points are geometrically unambiguous court features a human (or ICP-refined
# homography) can locate: line intersections, arc/circle extrema, corners.
KEYPOINT_NAMES: list[str] = [
    "baseline-left-corner",
    "baseline-right-corner",
    "paint-left-baseline",
    "paint-right-baseline",
    "ft-line-left",
    "ft-line-right",
    "ft-line-center",
    "ft-circle-top",
    "three-pt-arc-top",
    "corner-three-left",
    "corner-three-right",
    "rim-center",
    "halfcourt-left-corner",
    "halfcourt-right-corner",
    "halfcourt-center",
    "center-circle-near",
]


def keypoints_ft(profile: CourtProfile = NBA) -> np.ndarray:
    """Court-feet coordinates of the keypoint schema for a court level.

    Returns an (K, 2) array in `KEYPOINT_NAMES` order. Landmarks a profile does
    not define (e.g. corner threes on a pure-arc court) are `np.nan` so callers
    can drop them (no valid homography target).
    """
    lh = profile.lane_half_width_ft
    r = profile.three_pt_radius_ft
    if profile.corner_three_x_ft is None or profile.corner_three_y_ft is None:
        cl = (np.nan, np.nan)
        cr = (np.nan, np.nan)
    else:
        cl = (profile.corner_three_x_ft, profile.corner_three_y_ft)
        cr = (COURT_WIDTH_FT - profile.corner_three_x_ft, profile.corner_three_y_ft)
    coords: dict[str, tuple[float, float]] = {
        "baseline-left-corner": (0.0, 0.0),
        "baseline-right-corner": (COURT_WIDTH_FT, 0.0),
        "paint-left-baseline": (25.0 - lh, 0.0),
        "paint-right-baseline": (25.0 + lh, 0.0),
        "ft-line-left": (25.0 - lh, FT_LINE_Y),
        "ft-line-right": (25.0 + lh, FT_LINE_Y),
        "ft-line-center": (25.0, FT_LINE_Y),
        "ft-circle-top": (25.0, FT_LINE_Y + FT_CIRCLE_RADIUS),
        "three-pt-arc-top": (25.0, RIM_CENTER[1] + r),
        "corner-three-left": cl,
        "corner-three-right": cr,
        "rim-center": (RIM_CENTER[0], RIM_CENTER[1]),
        "halfcourt-left-corner": (0.0, COURT_LENGTH_FT),
        "halfcourt-right-corner": (COURT_WIDTH_FT, COURT_LENGTH_FT),
        "halfcourt-center": (25.0, COURT_LENGTH_FT),
        "center-circle-near": (25.0, COURT_LENGTH_FT - FT_CIRCLE_RADIUS),
    }
    return np.array([coords[name] for name in KEYPOINT_NAMES], dtype=np.float64)


# Default (NBA) schema — kept as module constants for backward compatibility.
KEYPOINT_COURT_FT: np.ndarray = keypoints_ft(NBA)
COURT_KEYPOINTS: list[tuple[str, float, float]] = [
    (name, float(x), float(y))
    for name, (x, y) in zip(KEYPOINT_NAMES, KEYPOINT_COURT_FT, strict=True)
]


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
