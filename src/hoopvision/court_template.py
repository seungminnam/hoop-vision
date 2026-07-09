"""NBA full-court template for the 33-point court-keypoint schema (v2 §4.2).

The Phase-1 detector (release v0.4.0) predicts 33 court keypoints in the schema
of the adopted `basketball-court-detection-2` dataset. That dataset ships **no**
real-world coordinates for its points, so Phase 2 recovers them here.

How the coordinates were obtained (honesty rule — reproducible by
`scripts/anchor_court_template.py`):

1. `scripts/recover_court_template.py` chained per-frame homographies to place
   all 33 points into one reference frame (they are homographies of one plane).
2. A homography from that recovered layout to NBA feet was fit from a confident,
   well-spread 15-point seed (both baselines' corners / lane edges / corner-3
   marks + the halfcourt line). Every *non-seed* point then landed on a real
   court feature (FT elbows, arc tops, corner-3 elbows, 28-ft coaching-box
   sideline hashes, baskets) — see the ADR and the committed overlay.
3. The idealized coordinates below (exact NBA geometry) were validated against
   **all 1,220 labeled frames independently**: fitting image->feet per frame
   from the visible planar points gives a reprojection error of median ~0.17 ft
   (p90 ~0.41 ft) over ~14k point observations — the labels are globally
   consistent with this template to a couple of inches.

Coordinate frame (feet). Origin at the near-left court corner; x runs 0..94
along the length (left baseline -> right baseline), y runs 0..50 across the
width (near sideline -> far sideline). Left basket floor-center (5.25, 25),
right (88.75, 25).

The index is a permanent contract (it is the detector's keypoint channel), so
this mapping is APPEND-ONLY — never reorder. It is a *different* schema from the
16-point halfcourt `court.KEYPOINT_NAMES` (our own pseudo-label factory); both
coexist (see docs/decisions.md ADR-004).
"""

from __future__ import annotations

import numpy as np

COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
RIM_INSET_FT = 5.25  # basket center, from baseline

# schema index -> (x, y) in feet. See module docstring for the coordinate frame.
NBA_FULLCOURT_FT: dict[int, tuple[float, float]] = {
    0: (0.0, 50.0),  # far-left corner (left baseline / far sideline)
    1: (0.0, 47.0),  # left corner-3 at baseline, far side (3 ft from sideline)
    2: (0.0, 33.0),  # left lane line at baseline, far side
    3: (0.0, 17.0),  # left lane line at baseline, near side
    4: (0.0, 3.0),  # left corner-3 at baseline, near side
    5: (0.0, 0.0),  # near-left corner
    6: (5.25, 25.0),  # left basket center (ELEVATED — see ELEVATED_KEYPOINTS)
    7: (14.0, 47.0),  # left corner-3 elbow, far (straight meets arc)
    8: (14.0, 3.0),  # left corner-3 elbow, near
    9: (19.0, 33.0),  # left free-throw elbow, far (FT line / lane line)
    10: (19.0, 25.0),  # left free-throw line center
    11: (19.0, 17.0),  # left free-throw elbow, near
    12: (28.0, 50.0),  # left coaching-box hash, far sideline (28 ft from baseline)
    13: (29.0, 25.0),  # left three-point arc top
    14: (28.0, 0.0),  # left coaching-box hash, near sideline
    15: (47.0, 50.0),  # halfcourt line at far sideline
    16: (47.0, 25.0),  # center court
    17: (47.0, 0.0),  # halfcourt line at near sideline
    18: (66.0, 50.0),  # right coaching-box hash, far sideline
    19: (65.0, 25.0),  # right three-point arc top
    20: (66.0, 0.0),  # right coaching-box hash, near sideline
    21: (75.0, 33.0),  # right free-throw elbow, far
    22: (75.0, 25.0),  # right free-throw line center
    23: (75.0, 17.0),  # right free-throw elbow, near
    24: (80.0, 47.0),  # right corner-3 elbow, far
    25: (80.0, 3.0),  # right corner-3 elbow, near
    26: (88.75, 25.0),  # right basket center (ELEVATED)
    27: (94.0, 50.0),  # far-right corner
    28: (94.0, 47.0),  # right corner-3 at baseline, far
    29: (94.0, 33.0),  # right lane line at baseline, far
    30: (94.0, 17.0),  # right lane line at baseline, near
    31: (94.0, 3.0),  # right corner-3 at baseline, near
    32: (94.0, 0.0),  # near-right corner
}

NUM_KEYPOINTS = 33

# short descriptive names, index-aligned with NBA_FULLCOURT_FT
KEYPOINT_NAMES: list[str] = [
    "far-left-corner",
    "left-corner3-baseline-far",
    "left-lane-baseline-far",
    "left-lane-baseline-near",
    "left-corner3-baseline-near",
    "near-left-corner",
    "left-basket",
    "left-corner3-elbow-far",
    "left-corner3-elbow-near",
    "left-ft-elbow-far",
    "left-ft-center",
    "left-ft-elbow-near",
    "left-hash-far",
    "left-arc-top",
    "left-hash-near",
    "center-far",
    "center",
    "center-near",
    "right-hash-far",
    "right-arc-top",
    "right-hash-near",
    "right-ft-elbow-far",
    "right-ft-center",
    "right-ft-elbow-near",
    "right-corner3-elbow-far",
    "right-corner3-elbow-near",
    "right-basket",
    "far-right-corner",
    "right-corner3-baseline-far",
    "right-lane-baseline-far",
    "right-lane-baseline-near",
    "right-corner3-baseline-near",
    "near-right-corner",
]

# The basket centers are ~10 ft above the court plane, so their image projection
# does not obey the court-plane homography (parallax). They validate ~1 ft worse
# than every planar point (median ~1.1 ft vs ~0.2 ft), so exclude them when
# fitting or scoring a planar homography.
ELEVATED_KEYPOINTS: frozenset[int] = frozenset({6, 26})

# indices safe to use for planar homography fitting / scoring
PLANAR_KEYPOINTS: tuple[int, ...] = tuple(
    i for i in range(NUM_KEYPOINTS) if i not in ELEVATED_KEYPOINTS
)


def template_array(indices: list[int] | None = None) -> np.ndarray:
    """(K, 2) feet coordinates for the given indices (default: all 33)."""
    idx = list(range(NUM_KEYPOINTS)) if indices is None else indices
    return np.array([NBA_FULLCOURT_FT[i] for i in idx], dtype=float)
