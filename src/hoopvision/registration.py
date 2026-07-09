"""Per-frame court registration: detected keypoints -> image↔feet homography.

v2 §4.2 Phase 2 runtime. The Phase-1 detector places the 33 court keypoints on a
broadcast frame; this maps them to the NBA feet template
(`court_template.NBA_FULLCOURT_FT`) with a RANSAC homography, and smooths the
result over time so the minimap does not jitter.

Design (mirrors the ball-coverage gate philosophy in v1): only *planar* points
(the two baskets are 10 ft up — excluded) drive the fit; a frame needs >=4
confident points or it is skipped and the last good homography carries forward
for a few frames, after which registration reports "unavailable" rather than
guessing.

Everything here is pure geometry (no video/model I/O) so it is unit-tested;
`scripts/register_court.py` supplies detected keypoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .court_template import PLANAR_KEYPOINTS, template_array

# a well-spread, always-planar basis for temporal smoothing (corners, center,
# arc tops) — projected to image each frame and EMA'd there
_SMOOTH_INDICES: tuple[int, ...] = (0, 5, 27, 32, 16, 13, 19)


def fit_homography(
    points: dict[int, tuple[float, float]],
    ransac_thresh_px: float = 8.0,
    min_points: int = 4,
) -> tuple[np.ndarray, list[int]] | None:
    """Fit image->feet homography from {schema_idx: (x, y) px}. None if too few.

    Only planar keypoints are used (elevated baskets would bias the plane).
    Returns (H_img2feet, inlier_indices) or None.
    """
    idx = [i for i in points if i in PLANAR_KEYPOINTS]
    if len(idx) < min_points:
        return None
    src = np.array([points[i] for i in idx], dtype=float)
    dst = template_array(idx)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh_px)
    if h is None:
        return None
    inliers = [i for i, m in zip(idx, mask.ravel(), strict=True) if m]
    if len(inliers) < min_points:
        return None
    return h, inliers


@dataclass
class CourtRegistrar:
    """Temporally-smoothed court registration with a last-good fallback.

    `update(points)` returns the current smoothed image->feet homography, or
    None when registration is unavailable (too few points for `max_misses`
    consecutive frames).
    """

    alpha: float = 0.35  # EMA weight on the newest frame
    min_points: int = 4
    max_misses: int = 15  # keep coasting on the last good H for this many frames
    ransac_thresh_px: float = 8.0

    _smooth_img: np.ndarray | None = field(default=None, init=False, repr=False)
    _H: np.ndarray | None = field(default=None, init=False, repr=False)
    misses: int = field(default=0, init=False)

    @property
    def homography(self) -> np.ndarray | None:
        return self._H

    def reset(self) -> None:
        self._smooth_img = None
        self._H = None
        self.misses = 0

    def update(self, points: dict[int, tuple[float, float]]) -> np.ndarray | None:
        fit = fit_homography(points, self.ransac_thresh_px, self.min_points)
        if fit is None:
            # coast on last good H for a while, then declare unavailable
            self.misses += 1
            if self._H is not None and self.misses <= self.max_misses:
                return self._H
            if self.misses > self.max_misses:
                self.reset()
            return None

        self.misses = 0
        h_img2feet, _ = fit
        feet = template_array(list(_SMOOTH_INDICES))
        h_feet2img = np.linalg.inv(h_img2feet)
        img_now = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), h_feet2img).reshape(-1, 2)

        if self._smooth_img is None:
            self._smooth_img = img_now
        else:
            self._smooth_img = (1 - self.alpha) * self._smooth_img + self.alpha * img_now

        # refit the smoothed feet<->image correspondence
        h_smooth, _ = cv2.findHomography(self._smooth_img, feet, 0)
        self._H = h_smooth if h_smooth is not None else h_img2feet
        return self._H


def image_to_court(H_img2feet: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Map (N, 2) image pixels to (N, 2) court feet."""
    p = np.asarray(pts_xy, float).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H_img2feet).reshape(-1, 2)


def court_polylines_ft() -> list[np.ndarray]:
    """NBA full-court line segments as (N, 2) feet polylines (for drawing)."""
    lines: list[np.ndarray] = []
    # outer boundary + halfcourt
    lines.append(np.array([[0, 0], [94, 0], [94, 50], [0, 50], [0, 0]], float))
    lines.append(np.array([[47, 0], [47, 50]], float))
    # center circle
    t = np.linspace(0, 2 * np.pi, 60)
    lines.append(np.column_stack([47 + 6 * np.cos(t), 25 + 6 * np.sin(t)]))
    for bx, s in ((0, 1), (94, -1)):
        ftx = bx + s * 19
        lines.append(np.array([[bx, 17], [ftx, 17], [ftx, 33], [bx, 33]], float))
        lines.append(np.column_stack([bx + s * 19 + 6 * np.cos(t), 25 + 6 * np.sin(t)]))
        # 3-pt: corner-3 straights + arc from the rim
        lines.append(np.array([[bx, 3], [bx + s * 14, 3]], float))
        lines.append(np.array([[bx, 47], [bx + s * 14, 47]], float))
        a = np.linspace(-1.2, 1.2, 60)
        lines.append(
            np.column_stack([bx + s * 5.25 + s * 23.75 * np.cos(a), 25 + 23.75 * np.sin(a)])
        )
    return lines
