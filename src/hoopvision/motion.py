"""Global camera-motion estimation + compensation for panning clips.

v1's homography and tracking assume a static camera. On a panning/zooming
camera (e.g. Hudl auto-tracking) every box shifts together each frame, which
violates ByteTrack's static-scene motion model and desynchronises a fixed
homography.

This estimates the *global* frame-to-frame motion from background optical flow
(players masked out) as a partial-affine transform (translation + rotation +
uniform scale), and accumulates it into a transform mapping each frame's pixels
back to a reference frame. Feeding the tracker boxes warped into that stabilised
reference frame lets ByteTrack see a nearly static scene; the same transform is
the cheap "pan/zoom" form of dynamic homography (a stepping stone to v2's
keypoint registration).

Parallax means a single global affine cannot be exact for a real 3-D scene, so
this helps pans/zooms, not arbitrary camera translation — stated honestly.
"""

from __future__ import annotations

import cv2
import numpy as np


def as_3x3(affine: np.ndarray) -> np.ndarray:
    """Promote a 2x3 affine to a 3x3 matrix."""
    m = np.eye(3, dtype=np.float64)
    m[:2] = affine
    return m


def warp_box(
    xyxy: tuple[float, float, float, float], affine: np.ndarray
) -> tuple[float, float, float, float]:
    """Transform a box by an affine: move the center, scale the size by the
    affine's uniform scale. Keeps boxes axis-aligned (good enough for IoU)."""
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    ncx, ncy = affine[:, :2] @ [cx, cy] + affine[:, 2]
    scale = float(np.sqrt(abs(np.linalg.det(affine[:, :2]))))
    nw, nh = w * scale, h * scale
    return (ncx - nw / 2, ncy - nh / 2, ncx + nw / 2, ncy + nh / 2)


def _foreground_mask(shape: tuple[int, int], boxes: list, pad: float = 0.15) -> np.ndarray:
    """255 on background, 0 inside (padded) player boxes — where features are OK."""
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        px1 = int(max(0, x1 - pad * bw))
        py1 = int(max(0, y1 - pad * bh))
        px2 = int(min(w, x2 + pad * bw))
        py2 = int(min(h, y2 + pad * bh))
        mask[py1:py2, px1:px2] = 0
    return mask


def estimate_affine(
    prev_gray: np.ndarray, cur_gray: np.ndarray, background_mask: np.ndarray | None = None
) -> np.ndarray:
    """Partial-affine mapping prev-frame pixels → cur-frame pixels (2x3).

    Tracks strong background corners from prev to cur with Lucas–Kanade and fits
    a similarity transform. Returns identity when there is too little signal.
    """
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pts_prev = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8, mask=background_mask
    )
    if pts_prev is None or len(pts_prev) < 6:
        return identity
    pts_cur, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts_prev, None)
    if pts_cur is None:
        return identity
    good = status.ravel() == 1
    if good.sum() < 6:
        return identity
    m, inliers = cv2.estimateAffinePartial2D(
        pts_prev[good], pts_cur[good], method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if m is None or inliers is None or int(inliers.sum()) < 6:
        return identity
    return m


class CameraMotionEstimator:
    """Accumulates frame→reference transforms across a clip.

    `update(gray, player_boxes)` returns the current transform mapping
    *current-frame* pixels to the *reference* (first) frame's pixels.
    """

    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._ref_from_prev = np.eye(3, dtype=np.float64)  # prev-frame → reference

    def update(self, gray: np.ndarray, player_boxes: list) -> np.ndarray:
        if self._prev_gray is None:
            self._prev_gray = gray
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mask = _foreground_mask(gray.shape[:2], player_boxes)
        prev_from_cur = estimate_affine(gray, self._prev_gray, mask)  # cur → prev
        # reference ← cur = (reference ← prev) @ (prev ← cur)
        ref_from_cur = self._ref_from_prev @ as_3x3(prev_from_cur)
        self._prev_gray = gray
        self._ref_from_prev = ref_from_cur
        return ref_from_cur[:2]
