"""Court-keypoint pseudo-labeling and augmentation for v2 field registration.

A working homography (manual or `auto_calibrate.py`) plus the fixed
`court.COURT_KEYPOINTS` schema is a free keypoint annotator: project every
landmark from court-feet into the image and record whether it lands inside the
frame. Augmenting those frames with random homography warps + color jitter
turns a handful of static clips into a pan/zoom keypoint dataset without a
single hand click.

Everything here is pure array math (no video I/O) so it is unit-testable;
`scripts/build_court_keypoints.py` wires it to clips and disk.
"""

from __future__ import annotations

import cv2
import numpy as np

from .court import KEYPOINT_COURT_FT, CourtCalibration

# COCO keypoint visibility flags.
V_ABSENT = 0  # not labeled (projects outside the frame)
V_VISIBLE = 2  # labeled and inside the frame

NUM_KEYPOINTS = len(KEYPOINT_COURT_FT)


def _mirror_index() -> np.ndarray:
    """Index map for a horizontal image flip, derived from court geometry.

    A left-right flip mirrors court x about the halfcourt center line
    (x → 50 − x), so keypoint i must be relabeled as whichever keypoint sits
    at the mirrored court location. Computed from `KEYPOINT_COURT_FT` rather
    than hand-typed so it can never drift out of sync with the schema.
    """
    from .court import COURT_WIDTH_FT

    mirrored = KEYPOINT_COURT_FT.copy()
    mirrored[:, 0] = COURT_WIDTH_FT - mirrored[:, 0]
    index = np.empty(NUM_KEYPOINTS, dtype=int)
    for i, (mx, my) in enumerate(mirrored):
        d = np.abs(KEYPOINT_COURT_FT[:, 0] - mx) + np.abs(KEYPOINT_COURT_FT[:, 1] - my)
        j = int(np.argmin(d))
        if d[j] > 1e-6:
            raise ValueError(
                f"keypoint {i} has no horizontal mirror in the schema "
                "(court layout is not left-right symmetric)"
            )
        index[i] = j
    return index


FLIP_INDEX = _mirror_index()


def project_keypoints(calibration: CourtCalibration, width: int, height: int) -> np.ndarray:
    """Project the court-keypoint schema into one frame.

    Returns an (K, 3) array of columns (x_px, y_px, visibility). A keypoint is
    `V_VISIBLE` when it lands inside the frame, else `V_ABSENT` with its pixel
    coordinates zeroed (no meaningful target for an off-frame landmark).
    """
    px = calibration.to_image(KEYPOINT_COURT_FT)
    out = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
    out[:, :2] = px
    inside = (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    out[inside, 2] = V_VISIBLE
    out[~inside, :2] = 0.0
    return out


def random_homography(
    width: int, height: int, rng: np.random.Generator, jitter: float = 0.12
) -> np.ndarray:
    """A random image→image homography emulating a camera pan/zoom/tilt.

    The frame's four corners are perturbed by up to `jitter · min(w, h)` and a
    homography is fit to the displacement. Larger `jitter` = more aggressive
    virtual camera motion. Returns a 3x3 matrix.
    """
    d = jitter * min(width, height)
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    dst = src + rng.uniform(-d, d, size=src.shape)
    h, _ = cv2.findHomography(src, dst, method=0)
    if h is None:  # pragma: no cover - only on degenerate jitter
        return np.eye(3)
    return h


def warp_keypoints(keypoints: np.ndarray, h: np.ndarray, width: int, height: int) -> np.ndarray:
    """Apply an image→image homography to (K, 3) keypoints, refreshing visibility.

    Points that were `V_ABSENT` stay absent; visible points that warp outside
    the frame become absent (coordinates zeroed).
    """
    out = keypoints.copy().astype(np.float64)
    visible = keypoints[:, 2] > 0
    if visible.any():
        pts = keypoints[visible, :2].reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, h).reshape(-1, 2)
        out[visible, :2] = warped
    inside = (out[:, 0] >= 0) & (out[:, 0] < width) & (out[:, 1] >= 0) & (out[:, 1] < height)
    keep = visible & inside
    out[:, 2] = np.where(keep, V_VISIBLE, V_ABSENT)
    out[~keep, :2] = 0.0
    return out


def flip_keypoints(keypoints: np.ndarray, width: int) -> np.ndarray:
    """Horizontally mirror keypoints for a left-right image flip.

    Reflects x about the frame center and relabels via `FLIP_INDEX` so, e.g.,
    the left baseline corner becomes the right one.
    """
    flipped = keypoints[FLIP_INDEX].copy()
    visible = flipped[:, 2] > 0
    flipped[visible, 0] = (width - 1) - flipped[visible, 0]
    return flipped


def color_jitter(
    frame: np.ndarray,
    rng: np.random.Generator,
    brightness: float = 0.2,
    contrast: float = 0.2,
    hue: float = 8.0,
) -> np.ndarray:
    """Random brightness/contrast/hue shift on a BGR uint8 frame.

    Court appearance varies with gym lighting and broadcast color grading;
    jitter keeps a keypoint model from latching onto one clip's exact tone.
    """
    out = frame.astype(np.float32)
    alpha = 1.0 + rng.uniform(-contrast, contrast)  # contrast
    beta = rng.uniform(-brightness, brightness) * 255.0  # brightness
    out = np.clip(out * alpha + beta, 0, 255).astype(np.uint8)
    if hue > 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.uniform(-hue, hue))) % 180
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out
