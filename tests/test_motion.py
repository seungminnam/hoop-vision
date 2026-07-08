"""Tests for camera-motion estimation (src/hoopvision/motion.py)."""

import cv2
import numpy as np

from hoopvision.motion import CameraMotionEstimator, as_3x3, estimate_affine, warp_box


def _textured_frame(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    # add strong corners the tracker can lock onto
    for _ in range(40):
        x, y = rng.integers(20, 300), rng.integers(20, 220)
        cv2.rectangle(img, (x, y), (x + 8, y + 8), int(rng.integers(0, 255)), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))


def test_estimate_recovers_translation():
    base = _textured_frame()
    shifted = _shift(base, 10, -6)  # camera moved so content shifted +10,-6
    # prev=base, cur=shifted → mapping prev→cur should be +10,-6
    m = estimate_affine(base, shifted)
    assert abs(m[0, 2] - 10) < 1.5
    assert abs(m[1, 2] - (-6)) < 1.5
    assert abs(m[0, 0] - 1.0) < 0.05  # near-identity scale/rotation


def test_estimate_identity_on_same_frame():
    base = _textured_frame()
    m = estimate_affine(base, base)
    assert abs(m[0, 2]) < 1.0 and abs(m[1, 2]) < 1.0


def test_warp_box_translation():
    m = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
    assert warp_box((10, 10, 20, 20), m) == (15.0, 7.0, 25.0, 17.0)


def test_warp_box_scale_changes_size():
    m = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])  # 2x zoom about origin
    x1, y1, x2, y2 = warp_box((10, 10, 20, 20), m)
    assert (x2 - x1) == 20.0 and (y2 - y1) == 20.0  # 10px box → 20px
    assert ((x1 + x2) / 2, (y1 + y2) / 2) == (30.0, 30.0)  # center scaled


def test_estimator_accumulates_reference_transform():
    base = _textured_frame()
    est = CameraMotionEstimator()
    est.update(base, [])  # reference frame → identity
    ref_from_cur = est.update(_shift(base, 8, 0), [])
    # cur is shifted +8; mapping cur→reference should undo it (≈ -8)
    assert abs(ref_from_cur[0, 2] - (-8)) < 2.0


def test_as_3x3_roundtrip():
    m = np.array([[1.0, 0.2, 3.0], [-0.1, 1.0, 4.0]])
    assert np.allclose(as_3x3(m)[:2], m)
    assert np.allclose(as_3x3(m)[2], [0, 0, 1])
