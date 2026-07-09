"""Tests for court-keypoint pseudo-labeling + augmentation (pure math)."""

import cv2
import numpy as np

from hoopvision.court import (
    COURT_LENGTH_FT,
    COURT_WIDTH_FT,
    KEYPOINT_COURT_FT,
    CourtCalibration,
)
from hoopvision.keypoints import (
    FLIP_INDEX,
    NUM_KEYPOINTS,
    V_ABSENT,
    V_VISIBLE,
    color_jitter,
    flip_keypoints,
    project_keypoints,
    random_homography,
    warp_keypoints,
)

CORNERS_FT = [
    (0.0, 0.0),
    (COURT_WIDTH_FT, 0.0),
    (COURT_WIDTH_FT, COURT_LENGTH_FT),
    (0.0, COURT_LENGTH_FT),
]
# A plausible fixed-camera view (from test_court) — image px for those corners.
IMAGE_PX = [(120, 620), (1180, 640), (980, 180), (300, 170)]
W, H = 1280, 720


def _calib() -> CourtCalibration:
    return CourtCalibration.from_points(IMAGE_PX, CORNERS_FT)


def test_project_marks_visibility_and_zeros_absent():
    kp = project_keypoints(_calib(), W, H)
    assert kp.shape == (NUM_KEYPOINTS, 3)
    # This wide view should place most landmarks on screen.
    assert (kp[:, 2] == V_VISIBLE).sum() >= NUM_KEYPOINTS - 2
    # Every visible keypoint sits inside the frame; absent ones are zeroed.
    for x, y, v in kp:
        if v == V_VISIBLE:
            assert 0 <= x < W and 0 <= y < H
        else:
            assert v == V_ABSENT and x == 0 and y == 0


def test_project_drops_offscreen_landmarks():
    # Tiny frame: almost nothing projects inside → most keypoints absent.
    kp = project_keypoints(_calib(), 40, 40)
    assert (kp[:, 2] == V_VISIBLE).sum() < NUM_KEYPOINTS


def test_flip_index_is_symmetric_involution():
    # Flipping labels twice returns the original ordering.
    np.testing.assert_array_equal(FLIP_INDEX[FLIP_INDEX], np.arange(NUM_KEYPOINTS))
    # And the mapped court point is the true horizontal mirror.
    mirrored = KEYPOINT_COURT_FT.copy()
    mirrored[:, 0] = COURT_WIDTH_FT - mirrored[:, 0]
    np.testing.assert_allclose(KEYPOINT_COURT_FT[FLIP_INDEX], mirrored, atol=1e-9)


def test_flip_keypoints_mirrors_x_and_relabels():
    kp = project_keypoints(_calib(), W, H)
    flipped = flip_keypoints(kp, W)
    # A visible keypoint's flipped partner mirrors its x about the frame center.
    for i in range(NUM_KEYPOINTS):
        j = FLIP_INDEX[i]
        if kp[i, 2] == V_VISIBLE:
            assert flipped[j, 2] == V_VISIBLE
            assert abs(flipped[j, 0] - (W - 1 - kp[i, 0])) < 1e-6
            assert abs(flipped[j, 1] - kp[i, 1]) < 1e-6


def test_warp_identity_is_noop_for_visible_points():
    kp = project_keypoints(_calib(), W, H)
    warped = warp_keypoints(kp, np.eye(3), W, H)
    vis = kp[:, 2] == V_VISIBLE
    np.testing.assert_allclose(warped[vis, :2], kp[vis, :2], atol=1e-6)
    np.testing.assert_array_equal(warped[:, 2], kp[:, 2])


def test_warp_matches_perspective_transform():
    kp = project_keypoints(_calib(), W, H)
    rng = np.random.default_rng(3)
    h = random_homography(W, H, rng, jitter=0.05)
    warped = warp_keypoints(kp, h, W, H)
    vis = kp[:, 2] == V_VISIBLE
    expect = cv2.perspectiveTransform(kp[vis, :2].reshape(-1, 1, 2), h).reshape(-1, 2)
    kept = warped[vis, 2] == V_VISIBLE
    np.testing.assert_allclose(warped[vis][kept, :2], expect[kept], atol=1e-6)


def test_warp_pushes_points_offscreen_to_absent():
    kp = project_keypoints(_calib(), W, H)
    shift = np.array([[1, 0, 5 * W], [0, 1, 5 * H], [0, 0, 1]], dtype=np.float64)
    warped = warp_keypoints(kp, shift, W, H)
    assert (warped[:, 2] == V_VISIBLE).sum() == 0


def test_random_homography_shape_and_invertible():
    rng = np.random.default_rng(0)
    h = random_homography(W, H, rng)
    assert h.shape == (3, 3)
    assert abs(np.linalg.det(h)) > 1e-9


def test_color_jitter_preserves_shape_and_dtype():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    out = color_jitter(frame, rng)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
