"""Court registration runtime: homography fit, smoothing, and fallback."""

import cv2
import numpy as np

from hoopvision import registration as reg
from hoopvision.court_template import NBA_FULLCOURT_FT, PLANAR_KEYPOINTS, template_array


def _synthetic_homography():
    """A plausible feet->image homography (broadcast-ish perspective)."""
    feet = np.array([[0, 0], [94, 0], [94, 50], [0, 50]], float)
    img = np.array([[100, 600], [1180, 640], [980, 220], [300, 210]], float)
    h, _ = cv2.findHomography(feet, img, 0)
    return h


def _project_points(h_feet2img, indices, noise=0.0, rng=None):
    pts = {}
    for i in indices:
        xy = np.array(NBA_FULLCOURT_FT[i], float).reshape(1, 1, 2)
        p = cv2.perspectiveTransform(xy, h_feet2img).reshape(2)
        if noise and rng is not None:
            p = p + rng.normal(0, noise, 2)
        pts[i] = (float(p[0]), float(p[1]))
    return pts


def test_fit_recovers_court_coordinates():
    h_f2i = _synthetic_homography()
    pts = _project_points(h_f2i, list(PLANAR_KEYPOINTS))
    fit = reg.fit_homography(pts)
    assert fit is not None
    H, inliers = fit
    assert len(inliers) >= 4
    # a known point maps back to its feet coordinate
    got = reg.image_to_court(H, np.array([pts[13]]))[0]  # left arc top
    np.testing.assert_allclose(got, template_array([13])[0], atol=0.05)


def test_fit_ignores_elevated_basket_points():
    h_f2i = _synthetic_homography()
    pts = _project_points(h_f2i, [0, 5, 27, 32])  # 4 planar corners
    pts[6] = (9999.0, 9999.0)  # garbage elevated point must be ignored
    fit = reg.fit_homography(pts)
    assert fit is not None
    assert 6 not in fit[1]


def test_fit_returns_none_below_min_points():
    h_f2i = _synthetic_homography()
    pts = _project_points(h_f2i, [0, 5, 27])  # only 3
    assert reg.fit_homography(pts) is None


def test_registrar_smoothing_reduces_jitter():
    h_f2i = _synthetic_homography()
    rng = np.random.default_rng(0)
    idx = list(PLANAR_KEYPOINTS)
    center_img_true = cv2.perspectiveTransform(
        np.array([[47.0, 25.0]]).reshape(1, 1, 2), h_f2i
    ).reshape(2)

    raw_err, smooth_err = [], []
    r = reg.CourtRegistrar(alpha=0.3)
    for _ in range(40):
        pts = _project_points(h_f2i, idx, noise=3.0, rng=rng)
        raw = reg.fit_homography(pts)[0]
        H = r.update(pts)
        # where does court center land back in the image under each H^-1?
        for H_est, bucket in ((raw, raw_err), (H, smooth_err)):
            c = cv2.perspectiveTransform(
                np.array([[47.0, 25.0]]).reshape(1, 1, 2), np.linalg.inv(H_est)
            ).reshape(2)
            bucket.append(np.linalg.norm(c - center_img_true))
    # skip warm-up; smoothed variance should be well below raw
    assert np.std(smooth_err[5:]) < np.std(raw_err[5:])


def test_registrar_fallback_then_unavailable():
    h_f2i = _synthetic_homography()
    r = reg.CourtRegistrar(max_misses=3)
    good = _project_points(h_f2i, list(PLANAR_KEYPOINTS))
    assert r.update(good) is not None

    # too few points -> coast on last good H
    for _ in range(3):
        H = r.update({0: (1.0, 2.0)})
        assert H is not None
    # exceeded max_misses -> unavailable
    assert r.update({0: (1.0, 2.0)}) is None
    assert r.homography is None


def test_registrar_recovers_after_gap():
    h_f2i = _synthetic_homography()
    r = reg.CourtRegistrar(max_misses=2)
    good = _project_points(h_f2i, list(PLANAR_KEYPOINTS))
    r.update(good)
    for _ in range(5):
        r.update({})  # nothing visible -> unavailable
    assert r.homography is None
    assert r.update(good) is not None  # re-acquires cleanly
