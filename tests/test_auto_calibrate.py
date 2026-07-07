"""Tests for the pure-logic parts of scripts/auto_calibrate.py."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from auto_calibrate import (  # noqa: E402
    PAINT_CORNERS_FT,
    order_paint_corners,
    paint_mask,
    pick_component,
    quad_from_mask,
)

# Real capture from hudl_static2: paint corners in the order
# (baseline-left, baseline-right, ft-left, ft-right).
HUDL_CORNERS_PX = np.array([[439.0, 130.0], [501.0, 163.0], [258.0, 162.0], [306.0, 193.0]])


def hudl_basket_px() -> tuple[float, float]:
    """Rim pixel implied by the hudl_static2 correspondences."""
    h, _ = cv2.findHomography(PAINT_CORNERS_FT, HUDL_CORNERS_PX)
    rim = cv2.perspectiveTransform(np.array([[[25.0, 5.25]]]), h).reshape(2)
    return float(rim[0]), float(rim[1])


@pytest.mark.parametrize("shift", range(4))
@pytest.mark.parametrize("reverse", [False, True])
def test_order_recovers_real_perspective_quad(shift: int, reverse: bool):
    quad = np.roll(HUDL_CORNERS_PX[[0, 1, 3, 2]], shift, axis=0)  # hull order, any start
    if reverse:
        quad = quad[::-1]
    ordered = order_paint_corners(quad, hudl_basket_px())
    np.testing.assert_allclose(ordered, HUDL_CORNERS_PX)


def test_order_with_elevated_rim_detection():
    # The detector finds the physical rim ~10 ft above the floor: in this
    # clip's frame the paint quad sits at y 128-198 but the rim is at (411, 63).
    quad = np.array([[261.3, 166.5], [316.6, 198.2], [488.6, 146.5], [438.7, 136.2]])
    ordered = order_paint_corners(quad, basket_px=(411.0, 63.0))
    np.testing.assert_allclose(
        ordered, np.array([[438.7, 136.2], [488.6, 146.5], [261.3, 166.5], [316.6, 198.2]])
    )


def test_order_overhead_view():
    # Overhead camera on the spectator side, basket below the baseline edge.
    bl, br, fl, fr = (200.0, 400.0), (440.0, 400.0), (200.0, 100.0), (440.0, 100.0)
    quad = np.array([fr, bl, fl, br])  # arbitrary order
    ordered = order_paint_corners(quad, basket_px=(320.0, 450.0))
    np.testing.assert_allclose(ordered, np.array([bl, br, fl, fr]))


def test_order_flip_swaps_left_right():
    bl, br, fl, fr = (200.0, 400.0), (440.0, 400.0), (200.0, 100.0), (440.0, 100.0)
    quad = np.array([bl, br, fr, fl])
    ordered = order_paint_corners(quad, basket_px=(320.0, 450.0), flip=True)
    np.testing.assert_allclose(ordered, np.array([br, bl, fr, fl]))


def test_order_rejects_bad_shape():
    with pytest.raises(ValueError):
        order_paint_corners(np.zeros((3, 2)), basket_px=(0.0, 0.0))


def test_segmentation_recovers_drawn_quad():
    frame = np.full((360, 640, 3), 120, np.uint8)  # gray floor (S=0, excluded)
    drawn = np.array([[420, 120], [500, 160], [300, 190], [250, 155]])  # convex, hull order-ish
    cv2.fillPoly(frame, [drawn], (40, 40, 220))  # saturated red paint
    mask = paint_mask(frame, s_min=120, v_min=165)
    component = pick_component(mask, min_area=300, basket_px=(460.0, 110.0), probe_px=None)
    quad = quad_from_mask(component)
    # every recovered corner is within a few px of a drawn corner
    for corner in quad:
        assert np.min(np.linalg.norm(drawn - corner, axis=1)) < 4.0


def test_pick_component_prefers_probe_and_basket_anchor():
    mask = np.zeros((100, 200), np.uint8)
    mask[10:40, 10:60] = 1  # far blob
    mask[60:90, 140:190] = 1  # near-basket blob
    by_basket = pick_component(mask, min_area=100, basket_px=(180.0, 80.0), probe_px=None)
    assert by_basket[70, 160] == 1 and by_basket[20, 20] == 0
    by_probe = pick_component(mask, min_area=100, basket_px=None, probe_px=(20.0, 20.0))
    assert by_probe[20, 20] == 1 and by_probe[70, 160] == 0
