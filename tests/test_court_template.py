"""Invariants for the 33-point NBA full-court template (v2 §4.2 Phase 2)."""

import numpy as np
import pytest

from hoopvision import court_template as ct

# left index -> mirror right index (across the halfcourt line x=47)
MIRROR_PAIRS = [
    (0, 27),
    (1, 28),
    (2, 29),
    (3, 30),
    (4, 31),
    (5, 32),
    (6, 26),
    (7, 24),
    (8, 25),
    (9, 21),
    (10, 22),
    (11, 23),
    (12, 18),
    (13, 19),
    (14, 20),
]
SELF_MIRROR = [15, 16, 17]  # on the halfcourt line


def test_thirty_three_points_and_names_aligned():
    assert ct.NUM_KEYPOINTS == 33
    assert len(ct.NBA_FULLCOURT_FT) == 33
    assert len(ct.KEYPOINT_NAMES) == 33
    assert sorted(ct.NBA_FULLCOURT_FT) == list(range(33))


def test_left_right_symmetry():
    for lo, hi in MIRROR_PAIRS:
        xl, yl = ct.NBA_FULLCOURT_FT[lo]
        xr, yr = ct.NBA_FULLCOURT_FT[hi]
        assert xr == pytest.approx(ct.COURT_LENGTH_FT - xl)
        assert yr == pytest.approx(yl)


def test_halfcourt_points_on_center_line():
    for i in SELF_MIRROR:
        x, _ = ct.NBA_FULLCOURT_FT[i]
        assert x == pytest.approx(ct.COURT_LENGTH_FT / 2)


def test_court_corners_and_dimensions():
    corners = {ct.NBA_FULLCOURT_FT[i] for i in (0, 5, 27, 32)}
    assert corners == {(0.0, 0.0), (0.0, 50.0), (94.0, 0.0), (94.0, 50.0)}


def test_lane_width_is_sixteen_feet():
    # left lane baseline edges (far/near) span the 16 ft painted lane
    y_far = ct.NBA_FULLCOURT_FT[2][1]
    y_near = ct.NBA_FULLCOURT_FT[3][1]
    assert abs(y_far - y_near) == pytest.approx(16.0)


def test_free_throw_line_nineteen_feet():
    for i in (9, 10, 11):  # left FT elbows + center
        assert ct.NBA_FULLCOURT_FT[i][0] == pytest.approx(19.0)


def test_arc_top_matches_23_75_radius():
    # left arc top x = basket inset + 3-pt radius
    assert ct.NBA_FULLCOURT_FT[13][0] == pytest.approx(ct.RIM_INSET_FT + 23.75)
    assert ct.NBA_FULLCOURT_FT[13][1] == pytest.approx(25.0)


def test_corner_three_three_feet_from_sideline():
    # baseline corner-3 marks sit 3 ft inside each sideline (y=3 and y=47)
    assert ct.NBA_FULLCOURT_FT[1][1] == pytest.approx(47.0)
    assert ct.NBA_FULLCOURT_FT[4][1] == pytest.approx(3.0)


def test_elevated_and_planar_sets():
    assert ct.ELEVATED_KEYPOINTS == frozenset({6, 26})
    assert set(ct.PLANAR_KEYPOINTS).isdisjoint(ct.ELEVATED_KEYPOINTS)
    assert len(ct.PLANAR_KEYPOINTS) == 31


def test_template_array_shapes():
    assert ct.template_array().shape == (33, 2)
    assert ct.template_array([0, 5, 27]).shape == (3, 2)
    np.testing.assert_allclose(ct.template_array([16])[0], [47.0, 25.0])


def test_cross_validates_against_roboflow_official_config():
    """Our derived template matches the dataset authors' published config."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from compare_court_template import comparison  # noqa: E402

    c = comparison()
    # mean agreement across all 33 points is a couple of inches
    assert c["mean_ft"] < 0.15
    # only the 4 sideline hashes + 4 corner-3 elbows differ meaningfully;
    # every other point is essentially identical (rounding only)
    assert c["within_0_1_ft"] >= 29
