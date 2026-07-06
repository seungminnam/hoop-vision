import numpy as np
import pytest

from hoopvision.court import COURT_LENGTH_FT, COURT_WIDTH_FT, CourtCalibration

CORNERS_FT = [
    (0.0, 0.0),
    (COURT_WIDTH_FT, 0.0),
    (COURT_WIDTH_FT, COURT_LENGTH_FT),
    (0.0, COURT_LENGTH_FT),
]


def test_identity_when_image_equals_court():
    calib = CourtCalibration.from_points(CORNERS_FT, CORNERS_FT)
    pts = np.array([[10.0, 10.0], [25.0, 40.0]])
    np.testing.assert_allclose(calib.to_court(pts), pts, atol=1e-8)
    assert calib.reprojection_error_ft() < 1e-8


def test_recovers_known_perspective():
    # A plausible fixed-camera view: court corners land at these pixels.
    image_points = [(120, 620), (1180, 640), (980, 180), (300, 170)]
    calib = CourtCalibration.from_points(image_points, CORNERS_FT)
    assert calib.reprojection_error_ft() < 1e-6

    # Round trip: image → court → image
    probe = np.array([[640.0, 400.0], [500.0, 300.0]])
    back = calib.to_image(calib.to_court(probe))
    np.testing.assert_allclose(back, probe, atol=1e-6)


def test_spec_accepts_sub_foot_reprojection():
    image_points = [(100, 700), (1150, 710), (900, 200), (350, 190)]
    calib = CourtCalibration.from_points(image_points, CORNERS_FT)
    assert calib.reprojection_error_ft() < 1.0  # SPEC §6 W3 acceptance


def test_requires_four_point_pairs():
    with pytest.raises(ValueError):
        CourtCalibration.from_points(CORNERS_FT[:3], CORNERS_FT[:3])
    with pytest.raises(ValueError):
        CourtCalibration.from_points(CORNERS_FT, CORNERS_FT[:3])


def test_in_bounds_mask():
    calib = CourtCalibration.from_points(CORNERS_FT, CORNERS_FT)
    pts = np.array([[25.0, 20.0], [-10.0, 5.0], [25.0, 60.0]])
    np.testing.assert_array_equal(calib.in_bounds(pts), [True, False, False])


def test_save_load_roundtrip(tmp_path):
    image_points = [(120, 620), (1180, 640), (980, 180), (300, 170)]
    calib = CourtCalibration.from_points(image_points, CORNERS_FT)
    path = tmp_path / "calib.json"
    calib.save(path)
    loaded = CourtCalibration.load(path)
    np.testing.assert_allclose(loaded.homography, calib.homography)
    probe = np.array([[640.0, 400.0]])
    np.testing.assert_allclose(loaded.to_court(probe), calib.to_court(probe))
