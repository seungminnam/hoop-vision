from hoopvision.events import ShotConfig, detect_shots, interpolate_track

# Rim: x 100..120 (width 20), top y=100, center y=105. Image y grows downward.
RIM = (100.0, 100.0, 120.0, 110.0)
FPS = 30.0
CFG = ShotConfig()


def rising_ball(frames: int = 10) -> list[tuple[float, float]]:
    """Ball travels toward the rim and ends above the rim top, in the window."""
    return [(60 + 5 * i, 200 - 13 * i) for i in range(frames)]  # ends (105, 83)


def test_made_shot_detected():
    track = rising_ball() + [(110.0, 95.0), (110.0, 103.0), (110.0, 112.0), (110.0, 130.0)]
    result = detect_shots(track, RIM, FPS, CFG)
    assert result.available
    assert [e.outcome for e in result.events] == ["made"]
    assert result.events[0].frame == 8  # first frame above rim top in window


def test_airball_miss_detected():
    # Ball sails over the rim and descends far outside the horizontal window.
    track = rising_ball() + [(140.0, 90.0), (170.0, 95.0), (200.0, 108.0), (230.0, 140.0)]
    result = detect_shots(track, RIM, FPS, CFG)
    assert result.available
    assert [e.outcome for e in result.events] == ["missed"]


def test_no_attempt_when_ball_stays_low():
    track = [(60.0 + 3 * i, 300.0) for i in range(40)]  # dribbling far below rim
    result = detect_shots(track, RIM, FPS, CFG)
    assert result.available
    assert result.events == []


def test_unresolved_attempt_at_clip_end_is_miss():
    result = detect_shots(rising_ball(), RIM, FPS, CFG)
    assert [e.outcome for e in result.events] == ["missed"]


def test_two_shots_in_one_clip():
    made = rising_ball() + [(110.0, 95.0), (110.0, 103.0), (110.0, 112.0)]
    gap = [(60.0, 300.0)] * (CFG.cooldown_frames + 1)
    result = detect_shots(made + gap + made, RIM, FPS, CFG)
    assert [e.outcome for e in result.events] == ["made", "made"]


def test_quality_gate_blocks_sparse_ball_track():
    track = [None] * 70 + [(110.0, 95.0)] * 30
    result = detect_shots(track, RIM, FPS, CFG)
    assert not result.available
    assert "coverage" in result.reason
    assert result.events == []


def test_no_rim_means_unavailable():
    result = detect_shots(rising_ball(), None, FPS, CFG)
    assert not result.available
    assert "rim" in result.reason


def test_interpolate_fills_small_gaps_only():
    track = [(0.0, 0.0), None, None, (3.0, 3.0), None, None, None, None, (8.0, 8.0)]
    out = interpolate_track(track, max_gap=2)
    assert out[1] == (1.0, 1.0) and out[2] == (2.0, 2.0)  # gap of 2: filled
    assert out[4] is None and out[7] is None  # gap of 4 > max_gap: left empty


def test_interpolation_bridges_detection_dropout_through_rim():
    # Ball detection drops out exactly at the rim crossing — interpolation
    # should still produce a MADE event.
    track = rising_ball() + [(110.0, 95.0), None, None, (110.0, 121.0), (110.0, 135.0)]
    result = detect_shots(track, RIM, FPS, CFG)
    assert [e.outcome for e in result.events] == ["made"]
