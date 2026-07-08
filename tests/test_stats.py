"""Tests for per-player movement stats (src/hoopvision/stats.py)."""

from dataclasses import dataclass

from hoopvision.court import COURT_LENGTH_FT, COURT_WIDTH_FT, CourtCalibration
from hoopvision.stats import MPH_PER_FPS, player_stats


@dataclass
class _Player:
    track_id: int
    xyxy: tuple
    team: int | None = None

    @property
    def foot(self):
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, y2)


@dataclass
class _Record:
    index: int
    players: list


@dataclass
class _Analysis:
    fps: float
    records: list


def _identity_calibration() -> CourtCalibration:
    # image px == court feet, so a foot point at (x, y) px is (x, y) ft
    corners = [(0, 0), (COURT_WIDTH_FT, 0), (COURT_WIDTH_FT, COURT_LENGTH_FT), (0, COURT_LENGTH_FT)]
    return CourtCalibration.from_points(corners, corners)


def _straight_line_analysis(fps=10.0, n=20, step_ft=1.0) -> _Analysis:
    # one player walking +x by step_ft each frame at court y=10
    records = []
    for i in range(n):
        x = 10.0 + i * step_ft
        players = [_Player(1, (x - 1, 9, x + 1, 10), team=0)]
        records.append(_Record(index=i, players=players))
    return _Analysis(fps=fps, records=records)


def test_distance_and_speed_on_straight_line():
    analysis = _straight_line_analysis(fps=10.0, n=20, step_ft=1.0)
    stats = player_stats(analysis, _identity_calibration(), min_frames=5, smooth_window=1)
    assert len(stats) == 1
    s = stats[0]
    assert s.track_id == 1 and s.team == 0
    assert s.frames == 20
    assert abs(s.distance_ft - 19.0) < 1e-6  # 19 one-foot steps
    # 10 ft/s over the whole span
    assert abs(s.avg_speed_mph - 10.0 * MPH_PER_FPS) < 0.1
    assert abs(s.top_speed_mph - 10.0 * MPH_PER_FPS) < 0.1


def test_short_tracks_are_dropped():
    analysis = _straight_line_analysis(n=6)
    assert player_stats(analysis, _identity_calibration(), min_frames=15) == []


def test_teleport_segments_rejected():
    # a single huge jump (bad detection) must not inflate distance
    cal = _identity_calibration()
    records = []
    for i in range(20):
        x = 10.0 if i != 10 else 49.0  # one-frame teleport across the court
        records.append(_Record(index=i, players=[_Player(1, (x - 1, 9, x + 1, 10))]))
    analysis = _Analysis(fps=10.0, records=records)
    s = player_stats(analysis, cal, min_frames=5, smooth_window=1)[0]
    # the ~39 ft round-trip jump implies ~390 ft/s, far above the human cap,
    # so both jump segments are rejected and distance stays near zero
    assert s.distance_ft < 5.0


def test_sorted_by_distance():
    cal = _identity_calibration()
    records = []
    for i in range(20):
        fast = _Player(1, (10 + i * 2 - 1, 9, 10 + i * 2 + 1, 10))  # 2 ft/frame
        slow = _Player(2, (10 - 1, 30 + i * 0.1, 10 + 1, 30 + i * 0.1 + 0.0))  # ~still
        records.append(_Record(index=i, players=[slow, fast]))
    analysis = _Analysis(fps=10.0, records=records)
    stats = player_stats(analysis, cal, min_frames=5, smooth_window=1)
    assert [s.track_id for s in stats][0] == 1  # the mover ranks first
