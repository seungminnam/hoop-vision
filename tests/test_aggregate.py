"""Tests for cross-segment per-number aggregation (src/hoopvision/aggregate.py)."""

from hoopvision.aggregate import MPH_PER_FPS, SegmentPlayerRow, aggregate_by_number


def _row(segment, track_id, number, seconds, distance, avg=5.0, top=10.0):
    return SegmentPlayerRow(segment, track_id, number, seconds, distance, avg, top)


def test_sums_across_segments():
    rows = [
        _row(0, 1, "23", 10.0, 100.0, top=12.0),
        _row(2, 7, "23", 20.0, 50.0, top=15.0),
    ]
    totals, meta = aggregate_by_number(rows)
    assert len(totals) == 1
    t = totals[0]
    assert t.number == "23"
    assert t.segments == 2
    assert t.seconds == 30.0
    assert t.distance_ft == 150.0
    assert t.top_speed_mph == 15.0  # max, not mean
    assert meta["identified_numbers"] == 1
    assert meta["anonymous_rows"] == 0


def test_avg_speed_from_totals_not_mean_of_means():
    # 100 ft in 10 s + 0 ft in 90 s -> 1 ft/s overall, NOT the ~5 ft/s a naive
    # mean of per-segment averages would claim.
    rows = [
        _row(0, 1, "5", 10.0, 100.0, avg=6.8),
        _row(1, 2, "5", 90.0, 0.0, avg=0.0),
    ]
    totals, _ = aggregate_by_number(rows)
    assert totals[0].avg_speed_mph == round(1.0 * MPH_PER_FPS, 1)


def test_anonymous_residual_reported():
    rows = [
        _row(0, 1, "23", 30.0, 100.0),
        _row(0, 2, None, 10.0, 40.0),
        _row(1, 3, None, 20.0, 60.0),
    ]
    totals, meta = aggregate_by_number(rows)
    assert len(totals) == 1
    assert meta["anonymous_rows"] == 2
    assert meta["anonymous_seconds"] == 30.0
    assert meta["anonymous_distance_ft"] == 100.0
    assert meta["identified_time_fraction"] == 0.5


def test_sorted_by_seconds_desc():
    rows = [
        _row(0, 1, "3", 5.0, 10.0),
        _row(0, 2, "30", 50.0, 10.0),
        _row(1, 3, "3", 10.0, 10.0),
    ]
    totals, _ = aggregate_by_number(rows)
    assert [t.number for t in totals] == ["30", "3"]


def test_same_segment_two_tracks_same_number_counts_one_segment():
    # duplicate number on two concurrent tracks in one segment (a known
    # precision failure mode) still counts that segment once
    rows = [
        _row(4, 1, "22", 10.0, 30.0),
        _row(4, 2, "22", 10.0, 30.0),
    ]
    totals, _ = aggregate_by_number(rows)
    assert totals[0].segments == 1
    assert totals[0].seconds == 20.0  # time still accumulates (honest over-count)


def test_empty():
    totals, meta = aggregate_by_number([])
    assert totals == []
    assert meta["identified_time_fraction"] == 0.0
    assert meta["identified_numbers"] == 0
