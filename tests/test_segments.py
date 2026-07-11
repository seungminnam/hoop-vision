"""Tests for the pure broadcast-segmenter logic (src/hoopvision/segments.py)."""

from hoopvision.segments import Segment, analysable_seconds, coverage, segment_registered

FPS = 30.0


def _samples(pattern: str, stride: int = 30) -> list[tuple[int, bool]]:
    """'RRR..RR' -> registered/failed samples spaced `stride` frames apart."""
    return [(i * stride, ch == "R") for i, ch in enumerate(pattern)]


def test_single_long_segment():
    segs = segment_registered(_samples("RRRRRRRRRR"), FPS, min_len_s=8.0)
    assert len(segs) == 1
    assert segs[0].start_frame == 0
    assert segs[0].end_frame == 9 * 30  # last registered sample
    assert segs[0].duration_s == 9.0


def test_drops_short_fragment():
    # 3 registered samples ~2 s < min_len 8 s
    assert segment_registered(_samples("RRR"), FPS, min_len_s=8.0) == []


def test_split_on_long_gap():
    # two 5-sample runs separated by 4 failed samples (~4 s > max_gap 2 s)
    segs = segment_registered(_samples("RRRRRFFFFRRRRR"), FPS, min_len_s=3.0, max_gap_s=2.0)
    assert len(segs) == 2
    assert segs[0].start_frame == 0 and segs[0].end_frame == 4 * 30
    assert segs[1].start_frame == 9 * 30 and segs[1].end_frame == 13 * 30


def test_bridges_short_dropout():
    # one failed sample (~1 s) inside a run is bridged, not split
    segs = segment_registered(_samples("RRRRRFRRRRR"), FPS, min_len_s=3.0, max_gap_s=2.0)
    assert len(segs) == 1
    assert segs[0].end_frame == 10 * 30


def test_trailing_run_emitted():
    segs = segment_registered(_samples("FFFRRRRRRRR"), FPS, min_len_s=3.0)
    assert len(segs) == 1
    assert segs[0].start_frame == 3 * 30


def test_unsorted_input_handled():
    samples = [(90, True), (0, True), (60, True), (30, True)]
    segs = segment_registered(samples, FPS, min_len_s=2.0)
    assert len(segs) == 1
    assert segs[0].start_frame == 0 and segs[0].end_frame == 90


def test_empty():
    assert segment_registered([], FPS) == []
    assert coverage([]) == 0.0
    assert analysable_seconds([]) == 0.0


def test_coverage_and_analysable():
    samples = _samples("RRRRRFFFFF")  # half registered
    assert coverage(samples) == 0.5
    segs = [Segment(0, 300, 10.0), Segment(600, 900, 10.0)]
    assert analysable_seconds(segs) == 20.0
