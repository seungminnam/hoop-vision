"""Tests for the pure tracklet-stitching logic (src/hoopvision/stitch.py)."""

import numpy as np

from hoopvision.stitch import CourtTracklet, Tracklet, stitch, stitch_court


def _feat(*vals) -> np.ndarray:
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _tk(tid, start, end, sx, sy, ex, ey, h=100.0, feat=(1.0, 0.0, 0.0)) -> Tracklet:
    return Tracklet(tid, start, end, (sx, sy), (ex, ey), h, h, _feat(*feat))


def test_merges_temporally_adjacent_nearby_similar():
    # one player: id 1 (frames 0-10) ends at (100,100); id 2 (13-25) starts at (108,102)
    tks = [
        _tk(1, 0, 10, 100, 100, 100, 100),
        _tk(2, 13, 25, 108, 102, 200, 150),
    ]
    remap = stitch(tks)
    assert remap[2] == remap[1] == 1  # canonical = smallest id


def test_keeps_overlapping_tracks_separate():
    # two players on court at the same time must never merge
    tks = [
        _tk(1, 0, 20, 100, 100, 100, 100),
        _tk(2, 5, 25, 110, 100, 210, 100),
    ]
    remap = stitch(tks)
    assert remap[1] != remap[2]


def test_rejects_far_reappearance():
    tks = [
        _tk(1, 0, 10, 100, 100, 100, 100, h=100),
        _tk(2, 12, 20, 800, 600, 800, 600, h=100),  # far away (> 2.5 * height)
    ]
    remap = stitch(tks)
    assert remap[1] != remap[2]


def test_rejects_dissimilar_appearance():
    tks = [
        _tk(1, 0, 10, 100, 100, 100, 100, feat=(1.0, 0.0, 0.0)),
        _tk(2, 12, 20, 105, 100, 105, 100, feat=(0.0, 0.0, 1.0)),  # different color
    ]
    remap = stitch(tks, min_similarity=0.5)
    assert remap[1] != remap[2]


def test_rejects_gap_too_large():
    tks = [
        _tk(1, 0, 10, 100, 100, 100, 100),
        _tk(2, 100, 120, 105, 100, 105, 100),  # 90-frame gap > max_gap 45
    ]
    remap = stitch(tks)
    assert remap[1] != remap[2]


def test_chains_three_fragments_into_one():
    tks = [
        _tk(1, 0, 10, 100, 100, 100, 100),
        _tk(2, 13, 22, 104, 100, 104, 100),
        _tk(3, 25, 35, 106, 100, 106, 100),
    ]
    remap = stitch(tks)
    assert remap[1] == remap[2] == remap[3] == 1


def test_two_players_two_chains():
    # left player fragments (1->3), right player fragments (2->4); no cross-merge
    tks = [
        _tk(1, 0, 10, 100, 100, 100, 100, feat=(1.0, 0.0, 0.0)),
        _tk(2, 0, 10, 400, 100, 400, 100, feat=(0.0, 1.0, 0.0)),
        _tk(3, 13, 20, 103, 100, 103, 100, feat=(1.0, 0.0, 0.0)),
        _tk(4, 13, 20, 403, 100, 403, 100, feat=(0.0, 1.0, 0.0)),
    ]
    remap = stitch(tks)
    assert remap[3] == remap[1]
    assert remap[4] == remap[2]
    assert remap[1] != remap[2]


def test_empty_input():
    assert stitch([]) == {}


# --- court-space stitching (stitch_court) ---

FPS = 30.0


def _ct(tid, start, end, sx, sy, ex, ey, feat=(1.0, 0.0, 0.0)) -> CourtTracklet:
    return CourtTracklet(tid, start, end, (sx, sy), (ex, ey), _feat(*feat))


def test_court_merges_temporally_adjacent_nearby_similar():
    # id 1 ends frame 10 at (40,25); id 2 starts frame 20 at (43,25), same colour
    tks = [
        _ct(1, 0, 10, 30, 25, 40, 25),
        _ct(2, 20, 40, 43, 25, 60, 25),
    ]
    remap = stitch_court(tks, FPS)
    assert remap[2] == remap[1] == 1


def test_court_keeps_overlapping_separate():
    tks = [
        _ct(1, 0, 20, 40, 25, 45, 25),
        _ct(2, 5, 25, 42, 25, 50, 25),  # overlaps in time
    ]
    remap = stitch_court(tks, FPS)
    assert remap[1] != remap[2]


def test_court_rejects_far_reappearance():
    # gap 2 frames (~0.067 s): max_dist = 3 + 25*0.067 ≈ 4.7 ft; 40 ft apart
    tks = [
        _ct(1, 0, 10, 10, 25, 10, 25),
        _ct(2, 12, 20, 50, 25, 50, 25),
    ]
    remap = stitch_court(tks, FPS)
    assert remap[1] != remap[2]


def test_court_speed_bound_scales_with_gap():
    # 20 ft apart is too far for a 2-frame gap but fine for a 1 s gap
    near = [
        _ct(1, 0, 10, 10, 25, 10, 25),
        _ct(2, 12, 20, 30, 25, 30, 25),
    ]
    assert stitch_court(near, FPS)[1] != stitch_court(near, FPS)[2]
    far_gap = [
        _ct(1, 0, 10, 10, 25, 10, 25),
        _ct(2, 40, 60, 30, 25, 30, 25),  # ~1 s gap -> max_dist ≈ 3 + 25 ≈ 28 ft
    ]
    remap = stitch_court(far_gap, FPS)
    assert remap[1] == remap[2]


def test_court_rejects_dissimilar_appearance():
    tks = [
        _ct(1, 0, 10, 40, 25, 40, 25, feat=(1.0, 0.0, 0.0)),
        _ct(2, 12, 20, 41, 25, 41, 25, feat=(0.0, 0.0, 1.0)),
    ]
    remap = stitch_court(tks, FPS, min_similarity=0.5)
    assert remap[1] != remap[2]


def test_court_rejects_gap_too_large():
    tks = [
        _ct(1, 0, 10, 40, 25, 40, 25),
        _ct(2, 100, 120, 41, 25, 41, 25),  # ~3 s gap > max_gap_s 1.5
    ]
    remap = stitch_court(tks, FPS)
    assert remap[1] != remap[2]


def test_court_chains_three_fragments():
    tks = [
        _ct(1, 0, 10, 40, 25, 41, 25),
        _ct(2, 15, 22, 42, 25, 43, 25),
        _ct(3, 27, 35, 44, 25, 45, 25),
    ]
    remap = stitch_court(tks, FPS)
    assert remap[1] == remap[2] == remap[3] == 1


def test_court_empty_input():
    assert stitch_court([], FPS) == {}
