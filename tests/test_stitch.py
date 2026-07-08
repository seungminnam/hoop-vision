"""Tests for the pure tracklet-stitching logic (src/hoopvision/stitch.py)."""

import numpy as np

from hoopvision.stitch import Tracklet, stitch


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
