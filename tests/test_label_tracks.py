"""Tests for the pure editing logic of scripts/label_tracks.py (no OpenCV)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from label_tracks import Box, LabelStore  # noqa: E402


def _fragmented_store() -> LabelStore:
    # One physical player, tracker split them into id 1 (frames 1-2) then id 2
    # (frame 3); a second player is a stable id 9 throughout.
    seq = {
        1: [(1, (0, 0, 10, 20), 0.9), (9, (100, 0, 10, 20), 0.9)],
        2: [(1, (1, 0, 10, 20), 0.9), (9, (101, 0, 10, 20), 0.9)],
        3: [(2, (2, 0, 10, 20), 0.9), (9, (102, 0, 10, 20), 0.9)],
    }
    return LabelStore.from_mot(seq)


def test_from_mot_shape():
    store = _fragmented_store()
    assert len(store.frames) == 3
    assert [len(f) for f in store.frames] == [2, 2, 2]
    assert store.frames[0][0].gid == 1


def test_relabel_track_collapses_fragment():
    store = _fragmented_store()
    # frame 3 (index 2), first det has id 2 → relabel its whole track to 1
    changed = store.relabel_track(2, 0, 1)
    assert changed == 1  # only the single id-2 box existed
    ids = [b.gid for frame in store.frames for b in frame]
    assert 2 not in ids
    # now id 1 spans all three frames for that player
    assert [store.frames[k][0].gid for k in range(3)] == [1, 1, 1]


def test_relabel_track_is_global_by_current_id():
    store = _fragmented_store()
    # relabel the id-1 track (2 boxes across frames 1-2) to 7
    changed = store.relabel_track(0, 0, 7)
    assert changed == 2
    assert store.frames[0][0].gid == 7 and store.frames[1][0].gid == 7


def test_override_box_touches_one_detection():
    store = _fragmented_store()
    store.override_box(1, 1, 42)  # frame 2, the id-9 box only
    assert store.frames[1][1].gid == 42
    assert store.frames[0][1].gid == 9 and store.frames[2][1].gid == 9


def test_undo_restores_previous_ids():
    store = _fragmented_store()
    store.relabel_track(0, 0, 7)
    assert store.undo() is True
    assert store.frames[0][0].gid == 1
    assert store.undo() is False  # nothing left to undo


def test_remove_track_deletes_all_boxes_of_id():
    store = _fragmented_store()
    removed = store.remove_track(0, 1)  # the id-9 track (a "bystander"), 3 boxes
    assert removed == 3
    assert all(b.gid != 9 for frame in store.frames for b in frame)
    assert [len(f) for f in store.frames] == [1, 1, 1]


def test_undo_reverses_a_removal():
    store = _fragmented_store()
    store.remove_track(0, 1)
    assert store.undo() is True
    assert [len(f) for f in store.frames] == [2, 2, 2]
    assert store.frames[0][1].gid == 9


def test_next_free_id():
    store = _fragmented_store()
    assert store.next_free_id() == 10  # max id is 9


def test_box_at_prefers_smallest_containing():
    frames = [[Box((0, 0, 100, 100), 0.9, 1), Box((10, 10, 20, 20), 0.9, 2)]]
    store = LabelStore(frames)
    assert store.box_at(0, 15, 15) == 1  # smaller box wins
    assert store.box_at(0, 95, 95) == 0
    assert store.box_at(0, 200, 200) is None


def test_save_load_roundtrip(tmp_path):
    store = _fragmented_store()
    store.relabel_track(2, 0, 1)
    out = tmp_path / "gt" / "clip.txt"
    store.save(out)
    reloaded = LabelStore.load_mot_file(out)
    assert reloaded.to_mot_lines() == store.to_mot_lines()
    assert [b.gid for b in reloaded.frames[2]] == [1, 9]
