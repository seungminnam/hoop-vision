"""Tests for the pure-logic parts of the tracking measurement scripts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_tracking import evaluate  # noqa: E402
from track_diagnostics import diagnose  # noqa: E402


def _box(cx, cy, w=20, h=40):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_diagnose_perfect_tracking():
    # 3 stable tracks present every frame → ratio 1.0, no churn, no switches.
    frames = [[(1, _box(50, 50)), (2, _box(150, 50)), (3, _box(250, 50))] for _ in range(30)]
    stats = diagnose(frames, effective_fps=30.0)
    assert stats["unique_ids"] == 3
    assert stats["median_simultaneous"] == 3
    assert stats["fragmentation_ratio"] == 1.0
    assert stats["new_ids_after_warmup"] == 0
    assert stats["id_switch_proxy"] == 0
    assert stats["track_len_median_frames"] == 30


def test_diagnose_detects_fragmentation_and_switch_proxy():
    # One physical player at a fixed spot, re-IDed every 10 frames.
    frames = []
    for i in range(30):
        tid = i // 10 + 1  # ids 1,2,3
        frames.append([(tid, _box(100, 100))])
    stats = diagnose(frames, effective_fps=10.0)
    assert stats["unique_ids"] == 3
    assert stats["median_simultaneous"] == 1
    assert stats["fragmentation_ratio"] == 3.0
    # id1 dies@9→id2 born@10 nearby; id2 dies@19→id3 born@20 nearby: 2 proxies
    assert stats["id_switch_proxy"] == 2
    # warmup = 1.0 * 10 = 10 frames; only id3 (born@20) is past it
    assert stats["new_ids_after_warmup"] == 1


def test_diagnose_switch_proxy_ignores_far_rebirths():
    # A track dies on the left; a new one is born far away on the right.
    frames = [[(1, _box(50, 50))] for _ in range(5)]
    frames += [[(2, _box(600, 400))] for _ in range(5)]
    stats = diagnose(frames, effective_fps=30.0, switch_dist_px=60.0)
    assert stats["id_switch_proxy"] == 0


def test_diagnose_empty():
    stats = diagnose([], effective_fps=30.0)
    assert stats["unique_ids"] == 0
    assert stats["fragmentation_ratio"] == 0.0


def _seq(frame_to_objs):
    return dict(frame_to_objs)


def test_evaluate_perfect_match():
    gt = {f: [(1, (0, 0, 20, 40)), (2, (100, 0, 20, 40))] for f in range(1, 6)}
    summary = evaluate(gt, gt)
    row = summary.loc["acc"]
    assert row["idf1"] == 1.0
    assert row["num_switches"] == 0
    assert row["mota"] == 1.0


def test_evaluate_counts_id_swap():
    gt = {f: [(1, (0, 0, 20, 40)), (2, (100, 0, 20, 40))] for f in range(1, 6)}
    pred = {}
    for f in range(1, 6):
        if f < 3:
            pred[f] = [(1, (0, 0, 20, 40)), (2, (100, 0, 20, 40))]
        else:  # IDs swapped between the two players
            pred[f] = [(2, (0, 0, 20, 40)), (1, (100, 0, 20, 40))]
    summary = evaluate(gt, pred)
    row = summary.loc["acc"]
    assert row["num_switches"] >= 1
    assert row["idf1"] < 1.0


def test_evaluate_penalizes_missed_detections():
    gt = {f: [(1, (0, 0, 20, 40)), (2, (100, 0, 20, 40))] for f in range(1, 6)}
    pred = {f: [(1, (0, 0, 20, 40))] for f in range(1, 6)}  # object 2 never detected
    summary = evaluate(gt, pred)
    row = summary.loc["acc"]
    assert row["mota"] < 1.0
    assert row["idr"] < 1.0
