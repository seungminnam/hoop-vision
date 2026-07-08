"""Supervised multi-object-tracking metrics (IDF1, MOTA, ID switches).

The honest counterpart to scripts/track_diagnostics.py: once player-ID ground
truth exists, this reports the standard MOTChallenge metrics via `motmetrics`.

File format (both GT and predictions), MOTChallenge CSV, one box per line:

    frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1

- `frame` is the 1-based processed-frame ordinal (matches --dump-mot output of
  scripts/track_diagnostics.py).
- GT lives in `data/labels/mot/gt/<clip>.txt`, predictions in
  `data/labels/mot/pred/<clip>.txt`.

Matching is IoU-based (default 0.5), the MOTChallenge convention.

    uv run python scripts/eval_tracking.py hudl_seg1 pickup_seg3 \
        --gt data/labels/mot/gt --pred data/labels/mot/pred

`motmetrics` is a dev dependency (offline eval tool, not needed by the
pipeline); run inside `uv run` so it is importable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# motmetrics 1.4.0 (latest release) still calls np.asfarray, removed in NumPy
# 2.0; the rest of the stack (torch, ultralytics) needs NumPy 2.x, so restore
# the shim instead of pinning numpy down. asfarray == asarray with a float dtype.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

# frame (int) -> list of (track_id, (x, y, w, h))
MotSequence = dict[int, list[tuple[int, tuple[float, float, float, float]]]]

METRICS = [
    "idf1",
    "idp",
    "idr",
    "mota",
    "motp",
    "num_switches",
    "mostly_tracked",
    "mostly_lost",
    "num_fragmentations",
    "num_unique_objects",
]


def load_mot(path: Path) -> MotSequence:
    seq: MotSequence = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        frame, tid = int(parts[0]), int(parts[1])
        x, y, w, h = (float(v) for v in parts[2:6])
        seq.setdefault(frame, []).append((tid, (x, y, w, h)))
    return seq


def evaluate(gt: MotSequence, pred: MotSequence, iou_threshold: float = 0.5):
    """Accumulate MOT metrics over aligned frames; returns a motmetrics summary."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    for frame in sorted(set(gt) | set(pred)):
        gt_objs = gt.get(frame, [])
        pred_objs = pred.get(frame, [])
        gt_ids = [tid for tid, _ in gt_objs]
        pred_ids = [tid for tid, _ in pred_objs]
        gt_boxes = np.array([b for _, b in gt_objs], dtype=float).reshape(-1, 4)
        pred_boxes = np.array([b for _, b in pred_objs], dtype=float).reshape(-1, 4)
        # iou_matrix returns 1-IoU distances; pairs with IoU < threshold -> NaN
        dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0 - iou_threshold)
        acc.update(gt_ids, pred_ids, dist, frameid=frame)

    mh = mm.metrics.create()
    return mh.compute(acc, metrics=METRICS, name="acc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="+", help="clip stems (no extension)")
    parser.add_argument("--gt", default="data/labels/mot/gt")
    parser.add_argument("--pred", default="data/labels/mot/pred")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import motmetrics as mm

    gt_dir, pred_dir = Path(args.gt), Path(args.pred)
    accs, names = [], []
    for stem in args.clips:
        gt_path, pred_path = gt_dir / f"{stem}.txt", pred_dir / f"{stem}.txt"
        if not gt_path.exists():
            raise SystemExit(f"No ground truth for {stem}: {gt_path} missing")
        if not pred_path.exists():
            raise SystemExit(
                f"No predictions for {stem}: {pred_path} missing "
                "(generate with scripts/track_diagnostics.py --dump-mot)"
            )
        summary = evaluate(load_mot(gt_path), load_mot(pred_path), args.iou)
        accs.append(summary)
        names.append(stem)

    import pandas as pd

    table = pd.concat(accs)
    table.index = names
    rename = mm.io.motchallenge_metric_names
    print(table.rename(columns=rename).to_markdown(floatfmt=".3f"))


if __name__ == "__main__":
    main()
