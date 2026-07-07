"""Evaluate shot detection against hand-labeled ground truth.

Ground truth: data/labels/<clip>.csv with rows `time_s,outcome` (made|missed),
one row per real shot attempt (labeled by reviewing the clip frame by frame).

Matching: a detected event is a true positive if a ground-truth attempt exists
within +/- tolerance seconds (greedy, one-to-one). Outcome accuracy is scored
over matched pairs only.

    uv run python scripts/eval_shots.py out/pickup_seg1/pickup_seg1_events.json \
        out/pickup_seg2/pickup_seg2_events.json --labels data/labels --tolerance 2.0

Every shot-detection number in the README comes from this script (honesty rule).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_labels(labels_dir: Path, clip_stem: str) -> list[tuple[float, str]]:
    path = labels_dir / f"{clip_stem}.csv"
    if not path.exists():
        raise SystemExit(f"No ground truth for {clip_stem}: {path} missing")
    with path.open() as f:
        return [(float(r["time_s"]), r["outcome"].strip()) for r in csv.DictReader(f)]


def match(
    events: list[dict], truths: list[tuple[float, str]], tolerance: float
) -> tuple[list[tuple[dict, tuple[float, str]]], list[dict], list[tuple[float, str]]]:
    unmatched_truths = list(truths)
    pairs = []
    false_positives = []
    for event in sorted(events, key=lambda e: e["time_s"]):
        best = None
        for truth in unmatched_truths:
            dt = abs(event["time_s"] - truth[0])
            if dt <= tolerance and (best is None or dt < abs(event["time_s"] - best[0])):
                best = truth
        if best is None:
            false_positives.append(event)
        else:
            unmatched_truths.remove(best)
            pairs.append((event, best))
    return pairs, false_positives, unmatched_truths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", nargs="+", help="events.json files from the pipeline")
    parser.add_argument("--labels", default="data/labels")
    parser.add_argument("--tolerance", type=float, default=2.0)
    args = parser.parse_args()

    labels_dir = Path(args.labels)
    tp = fp = fn = outcome_ok = 0
    print("| clip | GT attempts | detected | TP | FP | FN | outcomes correct |")
    print("|---|---|---|---|---|---|---|")
    for events_path in args.events:
        payload = json.loads(Path(events_path).read_text())
        clip_stem = Path(payload["video"]).stem
        truths = load_labels(labels_dir, clip_stem)
        pairs, fps_, fns_ = match(payload["events"], truths, args.tolerance)
        ok = sum(e["outcome"] == t[1] for e, t in pairs)
        tp += len(pairs)
        fp += len(fps_)
        fn += len(fns_)
        outcome_ok += ok
        print(
            f"| {clip_stem} | {len(truths)} | {len(payload['events'])} "
            f"| {len(pairs)} | {len(fps_)} | {len(fns_)} | {ok}/{len(pairs)} |"
        )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"\nattempt precision: {precision:.0%} ({tp}/{tp + fp})")
    print(f"attempt recall:    {recall:.0%} ({tp}/{tp + fn})")
    if tp:
        print(f"outcome accuracy on matched attempts: {outcome_ok}/{tp}")


if __name__ == "__main__":
    main()
