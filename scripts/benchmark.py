"""Benchmark detectors: mAP (val set), inference FPS, parameter count.

Every number in the README results tables must come from this script (or the
training logs) — see the honesty rule in ROADMAP.md.

Examples:
    # FPS + params for the COCO baseline on a clip
    uv run python scripts/benchmark.py --weights yolo11n.pt --video demo.mp4

    # Fine-tuned checkpoint: adds mAP from the dataset's val split
    uv run python scripts/benchmark.py --weights hoopvision_best.pt \
        --video demo.mp4 --data data/<dataset>/data.yaml

    # Include the from-scratch detector (see scratch_detector/)
    uv run python scripts/benchmark.py --weights hoopvision_best.pt \
        --video demo.mp4 --scratch-weights scratch_detector/best.pt
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoopvision.detect import Detector, YoloDetector, default_device  # noqa: E402
from hoopvision.ingest import frames  # noqa: E402


def measure_fps(detector: Detector, video: str, n_frames: int = 120, warmup: int = 10) -> float:
    times: list[float] = []
    for i, (_, frame) in enumerate(frames(video, max_frames=n_frames + warmup)):
        start = time.perf_counter()
        detector.detect(frame)
        if i >= warmup:
            times.append(time.perf_counter() - start)
    if not times:
        raise SystemExit(f"No frames read from {video}")
    return 1.0 / statistics.mean(times)


def yolo_row(weights: str, video: str | None, data: str | None) -> dict:
    detector = YoloDetector(weights=weights)
    params = sum(p.numel() for p in detector.model.model.parameters())
    row = {
        "model": Path(weights).stem,
        "params_m": params / 1e6,
        "fps": measure_fps(detector, video) if video else None,
        "map50": None,
    }
    if data:
        metrics = detector.model.val(data=data, verbose=False)
        row["map50"] = float(metrics.box.map50)
    return row


def scratch_row(weights: str, video: str | None) -> dict:
    from scratch_detector.adapter import ScratchDetector

    detector = ScratchDetector(weights)
    params = sum(p.numel() for p in detector.model.parameters())
    return {
        "model": "scratch (player only)",
        "params_m": params / 1e6,
        "fps": measure_fps(detector, video) if video else None,
        "map50": None,  # reported by scratch_detector/eval.py
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--video", default=None, help="clip for FPS measurement")
    parser.add_argument("--data", default=None, help="data.yaml for mAP (YOLO only)")
    parser.add_argument("--scratch-weights", default=None)
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()

    rows = [yolo_row(args.weights, args.video, args.data)]
    if args.scratch_weights:
        rows.append(scratch_row(args.scratch_weights, args.video))

    print(f"\ndevice: {default_device()}\n")
    print("| model | params (M) | FPS | mAP50 |")
    print("|---|---|---|---|")
    for r in rows:
        fps = f"{r['fps']:.1f}" if r["fps"] else "—"
        map50 = f"{r['map50']:.3f}" if r["map50"] is not None else "—"
        print(f"| {r['model']} | {r['params_m']:.2f} | {fps} | {map50} |")


if __name__ == "__main__":
    main()
