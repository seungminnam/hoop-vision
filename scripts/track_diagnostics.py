"""Unsupervised tracking-health diagnostics — no ground truth required.

The v1 tracker (ByteTrack, motion-only) visibly fragments on the demo clips.
Before investing in hand-labeled MOT ground truth (see ROADMAP.md v1.1), this
script quantifies the problem from the track set alone, so the "before" state
is a number instead of an impression.

For a clip it runs the real detection + tracking path (`pipeline.analyze`,
teams off) and reports, per clip:

- unique track IDs vs. the median number of players on screen at once. Their
  ratio — the **fragmentation ratio** — is roughly "how many IDs the average
  player was split into". 1.0 is perfect; higher is worse.
- track-length distribution (how long IDs survive), in frames and seconds.
- new IDs born after a warm-up second (churn during steady play).
- an **ID-switch proxy**: a track dying and a *different* track being born
  nearby within a few frames — the unsupervised signature of a swap.

These are proxies, not IDF1/HOTA (which need ground truth). They cannot say
the tracker is *correct*, only expose fragmentation; `scripts/eval_tracking.py`
does the supervised measurement once labels exist.

Usage:
    uv run python scripts/track_diagnostics.py data/clips/hudl_seg1.mp4 \
        data/clips/pickup_seg3.mp4 --weights hoopvision_best.pt

    # also write MOTChallenge-format predictions for scripts/eval_tracking.py:
    uv run python scripts/track_diagnostics.py data/clips/hudl_seg1.mp4 \
        --weights hoopvision_best.pt --dump-mot data/labels/mot/pred
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# per-track record: (track_id, center_xy) present in a processed frame
FrameTracks = list[tuple[int, tuple[float, float, float, float]]]


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def diagnose(
    frames: list[FrameTracks],
    effective_fps: float,
    warmup_s: float = 1.0,
    switch_gap: int = 5,
    switch_dist_px: float = 60.0,
) -> dict:
    """Compute unsupervised tracking-health metrics from per-frame tracks.

    `frames[i]` is the list of (track_id, xyxy) present in the i-th processed
    frame. Pure function so it is unit-testable without running the detector.
    """
    n_frames = len(frames)
    per_frame_counts = [len(f) for f in frames]
    total_boxes = sum(per_frame_counts)

    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    hits: dict[int, int] = {}
    birth_center: dict[int, tuple[float, float]] = {}
    death_center: dict[int, tuple[float, float]] = {}
    for i, frame in enumerate(frames):
        for tid, box in frame:
            center = _center(box)
            if tid not in first_seen:
                first_seen[tid] = i
                birth_center[tid] = center
            last_seen[tid] = i
            death_center[tid] = center
            hits[tid] = hits.get(tid, 0) + 1

    unique_ids = len(first_seen)
    median_simul = statistics.median(per_frame_counts) if per_frame_counts else 0.0
    frag_ratio = unique_ids / median_simul if median_simul else 0.0

    lengths = list(hits.values())
    lengths_sorted = sorted(lengths)
    mean_len = statistics.mean(lengths) if lengths else 0.0
    median_len = statistics.median(lengths) if lengths else 0.0
    p90_len = lengths_sorted[int(0.9 * (len(lengths_sorted) - 1))] if lengths_sorted else 0.0

    warmup_frames = warmup_s * effective_fps
    new_after_warmup = sum(1 for f in first_seen.values() if f > warmup_frames)

    # ID-switch proxy: a track dies mid-clip and a *different* track is born
    # within `switch_gap` frames and `switch_dist_px` of the death location.
    switch_proxy = 0
    deaths = [(tid, last_seen[tid]) for tid in first_seen if last_seen[tid] < n_frames - 1]
    births = [(tid, first_seen[tid]) for tid in first_seen]
    for dead_id, death_frame in deaths:
        dx, dy = death_center[dead_id]
        for born_id, birth_frame in births:
            if born_id == dead_id:
                continue
            if 0 < birth_frame - death_frame <= switch_gap:
                bx, by = birth_center[born_id]
                if (bx - dx) ** 2 + (by - dy) ** 2 <= switch_dist_px**2:
                    switch_proxy += 1
                    break

    return {
        "frames": n_frames,
        "total_boxes": total_boxes,
        "unique_ids": unique_ids,
        "median_simultaneous": round(median_simul, 1),
        "fragmentation_ratio": round(frag_ratio, 1),
        "track_len_mean_frames": round(mean_len, 1),
        "track_len_median_frames": round(median_len, 1),
        "track_len_p90_frames": round(float(p90_len), 1),
        "track_len_median_s": round(median_len / effective_fps, 2) if effective_fps else 0.0,
        "new_ids_after_warmup": new_after_warmup,
        "id_switch_proxy": switch_proxy,
    }


def _frames_from_analysis(analysis) -> list[FrameTracks]:
    return [[(p.track_id, p.xyxy) for p in r.players] for r in analysis.records]


def _dump_mot(analysis, path: Path) -> None:
    """Write MOTChallenge-format predictions (frame,id,x,y,w,h,conf,-1,-1,-1).

    Frame numbers are the processed-frame ordinal (1-based), matching the
    convention scripts/eval_tracking.py expects for ground-truth labels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for ordinal, record in enumerate(analysis.records, start=1):
        for p in record.players:
            x1, y1, x2, y2 = p.xyxy
            lines.append(
                f"{ordinal},{p.track_id},{x1:.1f},{y1:.1f},"
                f"{x2 - x1:.1f},{y2 - y1:.1f},{p.confidence:.3f},-1,-1,-1"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", help="clips to diagnose")
    parser.add_argument("--weights", default="hoopvision_best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--no-stitch",
        action="store_true",
        help="disable appearance track stitching (raw ByteTrack, for before/after)",
    )
    parser.add_argument(
        "--gmc",
        action="store_true",
        help="enable camera-motion compensation (for panning clips)",
    )
    parser.add_argument(
        "--dump-mot",
        default=None,
        help="directory to also write <clip>.txt MOTChallenge predictions",
    )
    args = parser.parse_args()

    from hoopvision.detect import YoloDetector
    from hoopvision.pipeline import analyze

    detector = YoloDetector(weights=args.weights, conf=args.conf)

    cols = [
        "clip",
        "frames",
        "unique_ids",
        "median_simultaneous",
        "fragmentation_ratio",
        "track_len_median_frames",
        "track_len_median_s",
        "new_ids_after_warmup",
        "id_switch_proxy",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for video in args.videos:
        analysis = analyze(
            video,
            detector,
            stride=args.stride,
            max_frames=args.max_frames,
            teams=False,
            stitch_tracks=not args.no_stitch,
            compensate_camera=args.gmc,
        )
        frames = _frames_from_analysis(analysis)
        stats = diagnose(frames, analysis.effective_fps)
        stem = Path(video).stem
        row = [stem] + [str(stats[c]) for c in cols[1:]]
        print("| " + " | ".join(row) + " |")
        if args.dump_mot:
            _dump_mot(analysis, Path(args.dump_mot) / f"{stem}.txt")


if __name__ == "__main__":
    main()
