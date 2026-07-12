"""Full-game automatic tracking box score — task H.

Point a full broadcast at this and it produces, with no human in the loop, a
per-player *tracking* box score: minutes on the game camera, distance covered,
average / top speed, and an occupancy heatmap — the tracking half of what the NBA
publishes (not PTS/REB/AST, which need ball-event understanding; see ADR-013).

Two passes (see `hoopvision.segments`):
  1. **Coarse** (H-1): sample ~1 fps and record whether the court registers.
     Registration is a free scene classifier — it succeeds only on game-camera
     frames — so the registered samples group into analysable *segments*, and
     the rest of the broadcast (close-ups, replays, ads) is skipped. Reports
     what fraction of the broadcast is game camera. Cached in segments.json.
  2. **Fine** (`--fine`, H-2): run the shared detect → register → track →
     identify pipeline (`hoopvision.identify_pipeline`) inside each segment,
     with a fresh tracker/registrar per segment (identity cannot survive a
     camera cut — only a jersey number can). Confirmed numbers key the
     cross-segment aggregation (`hoopvision.aggregate`); anonymous tracks are
     reported as an honest residual. Each segment's result is cached under
     `<output>/segments_cache/` so a crashed overnight run resumes for free.

    # download a full game yourself (never committed):
    yt-dlp -f "bv*[height<=720]" <url> -o data/clips/_nba_full.mp4
    uv run python scripts/game_report.py data/clips/_nba_full.mp4 --output report/
    uv run python scripts/game_report.py data/clips/_nba_full.mp4 --output report/ \
        --fine --fine-until-s 1800     # fine pass on the first ~quarter
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from hoopvision.aggregate import SegmentPlayerRow, aggregate_by_number
from hoopvision.identify_pipeline import PipelineModels, best_keypoints, collect, identify_tracks
from hoopvision.registration import fit_homography
from hoopvision.segments import Segment, analysable_seconds, coverage, segment_registered

ROOT = Path(__file__).resolve().parents[1]
SIZE = 640  # pose input (matches the 640-stretched training)


def coarse_pass(
    video: str,
    pose,
    sample_fps: float,
    kpt_conf: float,
    max_seconds: float | None,
) -> tuple[list[tuple[int, bool]], float]:
    """Sample ~`sample_fps` and record (frame_index, registered) for each sample."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_seconds is not None:
        total = min(total, int(max_seconds * fps))
    stride = max(1, round(fps / sample_fps))

    samples: list[tuple[int, bool]] = []
    for frame_idx in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        f640 = cv2.resize(frame, (SIZE, SIZE))
        res = pose.predict(f640, verbose=False, conf=0.25)[0]
        registered = fit_homography(best_keypoints(res, kpt_conf)) is not None
        samples.append((frame_idx, registered))
    cap.release()
    return samples, fps


def _coarse(args, out_dir: Path, pose_loader) -> tuple[dict, list[Segment], float]:
    """Coarse pass with a segments.json cache (delete the file to recompute)."""
    seg_path = out_dir / "segments.json"
    if seg_path.exists():
        payload = json.loads(seg_path.read_text())
        segments = [
            Segment(s["start_frame"], s["end_frame"], s["duration_s"]) for s in payload["segments"]
        ]
        print(f"reusing {seg_path} ({len(segments)} segments)")
        return payload["meta"], segments, payload["meta"]["fps"]

    samples, fps = coarse_pass(
        args.video, pose_loader(), args.sample_fps, args.kpt_conf, args.max_seconds
    )
    segments = segment_registered(samples, fps, min_len_s=args.min_seg_s, max_gap_s=args.max_gap_s)
    sampled_s = round(len(samples) / max(args.sample_fps, 1e-9), 1)
    meta = {
        "clip": Path(args.video).name,
        "fps": round(fps, 3),
        "coarse_samples": len(samples),
        "sampled_span_s": sampled_s,
        "game_camera_coverage": round(coverage(samples), 3),
        "segments": len(segments),
        "analysable_seconds": analysable_seconds(segments),
    }
    payload = {
        "meta": meta,
        "segments": [
            {"start_frame": s.start_frame, "end_frame": s.end_frame, "duration_s": s.duration_s}
            for s in segments
        ],
    }
    seg_path.write_text(json.dumps(payload, indent=2))
    print(f"segments -> {seg_path}")
    return meta, segments, fps


def _fine_segment(args, models: PipelineModels, seg: Segment, index: int, fps: float) -> dict:
    """Fine pass on one segment -> cacheable {rows, points, meta} dict."""
    got = collect(
        args.video,
        models,
        start=seg.start_frame / fps,
        seconds=seg.duration_s,
        kpt_conf=args.kpt_conf,
        number_conf=args.number_conf,
        read_every=args.read_every,
        stride=args.stride,
    )
    res = identify_tracks(got, min_votes=args.min_votes, min_frames=args.min_frames)

    rows = [
        {
            "segment": index,
            "track_id": r["track_id"],
            "number": r["number"],
            "seconds": r["seconds"],
            "distance_ft": r["distance_ft"],
            "avg_speed_mph": r["avg_speed_mph"],
            "top_speed_mph": r["top_speed_mph"],
        }
        for r in res.players
    ]
    # court points per confirmed number (for per-player heatmaps)
    points: dict[str, list[list[float]]] = {}
    for tid, number in res.numbers.items():
        obs = res.merged_paths.get(tid, [])
        points.setdefault(number, []).extend([round(xy[0], 1), round(xy[1], 1)] for _, xy, _ in obs)
    keep = (
        "registration_rate",
        "number_reads",
        "abstain_reads",
        "tracks_seen",
        "tracks_after_stitch",
        "tracks_identified",
        "numbers_on_multiple_players",
    )
    return {
        "segment": index,
        "start_frame": seg.start_frame,
        "duration_s": seg.duration_s,
        "rows": rows,
        "points": points,
        "meta": {k: res.meta[k] for k in keep},
    }


def _write_report_md(out_dir: Path, meta: dict, totals, agg_meta: dict) -> None:
    lines = [
        "# Full-game tracking box score",
        "",
        f"Clip: `{meta['clip']}` — game-camera coverage "
        f"{meta['game_camera_coverage']:.0%}, {meta['segments_analysed']} segments analysed "
        f"({meta['fine_analysed_seconds']:.0f} s).",
        "",
        "Numbers below are **game-camera** time/distance (not minutes played): a player",
        "off-camera accumulates nothing. Identification is jersey-number reading;",
        f"{agg_meta['anonymous_rows']} track-rows "
        f"({agg_meta['anonymous_seconds']:.0f} s) stayed anonymous — identified fraction "
        f"of tracked time: **{agg_meta['identified_time_fraction']:.0%}**.",
        "",
        "| # | segments | camera time (s) | distance (ft) | avg mph | top mph |",
        "|---|---------:|----------------:|--------------:|--------:|--------:|",
    ]
    for t in totals:
        lines.append(
            f"| {t.number} | {t.segments} | {t.seconds:.0f} | {t.distance_ft:.0f} "
            f"| {t.avg_speed_mph} | {t.top_speed_mph} |"
        )
    dupes = meta.get("numbers_on_multiple_players", {})
    if dupes:
        lines += ["", f"Duplicate-number warnings (same number on concurrent tracks): {dupes}"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"report -> {out_dir / 'report.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("--pose-weights", default=str(ROOT / "runs/pose/court_pose/weights/best.pt"))
    p.add_argument("--player-weights", default=str(ROOT / "hoopvision_best.pt"))
    p.add_argument(
        "--number-weights", default=str(ROOT / "runs/detect/number_detector/weights/best.pt")
    )
    p.add_argument(
        "--classifier-weights", default=str(ROOT / "runs/classify/number_classifier/best.pt")
    )
    p.add_argument("--output", default="report")
    p.add_argument("--sample-fps", type=float, default=1.0, help="coarse sampling rate")
    p.add_argument("--kpt-conf", type=float, default=0.5)
    p.add_argument("--min-seg-s", type=float, default=8.0, help="drop segments shorter than this")
    p.add_argument("--max-gap-s", type=float, default=2.0, help="bridge dropouts up to this long")
    p.add_argument("--max-seconds", type=float, default=None, help="cap for a quick test")
    p.add_argument("--fine", action="store_true", help="run the per-segment identity pipeline")
    p.add_argument("--stride", type=int, default=2, help="fine pass: analyse every n-th frame")
    p.add_argument("--read-every", type=int, default=15, help="analysed frames between reads")
    p.add_argument("--player-conf", type=float, default=0.3)
    p.add_argument("--number-conf", type=float, default=0.3)
    p.add_argument("--min-votes", type=int, default=3)
    p.add_argument("--min-frames", type=int, default=15)
    p.add_argument(
        "--fine-until-s",
        type=float,
        default=None,
        help="only fine-process segments starting before this video time (e.g. one quarter)",
    )
    p.add_argument("--heatmaps", type=int, default=6, help="per-player heatmaps for top N")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    def pose_loader():
        from ultralytics import YOLO

        return YOLO(args.pose_weights)

    coarse_meta, segments, fps = _coarse(args, out_dir, pose_loader)
    print(json.dumps(coarse_meta, indent=2))
    if not args.fine:
        return

    todo = [
        (k, s)
        for k, s in enumerate(segments)
        if args.fine_until_s is None or s.start_frame / fps < args.fine_until_s
    ]
    models = PipelineModels.load(
        args.pose_weights,
        args.player_weights,
        args.number_weights,
        args.classifier_weights,
        player_conf=args.player_conf,
    )
    cache_dir = out_dir / "segments_cache"
    cache_dir.mkdir(exist_ok=True)

    results: list[dict] = []
    for n, (k, seg) in enumerate(todo):
        cache = cache_dir / f"seg_{k:04d}.json"
        if cache.exists():
            results.append(json.loads(cache.read_text()))
            continue
        t0 = time.time()
        result = _fine_segment(args, models, seg, k, fps)
        cache.write_text(json.dumps(result))
        results.append(result)
        ided = result["meta"]["tracks_identified"]
        print(
            f"[{n + 1}/{len(todo)}] seg {k} @{seg.start_frame / fps:.0f}s "
            f"{seg.duration_s:.0f}s -> {ided} identified ({time.time() - t0:.0f}s)",
            flush=True,
        )

    rows = [SegmentPlayerRow(**r) for res in results for r in res["rows"]]
    totals, agg_meta = aggregate_by_number(rows)

    dupes: dict[str, int] = {}
    for res in results:
        for num, cnt in res["meta"]["numbers_on_multiple_players"].items():
            dupes[num] = max(dupes.get(num, 0), cnt)
    meta = coarse_meta | agg_meta
    meta["segments_analysed"] = len(results)
    meta["fine_analysed_seconds"] = round(sum(r["duration_s"] for r in results), 1)
    meta["numbers_on_multiple_players"] = dupes
    meta["knobs"] = {
        "stride": args.stride,
        "read_every": args.read_every,
        "min_votes": args.min_votes,
        "min_frames": args.min_frames,
        "fine_until_s": args.fine_until_s,
    }

    payload = {
        "schema_version": 1,  # contract for downstream consumers (Forecast Lab)
        "meta": meta,
        "players": [vars(t) for t in totals],
    }
    (out_dir / "box_tracking.json").write_text(json.dumps(payload, indent=2))
    print(f"box score -> {out_dir / 'box_tracking.json'}")
    _write_report_md(out_dir, meta, totals, agg_meta)

    # per-player heatmaps for the top-N by camera time
    import numpy as np

    from hoopvision.viz import render_court_heatmap

    points: dict[str, list[list[float]]] = {}
    for res in results:
        for num, pts in res["points"].items():
            points.setdefault(num, []).extend(pts)
    for t in totals[: args.heatmaps]:
        pts = np.array(points.get(t.number, []))
        out = out_dir / f"heatmap_{t.number}.png"
        render_court_heatmap(pts, out, f"#{t.number} — {t.seconds:.0f}s on game camera")
        print(f"heatmap -> {out}")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
