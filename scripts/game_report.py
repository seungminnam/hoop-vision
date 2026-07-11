"""Full-game automatic tracking box score — task H.

Point a full broadcast at this and it produces, with no human in the loop, a
per-player *tracking* box score: minutes on the game camera, distance covered,
average / top speed, and an occupancy heatmap — the tracking half of what the NBA
publishes (not PTS/REB/AST, which need ball-event understanding; see ADR-013).

Two passes (see `hoopvision.segments`):
  1. **Coarse** (this slice, H-1): sample ~1 fps and record whether the court
     registers. Registration is a free scene classifier — it succeeds only on
     game-camera frames — so the registered samples group into analysable
     *segments*, and the rest of the broadcast (close-ups, replays, ads) is
     skipped. Reports what fraction of the broadcast is game camera.
  2. **Fine** (H-2): run the full detect -> register -> track -> identify pipeline
     inside each segment, keyed by jersey number across cuts.

    # download a full game yourself (never committed):
    yt-dlp -f "bv*[height<=720]" <url> -o data/clips/_nba_full.mp4
    uv run python scripts/game_report.py data/clips/_nba_full.mp4 --output report/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from hoopvision.court_template import NUM_KEYPOINTS
from hoopvision.registration import fit_homography
from hoopvision.segments import Segment, analysable_seconds, coverage, segment_registered

ROOT = Path(__file__).resolve().parents[1]
SIZE = 640  # pose input (matches the 640-stretched training)


def _best_keypoints(res, conf: float) -> dict[int, tuple[float, float]]:
    if res.keypoints is None or len(res.keypoints) == 0:
        return {}
    confs = res.keypoints.conf.cpu().numpy()
    xy = res.keypoints.xy.cpu().numpy()
    b = int(confs.sum(axis=1).argmax())
    return {
        i: (float(xy[b, i, 0]), float(xy[b, i, 1]))
        for i in range(NUM_KEYPOINTS)
        if confs[b, i] >= conf
    }


def coarse_pass(
    video: str,
    pose_weights: str,
    sample_fps: float,
    kpt_conf: float,
    max_seconds: float | None,
) -> tuple[list[tuple[int, bool]], float]:
    """Sample ~`sample_fps` and record (frame_index, registered) for each sample."""
    from ultralytics import YOLO

    pose = YOLO(pose_weights)
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
        registered = fit_homography(_best_keypoints(res, kpt_conf)) is not None
        samples.append((frame_idx, registered))
    cap.release()
    return samples, fps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("--pose-weights", default=str(ROOT / "runs/pose/court_pose/weights/best.pt"))
    p.add_argument("--output", default="report")
    p.add_argument("--sample-fps", type=float, default=1.0, help="coarse sampling rate")
    p.add_argument("--kpt-conf", type=float, default=0.5)
    p.add_argument("--min-seg-s", type=float, default=8.0, help="drop segments shorter than this")
    p.add_argument("--max-gap-s", type=float, default=2.0, help="bridge dropouts up to this long")
    p.add_argument("--max-seconds", type=float, default=None, help="cap for a quick test")
    args = p.parse_args()

    samples, fps = coarse_pass(
        args.video, args.pose_weights, args.sample_fps, args.kpt_conf, args.max_seconds
    )
    segments: list[Segment] = segment_registered(
        samples, fps, min_len_s=args.min_seg_s, max_gap_s=args.max_gap_s
    )

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
    print(json.dumps(meta, indent=2))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "segments.json").write_text(json.dumps(payload, indent=2))
    print(f"segments -> {out_dir / 'segments.json'}")


if __name__ == "__main__":
    main()
