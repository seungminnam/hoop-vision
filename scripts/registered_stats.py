"""Court-coordinate player stats on a moving-camera clip (v2 §4.3).

v1 could only produce physical player stats on a *static* clip with a hand
calibration. This does it on a **panning broadcast**: every frame is registered
to NBA feet by the Phase-2 `CourtRegistrar`, so each tracked player's foot point
becomes a court coordinate — and the distance/speed math is identical to v1
(`stats.stats_from_paths`, which is coordinate-frame agnostic).

Because court coordinates are camera-invariant, a panning camera needs no
separate motion compensation — the homography absorbs it. This is the honest
win over v1.1's camera-motion estimator, which did not improve image-space
tracking (see ROADMAP §3.2 / decisions).

    uv run python scripts/registered_stats.py --start 2 --seconds 30 \
        --json docs/registered_stats_nba.json \
        --heatmap docs/registered_occupancy_nba.png
        # optional: --trails PATH draws the top-N longest tracks

Needs the pose weights (`runs/pose/court_pose/weights/best.pt`, release v0.4.0)
and the v1 player detector (`hoopvision_best.pt`, release v0.2.0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hoopvision.court_template import COURT_LENGTH_FT, COURT_WIDTH_FT, NUM_KEYPOINTS
from hoopvision.registration import CourtRegistrar, image_to_court
from hoopvision.stats import PlayerStat, TrackPath, stats_from_paths

ROOT = Path(__file__).resolve().parents[1]
SIZE = 640  # detector input (matches the 640x640-stretched training data)
BOUNDS_MARGIN = 2.0  # feet outside the lines still counts (baseline players)


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


def _in_bounds(xy: np.ndarray, margin: float = BOUNDS_MARGIN) -> bool:
    x, y = xy
    return -margin <= x <= COURT_LENGTH_FT + margin and -margin <= y <= COURT_WIDTH_FT + margin


def collect_paths(
    video: str,
    pose_weights: str,
    player_weights: str,
    start: float,
    seconds: float,
    kpt_conf: float,
    player_conf: float,
) -> tuple[dict[int, TrackPath], dict]:
    """Run detection+registration over the clip → per-track court-feet paths."""
    from ultralytics import YOLO

    from hoopvision.detect import YoloDetector
    from hoopvision.track import PlayerTracker

    pose = YOLO(pose_weights)
    players = YoloDetector(player_weights, conf=player_conf)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    n_frames = int(seconds * fps)

    reg = CourtRegistrar(alpha=0.35, max_misses=20)
    tracker = PlayerTracker(frame_rate=fps)
    paths: dict[int, TrackPath] = {}
    processed = registered = 0

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        processed += 1
        t = i / fps
        f640 = cv2.resize(frame, (SIZE, SIZE))

        res = pose.predict(f640, verbose=False, conf=0.25)[0]
        H = reg.update(_best_keypoints(res, kpt_conf))
        tracked = tracker.update(players.detect(f640))
        if H is None:
            continue
        registered += 1
        for p in tracked:
            court = image_to_court(H, np.array([p.foot]))[0]
            if _in_bounds(court):
                paths.setdefault(p.track_id, []).append(
                    (t, (float(court[0]), float(court[1])), p.team)
                )

    cap.release()
    meta = {
        "clip": Path(video).name,
        "window_s": [round(start, 1), round(start + seconds, 1)],
        "frames_processed": processed,
        "frames_registered": registered,
        "registration_rate": round(registered / max(processed, 1), 3),
        "tracks_seen": len(paths),
    }
    return paths, meta


def render_heatmap(paths: dict[int, TrackPath], out: Path, title: str) -> None:
    """Whole-clip occupancy heatmap (rendering shared with game_report, H-2)."""
    from hoopvision.viz import render_court_heatmap

    pts = np.array([xy for obs in paths.values() for _, xy, _ in obs])
    render_court_heatmap(pts, out, title)


def render_trails(
    paths: dict[int, TrackPath], out: Path, title: str, min_frames: int = 15, top_n: int = 12
) -> None:
    """Smoothed movement trails of the `top_n` longest tracks (readable subset)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hoopvision.stats import _smooth
    from hoopvision.viz import draw_court_ft

    fig, ax = plt.subplots(figsize=(9, 5))
    draw_court_ft(ax)
    cmap = plt.get_cmap("tab20")
    longest = [obs for obs in paths.values() if len(obs) >= min_frames]
    longest.sort(key=len, reverse=True)
    for k, obs in enumerate(longest[:top_n]):
        xy = _smooth(np.array([o[1] for o in obs]), window=9)
        ax.plot(xy[:, 0], xy[:, 1], "-", color=cmap(k % 20), lw=1.6, alpha=0.85, zorder=2)
    ax.set_title(f"{title} (top {min(top_n, len(longest))} of {len(longest)} tracks)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default=str(ROOT / "data/clips/_nba_raw.webm"))
    p.add_argument("--pose-weights", default=str(ROOT / "runs/pose/court_pose/weights/best.pt"))
    p.add_argument("--player-weights", default=str(ROOT / "hoopvision_best.pt"))
    p.add_argument("--start", type=float, default=2.0)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--kpt-conf", type=float, default=0.5)
    p.add_argument("--player-conf", type=float, default=0.3)
    p.add_argument("--min-frames", type=int, default=15)
    p.add_argument("--json", default=None)
    p.add_argument("--heatmap", default=None)
    p.add_argument("--trails", default=None)
    p.add_argument("--gate", type=float, default=0.8, help="min registration rate to trust stats")
    args = p.parse_args()

    paths, meta = collect_paths(
        args.video,
        args.pose_weights,
        args.player_weights,
        args.start,
        args.seconds,
        args.kpt_conf,
        args.player_conf,
    )
    stats: list[PlayerStat] = stats_from_paths(paths, min_frames=args.min_frames)
    meta["tracks_reported"] = len(stats)
    meta["court_analytics"] = "ok" if meta["registration_rate"] >= args.gate else "unavailable"

    payload = {
        "meta": meta,
        "players": [vars(s) for s in stats],
    }
    print(json.dumps({**meta, "top_tracks": [vars(s) for s in stats[:5]]}, indent=2))
    if meta["court_analytics"] == "unavailable":
        print(f"WARNING: registration rate {meta['registration_rate']} < gate {args.gate}")

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"stats -> {args.json}")
    if args.heatmap:
        render_heatmap(paths, Path(args.heatmap), "NBA broadcast occupancy (registered)")
        print(f"heatmap -> {args.heatmap}")
    if args.trails:
        render_trails(
            paths, Path(args.trails), "Player trails on registered court", args.min_frames
        )
        print(f"trails -> {args.trails}")


if __name__ == "__main__":
    main()
