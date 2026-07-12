"""Per-player stats by reading jersey numbers — task D (D-4).

Extends the §4.3 registered-stats runner with identity: it detects each jersey
number at native resolution, reads it with the D-3 classifier, matches it to a
player track (IoS ≥ 0.9), votes over time, and merges same-number tracks that
never overlap in time. Fragmented anonymous tracks become a per-player box
score ("player #23: distance ...").

The pipeline itself (models, collect pass, stitch → vote → merge, honesty
telemetry) lives in `hoopvision.identify_pipeline` — shared with the
full-game runner `scripts/game_report.py` (task H) — so this script is just
the single-clip CLI around it. See that module's docstring for the coordinate
handling (640-stretch for detection/registration, NATIVE 1280 for number
crops) and the stitching-before-reading design.

The honest headline is the **read rate**: how many tracks actually get a
confirmed number on a 720p panning broadcast. It is reported, not hidden — a
low rate still yields a valid hybrid (named where read, per-track otherwise).

    uv run python scripts/identify_players.py --start 2 --seconds 30 \
        --json docs/player_identity_nba.json          # --no-stitch for baseline

Needs pose weights (release v0.4.0), the v1 player detector (v0.2.0), and the
D-2 number detector + D-3 classifier (v0.5.0: number_detector.pt,
number_classifier.pt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hoopvision.identify_pipeline import PipelineModels, collect, identify_tracks

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default=str(ROOT / "data/clips/_nba_raw.webm"))
    p.add_argument("--pose-weights", default=str(ROOT / "runs/pose/court_pose/weights/best.pt"))
    p.add_argument("--player-weights", default=str(ROOT / "hoopvision_best.pt"))
    p.add_argument(
        "--number-weights", default=str(ROOT / "runs/detect/number_detector/weights/best.pt")
    )
    p.add_argument(
        "--classifier-weights", default=str(ROOT / "runs/classify/number_classifier/best.pt")
    )
    p.add_argument("--start", type=float, default=2.0)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--kpt-conf", type=float, default=0.5)
    p.add_argument("--player-conf", type=float, default=0.3)
    p.add_argument("--number-conf", type=float, default=0.3)
    p.add_argument("--read-every", type=int, default=5, help="frames between number reads")
    p.add_argument("--min-frames", type=int, default=15)
    p.add_argument("--min-votes", type=int, default=3)
    p.add_argument(
        "--stitch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="court-space appearance stitching before reading (--no-stitch to disable)",
    )
    p.add_argument("--json", default=None)
    args = p.parse_args()

    models = PipelineModels.load(
        args.pose_weights,
        args.player_weights,
        args.number_weights,
        args.classifier_weights,
        player_conf=args.player_conf,
    )
    got = collect(
        args.video,
        models,
        args.start,
        args.seconds,
        kpt_conf=args.kpt_conf,
        number_conf=args.number_conf,
        read_every=args.read_every,
    )
    result = identify_tracks(
        got, stitch=args.stitch, min_votes=args.min_votes, min_frames=args.min_frames
    )

    payload = {"meta": result.meta, "players": result.players}
    named = [r for r in result.players if r["number"]]
    print(json.dumps({**result.meta, "named_players": named[:10]}, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"stats -> {args.json}")


if __name__ == "__main__":
    main()
