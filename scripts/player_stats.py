"""Per-player distance & speed + a court occupancy heatmap.

Runs the pipeline on a calibrated fixed-camera clip and reports, per track,
distance covered and average / top speed in physical units, plus a heatmap
PNG. Numbers are per track (not per named player — jersey OCR is future work)
and only meaningful with a real homography, so a calibration is required.

    uv run python scripts/player_stats.py data/clips/hudl_static2.mp4 \
        --calibration calib_hudl_static2.json --weights hoopvision_best.pt \
        --heatmap docs/occupancy_hudl_static2.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--calibration", required=True, help="court calibration JSON")
    parser.add_argument("--weights", default="hoopvision_best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1.0, help="drop shorter tracks")
    parser.add_argument("--heatmap", default=None, help="output PNG for the occupancy heatmap")
    args = parser.parse_args()

    from hoopvision.court import CourtCalibration
    from hoopvision.detect import YoloDetector
    from hoopvision.pipeline import analyze
    from hoopvision.stats import court_heatmap, player_stats

    calibration = CourtCalibration.load(args.calibration)
    detector = YoloDetector(weights=args.weights, conf=args.conf)
    analysis = analyze(args.video, detector, calibration=calibration, stride=args.stride)

    min_frames = int(args.min_seconds * analysis.effective_fps)
    stats = player_stats(analysis, calibration, min_frames=min_frames)

    print(
        f"\nPlayer movement — {Path(args.video).name} "
        f"({len(analysis.records)} frames, {analysis.effective_fps:.1f} fps)\n"
    )
    print("| track | team | seconds | distance (ft) | avg (mph) | top (mph) |")
    print("|---|---|---|---|---|---|")
    for s in stats:
        team = "—" if s.team is None else s.team
        print(
            f"| {s.track_id} | {team} | {s.seconds} | {s.distance_ft} "
            f"| {s.avg_speed_mph} | {s.top_speed_mph} |"
        )
    if not stats:
        print("| (no track met the minimum duration) |")

    if args.heatmap:
        court_heatmap(
            analysis,
            calibration,
            output_path=args.heatmap,
            title=f"Player occupancy — {Path(args.video).stem}",
        )
        print(f"\nSaved heatmap: {args.heatmap}")


if __name__ == "__main__":
    main()
