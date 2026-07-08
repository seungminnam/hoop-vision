"""End-to-end pipeline: video in → annotated video + events.json (+ shot chart).

Two passes over the clip:
  pass 1  detect + track + collect (team color features, ball track, rim boxes)
  pass 2  render annotations with team colors and the minimap overlay

Run:  python -m hoopvision.pipeline clip.mp4 --calibration calib.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from .court import CourtCalibration
from .detect import BALL, PLAYER, RIM, Detection, Detector, YoloDetector
from .events import ShotConfig, ShotResult, detect_shots
from .ingest import frames, video_info
from .motion import CameraMotionEstimator, as_3x3, warp_box
from .shotchart import render_shot_chart
from .stitch import TrackletBuilder, stitch
from .teams import TeamAssigner
from .track import PlayerTracker, TrackedPlayer
from .viz import VideoSink, annotate_frame, overlay_minimap, render_minimap


@dataclass
class FrameRecord:
    index: int  # true frame index in the source video
    players: list[TrackedPlayer]
    ball: Detection | None
    rim: Detection | None


@dataclass
class ClipAnalysis:
    video: Path
    fps: float
    stride: int
    records: list[FrameRecord] = field(default_factory=list)
    rim_box: tuple[float, float, float, float] | None = None
    shots: ShotResult = field(default_factory=ShotResult)
    shot_details: list[dict] = field(default_factory=list)  # +court_xy/shooter

    @property
    def effective_fps(self) -> float:
        return self.fps / self.stride


def _best(detections: list[Detection], class_name: str) -> Detection | None:
    candidates = [d for d in detections if d.class_name == class_name]
    return max(candidates, key=lambda d: d.confidence) if candidates else None


def _median_rim_box(
    records: list[FrameRecord], min_hits: int = 5
) -> tuple[float, float, float, float] | None:
    boxes = [r.rim.xyxy for r in records if r.rim is not None]
    if len(boxes) < min_hits:
        return None
    return tuple(np.median(np.array(boxes), axis=0).tolist())


def _attribute_shooter(
    record: FrameRecord, ball_xy: tuple[float, float], max_dist_px: float = 180.0
) -> TrackedPlayer | None:
    """Nearest tracked player to the ball at the attempt trigger, if close."""
    best, best_dist = None, max_dist_px
    bx, by = ball_xy
    for p in record.players:
        px = (p.xyxy[0] + p.xyxy[2]) / 2
        py = (p.xyxy[1] + p.xyxy[3]) / 2
        dist = float(np.hypot(px - bx, py - by))
        if dist < best_dist:
            best, best_dist = p, dist
    return best


def analyze(
    video: str | Path,
    detector: Detector | None = None,
    calibration: CourtCalibration | None = None,
    shot_config: ShotConfig | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    teams: bool = True,
    stitch_tracks: bool = True,
    compensate_camera: bool = False,
) -> ClipAnalysis:
    video = Path(video)
    info = video_info(video)
    detector = detector or YoloDetector()
    tracker = PlayerTracker(frame_rate=info.fps / stride)
    assigner = TeamAssigner()
    builder = TrackletBuilder() if stitch_tracks else None
    motion = CameraMotionEstimator() if compensate_camera else None
    analysis = ClipAnalysis(video=video, fps=info.fps, stride=stride)

    for ordinal, (index, frame) in enumerate(frames(video, stride=stride, max_frames=max_frames)):
        detections = detector.detect(frame)
        if motion is not None:
            # Track in a camera-stabilised reference frame, then map ids back.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            player_boxes = [d.xyxy for d in detections if d.class_name == PLAYER]
            ref_from_cur = motion.update(gray, player_boxes)
            ref_dets = [
                replace(d, xyxy=warp_box(d.xyxy, ref_from_cur))
                for d in detections
                if d.class_name == PLAYER
            ]
            cur_from_ref = np.linalg.inv(as_3x3(ref_from_cur))[:2]
            players = [
                replace(tp, xyxy=warp_box(tp.xyxy, cur_from_ref)) for tp in tracker.update(ref_dets)
            ]
        else:
            players = tracker.update(detections)
        if builder is not None:
            for p in players:
                builder.observe(p.track_id, ordinal, p.xyxy, frame)
        if teams:
            for p in players:
                assigner.observe(p.track_id, frame, p.xyxy)
        analysis.records.append(
            FrameRecord(index, players, _best(detections, BALL), _best(detections, RIM))
        )

    if builder is not None:
        remap = stitch(builder.build())
        for record in analysis.records:
            record.players = [
                replace(p, track_id=remap.get(p.track_id, p.track_id)) for p in record.players
            ]
        if teams:
            assigner.remap_ids(remap)

    if teams:
        assigner.fit()
        for record in analysis.records:
            record.players = [p.with_team(assigner.team_of(p.track_id)) for p in record.players]

    analysis.rim_box = _median_rim_box(analysis.records)
    ball_centers = [r.ball.center if r.ball else None for r in analysis.records]
    analysis.shots = detect_shots(
        ball_centers, analysis.rim_box, analysis.effective_fps, shot_config
    )

    for event in analysis.shots.events:
        record = analysis.records[event.frame]
        shooter = _attribute_shooter(record, event.ball_xy)
        detail: dict = {
            "frame": record.index,
            "time_s": round(record.index / analysis.fps, 3),
            "outcome": event.outcome,
            "shooter_track_id": shooter.track_id if shooter else None,
            "shooter_team": shooter.team if shooter else None,
            "court_xy": None,
        }
        if calibration is not None:
            anchor = shooter.foot if shooter else event.ball_xy
            court_xy = calibration.to_court([anchor])[0]
            detail["court_xy"] = [round(float(v), 2) for v in court_xy]
        analysis.shot_details.append(detail)

    return analysis


def render(
    analysis: ClipAnalysis,
    output_path: str | Path,
    calibration: CourtCalibration | None = None,
) -> Path:
    output_path = Path(output_path)
    info = video_info(analysis.video)
    records = {r.index: r for r in analysis.records}
    with VideoSink(output_path, analysis.effective_fps, info.width, info.height) as sink:
        for index, frame in frames(analysis.video, stride=analysis.stride):
            record = records.get(index)
            if record is None:
                continue
            out = annotate_frame(frame, record.players, record.ball, analysis.rim_box)
            if calibration is not None and record.players:
                feet = np.array([p.foot for p in record.players])
                court_pts = calibration.to_court(feet)
                mask = calibration.in_bounds(court_pts)
                positions = [
                    (p.track_id, float(x), float(y), p.team)
                    for p, (x, y), ok in zip(record.players, court_pts, mask, strict=True)
                    if ok
                ]
                out = overlay_minimap(out, render_minimap(positions))
            sink.write(out)
    return output_path


def events_payload(analysis: ClipAnalysis) -> dict:
    return {
        "video": analysis.video.name,
        "fps": analysis.fps,
        "stride": analysis.stride,
        "frames_processed": len(analysis.records),
        "rim_detected": analysis.rim_box is not None,
        "ball_coverage": round(analysis.shots.ball_coverage, 3),
        "shot_analytics_available": analysis.shots.available,
        "reason": analysis.shots.reason,
        "events": analysis.shot_details,
    }


def run(
    video: str | Path,
    weights: str = "yolo11n.pt",
    calibration_path: str | Path | None = None,
    output_dir: str | Path = "out",
    conf: float = 0.25,
    stride: int = 1,
    max_frames: int | None = None,
) -> dict:
    video = Path(video)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = CourtCalibration.load(calibration_path) if calibration_path else None

    started = time.perf_counter()
    detector = YoloDetector(weights=weights, conf=conf)
    analysis = analyze(video, detector, calibration, stride=stride, max_frames=max_frames)
    annotated = render(analysis, output_dir / f"{video.stem}_annotated.mp4", calibration)
    elapsed = time.perf_counter() - started

    payload = events_payload(analysis)
    events_path = output_dir / f"{video.stem}_events.json"
    events_path.write_text(json.dumps(payload, indent=2))

    chart_path = None
    chart_shots = [s for s in analysis.shot_details if s["court_xy"]]
    if analysis.shots.available and chart_shots:
        chart_path = output_dir / f"{video.stem}_shotchart.png"
        render_shot_chart(
            [{"court_xy": s["court_xy"], "outcome": s["outcome"]} for s in chart_shots],
            chart_path,
            title=f"Shot chart — {video.stem}",
        )

    fps_processed = len(analysis.records) / elapsed if elapsed else 0.0
    print(f"Processed {len(analysis.records)} frames in {elapsed:.1f}s ({fps_processed:.1f} FPS)")
    print(f"Annotated video: {annotated}")
    print(f"Events: {events_path}")
    if not analysis.shots.available:
        print(f"Shot analytics unavailable: {analysis.shots.reason}")
    else:
        made = sum(e["outcome"] == "made" for e in analysis.shot_details)
        print(f"Shots: {len(analysis.shot_details)} attempts, {made} made")
    if chart_path:
        print(f"Shot chart: {chart_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="input clip (mp4/mov)")
    parser.add_argument(
        "--weights", default="yolo11n.pt", help="YOLO weights (COCO pretrained or fine-tuned)"
    )
    parser.add_argument(
        "--calibration", default=None, help="court calibration JSON from scripts/calibrate.py"
    )
    parser.add_argument("--output", default="out", help="output directory")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=1, help="process every n-th frame")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    run(
        args.video,
        weights=args.weights,
        calibration_path=args.calibration,
        output_dir=args.output,
        conf=args.conf,
        stride=args.stride,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
