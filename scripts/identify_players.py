"""Per-player stats by reading jersey numbers — task D (D-4).

Extends the §4.3 registered-stats runner with identity: it detects each jersey
number at native resolution, reads it with the D-3 classifier, matches it to a
player track (IoS ≥ 0.9), votes over time, and merges same-number tracks that
never overlap in time. Fragmented anonymous tracks become a per-player box
score ("player #23: distance ...").

Coordinate handling (the crux): player detection + court registration run on a
640×640 stretch of each frame (matching their training), but number boxes are
~12-17 px and vanish at 640, so number detection runs on the NATIVE frame at
imgsz=1280 and crops come from the native frame too. Native number boxes are
then scaled into 640 space to match track boxes for IoS.

The honest headline is the **read rate**: how many tracks actually get a
confirmed number on a 720p panning broadcast. It is reported, not hidden — a
low rate still yields a valid hybrid (named where read, per-track otherwise).

    uv run python scripts/identify_players.py --start 2 --seconds 30 \
        --json docs/player_identity_nba.json

Needs pose weights (release v0.4.0), the v1 player detector (v0.2.0), and the
D-2 number detector + D-3 classifier (v0.5.0: number_detector.pt,
number_classifier.pt).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from hoopvision.court_template import NUM_KEYPOINTS
from hoopvision.identity import NumberRead, TrackBox, identify
from hoopvision.registration import CourtRegistrar, image_to_court
from hoopvision.stats import PlayerStat, TrackPath, stats_from_paths

ROOT = Path(__file__).resolve().parents[1]
SIZE = 640  # detector / registration input (matches 640-stretched training)
BOUNDS_MARGIN = 2.0


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
    from hoopvision.court_template import COURT_LENGTH_FT, COURT_WIDTH_FT

    x, y = xy
    return -margin <= x <= COURT_LENGTH_FT + margin and -margin <= y <= COURT_WIDTH_FT + margin


class NumberReader:
    """D-2 detector + D-3 classifier: native frame → number reads (native px)."""

    def __init__(self, detector_weights: str, classifier_weights: str, device: str):
        import torch
        from torchvision import models, transforms
        from ultralytics import YOLO

        self.torch = torch
        self.device = device
        self.det = YOLO(detector_weights)
        self.number_id = next((i for i, n in self.det.names.items() if n == "number"), 2)

        ckpt = torch.load(classifier_weights, map_location=device, weights_only=True)
        self.classes: list[str] = ckpt["classes"]
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)
        self.model = model
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def read(self, frame: np.ndarray, det_conf: float, pad: int = 10) -> list[tuple[tuple, str]]:
        """Return [(native_xyxy, number)] for every number box in the frame."""
        res = self.det.predict(
            frame, imgsz=1280, conf=det_conf, classes=[self.number_id], verbose=False
        )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        h, w = frame.shape[:2]
        crops, native_boxes = [], []
        for xyxy in boxes.xyxy.tolist():
            x1, y1, x2, y2 = xyxy
            cx1, cy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
            cx2, cy2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            crop = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
            crops.append(self.tf(crop))
            native_boxes.append((x1, y1, x2, y2))
        if not crops:
            return []
        batch = self.torch.stack(crops).to(self.device)
        with self.torch.no_grad():
            pred = self.model(batch).argmax(1).cpu().tolist()
        return [(box, self.classes[p]) for box, p in zip(native_boxes, pred, strict=True)]


def collect(
    video: str,
    pose_weights: str,
    player_weights: str,
    reader: NumberReader,
    start: float,
    seconds: float,
    kpt_conf: float,
    player_conf: float,
    number_conf: float,
    read_every: int,
) -> tuple[
    dict[int, TrackPath],
    list[NumberRead],
    list[TrackBox],
    dict[int, tuple[int, int]],
    dict,
]:
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
    track_boxes: list[TrackBox] = []
    spans: dict[int, tuple[int, int]] = {}
    reads: list[NumberRead] = []
    processed = registered = read_frames = 0

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        processed += 1
        t = i / fps
        h, w = frame.shape[:2]
        sx, sy = SIZE / w, SIZE / h
        f640 = cv2.resize(frame, (SIZE, SIZE))

        res = pose.predict(f640, verbose=False, conf=0.25)[0]
        H = reg.update(_best_keypoints(res, kpt_conf))
        tracked = tracker.update(players.detect(f640))

        for p in tracked:
            track_boxes.append(TrackBox(i, p.track_id, p.xyxy))  # 640-space
            lo, hi = spans.get(p.track_id, (i, i))
            spans[p.track_id] = (min(lo, i), max(hi, i))
            if H is not None:
                court = image_to_court(H, np.array([p.foot]))[0]
                if _in_bounds(court):
                    paths.setdefault(p.track_id, []).append(
                        (t, (float(court[0]), float(court[1])), p.team)
                    )
        if H is not None:
            registered += 1

        if i % read_every == 0:
            read_frames += 1
            for native_box, number in reader.read(frame, number_conf):
                nx1, ny1, nx2, ny2 = native_box
                box640 = (nx1 * sx, ny1 * sy, nx2 * sx, ny2 * sy)  # into 640 space
                reads.append(NumberRead(i, box640, number))

    cap.release()
    meta = {
        "clip": Path(video).name,
        "window_s": [round(start, 1), round(start + seconds, 1)],
        "frames_processed": processed,
        "frames_registered": registered,
        "registration_rate": round(registered / max(processed, 1), 3),
        "frames_read": read_frames,
        "read_every": read_every,
        "number_reads": len(reads),
        "tracks_seen": len(spans),
    }
    return paths, reads, track_boxes, spans, meta


def apply_remap(paths: dict[int, TrackPath], remap: dict[int, int]) -> dict[int, TrackPath]:
    merged: dict[int, TrackPath] = {}
    for tid, obs in paths.items():
        merged.setdefault(remap.get(tid, tid), []).extend(obs)
    for obs in merged.values():
        obs.sort(key=lambda o: o[0])
    return merged


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
    p.add_argument("--json", default=None)
    args = p.parse_args()

    from hoopvision.detect import default_device

    device = default_device()
    reader = NumberReader(args.number_weights, args.classifier_weights, device)

    paths, reads, track_boxes, spans, meta = collect(
        args.video,
        args.pose_weights,
        args.player_weights,
        reader,
        args.start,
        args.seconds,
        args.kpt_conf,
        args.player_conf,
        args.number_conf,
        args.read_every,
    )

    remap, numbers = identify(reads, track_boxes, spans, min_ios=0.9, min_votes=args.min_votes)
    merged_paths = apply_remap(paths, remap)
    stats: list[PlayerStat] = stats_from_paths(merged_paths, min_frames=args.min_frames)

    identified = {tid for tid, canon in remap.items() if canon in numbers}
    meta["tracks_after_merge"] = len(merged_paths)
    meta["tracks_identified"] = len(numbers)
    meta["read_rate"] = round(len(identified) / max(len(spans), 1), 3)

    # Honesty telemetry: what the classifier actually read, and whether a number
    # got confirmed on two players at once (concurrent same-number tracks can't
    # be the same person, so a duplicated number exposes read-precision limits).
    meta["number_read_histogram"] = dict(Counter(r.number for r in reads).most_common())
    number_to_canons: dict[str, set[int]] = {}
    for canon, num in numbers.items():
        number_to_canons.setdefault(num, set()).add(canon)
    meta["numbers_on_multiple_players"] = {
        num: len(canons) for num, canons in number_to_canons.items() if len(canons) > 1
    }

    players = []
    for s in stats:
        row = vars(s) | {"number": numbers.get(s.track_id)}
        players.append(row)

    payload = {"meta": meta, "players": players}
    named = [r for r in players if r["number"]]
    print(json.dumps({**meta, "named_players": named[:10]}, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"stats -> {args.json}")


if __name__ == "__main__":
    main()
