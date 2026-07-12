"""Reusable detect → register → track → identify pipeline — tasks D/E/G, shared by H.

`scripts/identify_players.py` (single-clip identity stats) and
`scripts/game_report.py` (full-game fine pass, task H-2) run the exact same
per-window pipeline; this module holds it once so the two runners cannot
drift. The flow:

  1. `PipelineModels.load()` — pose keypoints + player detector + D-2 number
     detector + D-3 classifier, loaded **once** (a full game has hundreds of
     segments; reloading YOLO per segment would dominate runtime).
  2. `collect()` — one pass over a frame window: detect players (640-stretch),
     register the court, track, crop + read jersey numbers at NATIVE
     resolution, gather per-track court paths / appearance / spans.
     A fresh tracker + registrar per call, so calling it per broadcast segment
     gives the boundary reset for free (identity cannot survive a camera cut).
  3. `identify_tracks()` — court-space stitching (E-1), IoS matching + temporal
     vote + same-number merge (D-4), per-track movement stats, and the honesty
     telemetry (read rate, vote counts, number histogram, duplicate numbers).

Model/video I/O lives here; the pure logic stays in `identity`, `stitch`,
`stats`, and `aggregate` where it is unit-tested.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .court_template import COURT_LENGTH_FT, COURT_WIDTH_FT, NUM_KEYPOINTS
from .identity import NumberRead, TrackBox, identify, match_reads_to_tracks
from .registration import CourtRegistrar, image_to_court
from .stats import PlayerStat, TrackPath, stats_from_paths
from .stitch import CourtTracklet, stitch_court
from .teams import color_histogram, torso_crop

SIZE = 640  # detector / registration input (matches 640-stretched training)
BOUNDS_MARGIN = 2.0
ABSTAIN_LABEL = "unreadable"  # classifier's abstain class (task G); dropped from voting


def best_keypoints(res, conf: float) -> dict[int, tuple[float, float]]:
    """Confident keypoints of the best court instance in a pose result."""
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


@dataclass
class PipelineModels:
    """All models the pipeline needs, loaded once and reused across windows."""

    pose: object  # ultralytics YOLO (pose)
    players: object  # hoopvision.detect.YoloDetector
    reader: NumberReader
    device: str

    @classmethod
    def load(
        cls,
        pose_weights: str,
        player_weights: str,
        number_weights: str,
        classifier_weights: str,
        player_conf: float = 0.3,
    ) -> PipelineModels:
        from ultralytics import YOLO

        from .detect import YoloDetector, default_device

        device = default_device()
        return cls(
            pose=YOLO(pose_weights),
            players=YoloDetector(player_weights, conf=player_conf),
            reader=NumberReader(number_weights, classifier_weights, device),
            device=device,
        )


@dataclass
class Collected:
    """Everything one pass over a frame window yields (collect() is called once)."""

    paths: dict[int, TrackPath]
    reads: list[NumberRead]
    track_boxes: list[TrackBox]
    spans: dict[int, tuple[int, int]]
    tracklets: list[CourtTracklet]  # per-track court endpoints + appearance
    fps: float
    meta: dict


def _build_tracklets(
    first_court: dict[int, tuple[int, tuple[float, float]]],
    last_court: dict[int, tuple[int, tuple[float, float]]],
    feats: dict[int, list[np.ndarray]],
) -> list[CourtTracklet]:
    """Per-track court endpoints + mean torso histogram → stitching tracklets."""
    tracklets = []
    for tid, (sf, sft) in first_court.items():
        lf, lft = last_court[tid]
        flist = feats.get(tid, [])
        if flist:
            mean = np.mean(flist, axis=0)
            norm = float(np.linalg.norm(mean))
            feature = (mean / norm).astype(np.float32) if norm > 0 else mean.astype(np.float32)
        else:
            feature = np.zeros(1, dtype=np.float32)
        tracklets.append(CourtTracklet(tid, sf, lf, sft, lft, feature))
    return tracklets


def collect(
    video: str,
    models: PipelineModels,
    start: float,
    seconds: float,
    kpt_conf: float = 0.5,
    number_conf: float = 0.3,
    read_every: int = 5,
    stride: int = 1,
) -> Collected:
    """One pass over `[start, start+seconds)`: tracks, court paths, number reads.

    Fresh tracker + registrar per call — callers running per broadcast segment
    get the boundary reset (task H design ①) for free. `stride` analyses every
    n-th frame (budget knob; grabbed-but-skipped frames cost no decode work),
    and `read_every` counts *analysed* frames between number reads. Frame
    indices in the result are window-relative original-frame indices, and
    times are real seconds, so speeds stay correct at any stride.
    """
    from .track import PlayerTracker

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    n_frames = int(seconds * fps)

    reg = CourtRegistrar(alpha=0.35, max_misses=max(1, 20 // stride))
    tracker = PlayerTracker(frame_rate=fps / stride)
    paths: dict[int, TrackPath] = {}
    track_boxes: list[TrackBox] = []
    spans: dict[int, tuple[int, int]] = {}
    reads: list[NumberRead] = []
    feats: dict[int, list[np.ndarray]] = {}  # torso histograms per track (640 space)
    first_court: dict[int, tuple[int, tuple[float, float]]] = {}
    last_court: dict[int, tuple[int, tuple[float, float]]] = {}
    processed = registered = read_frames = abstain_reads = 0

    for i in range(n_frames):
        if i % stride:
            if not cap.grab():  # skip decode entirely on strided-out frames
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        processed += 1
        t = i / fps
        h, w = frame.shape[:2]
        sx, sy = SIZE / w, SIZE / h
        f640 = cv2.resize(frame, (SIZE, SIZE))

        res = models.pose.predict(f640, verbose=False, conf=0.25)[0]
        H = reg.update(best_keypoints(res, kpt_conf))
        tracked = tracker.update(models.players.detect(f640))

        for p in tracked:
            track_boxes.append(TrackBox(i, p.track_id, p.xyxy))  # 640-space
            lo, hi = spans.get(p.track_id, (i, i))
            spans[p.track_id] = (min(lo, i), max(hi, i))
            crop = torso_crop(f640, p.xyxy)  # appearance for stitching
            if crop is not None:
                feats.setdefault(p.track_id, []).append(color_histogram(crop))
            if H is not None:
                court = image_to_court(H, np.array([p.foot]))[0]
                if _in_bounds(court):
                    ft = (float(court[0]), float(court[1]))
                    paths.setdefault(p.track_id, []).append((t, ft, p.team))
                    if p.track_id not in first_court:
                        first_court[p.track_id] = (i, ft)
                    last_court[p.track_id] = (i, ft)
        if H is not None:
            registered += 1

        if (processed - 1) % read_every == 0:
            read_frames += 1
            for native_box, number in models.reader.read(frame, number_conf):
                if number == ABSTAIN_LABEL:  # classifier said "can't read" -> no vote
                    abstain_reads += 1
                    continue
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
        "abstain_reads": abstain_reads,
        "tracks_seen": len(spans),
    }
    tracklets = _build_tracklets(first_court, last_court, feats)
    return Collected(paths, reads, track_boxes, spans, tracklets, fps, meta)


def apply_remap(paths: dict[int, TrackPath], remap: dict[int, int]) -> dict[int, TrackPath]:
    merged: dict[int, TrackPath] = {}
    for tid, obs in paths.items():
        merged.setdefault(remap.get(tid, tid), []).extend(obs)
    for obs in merged.values():
        obs.sort(key=lambda o: o[0])
    return merged


def _remap_spans(
    spans: dict[int, tuple[int, int]], remap: dict[int, int]
) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for tid, (lo, hi) in spans.items():
        c = remap.get(tid, tid)
        if c in out:
            out[c] = (min(out[c][0], lo), max(out[c][1], hi))
        else:
            out[c] = (lo, hi)
    return out


def _remap_boxes(track_boxes: list[TrackBox], remap: dict[int, int]) -> list[TrackBox]:
    return [TrackBox(tb.frame, remap.get(tb.track_id, tb.track_id), tb.xyxy) for tb in track_boxes]


@dataclass
class Identified:
    """identify_tracks() result: named/anonymous player rows + merged paths."""

    players: list[dict]  # vars(PlayerStat) | {"number": str | None}
    merged_paths: dict[int, TrackPath]
    numbers: dict[int, str]  # canonical track id -> confirmed number
    meta: dict  # the Collected meta, extended with identity telemetry


def identify_tracks(
    got: Collected,
    stitch: bool = True,
    min_votes: int = 3,
    min_frames: int = 15,
) -> Identified:
    """Stage 1 stitch + stage 2 identify + stats + honesty telemetry.

    Stage 1 stitches fragmented tracks in court space (appearance + speed
    gate) BEFORE reading, so a player's sparse reads pool onto one track and
    can clear the vote threshold. Stage 2 runs the number match/vote/merge on
    the stitched tracks. The returned meta extends `got.meta` with the same
    telemetry `identify_players.py` has always reported.
    """
    meta = dict(got.meta)
    stitch_remap = stitch_court(got.tracklets, got.fps) if stitch else {}
    boxes_s = _remap_boxes(got.track_boxes, stitch_remap)
    spans_s = _remap_spans(got.spans, stitch_remap)
    paths_s = apply_remap(got.paths, stitch_remap)

    num_remap, numbers = identify(got.reads, boxes_s, spans_s, min_ios=0.9, min_votes=min_votes)
    merged_paths = apply_remap(paths_s, num_remap)
    stats: list[PlayerStat] = stats_from_paths(merged_paths, min_frames=min_frames)

    meta["stitching"] = "on" if stitch else "off"
    meta["tracks_after_stitch"] = len(spans_s)
    meta["tracks_after_merge"] = len(merged_paths)
    meta["tracks_identified"] = len(numbers)
    # read rate = identified players / player-tracks after stitch (same metric
    # for stitch on/off, so the two runs compare directly).
    meta["read_rate"] = round(len(numbers) / max(len(spans_s), 1), 3)

    # Did stitching actually thicken per-track reads? Report the vote counts on
    # the (stitched) tracks that received any number read.
    votes = match_reads_to_tracks(got.reads, boxes_s)
    vote_counts = sorted((len(v) for v in votes.values()), reverse=True)
    meta["tracks_with_any_read"] = len(vote_counts)
    meta["max_votes_on_a_track"] = vote_counts[0] if vote_counts else 0
    meta["median_votes_among_read_tracks"] = float(np.median(vote_counts)) if vote_counts else 0.0

    # Honesty telemetry: what the classifier actually read, and whether a number
    # got confirmed on two players at once (concurrent same-number tracks can't
    # be the same person, so a duplicated number exposes read-precision limits).
    meta["number_read_histogram"] = dict(Counter(r.number for r in got.reads).most_common())
    number_to_canons: dict[str, set[int]] = {}
    for canon, num in numbers.items():
        number_to_canons.setdefault(num, set()).add(canon)
    meta["numbers_on_multiple_players"] = {
        num: len(canons) for num, canons in number_to_canons.items() if len(canons) > 1
    }

    players = [vars(s) | {"number": numbers.get(s.track_id)} for s in stats]
    return Identified(players, merged_paths, numbers, meta)
