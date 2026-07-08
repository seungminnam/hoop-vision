"""Interactive player-ID labeling tool for MOT ground truth.

Bootstraps from the tracker's own output (fragmented IDs and all), then you
correct IDs frame by frame and save MOTChallenge ground truth for
`scripts/eval_tracking.py`. The dominant error is fragmentation — one player
split into many IDs over time — so the main gesture is "relabel this whole
track to N", which collapses a fragment into a consistent player number.

Bootstrap source (one of):
  --boxes PRED.txt   a MOTChallenge file (from track_diagnostics.py --dump-mot)
  --weights W.pt     run detection+tracking now to generate boxes

Controls (shown on screen too):
  a / d            previous / next frame        , / .   jump 10 frames
  left-click       select the box under cursor
  0-9              type a player number          Backspace  delete a digit
  Enter            assign number to the whole track of the selected box
  x                assign number to only the selected box (fixes a swap)
  f                give the selected track the next unused number
  u                undo last change              s  save        q  quit

Saves to data/labels/mot/gt/<clip>.txt by default (1-based processed-frame
ordinals; see data/README.md). Re-open the same output to resume.

    uv run python scripts/label_tracks.py data/clips/hudl_seg1.mp4 \
        --boxes data/labels/mot/pred/hudl_seg1.txt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@dataclass
class Box:
    xywh: tuple[float, float, float, float]
    conf: float
    gid: int

    def contains(self, px: float, py: float) -> bool:
        x, y, w, h = self.xywh
        return x <= px <= x + w and y <= py <= y + h

    @property
    def area(self) -> float:
        return self.xywh[2] * self.xywh[3]


@dataclass
class LabelStore:
    """Per-frame boxes with editable player IDs. Pure logic (no OpenCV)."""

    frames: list[list[Box]] = field(default_factory=list)
    _undo: list[list[list[int]]] = field(default_factory=list)

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_mot(cls, sequence: dict[int, list[tuple]], n_frames: int | None = None) -> LabelStore:
        """Build from {ordinal(1-based): [(id, (x,y,w,h), conf?), ...]}."""
        count = n_frames if n_frames is not None else (max(sequence) if sequence else 0)
        frames: list[list[Box]] = [[] for _ in range(count)]
        for ordinal, dets in sequence.items():
            for det in dets:
                gid, xywh = det[0], det[1]
                conf = det[2] if len(det) > 2 else 1.0
                frames[ordinal - 1].append(Box(tuple(map(float, xywh)), float(conf), int(gid)))
        return cls(frames)

    @classmethod
    def load_mot_file(cls, path: Path) -> LabelStore:
        sequence: dict[int, list[tuple]] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            ordinal, gid = int(p[0]), int(p[1])
            xywh = tuple(float(v) for v in p[2:6])
            conf = float(p[6]) if len(p) > 6 and p[6] not in ("", "-1") else 1.0
            sequence.setdefault(ordinal, []).append((gid, xywh, conf))
        return cls.from_mot(sequence)

    # ---- editing ----------------------------------------------------------

    def _push_undo(self) -> None:
        self._undo.append([[b.gid for b in frame] for frame in self.frames])
        if len(self._undo) > 200:
            self._undo.pop(0)

    def undo(self) -> bool:
        if not self._undo:
            return False
        snapshot = self._undo.pop()
        for frame, gids in zip(self.frames, snapshot, strict=True):
            for box, gid in zip(frame, gids, strict=True):
                box.gid = gid
        return True

    def relabel_track(self, frame_i: int, det_i: int, new_id: int) -> int:
        """Reassign every box sharing the selected box's current id. Returns count."""
        current = self.frames[frame_i][det_i].gid
        if current == new_id:
            return 0
        self._push_undo()
        changed = 0
        for frame in self.frames:
            for box in frame:
                if box.gid == current:
                    box.gid = new_id
                    changed += 1
        return changed

    def override_box(self, frame_i: int, det_i: int, new_id: int) -> None:
        """Reassign a single box only (for fixing an ID swap mid-track)."""
        if self.frames[frame_i][det_i].gid == new_id:
            return
        self._push_undo()
        self.frames[frame_i][det_i].gid = new_id

    def next_free_id(self) -> int:
        used = {box.gid for frame in self.frames for box in frame}
        return max(used, default=0) + 1

    def box_at(self, frame_i: int, px: float, py: float) -> int | None:
        """Index of the smallest box containing (px, py) in a frame, or None."""
        hits = [(b.area, i) for i, b in enumerate(self.frames[frame_i]) if b.contains(px, py)]
        return min(hits)[1] if hits else None

    # ---- serialization ----------------------------------------------------

    def to_mot_lines(self) -> list[str]:
        lines = []
        for ordinal, frame in enumerate(self.frames, start=1):
            for b in frame:
                x, y, w, h = b.xywh
                lines.append(
                    f"{ordinal},{b.gid},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{b.conf:.3f},-1,-1,-1"
                )
        return lines

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.to_mot_lines()) + "\n")


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def store_from_weights(video: str, weights: str, stride: int, conf: float):
    """Run detection+tracking and return (LabelStore, ordinal->true_frame_index)."""
    from hoopvision.detect import YoloDetector
    from hoopvision.pipeline import analyze

    analysis = analyze(video, YoloDetector(weights=weights, conf=conf), stride=stride, teams=False)
    sequence: dict[int, list[tuple]] = {}
    index_map: dict[int, int] = {}
    for ordinal, record in enumerate(analysis.records, start=1):
        index_map[ordinal] = record.index
        dets = []
        for p in record.players:
            x1, y1, x2, y2 = p.xyxy
            dets.append((p.track_id, (x1, y1, x2 - x1, y2 - y1), p.confidence))
        sequence[ordinal] = dets
    return LabelStore.from_mot(sequence, n_frames=len(analysis.records)), index_map


# ---------------------------------------------------------------------------
# OpenCV UI (thin; logic lives in LabelStore)
# ---------------------------------------------------------------------------


def _color(gid: int) -> tuple[int, int, int]:
    import colorsys

    h = (gid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def run_ui(store: LabelStore, video: str, index_map: dict[int, int], out_path: Path) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")

    frame_cache: dict = {}

    def get_image(ordinal: int):
        true_index = index_map.get(ordinal, ordinal - 1)
        if true_index in frame_cache:
            return frame_cache[true_index]
        cap.set(cv2.CAP_PROP_POS_FRAMES, true_index)
        ok, img = cap.read()
        if not ok:
            return None
        if len(frame_cache) > 48:
            frame_cache.pop(next(iter(frame_cache)))
        frame_cache[true_index] = img
        return img

    n = len(store.frames)
    state = {"i": 0, "sel": None, "buf": "", "dirty": False}
    window = "label tracks — [s]ave [q]uit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["sel"] = store.box_at(state["i"], float(x), float(y))

    cv2.setMouseCallback(window, on_mouse)

    while True:
        i = state["i"]
        img = get_image(i + 1)
        canvas = img.copy() if img is not None else _blank()
        for idx, box in enumerate(store.frames[i]):
            x, y, w, h = (int(v) for v in box.xywh)
            selected = idx == state["sel"]
            color = (255, 255, 255) if selected else _color(box.gid)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3 if selected else 2)
            cv2.putText(
                canvas,
                str(box.gid),
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        _hud(cv2, canvas, i, n, store, state, out_path)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key in (ord("q"), 27):
            if not state["dirty"] or _confirm_quit(cv2, window, canvas):
                break
        elif key == ord("s"):
            store.save(out_path)
            state["dirty"] = False
        elif key == ord("d"):
            state["i"] = min(i + 1, n - 1)
            state["sel"] = None
        elif key == ord("a"):
            state["i"] = max(i - 1, 0)
            state["sel"] = None
        elif key == ord("."):
            state["i"] = min(i + 10, n - 1)
            state["sel"] = None
        elif key == ord(","):
            state["i"] = max(i - 10, 0)
            state["sel"] = None
        elif ord("0") <= key <= ord("9"):
            state["buf"] += chr(key)
        elif key == 8:  # backspace
            state["buf"] = state["buf"][:-1]
        elif key in (13, 10):  # enter -> relabel whole track
            _apply(store, state, whole_track=True)
        elif key == ord("x"):  # override single box
            _apply(store, state, whole_track=False)
        elif key == ord("f") and state["sel"] is not None:
            store.relabel_track(i, state["sel"], store.next_free_id())
            state["dirty"] = True
        elif key == ord("u"):
            state["dirty"] = store.undo() or state["dirty"]

    cap.release()
    cv2.destroyAllWindows()


def _apply(store: LabelStore, state: dict, whole_track: bool) -> None:
    if state["sel"] is None or not state["buf"]:
        return
    new_id = int(state["buf"])
    if whole_track:
        store.relabel_track(state["i"], state["sel"], new_id)
    else:
        store.override_box(state["i"], state["sel"], new_id)
    state["buf"] = ""
    state["dirty"] = True


def _hud(cv2, canvas, i, n, store, state, out_path) -> None:
    sel = state["sel"]
    sel_txt = f"id={store.frames[i][sel].gid}" if sel is not None else "none"
    dirty = "*" if state["dirty"] else ""
    line1 = f"frame {i + 1}/{n}  selected: {sel_txt}  typing: [{state['buf']}]  {dirty}"
    line2 = "click box; digits+Enter=relabel track; x=this box only; f=new id; u=undo; s=save"
    for k, text in enumerate((line1, line2)):
        y = 22 + k * 26
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)


def _confirm_quit(cv2, window, canvas) -> bool:
    overlay = canvas.copy()
    cv2.putText(
        overlay,
        "Unsaved changes. Press y to quit, any key to stay.",
        (10, canvas.shape[0] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.imshow(window, overlay)
    return (cv2.waitKey(0) & 0xFF) == ord("y")


def _blank():
    import numpy as np

    return np.zeros((480, 854, 3), dtype="uint8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video")
    parser.add_argument("--boxes", default=None, help="bootstrap MOT file (pred)")
    parser.add_argument(
        "--weights", default="hoopvision_best.pt", help="if --boxes absent, run this to bootstrap"
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--output", default=None, help="GT path (default data/labels/mot/gt/<clip>.txt)"
    )
    args = parser.parse_args()

    stem = Path(args.video).stem
    out_path = Path(args.output) if args.output else Path(f"data/labels/mot/gt/{stem}.txt")

    if out_path.exists():
        store = LabelStore.load_mot_file(out_path)
        index_map = {o: (o - 1) * args.stride for o in range(1, len(store.frames) + 1)}
        print(f"Resuming from existing labels: {out_path}")
    elif args.boxes:
        store = LabelStore.load_mot_file(Path(args.boxes))
        index_map = {o: (o - 1) * args.stride for o in range(1, len(store.frames) + 1)}
    else:
        print("No --boxes given; running detection+tracking to bootstrap...")
        store, index_map = store_from_weights(args.video, args.weights, args.stride, args.conf)

    run_ui(store, args.video, index_map, out_path)


if __name__ == "__main__":
    main()
