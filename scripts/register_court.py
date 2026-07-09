"""Per-frame NBA court registration demo on a panning broadcast clip (v2 §4.2).

Runs the Phase-1 pose detector each frame, fits + temporally smooths an
image->feet homography (`CourtRegistrar`), and renders:

  * the NBA court model **reprojected onto the broadcast** (lines should hug the
    real court as the camera pans) — the visible proof of registration;
  * a top-down **minimap** where detected players (v1 detector) are placed by
    mapping each player's foot point through the homography.

Frames are resized to 640x640 to match the detector's training (the dataset was
stretched to 640x640); the homography absorbs the anisotropic scale.

    uv run python scripts/register_court.py --video data/clips/_nba_raw.webm \
        --out docs/court_registration_nba.mp4 --gif docs/court_registration_nba.gif \
        --players --seconds 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hoopvision.court_template import COURT_LENGTH_FT, COURT_WIDTH_FT, NUM_KEYPOINTS
from hoopvision.registration import CourtRegistrar, court_polylines_ft, image_to_court

ROOT = Path(__file__).resolve().parents[1]
SIZE = 640  # detector input (matches the 640x640-stretched training data)


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


def _draw_reprojection(frame640, H_img2feet, color=(0, 235, 255)) -> None:
    H_feet2img = np.linalg.inv(H_img2feet)
    for poly_ft in court_polylines_ft():
        img = cv2.perspectiveTransform(poly_ft.reshape(-1, 1, 2).astype(float), H_feet2img).reshape(
            -1, 2
        )
        cv2.polylines(frame640, [img.astype(np.int32)], False, color, 2, cv2.LINE_AA)


def _minimap(players_ft: list[tuple[float, float]], w=300) -> np.ndarray:
    h = int(w * COURT_WIDTH_FT / COURT_LENGTH_FT)
    m = np.full((h + 16, w + 16, 3), 40, np.uint8)

    def px(x, y):
        return (int(8 + x / COURT_LENGTH_FT * w), int(8 + y / COURT_WIDTH_FT * h))

    for poly_ft in court_polylines_ft():
        pts = np.array([px(x, y) for x, y in poly_ft], np.int32)
        cv2.polylines(m, [pts], False, (200, 200, 200), 1, cv2.LINE_AA)
    for x, y in players_ft:
        if -2 <= x <= 96 and -2 <= y <= 52:
            cv2.circle(m, px(x, y), 4, (0, 165, 255), -1)
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default=str(ROOT / "data/clips/_nba_raw.webm"))
    p.add_argument("--weights", default=str(ROOT / "runs/pose/court_pose/weights/best.pt"))
    p.add_argument("--player-weights", default=str(ROOT / "hoopvision_best.pt"))
    p.add_argument("--out", default=str(ROOT / "docs/court_registration_nba.mp4"))
    p.add_argument("--gif", default=None)
    p.add_argument("--players", action="store_true")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--start", type=float, default=0.0)
    args = p.parse_args()

    from ultralytics import YOLO

    pose = YOLO(args.weights)
    player_det = None
    if args.players and Path(args.player_weights).exists():
        from hoopvision.detect import YoloDetector

        player_det = YoloDetector(args.player_weights, conf=0.3)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    n_frames = int(args.seconds * fps)

    reg = CourtRegistrar(alpha=0.35, max_misses=20)
    writer = None
    registered = 0
    processed = 0

    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        processed += 1
        f640 = cv2.resize(frame, (SIZE, SIZE))
        res = pose.predict(f640, verbose=False, conf=0.25)[0]
        H = reg.update(_best_keypoints(res, args.conf))

        canvas = f640.copy()
        players_ft: list[tuple[float, float]] = []
        if H is not None:
            registered += 1
            _draw_reprojection(canvas, H)
            if player_det is not None:
                for d in player_det.detect(f640):
                    if d.class_name == "player":
                        fx, fy = d.foot
                        players_ft.append(tuple(image_to_court(H, np.array([[fx, fy]]))[0]))
                        cv2.circle(canvas, (int(fx), int(fy)), 4, (0, 165, 255), -1)
            status, scolor = "REGISTERED", (0, 220, 0)
        else:
            status, scolor = "searching...", (0, 0, 230)
        cv2.putText(canvas, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, scolor, 2, cv2.LINE_AA)

        out = cv2.resize(canvas, (1280, 720))
        mini = _minimap(players_ft)
        mh, mw = mini.shape[:2]
        out[720 - mh - 10 : 720 - 10, 1280 - mw - 10 : 1280 - 10] = mini

        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
        writer.write(out)

    cap.release()
    if writer:
        writer.release()
    rate = registered / max(processed, 1)
    print(f"processed {processed} frames, registered {registered} ({rate:.0%}) -> {args.out}")

    if args.gif:
        import subprocess

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                args.out,
                "-vf",
                "fps=10,scale=640:-1:flags=lanczos",
                "-loop",
                "0",
                args.gif,
            ],
            check=True,
            capture_output=True,
        )
        print(f"gif -> {args.gif}")


if __name__ == "__main__":
    main()
