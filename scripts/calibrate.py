"""Manual 4-point court calibration tool.

Opens a reference frame in an OpenCV window and asks you to click each named
landmark in order. Press `u` to undo the last click, `q` to abort. When all
landmarks are clicked the homography is computed, the reprojection error is
printed, and the calibration is saved as JSON for the pipeline.

Usage:
    uv run python scripts/calibrate.py clip.mp4 --output calib.json \
        --landmarks baseline-left-corner,baseline-right-corner,ft-line-left,ft-line-right

Custom points (e.g. a court with visible halfcourt corners) can mix names from
hoopvision.court.LANDMARKS with raw court coordinates as `x,y` in feet:
    --landmarks baseline-left-corner,baseline-right-corner,25;19,0;47
(`;` separates x;y inside one landmark so `,` can separate landmarks.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hoopvision.court import LANDMARKS, CourtCalibration  # noqa: E402
from hoopvision.ingest import frames  # noqa: E402

DEFAULT_LANDMARKS = "baseline-left-corner,baseline-right-corner,ft-line-left,ft-line-right"


def parse_landmarks(spec: str) -> list[tuple[str, tuple[float, float]]]:
    out = []
    for token in spec.split(","):
        token = token.strip()
        if token in LANDMARKS:
            out.append((token, LANDMARKS[token]))
        elif ";" in token:
            x, y = token.split(";")
            out.append((f"({x},{y}) ft", (float(x), float(y))))
        else:
            known = ", ".join(LANDMARKS)
            raise SystemExit(f"Unknown landmark '{token}'. Known: {known}")
    if len(out) < 4:
        raise SystemExit("Need at least 4 landmarks for a homography.")
    return out


def grab_frame(video: str, frame_index: int):
    for index, frame in frames(video):
        if index >= frame_index:
            return frame
    raise SystemExit(f"Video has fewer than {frame_index} frames.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--frame", type=int, default=0, help="reference frame index")
    parser.add_argument("--output", default="calib.json")
    parser.add_argument("--landmarks", default=DEFAULT_LANDMARKS)
    args = parser.parse_args()

    landmarks = parse_landmarks(args.landmarks)
    frame = grab_frame(args.video, args.frame)
    clicks: list[tuple[float, float]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(landmarks):
            clicks.append((float(x), float(y)))

    window = "calibrate — click the highlighted landmark"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = frame.copy()
        for (x, y), (name, _) in zip(clicks, landmarks, strict=False):
            cv2.circle(canvas, (int(x), int(y)), 6, (0, 220, 255), -1)
            cv2.putText(
                canvas,
                name,
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                2,
            )
        if len(clicks) < len(landmarks):
            name, court_xy = landmarks[len(clicks)]
            msg = f"Click: {name}  (court {court_xy} ft)   [u]ndo  [q]uit"
            cv2.putText(canvas, msg, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            raise SystemExit("Aborted.")
        if key == ord("u") and clicks:
            clicks.pop()
        if len(clicks) == len(landmarks):
            break
    cv2.destroyAllWindows()

    calib = CourtCalibration.from_points(clicks, [xy for _, xy in landmarks])
    error = calib.reprojection_error_ft()
    calib.save(args.output)
    print(
        f"Saved {args.output} — reprojection error {error:.3f} ft "
        f"({'OK' if error < 1.0 else 'WARNING: > 1 ft, re-click more carefully'})"
    )


if __name__ == "__main__":
    main()
