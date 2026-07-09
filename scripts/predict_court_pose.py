"""Run the trained court-keypoint model on an image and draw the predictions.

Qualitative check for v2 §4.2 Phase 1: does the YOLO11-pose model place the
33 court landmarks correctly on a real (broadcast) frame? Draws each predicted
keypoint with its schema index and confidence.

    uv run python scripts/predict_court_pose.py FRAME.jpg \
        --weights runs/pose/court_pose/weights/best.pt --out overlay.jpg
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--weights", default="runs/pose/court_pose/weights/best.pt")
    parser.add_argument("--out", default="court_pose_pred.jpg")
    parser.add_argument("--conf", type=float, default=0.25, help="keypoint confidence to draw")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    result = model(args.image, verbose=False)[0]
    img = cv2.imread(args.image)

    if result.keypoints is None or len(result.keypoints) == 0:
        print("no court detected")
        cv2.imwrite(args.out, img)
        return

    # highest-confidence court instance
    best = int(np.argmax(result.boxes.conf.cpu().numpy())) if result.boxes is not None else 0
    xy = result.keypoints.xy.cpu().numpy()[best]
    conf = result.keypoints.conf.cpu().numpy()[best] if result.keypoints.conf is not None else None

    drawn = 0
    for i, (x, y) in enumerate(xy):
        c = float(conf[i]) if conf is not None else 1.0
        if (x == 0 and y == 0) or c < args.conf:
            continue
        p = (int(round(x)), int(round(y)))
        cv2.circle(img, p, 5, (0, 255, 0), -1)
        cv2.circle(img, p, 6, (0, 0, 0), 1)
        cv2.putText(
            img,
            str(i),
            (p[0] + 6, p[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        drawn += 1

    cv2.imwrite(args.out, img)
    print(f"drew {drawn} keypoints (conf>={args.conf}) -> {args.out}")


if __name__ == "__main__":
    main()
