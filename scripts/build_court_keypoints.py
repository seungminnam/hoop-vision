"""Build a court-keypoint dataset from calibrated static clips (v2, §4.1).

A static clip plus its homography JSON is a free keypoint annotator: project
the fixed `court.COURT_KEYPOINTS` schema into every sampled frame, then augment
each with random homography warps (virtual pans/zooms), optional horizontal
flips, and color jitter. The result is a COCO-keypoints dataset that a v2
registration model trains on — no hand labeling.

The image bytes live under a gitignored output dir (regenerable, and we never
commit raw broadcast frames); commit only the script, a spot-check overlay, and
the printed summary numbers.

Usage:
    uv run python scripts/build_court_keypoints.py \
        --source data/clips/hudl_static2.mp4 calib_hudl_static2.json \
        --output data/court_kpts --stride 15 --augment 4 \
        --overlay docs/court_keypoints_sample.jpg

`--source CLIP CALIB` is repeatable to pool several clips into one dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hoopvision.court import KEYPOINT_NAMES, CourtCalibration  # noqa: E402
from hoopvision.ingest import frames  # noqa: E402
from hoopvision.keypoints import (  # noqa: E402
    NUM_KEYPOINTS,
    color_jitter,
    flip_keypoints,
    project_keypoints,
    random_homography,
    warp_keypoints,
)

MIN_VISIBLE = 4  # a sample with <4 landmarks cannot constrain a homography


def _bbox(keypoints: np.ndarray) -> list[float]:
    """COCO bbox [x, y, w, h] around the visible keypoints (empty → zeros)."""
    vis = keypoints[keypoints[:, 2] > 0, :2]
    if len(vis) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0 = vis.min(axis=0)
    x1, y1 = vis.max(axis=0)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def _record(keypoints: np.ndarray, image_id: int, ann_id: int) -> dict:
    flat = [round(float(v), 2) for row in keypoints for v in row]
    bx, by, bw, bh = _bbox(keypoints)
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "keypoints": flat,
        "num_keypoints": int((keypoints[:, 2] > 0).sum()),
        "bbox": [bx, by, bw, bh],
        "area": bw * bh,
        "iscrowd": 0,
    }


def draw_overlay(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Annotate a frame with the projected keypoints for visual spot-check."""
    out = frame.copy()
    for i, (x, y, v) in enumerate(keypoints):
        if v <= 0:
            continue
        p = (int(round(x)), int(round(y)))
        cv2.circle(out, p, 5, (0, 255, 0), -1)
        cv2.circle(out, p, 6, (0, 0, 0), 1)
        cv2.putText(
            out, str(i), (p[0] + 7, p[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def build(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    images: list[dict] = []
    annotations: list[dict] = []
    per_source: dict[str, int] = {}
    overlay_saved = False

    for clip_path, calib_path in args.source:
        calib = CourtCalibration.load(calib_path)
        source = Path(clip_path).name
        per_source[source] = 0
        for _, frame in frames(clip_path, stride=args.stride, max_frames=args.max_frames):
            h, w = frame.shape[:2]
            base_kp = project_keypoints(calib, w, h)
            if int((base_kp[:, 2] > 0).sum()) < MIN_VISIBLE:
                continue

            if args.overlay and not overlay_saved:
                cv2.imwrite(args.overlay, draw_overlay(frame, base_kp))
                overlay_saved = True

            # one clean sample + N augmented virtual-camera views
            variants: list[tuple[np.ndarray, np.ndarray]] = [(frame, base_kp)]
            for _ in range(args.augment):
                warp = random_homography(w, h, rng, jitter=args.jitter)
                aug_img = cv2.warpPerspective(frame, warp, (w, h))
                aug_kp = warp_keypoints(base_kp, warp, w, h)
                if rng.random() < 0.5:
                    aug_img = cv2.flip(aug_img, 1)
                    aug_kp = flip_keypoints(aug_kp, w)
                aug_img = color_jitter(aug_img, rng)
                variants.append((aug_img, aug_kp))

            for img, kp in variants:
                if int((kp[:, 2] > 0).sum()) < MIN_VISIBLE:
                    continue
                image_id = len(images)
                name = f"{image_id:06d}.jpg"
                cv2.imwrite(str(img_dir / name), img)
                images.append(
                    {
                        "id": image_id,
                        "file_name": f"images/{name}",
                        "width": w,
                        "height": h,
                        "source": source,
                    }
                )
                annotations.append(_record(kp, image_id, len(annotations)))
                per_source[source] += 1

    coco = {
        "info": {"description": "Hoop Vision court keypoints (pseudo-labeled)"},
        "categories": [
            {
                "id": 1,
                "name": "court",
                "keypoints": KEYPOINT_NAMES,
                "skeleton": [],
            }
        ],
        "images": images,
        "annotations": annotations,
    }
    (out_dir / "annotations.json").write_text(json.dumps(coco))

    visible = [a["num_keypoints"] for a in annotations]
    return {
        "images": len(images),
        "annotations": len(annotations),
        "keypoints_per_image_mean": round(float(np.mean(visible)), 2) if visible else 0.0,
        "keypoints_total": NUM_KEYPOINTS,
        "per_source": per_source,
        "overlay": args.overlay if overlay_saved else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        nargs=2,
        action="append",
        metavar=("CLIP", "CALIB"),
        required=True,
        help="clip video + its calibration JSON (repeatable)",
    )
    parser.add_argument("--output", default="data/court_kpts", help="dataset dir")
    parser.add_argument("--stride", type=int, default=15, help="keep every n-th frame")
    parser.add_argument("--max-frames", type=int, default=None, help="cap frames per clip")
    parser.add_argument("--augment", type=int, default=4, help="augmented views per frame")
    parser.add_argument("--jitter", type=float, default=0.12, help="warp corner jitter fraction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlay", default=None, help="write a spot-check overlay JPEG")
    args = parser.parse_args()

    summary = build(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
