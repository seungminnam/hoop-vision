"""End-to-end court-registration accuracy on the held-out NBA test split.

Runs the Phase-1 pose detector on each test image, fits an image->feet
homography from the *predicted* planar keypoints, then scores it on the
*ground-truth* keypoint pixels (independent of the points that fit the
homography): how many feet off is a true court point localized to?

This isolates the registration step from raw detection quality (Phase-1 keypoint
mAP is 0.985). Reproduces the number in the README v2 section.

    uv run python scripts/eval_registration.py \
        --weights runs/pose/court_pose/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hoopvision.court_template import ELEVATED_KEYPOINTS, NUM_KEYPOINTS, template_array
from hoopvision.registration import fit_homography, image_to_court

ROOT = Path(__file__).resolve().parents[1]


def _gt_keypoints(coco_root: Path, split: str) -> dict[str, np.ndarray]:
    d = json.loads((coco_root / split / "_annotations.coco.json").read_text())
    cat = next(c for c in d["categories"] if c.get("keypoints"))
    images = {im["id"]: im for im in d["images"]}
    out = {}
    for a in d["annotations"]:
        if a["category_id"] == cat["id"]:
            out[images[a["image_id"]]["file_name"]] = np.array(a["keypoints"], float).reshape(-1, 3)
    return out


def evaluate(weights: Path, coco_root: Path, split: str, conf: float) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    gt = _gt_keypoints(coco_root, split)
    img_dir = coco_root / split

    errs: list[float] = []
    n_registered = 0
    n_total = 0
    for fname, kp_gt in gt.items():
        n_total += 1
        res = model.predict(str(img_dir / fname), verbose=False, conf=0.25)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            continue
        # highest-confidence court instance
        confs = res.keypoints.conf.cpu().numpy()  # (N, 33)
        best = int(confs.sum(axis=1).argmax())
        xy = res.keypoints.xy.cpu().numpy()[best]  # (33, 2)
        cf = confs[best]

        pred = {
            i: (float(xy[i, 0]), float(xy[i, 1]))
            for i in range(NUM_KEYPOINTS)
            if cf[i] >= conf and i not in ELEVATED_KEYPOINTS
        }
        fit = fit_homography(pred)
        if fit is None:
            continue
        n_registered += 1
        H, _ = fit

        # score on GT pixels of visible planar points (not the fitting set)
        gt_idx = [
            i for i in range(NUM_KEYPOINTS) if kp_gt[i, 2] > 0 and i not in ELEVATED_KEYPOINTS
        ]
        if not gt_idx:
            continue
        gt_px = np.array([kp_gt[i, :2] for i in gt_idx], float)
        feet_pred = image_to_court(H, gt_px)
        feet_true = template_array(gt_idx)
        errs.extend(np.linalg.norm(feet_pred - feet_true, axis=1).tolist())

    a = np.array(errs)
    return {
        "split": split,
        "images": n_total,
        "registered": n_registered,
        "registration_rate": round(n_registered / max(n_total, 1), 3),
        "point_observations": int(a.size),
        "median_ft": round(float(np.median(a)), 3),
        "mean_ft": round(float(a.mean()), 3),
        "p90_ft": round(float(np.percentile(a, 90)), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default=str(ROOT / "runs/pose/court_pose/weights/best.pt"))
    p.add_argument("--coco-root", default=str(ROOT / "data/basketball-court-detection-2-13"))
    p.add_argument("--split", default="test")
    p.add_argument("--conf", type=float, default=0.5, help="min keypoint confidence")
    args = p.parse_args()
    print(
        json.dumps(
            evaluate(Path(args.weights), Path(args.coco_root), args.split, args.conf), indent=2
        )
    )


if __name__ == "__main__":
    main()
