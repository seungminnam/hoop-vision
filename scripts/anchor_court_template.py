"""Validate (and visualize) the 33-point NBA feet template — v2 §4.2 Phase 2.

The template itself lives in `hoopvision.court_template.NBA_FULLCOURT_FT`. This
script reproduces the honesty-rule evidence for it and renders the committed
court diagram:

  * `--validate` fits an image->feet homography per labeled dataset frame from
    the visible points and reports the reprojection error in feet (independent
    of how the coordinates were derived). Needs the dataset downloaded.
  * `--diagram PATH` draws the 33 points on a to-scale NBA court (no dataset
    needed) — this is the committed reference figure.

    uv run python scripts/anchor_court_template.py --validate \
        --diagram docs/court_template_nba.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hoopvision.court_template import (
    COURT_LENGTH_FT,
    COURT_WIDTH_FT,
    ELEVATED_KEYPOINTS,
    KEYPOINT_NAMES,
    NBA_FULLCOURT_FT,
    NUM_KEYPOINTS,
    template_array,
)

ROOT = Path(__file__).resolve().parents[1]
MIN_VIS = 6  # a stable homography needs a well-spread handful of points


def _frame_annotations(coco_root: Path):
    for split in ("train", "valid", "test"):
        f = coco_root / split / "_annotations.coco.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        cat = next(c for c in d["categories"] if c.get("keypoints"))
        for a in d["annotations"]:
            if a["category_id"] == cat["id"]:
                yield np.array(a["keypoints"], float).reshape(-1, 3)


def validate(coco_root: Path) -> dict:
    """Per-frame image->feet reprojection error over planar points, in feet."""
    errs: list[float] = []
    per_point: list[list[float]] = [[] for _ in range(NUM_KEYPOINTS)]
    n_frames = 0
    for kp in _frame_annotations(coco_root):
        idx = [i for i, (x, y, v) in enumerate(kp) if v > 0 and i not in ELEVATED_KEYPOINTS]
        if len(idx) < MIN_VIS:
            continue
        src = np.array([kp[i, :2] for i in idx], float)
        dst = template_array(idx)
        h, _ = cv2.findHomography(src, dst, cv2.RANSAC, 1.0)
        if h is None:
            continue
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h).reshape(-1, 2)
        e = np.linalg.norm(proj - dst, axis=1)
        errs.extend(e.tolist())
        for k, i in enumerate(idx):
            per_point[i].append(float(e[k]))
        n_frames += 1
    a = np.array(errs)
    return {
        "frames": n_frames,
        "observations": int(a.size),
        "mean_ft": float(a.mean()),
        "median_ft": float(np.median(a)),
        "p90_ft": float(np.percentile(a, 90)),
        "p99_ft": float(np.percentile(a, 99)),
        "per_point_median_ft": {
            i: (float(np.median(v)) if v else None) for i, v in enumerate(per_point)
        },
    }


def draw_diagram(path: Path, scale: int = 12, pad: int = 40) -> None:
    w = int(COURT_LENGTH_FT * scale + 2 * pad)
    h = int(COURT_WIDTH_FT * scale + 2 * pad)
    img = np.full((h, w, 3), 255, np.uint8)

    def px(x, y):
        return (int(pad + x * scale), int(pad + y * scale))

    def line(a, b):
        cv2.line(img, px(*a), px(*b), (60, 60, 60), 1, cv2.LINE_AA)

    # outer boundary + halfcourt + center circle
    line((0, 0), (94, 0))
    line((0, 50), (94, 50))
    line((0, 0), (0, 50))
    line((94, 0), (94, 50))
    line((47, 0), (47, 50))
    cv2.circle(img, px(47, 25), int(6 * scale), (60, 60, 60), 1)
    for bx, s in ((0, 1), (94, -1)):
        ftx = bx + s * 19
        line((bx, 17), (ftx, 17))
        line((bx, 33), (ftx, 33))
        line((ftx, 17), (ftx, 33))
        cv2.circle(img, px(bx + s * 19, 25), int(6 * scale), (60, 60, 60), 1)
        cv2.circle(img, px(bx + s * 5.25, 25), int(0.75 * scale), (60, 60, 60), 1)
        line((bx, 3), (bx + s * 14, 3))
        line((bx, 47), (bx + s * 14, 47))
        ang = np.linspace(-1.2, 1.2, 80)
        arc = [(bx + s * 5.25 + s * 23.75 * np.cos(t), 25 + 23.75 * np.sin(t)) for t in ang]
        for j in range(len(arc) - 1):
            line(arc[j], arc[j + 1])

    for i in range(NUM_KEYPOINTS):
        x, y = NBA_FULLCOURT_FT[i]
        p = px(x, y)
        elevated = i in ELEVATED_KEYPOINTS
        color = (0, 140, 255) if elevated else (0, 0, 220)
        cv2.circle(img, p, 5, color, -1)
        cv2.circle(img, p, 6, (255, 255, 255), 1)
        cv2.putText(
            img,
            str(i),
            (p[0] + 6, p[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 0, 0),
            2,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco-root", default=str(ROOT / "data/basketball-court-detection-2-13"))
    p.add_argument("--validate", action="store_true", help="reproject error over all frames")
    p.add_argument("--diagram", default=None, help="write the court diagram PNG here")
    args = p.parse_args()

    if args.validate:
        stats = validate(Path(args.coco_root))
        keys = ("frames", "observations", "mean_ft", "median_ft", "p90_ft", "p99_ft")
        print(json.dumps({k: stats[k] for k in keys}, indent=2))
        worst = sorted(
            ((v, i) for i, v in stats["per_point_median_ft"].items() if v is not None),
            reverse=True,
        )[:3]
        print("worst planar points (median ft):", [(i, round(v, 2)) for v, i in worst])
        print("names:", {i: KEYPOINT_NAMES[i] for _, i in worst})

    if args.diagram:
        out = Path(args.diagram)
        draw_diagram(out)
        print(f"saved diagram -> {out}")


if __name__ == "__main__":
    main()
