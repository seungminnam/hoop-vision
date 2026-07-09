"""Recover the 33-point court template geometry from the labeled dataset.

The dataset ships no real-world coordinates for its 33 court keypoints. But
every labeled frame is a homography of the *same* planar court, so points
shared between frames let us chain homographies and place all 33 schema points
into one common reference frame — no manual guessing. Reprojection error then
tells us whether the recovered layout is globally consistent (a planar court).

This is v2 §4.2 Phase 2 groundwork. The recovered template is in the reference
image's (perspective) frame; a later step anchors it to real feet via a few
identified correspondences and NBA court dimensions.

    uv run python scripts/recover_court_template.py \
        --coco-root data/basketball-court-detection-2-13 --out docs/court_template_recovered.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MIN_VIS = 6  # ignore frames with too few points to be a useful constraint
MIN_SHARED = 4  # a homography needs >= 4 correspondences


def load_observations(
    coco_root: Path,
) -> tuple[list[dict[int, np.ndarray]], int, tuple[Path, dict]]:
    obs: list[dict[int, np.ndarray]] = []
    num_kpts = 0
    ref_meta = None  # (image_path, {idx: xy}) of the frame with the most points
    best_vis = -1
    for split in ("train", "valid", "test"):
        f = coco_root / split / "_annotations.coco.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        cat = next(c for c in d["categories"] if c.get("keypoints"))
        num_kpts = len(cat["keypoints"])
        images = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            if a["category_id"] != cat["id"]:
                continue
            kp = np.array(a["keypoints"]).reshape(-1, 3)
            pts = {i: np.array([x, y], float) for i, (x, y, v) in enumerate(kp) if v > 0}
            if len(pts) < MIN_VIS:
                continue
            obs.append(pts)
            if len(pts) > best_vis:
                best_vis = len(pts)
                ref_meta = (coco_root / split / images[a["image_id"]]["file_name"], pts)
    return obs, num_kpts, ref_meta


def _homography(pts: dict[int, np.ndarray], template: dict[int, np.ndarray]):
    shared = [i for i in pts if i in template]
    if len(shared) < MIN_SHARED:
        return None, shared
    src = np.array([pts[i] for i in shared])
    dst = np.array([template[i] for i in shared])
    h, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return h, shared


def recover(obs, ref_pts, passes: int = 4) -> dict[int, np.ndarray]:
    template = {i: xy.copy() for i, xy in ref_pts.items()}
    est: dict[int, list[np.ndarray]] = {i: [xy.copy()] for i, xy in ref_pts.items()}
    for _ in range(passes):
        for pts in obs:
            h, shared = _homography(pts, template)
            if h is None:
                continue
            src = np.array([pts[i] for i in pts])
            proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h).reshape(-1, 2)
            for (i, _), p in zip(pts.items(), proj, strict=True):
                est.setdefault(i, []).append(p)
        template = {i: np.median(np.array(v), axis=0) for i, v in est.items()}
    return template


def reprojection_errors(obs, template) -> np.ndarray:
    errs = []
    for pts in obs:
        h, shared = _homography(pts, template)
        if h is None:
            continue
        src = np.array([pts[i] for i in shared])
        dst = np.array([template[i] for i in shared])
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h).reshape(-1, 2)
        errs.extend(np.linalg.norm(proj - dst, axis=1))
    return np.array(errs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", default="data/basketball-court-detection-2-13")
    parser.add_argument("--out", default="docs/court_template_recovered.jpg")
    parser.add_argument("--json-out", default="data/court_template_recovered.json")
    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    obs, num_kpts, ref_meta = load_observations(coco_root)
    ref_path, ref_pts = ref_meta
    print(f"{len(obs)} usable frames, {num_kpts} schema points; reference has {len(ref_pts)}")

    template = recover(obs, ref_pts)
    errs = reprojection_errors(obs, template)
    print(f"recovered {len(template)}/{num_kpts} points")
    print(
        f"reprojection error (px in ref frame): "
        f"mean {errs.mean():.2f}  median {np.median(errs):.2f}  p90 {np.percentile(errs, 90):.2f}"
    )

    # Overlay every recovered point on the reference image (incl. ones occluded
    # there) — a visual sanity check that the layout is a coherent court.
    img = cv2.imread(str(ref_path))
    for i, xy in sorted(template.items()):
        p = (int(round(xy[0])), int(round(xy[1])))
        seen = i in ref_pts
        cv2.circle(img, p, 5, (0, 255, 0) if seen else (0, 165, 255), -1)
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
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, img)
    Path(args.json_out).write_text(
        json.dumps({str(i): xy.tolist() for i, xy in template.items()}, indent=2)
    )
    print(f"saved overlay -> {args.out}  (green=visible in ref, orange=recovered)")
    print(f"saved template -> {args.json_out}")


if __name__ == "__main__":
    main()
