"""Fine-tune YOLO11-pose to detect the 33-point NBA court keypoint schema.

Trains on the Roboflow basketball court-keypoint dataset (converted to
YOLO-pose by scripts/convert_court_coco_to_yolo_pose.py). This is v2 §4.2
Phase 1: a detector that places court landmarks on real NBA broadcast frames.
Phase 2 (separate) reverse-engineers the 33-point real-world template and adds
the keypoint -> homography step.

`fliplr` is forced to 0: the 33-point schema has no known left-right mirror
map, so a horizontal flip would swap point identities and corrupt the labels.

    uv run python scripts/train_court_pose.py --epochs 100 --device mps
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/court_pose/court_pose.yaml"))
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--name", default="court_pose")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        fliplr=0.0,  # no known L/R keypoint mirror — see module docstring
        project=str(ROOT / "runs/pose"),
        name=args.name,
        exist_ok=True,
    )
    metrics = model.val(split="test")
    print("test metrics:", metrics.results_dict)


if __name__ == "__main__":
    main()
