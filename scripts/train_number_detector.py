"""Fine-tune YOLO11n to detect jersey `number` boxes — task D (D-2).

Trains on `basketball-player-detection-3-ycjdo` (10-class detection, ADR-008).
We consume only the `number` class downstream (D-4), but train on all classes
the dataset labels — `referee` is a useful bonus (v1 lacks it) and the rest cost
nothing. The action classes are too rare to learn (see ADR-008); their AP will
be low and that is expected.

**Resolution matters.** Number boxes are ~12-17 px at native 1280x720 (~6 px at
640), so we train at `imgsz=1280` (the dataset's own image size) — the court
detector's 640 would erase them. Inference must likewise run near native res.

The Roboflow YOLOv11 export ships a `data.yaml` with `../train/images` paths and
no `path:` key, which ultralytics resolves wrong; we rewrite a corrected yaml
with an absolute `path:` next to it (idempotent) so this is reproducible.

    uv run python scripts/train_number_detector.py --epochs 100 --device mps
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/basketball-player-detection-3-ycjdo-1"


def corrected_yaml(dataset_dir: Path) -> Path:
    """Write a data.yaml with an absolute `path:` (Roboflow's relative one is wrong)."""
    names = [
        "ball",
        "ball-in-basket",
        "number",
        "player",
        "player-in-possession",
        "player-jump-shot",
        "player-layup-dunk",
        "player-shot-block",
        "referee",
        "rim",
    ]
    out = dataset_dir / "data_fixed.yaml"
    out.write_text(
        f"path: {dataset_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(DEFAULT_DATA))
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280, help="number boxes are tiny")
    p.add_argument("--batch", type=int, default=4, help="imgsz=1280 is memory-heavy on MPS")
    p.add_argument("--device", default="mps")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--name", default="number_detector")
    args = p.parse_args()

    data_yaml = corrected_yaml(Path(args.dataset))

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(ROOT / "runs/detect"),
        name=args.name,
        exist_ok=True,
    )

    metrics = model.val(data=str(data_yaml), split="test", imgsz=args.imgsz)
    names = metrics.names
    print("\n=== test metrics ===")
    print(f"overall mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}")
    for i, idx in enumerate(metrics.box.ap_class_index):
        tag = "  <- number" if names[int(idx)] == "number" else ""
        print(f"  {names[int(idx)]:>20}: AP50={metrics.box.ap50[i]:.3f}{tag}")


if __name__ == "__main__":
    main()
