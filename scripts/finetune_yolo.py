"""Fine-tune YOLO on the basketball dataset (player / ball / rim).

Designed to run identically on the MacBook (MPS, small runs) and on free
Colab/Kaggle GPUs (real runs). $0 setup:

Colab (free T4):
    !git clone https://github.com/<you>/hoop-vision && cd hoop-vision
    !pip install ultralytics
    # upload or download the dataset (scripts/download_data.py), then:
    !python scripts/finetune_yolo.py --data data/<dataset>/data.yaml --epochs 60

Kaggle (free P100, 30 h/week): same commands in a notebook cell.

After training, copy the best checkpoint back to the repo root as
`hoopvision_best.pt` and run the pipeline with `--weights hoopvision_best.pt`.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="path to dataset data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="base checkpoint")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960, help="ball/rim are small")
    parser.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    parser.add_argument("--name", default="hoopvision")
    parser.add_argument("--device", default=None,
                        help="cuda/mps/cpu; default: auto (MPS on Apple Silicon)")
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.device is None:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from hoopvision.detect import default_device

        args.device = default_device()
    if args.device == "mps" and args.batch == -1:
        args.batch = 8  # autobatch is CUDA-only

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=15,
    )

    metrics = model.val(data=args.data)
    print("\n=== Validation metrics (copy into README results table) ===")
    print(f"mAP50:    {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    names = metrics.names
    for i, idx in enumerate(metrics.box.ap_class_index):
        print(
            f"  {names[int(idx)]:>8}: AP50={metrics.box.ap50[i]:.3f} "
            f"AP50-95={metrics.box.ap[i]:.3f}"
        )
    print(f"\nBest weights: runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
