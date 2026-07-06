"""Train the from-scratch detector on the basketball dataset (player class).

Runs on MPS (MacBook), CUDA (Colab/Kaggle free GPU), or CPU:
    uv run python -m scratch_detector.train --data data/<dataset>/data.yaml \
        --epochs 40 --batch 16

Writes checkpoints and a loss log (for training curves) to scratch_detector/runs/.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hoopvision.detect import default_device  # noqa: E402

from .data import PlayerDataset  # noqa: E402
from .loss import detection_loss  # noqa: E402
from .model import CenterNetLite  # noqa: E402


def run_epoch(model, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["image"].to(device)
            targets = {k: batch[k].to(device) for k in ("heatmap", "wh", "offset", "mask")}
            losses = detection_loss(model(images), targets)
            if training:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
            total += float(losses["total"]) * len(images)
    return total / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dataset data.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default="scratch_detector/runs")
    args = parser.parse_args()

    device = default_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = PlayerDataset(args.data, "train", args.img_size)
    val_set = PlayerDataset(args.data, "val", args.img_size)
    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=args.workers)
    print(f"device={device}  train={len(train_set)}  val={len(val_set)}")

    model = CenterNetLite().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    log_path = out_dir / "losses.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss = run_epoch(model, train_loader, device, optimizer)
            val_loss = run_epoch(model, val_loader, device)
            scheduler.step()
            seconds = time.perf_counter() - started
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.4f}",
                    f"{val_loss:.4f}",
                    f"{scheduler.get_last_lr()[0]:.2e}",
                    f"{seconds:.1f}",
                ]
            )
            f.flush()
            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), out_dir / "best.pt")
                marker = "  ← best"
            print(
                f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}"
                f"  {seconds:.0f}s{marker}"
            )

    torch.save(model.state_dict(), out_dir / "last.pt")
    print(f"Done. best.pt / last.pt / losses.csv in {out_dir}")


if __name__ == "__main__":
    main()
