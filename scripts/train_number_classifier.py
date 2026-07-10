"""Fine-tune resnet18 to read jersey numbers from crops — task D (D-3).

Trains on `basketball-jersey-numbers-ocr` (3,188 NBA-broadcast crops, 224x224,
ADR-008): a 40-way closed-set classification over the numbers that appear in
the data. Closed-set is enough for our purpose (merging tracks that share a
number); open-set digit decomposition would be over-engineering.

Crops with an empty label (unreadable, 52 across splits) are excluded from
training. This script also measures, as an honesty check, how a softmax
threshold behaves on them: a plain resnet18 has no "none" class, so it labels
unreadable crops confidently anyway (0/52 fall below 0.5). A threshold does
help suppress *wrong* reads (correct crops sit at p10~0.95, wrong reads at
median ~0.80), but it cannot reject unreadable ones -- so D-4 must gate
"unknown" via temporal voting, not per-crop confidence alone.

The dataset authors report ResNet-32 at 93% on this data (their blog); their
split differs from this export, so the numbers are not directly comparable.

    uv run python scripts/train_number_classifier.py --epochs 30 --device mps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/basketball-jersey-numbers-ocr-3"

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose(
    [
        transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)
EVAL_TF = transforms.Compose([transforms.ToTensor(), NORMALIZE])


def read_split(dataset_dir: Path, split: str) -> tuple[list[tuple[Path, str]], list[Path]]:
    """Return (labeled crop paths, unreadable crop paths) for a split."""
    split_dir = dataset_dir / split
    labeled: list[tuple[Path, str]] = []
    unreadable: list[Path] = []
    with open(split_dir / "annotations.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            img = split_dir / row["image"]
            if row["suffix"]:
                labeled.append((img, row["suffix"]))
            else:
                unreadable.append(img)
    return labeled, unreadable


class CropDataset(Dataset):
    def __init__(self, items: list[tuple[Path, str]], classes: list[str], tf) -> None:
        self.items = items
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.tf = tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), self.class_to_idx[label]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += int((pred == y).sum())
        total += len(y)
    return correct / total


@torch.no_grad()
def max_probs(model: nn.Module, paths: list[Path], device: str) -> list[float]:
    model.eval()
    out = []
    for path in paths:
        x = EVAL_TF(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        out.append(float(model(x).softmax(1).max()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(DEFAULT_DATA))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="mps")
    p.add_argument("--threshold", type=float, default=0.5, help="unknown cutoff (D-4)")
    p.add_argument("--out", default=str(ROOT / "runs/classify/number_classifier"))
    args = p.parse_args()

    dataset_dir = Path(args.dataset)
    train_items, train_empty = read_split(dataset_dir, "train")
    valid_items, valid_empty = read_split(dataset_dir, "valid")
    test_items, test_empty = read_split(dataset_dir, "test")
    classes = sorted({label for _, label in train_items}, key=lambda s: (len(s), s))
    print(
        f"classes: {len(classes)}  train {len(train_items)}"
        f"  valid {len(valid_items)}  test {len(test_items)}"
    )

    loaders = {
        "train": DataLoader(
            CropDataset(train_items, classes, TRAIN_TF),
            batch_size=args.batch,
            shuffle=True,
            num_workers=2,
        ),
        "valid": DataLoader(
            CropDataset(valid_items, classes, EVAL_TF), batch_size=args.batch, num_workers=2
        ),
        "test": DataLoader(
            CropDataset(test_items, classes, EVAL_TF), batch_size=args.batch, num_workers=2
        ),
    }

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in loaders["train"]:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(args.device)), y.to(args.device))
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(y)
        scheduler.step()
        val_acc = evaluate(model, loaders["valid"], args.device)
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = "  *"
        avg_loss = running / len(train_items)
        print(f"epoch {epoch:3d}  loss {avg_loss:.4f}  val acc {val_acc:.4f}{marker}")

    assert best_state is not None
    model.load_state_dict(best_state)
    test_acc = evaluate(model, loaders["test"], args.device)

    # Honesty check: unreadable crops (never trained on) get labeled confidently
    # anyway -- softmax alone cannot reject them (see the module docstring).
    empties = train_empty + valid_empty + test_empty
    empty_probs = max_probs(model, empties, args.device)
    below = sum(prob < args.threshold for prob in empty_probs)
    test_probs = max_probs(model, [path for path, _ in test_items], args.device)
    covered = sum(prob >= args.threshold for prob in test_probs)

    print("\n=== held-out metrics ===")
    print(f"valid acc {best_acc:.4f}  test acc {test_acc:.4f}  ({len(test_items)} crops)")
    print(
        f"threshold {args.threshold}: test coverage {covered}/{len(test_items)}"
        f" ({covered / len(test_items):.1%}),"
        f" unreadable below threshold {below}/{len(empties)} ({below / len(empties):.1%})"
    )

    torch.save(
        {
            "state_dict": best_state,
            "classes": classes,
            "arch": "resnet18",
            "val_acc": best_acc,
            "test_acc": test_acc,
        },
        out_dir / "best.pt",
    )
    print(f"saved {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
