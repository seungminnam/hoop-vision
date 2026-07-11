"""Fine-tune resnet18 to read jersey numbers from crops — task D (D-3) / G (G-1).

Trains on `basketball-jersey-numbers-ocr` (3,188 NBA-broadcast crops, 224x224,
ADR-008): a closed-set read over the numbers that appear in the data.

**Task G adds two levers against the read-*precision* wall** (D-4/E-1 found that
the classifier collapses small in-game numbers onto "22"):

1. **Degradation augmentation** (`--degrade`, default on) mirrors what a live
   crop suffers: the number box is ~12-17 px at native res, padded and upscaled
   to 224, so real inputs are heavily downscaled + motion-blurred. Training crops
   get the same treatment so the model sees its deployment distribution, not just
   curated close crops.
2. **An "unreadable" abstain class** (`--abstain`, default on) lets the model say
   "I can't read this" instead of guessing a number. Seeds: the 52 empty-label
   crops in the OCR set; the rest are mined for free from
   `basketball-player-detection-3-ycjdo` — torso patches of players whose number
   box was not detected (a numberless view), no manual labelling. D-4's runner
   drops any read classified `unreadable` (it abstains from voting).

`--no-degrade --no-abstain` reproduces the original D-3 model. `--eval-only
--weights PATH` scores an existing checkpoint on the same 3-tier eval (clean /
degraded number accuracy + abstain recall) for an honest old-vs-new comparison.

    uv run python scripts/train_number_classifier.py --epochs 30 --device mps
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/basketball-jersey-numbers-ocr-3"
DEFAULT_DET = ROOT / "data/basketball-player-detection-3-ycjdo-1"
ABSTAIN = "unreadable"
NUMBER_CLS, PLAYER_CLS = 2, 3  # player-detection class ids (data_fixed.yaml)

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# --- degradation: mirror the live crop pipeline (tiny box upscaled + blurred) ---


def _motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return cv2.filter2D(img, -1, kernel)


def degrade_image(
    pil: Image.Image, rng: np.random.Generator, min_px: int, max_px: int, blur_max: int
) -> Image.Image:
    """Downscale to a tiny size then back to 224 (as the live crop is), + motion blur."""
    img = np.asarray(pil)
    target = int(rng.integers(min_px, max_px + 1))
    small = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (224, 224), interpolation=cv2.INTER_LINEAR)
    if blur_max >= 3 and rng.random() < 0.7:
        length = int(rng.integers(3, blur_max + 1))
        img = _motion_blur(img, length, float(rng.uniform(0, 180)))
    return Image.fromarray(img)


class RandomDegrade:
    """Apply degradation with probability `p` (each worker gets its own RNG)."""

    def __init__(self, p: float, min_px: int, max_px: int, blur_max: int) -> None:
        self.p, self.min_px, self.max_px, self.blur_max = p, min_px, max_px, blur_max
        self.rng = np.random.default_rng()

    def __call__(self, pil: Image.Image) -> Image.Image:
        if self.rng.random() < self.p:
            return degrade_image(pil, self.rng, self.min_px, self.max_px, self.blur_max)
        return pil


def build_train_tf(degrade: bool, min_px: int, max_px: int, blur_max: int):
    steps = [transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1))]
    if degrade:
        steps.append(RandomDegrade(0.7, min_px, max_px, blur_max))
    steps += [
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        NORMALIZE,
    ]
    return transforms.Compose(steps)


EVAL_TF = transforms.Compose([transforms.ToTensor(), NORMALIZE])


# --- data ---


@dataclass
class Item:
    path: Path
    label: str
    box: tuple[float, float, float, float] | None = None  # normalized xyxy crop; None = whole


class CropDataset(Dataset):
    def __init__(self, items: list[Item], classes: list[str], tf) -> None:
        self.items = items
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.tf = tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        item = self.items[i]
        img = Image.open(item.path).convert("RGB")
        if item.box is not None:
            w, h = img.size
            x1, y1, x2, y2 = item.box
            img = img.crop((x1 * w, y1 * h, x2 * w, y2 * h))
        if img.size != (224, 224):
            img = img.resize((224, 224))
        return self.tf(img), self.class_to_idx[item.label]


def read_ocr_split(dataset_dir: Path, split: str) -> tuple[list[Item], list[Item]]:
    """Return (number items, abstain-seed items) for an OCR split."""
    split_dir = dataset_dir / split
    numbers: list[Item] = []
    abstain: list[Item] = []
    with open(split_dir / "annotations.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            img = split_dir / row["image"]
            if row["suffix"]:
                numbers.append(Item(img, row["suffix"]))
            else:
                abstain.append(Item(img, ABSTAIN))
    return numbers, abstain


def mine_abstain(det_dir: Path, split: str, max_count: int, seed: int) -> list[Item]:
    """Torso patches of players whose number box was NOT detected (numberless views).

    A free source of realistic "no readable number" crops: parse the YOLO labels,
    take each player box that contains no number-box center, and crop its
    upper-torso region (where a number would sit). No manual labelling.
    """
    lbl_dir = det_dir / split / "labels"
    img_dir = det_dir / split / "images"
    mined: list[Item] = []
    for lbl in sorted(lbl_dir.glob("*.txt")):
        img = img_dir / (lbl.stem + ".jpg")
        if not img.exists():
            continue
        players, numbers = [], []
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            if cls == PLAYER_CLS:
                players.append((cx, cy, bw, bh))
            elif cls == NUMBER_CLS:
                numbers.append((cx, cy))
        for cx, cy, bw, bh in players:
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            if any(x1 <= nx <= x2 and y1 <= ny <= y2 for nx, ny in numbers):
                continue  # this player shows a (detected) number -> not an abstain crop
            torso = (x1 + 0.2 * bw, y1 + 0.15 * bh, x1 + 0.8 * bw, y1 + 0.55 * bh)
            mined.append(Item(img, ABSTAIN, torso))
    rng = np.random.default_rng(seed)
    rng.shuffle(mined)
    return mined[:max_count]


# --- evaluation ---


@torch.no_grad()
def _predict(model: nn.Module, items: list[Item], classes: list[str], tf, device: str) -> list[int]:
    model.eval()
    preds = []
    for item in items:
        img = Image.open(item.path).convert("RGB")
        if item.box is not None:
            w, h = img.size
            x1, y1, x2, y2 = item.box
            img = img.crop((x1 * w, y1 * h, x2 * w, y2 * h))
        if img.size != (224, 224):
            img = img.resize((224, 224))
        preds.append(int(model(tf(img).unsqueeze(0).to(device)).argmax(1)))
    return preds


def _degraded_tf(seed: int, min_px: int, max_px: int, blur_max: int):
    rng = np.random.default_rng(seed)

    def tf(pil: Image.Image) -> torch.Tensor:
        return EVAL_TF(degrade_image(pil, rng, min_px, max_px, blur_max))

    return tf


def three_tier_eval(
    model: nn.Module,
    classes: list[str],
    number_items: list[Item],
    abstain_items: list[Item],
    device: str,
    degrade_cfg: tuple[int, int, int],
) -> dict:
    """Clean & degraded number accuracy + abstain recall (all on held-out test)."""
    idx = {c: i for i, c in enumerate(classes)}
    min_px, max_px, blur_max = degrade_cfg

    def number_acc(tf) -> float:
        preds = _predict(model, number_items, classes, tf, device)
        correct = sum(idx.get(it.label, -1) == p for it, p in zip(number_items, preds, strict=True))
        return correct / max(len(number_items), 1)

    clean = number_acc(EVAL_TF)
    degraded = number_acc(_degraded_tf(0, min_px, max_px, blur_max))
    result = {"clean_number_acc": clean, "degraded_number_acc": degraded}
    if ABSTAIN in idx and abstain_items:
        # abstain recall on degraded abstain crops (the realistic case)
        tf = _degraded_tf(1, min_px, max_px, blur_max)
        preds = _predict(model, abstain_items, classes, tf, device)
        result["abstain_recall"] = sum(p == idx[ABSTAIN] for p in preds) / len(abstain_items)
    else:
        result["abstain_recall"] = None  # model has no abstain class
    return result


def load_model(classes: list[str], device: str, weights: Path | None = None) -> nn.Module:
    model = models.resnet18(weights=None if weights else models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    if weights:
        ckpt = torch.load(weights, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
    return model.to(device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(DEFAULT_DATA))
    p.add_argument("--det-dataset", default=str(DEFAULT_DET))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="mps")
    p.add_argument("--degrade", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--abstain", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--degrade-min", type=int, default=20, help="min downscale px before upscale")
    p.add_argument("--degrade-max", type=int, default=48)
    p.add_argument("--blur-max", type=int, default=7)
    p.add_argument("--mine-train", type=int, default=250, help="abstain crops mined for train")
    p.add_argument("--mine-eval", type=int, default=40, help="abstain crops mined per eval split")
    p.add_argument("--eval-only", metavar="WEIGHTS", default=None, help="score a checkpoint only")
    p.add_argument("--out", default=str(ROOT / "runs/classify/number_classifier"))
    args = p.parse_args()

    dataset_dir, det_dir = Path(args.dataset), Path(args.det_dataset)
    degrade_cfg = (args.degrade_min, args.degrade_max, args.blur_max)

    train_num, train_ab = read_ocr_split(dataset_dir, "train")
    valid_num, valid_ab = read_ocr_split(dataset_dir, "valid")
    test_num, test_ab = read_ocr_split(dataset_dir, "test")

    if args.abstain:
        train_ab += mine_abstain(det_dir, "train", args.mine_train, seed=0)
        valid_ab += mine_abstain(det_dir, "valid", args.mine_eval, seed=1)
        test_ab += mine_abstain(det_dir, "test", args.mine_eval, seed=2)

    number_classes = sorted({it.label for it in train_num}, key=lambda s: (len(s), s))
    classes = number_classes + ([ABSTAIN] if args.abstain else [])

    # eval-only: score an existing checkpoint on the same held-out tiers, then stop.
    if args.eval_only:
        ckpt = torch.load(args.eval_only, map_location=args.device, weights_only=True)
        old_classes = ckpt["classes"]
        model = load_model(old_classes, args.device, Path(args.eval_only))
        res = three_tier_eval(model, old_classes, test_num, test_ab, args.device, degrade_cfg)
        print(f"=== eval-only: {args.eval_only} ({len(old_classes)} classes) ===")
        _print_eval(res)
        return

    train_items = train_num + (train_ab if args.abstain else [])
    n_ab = len(train_ab) if args.abstain else 0
    print(
        f"classes {len(classes)} (abstain={args.abstain}, degrade={args.degrade})  "
        f"train {len(train_items)} ({len(train_num)} num + {n_ab} abstain)"
    )

    train_tf = build_train_tf(args.degrade, *degrade_cfg)
    loaders = {
        "train": DataLoader(
            CropDataset(train_items, classes, train_tf),
            batch_size=args.batch,
            shuffle=True,
            num_workers=2,
        ),
        "valid": DataLoader(
            CropDataset(valid_num + (valid_ab if args.abstain else []), classes, EVAL_TF),
            batch_size=args.batch,
            num_workers=2,
        ),
    }

    model = load_model(classes, args.device)
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

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loaders["valid"]:
                correct += int((model(x.to(args.device)).argmax(1).cpu() == y).sum())
                total += len(y)
        val_acc = correct / max(total, 1)
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = "  *"
        avg_loss = running / len(train_items)
        print(f"epoch {epoch:3d}  loss {avg_loss:.4f}  val acc {val_acc:.4f}{marker}")

    assert best_state is not None
    model.load_state_dict({k: v.to(args.device) for k, v in best_state.items()})

    res = three_tier_eval(model, classes, test_num, test_ab, args.device, degrade_cfg)
    print("\n=== held-out metrics (new model) ===")
    print(f"valid acc {best_acc:.4f}  ({len(test_num)} number + {len(test_ab)} abstain test crops)")
    _print_eval(res)

    torch.save(
        {
            "state_dict": best_state,
            "classes": classes,
            "arch": "resnet18",
            "val_acc": best_acc,
            "eval": res,
            "degrade": args.degrade,
            "abstain": args.abstain,
        },
        out_dir / "best.pt",
    )
    print(f"saved {out_dir / 'best.pt'}")


def _print_eval(res: dict) -> None:
    print(f"  clean number acc    {res['clean_number_acc']:.4f}")
    print(f"  degraded number acc {res['degraded_number_acc']:.4f}")
    if res["abstain_recall"] is None:
        print("  abstain recall      n/a (model has no unreadable class)")
    else:
        print(f"  abstain recall      {res['abstain_recall']:.4f}")


if __name__ == "__main__":
    main()
