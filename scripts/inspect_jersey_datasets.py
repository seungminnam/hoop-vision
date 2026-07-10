"""Inspect the two Roboflow datasets that unblock jersey-number identity (task D).

Reproduces the counts cited in ADR-008 / data/README:
  * basketball-jersey-numbers-ocr (v3) — text-image-pairs (crop -> number string)
  * basketball-player-detection-3-ycjdo (v1) — 10-class detection incl. `number`

Both are gitignored (real broadcast crops); this only prints statistics.

    uv run python scripts/inspect_jersey_datasets.py
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAYER_NAMES = [
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


def inspect_ocr(root: Path) -> dict:
    labels: Counter[str] = Counter()
    splits = {}
    for split in ("train", "valid", "test"):
        ann = root / split / "annotations.jsonl"
        if not ann.exists():
            continue
        n = 0
        for line in ann.read_text().splitlines():
            d = json.loads(line)
            labels[d["suffix"]] += 1
            n += 1
        splits[split] = n
    numbers = {k: v for k, v in labels.items() if k != ""}
    return {
        "splits": splits,
        "total": sum(splits.values()),
        "number_classes": len(numbers),
        "empty_unreadable": labels.get("", 0),
        "most_common": labels.most_common(5),
        "min_count": min(numbers.values()) if numbers else 0,
        "classes_under_5": sum(1 for v in numbers.values() if v < 5),
    }


def inspect_detection(root: Path) -> dict:
    counts: Counter[int] = Counter()
    num_wh: list[tuple[float, float]] = []
    splits = {}
    for split in ("train", "valid", "test"):
        lbls = glob.glob(str(root / split / "labels" / "*.txt"))
        splits[split] = len(lbls)
        for lf in lbls:
            for line in Path(lf).read_text().splitlines():
                p = line.split()
                if not p:
                    continue
                c = int(p[0])
                counts[c] += 1
                if c == 2:  # number
                    num_wh.append((float(p[3]), float(p[4])))
    nw = np.array([x[0] for x in num_wh])
    nh = np.array([x[1] for x in num_wh])
    # dataset images are 1280x1280; our clip is native 1280x720
    px_native = (float(np.median(nw) * 1280), float(np.median(nh) * 720))
    px_640 = (float(np.median(nw) * 640), float(np.median(nh) * 640))
    return {
        "splits": splits,
        "total": sum(splits.values()),
        "class_counts": {PLAYER_NAMES[i]: counts[i] for i in range(len(PLAYER_NAMES))},
        "number_boxes": len(num_wh),
        "number_px_native_1280x720": [round(px_native[0], 1), round(px_native[1], 1)],
        "number_px_640sq": [round(px_640[0], 1), round(px_640[1], 1)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ocr-root", default=str(ROOT / "data/basketball-jersey-numbers-ocr-3"))
    ap.add_argument("--det-root", default=str(ROOT / "data/basketball-player-detection-3-ycjdo-1"))
    args = ap.parse_args()

    out = {}
    if Path(args.ocr_root).exists():
        out["jersey_ocr"] = inspect_ocr(Path(args.ocr_root))
    if Path(args.det_root).exists():
        out["player_detection"] = inspect_detection(Path(args.det_root))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
