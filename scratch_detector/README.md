# From-scratch chapter: a minimal anchor-free detector

This package continues the "CNN from scratch" series: a CenterNet-style,
single-class (player) object detector implemented in plain PyTorch — no
`ultralytics` calls — benchmarked honestly against the fine-tuned YOLO baseline.

## Architecture

```
input 512×512 ── ResNet18 (stride 32) ── 3× [Upsample ×2 + Conv-BN-ReLU] ── stride-4 features
                                                                              ├── heatmap head (1ch): object-center probability
                                                                              ├── wh head      (2ch): box size in grid units
                                                                              └── offset head  (2ch): sub-cell center offset
```

Hand-implemented pieces (see the source — each is short):

| Piece | File | Idea |
|---|---|---|
| Gaussian target splatting | `data.py` | soft positives around each GT center (CornerNet radius) |
| Penalty-reduced focal loss | `loss.py` | down-weight negatives near GT centers |
| Max-pool NMS decoding | `model.py` | a peak survives iff it's the 3×3 local max |
| AP50 evaluation | `eval.py` | greedy IoU matching + PR-curve area, by hand |

## Train / evaluate ($0: Colab or Kaggle free GPU, or MPS locally)

```bash
uv run python -m scratch_detector.train --data data/<dataset>/data.yaml --epochs 40
uv run python -m scratch_detector.eval  --data data/<dataset>/data.yaml
uv run python scripts/benchmark.py --weights hoopvision_best.pt --video demo.mp4 \
    --scratch-weights scratch_detector/runs/best.pt
```

## Results (to be measured — see honesty rule in ROADMAP.md)

| model | player AP50 | FPS (M-series MPS) | params (M) |
|---|---|---|---|
| fine-tuned YOLO11n | TBD | TBD | TBD |
| scratch CenterNet-lite | TBD | TBD | TBD |

Training curves: `runs/losses.csv` (plotted in the write-up).

Expected outcome: the scratch detector loses to YOLO — the write-up documents
*why* (anchor-free head vs. YOLO's TAL assignment, augmentation gap, multi-scale
features), which is the actual point of the chapter.
