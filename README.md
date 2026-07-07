# 🏀 Hoop Vision

**Basketball video analytics from raw footage**: player detection & tracking with
persistent IDs, team assignment, court-to-2D homography (live minimap), shot attempt
detection, and automatic shot charts — plus a from-scratch object detector chapter
benchmarked against the YOLO baseline.

> _NBA Forecast Lab predicted games from tabular stats; Hoop Vision extracts those
> stats from raw video. Statistics → perception._

![Fine-tuned detection on an NBA broadcast frame](docs/sample_detection.jpg)
*Fine-tuned YOLO11n on a held-out val frame: 10 players, ball, and rim detected —
referees excluded by design.*

![Live minimap on a static game clip](docs/sample_minimap.jpg)
*Court homography in action during a free throw: tracked players project onto
the 2D halfcourt (picture-in-picture) with team colors and the rim marker.*

## What it does

| Capability | How |
|---|---|
| Detect players / ball / rim | YOLO11n fine-tuned on a Roboflow basketball dataset |
| Track players with stable IDs | ByteTrack (via `supervision`) over player detections |
| Assign teams | jersey-crop LAB color features + k-means (k=2), smoothed per track |
| Map players to court coordinates | manual 4-point homography → NBA halfcourt (50×47 ft) |
| Detect shot attempts & outcomes | trajectory state machine over ball + rim tracks |
| Render | annotated video + picture-in-picture minimap + shot chart |
| From-scratch chapter | CenterNet-style detector in plain PyTorch ([details](scratch_detector/README.md)) |

## Architecture

```
video ──▶ ingest ──▶ detect (YOLO ▮ Detector protocol ▮ scratch) ──▶ track (ByteTrack)
                                                                        │
        ┌───────────────────────────────────────────────────────────────┤
        ▼                          ▼                                    ▼
  teams (k-means)        court (homography H)                events (state machine)
        │                          │                                    │
        └──────────────┬───────────┴───────────────┬────────────────────┘
                       ▼                           ▼
        viz: annotated video + minimap      events.json + shot chart
```

Design rule: `detect.py` exposes a `Detector` protocol, so the fine-tuned YOLO
baseline and the from-scratch detector are interchangeable in the pipeline and in
`scripts/benchmark.py`.

## Quick start

```bash
git clone https://github.com/seungminnam/hoop-vision && cd hoop-vision
uv sync                                  # installs Python 3.12 env + deps

# 1. Baseline (COCO pretrained; players + ball, no rim):
uv run python -m hoopvision.pipeline clip.mp4 --output out

# 2. Calibrate the court on a fixed-camera clip (click 4 landmarks):
uv run python scripts/calibrate.py clip.mp4 --output calib.json

# 2b. ...or recover it automatically from the painted key (colored paint,
#     static lined court) — writes an overlay JPEG so you can inspect the fit:
uv run python scripts/auto_calibrate.py clip.mp4 --weights hoopvision_best.pt \
    --output calib.json --overlay check.jpg

# 3. Full pipeline with minimap + shot analytics (needs fine-tuned weights):
uv run python -m hoopvision.pipeline clip.mp4 --weights hoopvision_best.pt \
    --calibration calib.json --output out

# Demo app:
uv run streamlit run app/streamlit_app.py

# Tests / lint:
uv run pytest -q && uv run ruff check .
```

Fine-tuning and the from-scratch chapter run on **free** Colab/Kaggle GPUs — see
[`scripts/finetune_yolo.py`](scripts/finetune_yolo.py) and
[`scratch_detector/README.md`](scratch_detector/README.md).

## Results

Numbers appear here only when measured (every figure must be reproducible by
`scripts/benchmark.py`, `scripts/finetune_yolo.py` logs, or a committed notebook —
no placeholders presented as results).

**Detection (fine-tuned YOLO11n)** — measured 2026-07-07 on the
[basketball-computer-vision v14](https://universe.roboflow.com/basketballcomputervision/basketball-computer-vision/dataset/14)
val split (46 images; small dataset — 235 images total, so treat as indicative):

| class | AP50 | AP50-95 |
|---|---|---|
| player | 0.965 | 0.602 |
| ball | 0.814 | 0.423 |
| rim | 0.995 | 0.512 |
| **all (incl. referee)** | **0.919** | **0.526** |

Training: `scripts/finetune_yolo.py --epochs 60 --imgsz 960`, early-stopped at
epoch 42, 0.5 h on Apple M4 (MPS) — no cloud GPU needed for this dataset size.
Inference: **30.6 FPS** at 640 px on M4 MPS, 2.59 M params (`scripts/benchmark.py`).
Ball is the weakest class as expected (small object, motion blur) — this is why
the pipeline has the ball-coverage quality gate.

**Shot detection vs hand labels** — measured 2026-07-07 on three fixed-camera
pickup-game clips (1080p, static camera verified by frame-blend test). Ground
truth: frame-by-frame review (`data/labels/*.csv`); numbers reproduced by
`scripts/eval_shots.py`:

| clip | GT attempts | TP | FP | FN | outcomes correct |
|---|---|---|---|---|---|
| pickup_seg1 | 2 | 2 | 0 | 0 | 2/2 |
| pickup_seg2 | 1 | 1 | 0 | 0 | 1/1 |
| pickup_seg3 | 2 | 1 | 1 | 1 | 1/1 |
| **overall** | **5** | **4** | **1** | **1** | **4/4** |

Attempt **precision 80% / recall 80%** (n=5 — small sample, stated plainly);
made/missed outcome accuracy 4/4 on matched attempts. The miss was an airball
whose arc stayed outside the horizontal attempt window; the false positive was
a high pass crossing above rim level. Ball-track coverage on these clips:
42–77% (vs 2% on 360p footage — resolution is the ball detector's bottleneck).

**Scratch detector vs YOLO** — measured 2026-07-08, same val split and clip,
Apple M4 MPS (`scripts/benchmark.py`; details + training curves in
[scratch_detector/README.md](scratch_detector/README.md)):

| model | player AP50 | FPS | params (M) |
|---|---|---|---|
| fine-tuned YOLO11n | 0.965 | 34.0 | 2.59 |
| scratch CenterNet-lite (plain PyTorch) | 0.578 | 40.1 | 12.84 |

The from-scratch detector is *faster* but far less accurate on 165 training
images — the write-up documents why (pretraining, augmentation recipe,
multi-scale assignment), which is the point of the chapter.

## $0 operations

Everything in this project runs on free tiers:

| Need | Free service |
|---|---|
| GPU fine-tuning | Google Colab free T4 / Kaggle (30 h/wk P100) |
| Dataset | Roboflow Universe public datasets (free API key) |
| Local inference | MacBook Apple Silicon (PyTorch MPS) |
| Demo hosting | Streamlit Community Cloud |
| Repo + CI | GitHub public repo + Actions free minutes |

## Honest limitations (v1)

- **Fixed-camera clips only** for homography/minimap and shot analytics; broadcast
  pans/cuts get detection + tracking + teams only. Keypoint-based dynamic homography
  is future work.
- **Ball detection is flaky** (small object, motion blur). The pipeline linearly
  interpolates short gaps, and if ball-track coverage is <40% of frames it reports
  shot analytics as *unavailable* rather than emitting low-confidence events.
- No jersey-number OCR or cross-cut re-identification; offline processing only.

## Repo map

```
src/hoopvision/     pipeline modules (ingest → detect → track → teams → court → events → viz)
scratch_detector/   from-scratch CenterNet-style detector + training/eval
scripts/            calibrate, auto_calibrate, download_data, finetune_yolo, benchmark
app/                Streamlit demo (+ precomputed samples for the free-tier deploy)
tests/              unit tests for the pure-logic modules (court, events, teams)
data/               gitignored; dataset/clip documentation in data/README.md
```

## Status

Building in public, Jul–Aug 2026 ([SPEC.md](SPEC.md) has the weekly milestones):

- [x] W1 — baseline pipeline: detection, ByteTrack IDs, annotated video, tests
- [x] W2a — fine-tune YOLO11n on player/ball/rim (mAP50 0.919, table above)
- [x] W2b — team assignment verified on a real game clip (white vs red jerseys
  mostly separated at 360p; occasional flips on small crops — noted limitation)
- [x] W3 — homography + minimap, verified on a static lined-court clip
  (`docs/sample_minimap.jpg`; calibration = paint-region corners + curve
  refinement, paint-corner reprojection 1.7 ft — above the 1 ft target because
  the source is 360p; see `calib_hudl_static2.json`)
- [x] W4 — shot events vs hand-labeled ground truth (80%/80% on 3 clips, n=5;
  shot-chart court coordinates pending W3 calibration)
- [x] W5 — from-scratch detector benchmark (player AP50 0.578 vs YOLO 0.965;
  table above, analysis in `scratch_detector/README.md`)
- [ ] W6 — deployed demo + write-up

## License

[MIT](LICENSE)
