# 🏀 Hoop Vision

**Basketball video analytics from raw footage**: player detection & tracking with
persistent IDs, team assignment, court-to-2D homography (live minimap), shot attempt
detection, and automatic shot charts — plus a from-scratch object detector chapter
benchmarked against the YOLO baseline.

> _NBA Forecast Lab predicted games from tabular stats; Hoop Vision extracts those
> stats from raw video. Statistics → perception._

<!-- demo GIF goes here at W6: docs/demo.gif -->
📽️ *Demo GIF and deployed app link coming with milestone W6 (Aug 2026).*

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

**Detection (fine-tuned YOLO11n, Roboflow val split)** — *pending W2*

| class | AP50 | AP50-95 |
|---|---|---|
| player | TBD | TBD |
| ball | TBD | TBD |
| rim | TBD | TBD |

**Shot detection vs hand labels (≥3 clips)** — *pending W4*

| clip | camera | attempts (GT) | precision | recall |
|---|---|---|---|---|
| TBD | fixed | TBD | TBD | TBD |

**Scratch detector vs YOLO** — *pending W5* (table in [scratch_detector/README.md](scratch_detector/README.md))

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
scripts/            calibrate, download_data, finetune_yolo, benchmark
app/                Streamlit demo (+ precomputed samples for the free-tier deploy)
tests/              unit tests for the pure-logic modules (court, events, teams)
data/               gitignored; dataset/clip documentation in data/README.md
```

## Status

Building in public, Jul–Aug 2026 ([SPEC.md](SPEC.md) has the weekly milestones):

- [x] W1 — baseline pipeline: detection, ByteTrack IDs, annotated video, tests
- [ ] W2 — fine-tune YOLO on player/ball/rim + team assignment metrics
- [ ] W3 — homography + minimap on fixed-camera clips
- [ ] W4 — shot events vs hand-labeled ground truth
- [ ] W5 — from-scratch detector benchmark
- [ ] W6 — deployed demo + write-up

## License

[MIT](LICENSE)
