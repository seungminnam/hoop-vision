# SPEC — Hoop Vision: Basketball Broadcast Video Analytics

> **Audience:** an AI coding agent (or human) implementing this project from scratch.
> This spec is self-contained. Copy it into the new repo as `SPEC.md` and work from it.
> **Owner:** Ryan Nam · **Window:** Jul 6 – Aug 16, 2026 (6 weeks, main project)
> **Repo name:** `hoop-vision` (public)

## 1. Pitch

A computer-vision pipeline that takes basketball game footage and produces structured
analytics: player detection and tracking with persistent IDs, ball/rim detection,
court-to-2D homography mapping (live minimap), shot attempt detection, and automatic
shot charts. Includes a **from-scratch chapter**: a small object detector implemented
manually in PyTorch and benchmarked against the fine-tuned YOLO baseline, continuing the
owner's "CNN from scratch" series.

**Portfolio narrative:** NBA Forecast Lab predicted games from tabular stats; Hoop Vision
extracts those stats from raw video. Statistics → perception.

## 2. Goals & non-goals

**Goals**
- G1: Detect players, ball, and rim in broadcast/fixed-camera clips (fine-tuned model, measured mAP).
- G2: Track players across frames with persistent IDs (ByteTrack), with team assignment via jersey-color clustering.
- G3: Map player positions onto a 2D court minimap via homography.
- G4: Detect shot attempts and made/missed outcomes; render a shot chart.
- G5: From-scratch chapter — implement a minimal single-class detector in pure PyTorch; benchmark vs the YOLO baseline (mAP, FPS, params).
- G6: Ship a Streamlit demo app + a ≤90-second demo video + README with results tables.

**Non-goals (v1)**
- Live-stream / real-time processing (offline clips only).
- Jersey number OCR, player re-identification across camera cuts, multi-game aggregation.
- Moving-broadcast-camera homography that survives cuts (see fallbacks §8).

## 3. Tech stack

- Python 3.11+, managed with `uv` (or venv + pip). Repo uses `pyproject.toml`.
- PyTorch (MPS backend on the owner's Apple Silicon MacBook for inference/small training).
  Heavy fine-tuning runs on Colab/Kaggle free GPU; keep training scripts runnable both places.
- `ultralytics` (YOLO, latest stable major at implementation time), `supervision` (Roboflow)
  for ByteTrack tracking + annotation utilities, `opencv-python`, `numpy`, `streamlit`.
- `yt-dlp` for sourcing clips (see data ethics note §4).
- Verify current library APIs from official docs at implementation time; do not trust
  memorized ultralytics/supervision APIs — both change often.

## 4. Data

1. **Detection fine-tuning:** use a public basketball dataset from Roboflow Universe
   (search "basketball players ball rim detection"; several thousand-image datasets with
   player/ball/rim/net classes exist). Export in YOLO format. Record dataset name, URL,
   license, and image counts in `data/README.md`.
2. **Evaluation clips:** 5–10 short clips (10–30 s each): mix of fixed-camera amateur
   footage and broadcast footage. At least 2 clips reserved as a held-out demo set never
   used for tuning decisions.
3. **Ethics/legal:** clips are for research/demo. Do not redistribute raw broadcast video
   in the repo; store locally, commit only annotated output GIFs/screenshots of short
   excerpts. Document sources.

## 5. Architecture

```
hoop-vision/
├── SPEC.md
├── pyproject.toml
├── data/                    # gitignored except README.md
├── src/hoopvision/
│   ├── ingest.py            # video → frames iterator (OpenCV), fps/resize handling
│   ├── detect.py            # Detector interface; YoloDetector impl
│   ├── track.py             # ByteTrack wrapper → per-frame [track_id, class, bbox, conf]
│   ├── teams.py             # team assignment: jersey-crop color features + k-means (k=2)
│   ├── court.py             # homography: 4-point manual calibration → 3x3 H; px → court coords
│   ├── events.py            # shot attempt + outcome detection (state machine, §5.2)
│   ├── shotchart.py         # aggregate events → matplotlib/plotly halfcourt chart
│   ├── viz.py               # annotated video writer + minimap overlay
│   └── pipeline.py          # orchestrates: video in → annotated video + events.json out
├── scratch_detector/        # G5: from-scratch chapter (own README with results)
│   ├── model.py             # minimal anchor-free FCOS/CenterNet-style single-class detector
│   ├── loss.py  train.py  eval.py
├── app/streamlit_app.py     # upload clip or pick sample → run pipeline → results tabs
├── scripts/                 # download_data.py, finetune_yolo.py, benchmark.py
└── tests/                   # unit tests for court.py, events.py, teams.py (pure logic)
```

Design rule: `detect.py` exposes a `Detector` protocol (`detect(frame) -> list[Detection]`)
so the YOLO baseline and the scratch detector are interchangeable in the pipeline and in
benchmarks.

### 5.1 Court homography (v1 = manual calibration)

- CLI/Streamlit step: user clicks ≥4 known court landmarks (corners, free-throw line
  intersections) on a reference frame → `cv2.findHomography` → store H per clip (JSON).
- Works for fixed-camera clips. Broadcast pans/cuts are out of scope for v1 (§8).
- Court model: NBA halfcourt coordinate system in feet (50 × 47), documented in `court.py`.

### 5.2 Shot event detection (heuristic state machine)

- Inputs: ball track + rim bbox per frame.
- Attempt: ball trajectory goes above rim top-edge within a horizontal window around the rim.
- Outcome: MADE if ball center passes downward through rim bbox interior within N frames
  after the attempt apex; MISS otherwise (timeout or trajectory exits window).
- All thresholds in a dataclass config; unit-test with synthetic trajectories.
- Ball detection is flaky (small, motion blur) → interpolate ball track gaps ≤ K frames;
  attempt detection also has a fallback trigger: player in shooting pose is out of scope,
  so if ball track is unusable on a clip, report shot events as unavailable for that clip
  rather than producing garbage (quality gate in §7).

## 6. Milestones (weekly, each with acceptance criteria)

**W1 — Baseline pipeline (Jul 6–12)**
- Repo scaffolding, `uv` env, CI-less but `pytest` + `ruff` runnable.
- Pretrained COCO YOLO detects `person` + `sports ball` on sample clips; supervision
  ByteTrack assigns IDs; annotated output video renders.
- ✅ Accept: `python -m hoopvision.pipeline demo.mp4` produces annotated video with
  stable player IDs on a fixed-camera clip.

**W2 — Fine-tune + teams (Jul 13–19)**
- Fine-tune YOLO (nano or small) on the Roboflow dataset: classes `player, ball, rim`.
  Train on Colab GPU; log mAP50 / mAP50-95 per class on the dataset's val split.
- Team assignment via jersey-crop color k-means; smoothed over each track's history.
- ✅ Accept: metrics table committed (README); ball & rim detected on demo clips;
  players colored by team in output video.

**W3 — Homography + minimap (Jul 20–26)**
- Manual 4-point calibration tool; feet-coordinate projection; minimap rendered
  side-by-side/picture-in-picture in the output video.
- ✅ Accept: on a fixed-camera clip, projected player dots track visually correctly;
  reprojection error of calibration points < 1 ft; `court.py` unit tests pass.

**W4 — Shot events + shot chart (Jul 27–Aug 2)**
- Event state machine; `events.json` (timestamp, shooter track_id if attributable,
  court x/y, outcome); shot chart rendering.
- Label ground truth by hand for the demo clips (a simple CSV: timestamp, outcome).
- ✅ Accept: shot-attempt precision/recall vs hand labels reported on ≥ 3 clips
  (target: ≥80% recall on fixed-camera clips; report honestly whatever it is).

**W5 — From-scratch chapter (Aug 3–9)**
- Implement a minimal anchor-free detector (FCOS/CenterNet-style: backbone = small
  ResNet, single detection head, focal + IoU loss) for the `player` class only.
- Train on the same Roboflow dataset (subset OK). Benchmark table: scratch vs fine-tuned
  YOLO — mAP50 (player class), inference FPS on M-series MPS, parameter count.
- ✅ Accept: `scratch_detector/README.md` with the table + training curves; scratch model
  plugs into the pipeline via the `Detector` protocol (even if worse — that's the point).

**W6 — App, demo, write-up (Aug 10–16)**
- Streamlit app: pick sample clip (or upload) → tabs: annotated video, minimap, shot
  chart, events table. Deploy on Streamlit Community Cloud with bundled sample outputs
  (pre-computed if runtime limits bite; note this in README).
- ≤90 s screen-recorded demo video; README polished (pitch, GIF, architecture diagram,
  results tables, honest limitations section); technical write-up on the from-scratch
  chapter.
- ✅ Accept: public URL works; README meets the Definition of Done in ROADMAP.md.

## 7. Quality gates & verification

- `pytest` green for `court`, `events`, `teams` logic; `ruff check` clean.
- Every README number generated by `scripts/benchmark.py` or a committed notebook.
- Per-clip quality gate: if ball-track coverage < 40% of frames, the pipeline marks shot
  analytics "unavailable" for that clip instead of emitting low-confidence events.

## 8. Risks & fallbacks

| Risk | Likelihood | Fallback |
|---|---|---|
| Ball detection too unreliable on broadcast clips | High | Restrict shot analytics to fixed-camera clips; broadcast clips get detection+tracking+teams only |
| Homography on moving broadcast cameras | Certain (out of scope) | v1 fixed-camera only; list "keypoint-based dynamic homography" as future work |
| Colab GPU limits | Medium | YOLO-nano + smaller image size; Kaggle as backup (30 h/wk) |
| Scratch detector too weak to be presentable | Medium | That's acceptable — frame the chapter as "why YOLO wins"; report the gap honestly |
| Streamlit Cloud can't run video inference | Medium | Ship pre-computed results for sample clips; app becomes an explorer, not a live runner |

## 9. Resume bullets (numbers measured; sources in README results tables)

- "Built an end-to-end basketball video-analytics pipeline (YOLO fine-tuned to
  player/ball/rim, 0.92 mAP50; ByteTrack multi-object tracking) that maps players to court
  coordinates via homography and auto-generates shot charts from raw game footage."
- "Implemented an anchor-free object detector from scratch in PyTorch and benchmarked it
  against the fine-tuned YOLO baseline (0.58 vs 0.97 player AP50, 40 vs 34 FPS on Apple
  M4 MPS), documenting the architecture trade-offs."
- "Detected shot attempts/outcomes with a trajectory state machine at 80% precision and
  80% recall (n=5) against hand-labeled ground truth, 4/4 outcomes correct; shipped an
  interactive Streamlit demo."
