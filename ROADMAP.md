# Hoop Vision — Post-v1 Roadmap (v1.1 → v3)

> **Who this is for.** Any agent or contributor picking up this project after
> v1. It assumes v1 is done (it is — see the snapshot below) and defines the
> next three phases with enough context to start work without re-deriving
> decisions. [SPEC.md](SPEC.md) is the v1 spec and stays frozen as a record;
> this file owns everything after it.
>
> **How to use this file.** Work top-down inside a phase; phases are ordered
> by dependency (v1.1 unblocks v2's quality, v2 unblocks most of v3). Each
> phase lists motivation, workstreams with acceptance criteria, deliverables,
> and risks. Update the status checkboxes here as work merges, the same way
> README's status section tracked W1–W6.

---

## 0. Current state (v1, shipped 2026-07-08)

Everything below is merged to `main` and measured — sources for every number
are committed scripts (honesty rule, §2).

| Piece | State | Key numbers |
|---|---|---|
| Detection | YOLO11n fine-tuned on player/ball/rim ([release v0.2.0](https://github.com/seungminnam/hoop-vision/releases/tag/v0.2.0)) | mAP50 0.919 (player 0.965 / ball 0.814 / rim 0.995), 34 FPS @ M4 MPS |
| Tracking | ByteTrack via `supervision` (motion-only, players) | **no MOT metrics yet — v1.1's first job** |
| Teams | torso-crop LAB + k-means(2), per-track majority | qualitative only |
| Court | manual 4-point (`scripts/calibrate.py`) + automatic (`scripts/auto_calibrate.py`: paint segmentation → quad → ICP vs court lines) | 1.7 ft corner reprojection @ 360p (static clips only) |
| Shot events | trajectory state machine + 40% ball-coverage quality gate | precision 80% / recall 80% (n=5), outcomes 4/4 |
| From-scratch chapter | CenterNet-lite ([release v0.3.0](https://github.com/seungminnam/hoop-vision/releases/tag/v0.3.0)) | player AP50 0.578, 40 FPS, 12.84M params |
| Demo | [hoop-vision.streamlit.app](https://hoop-vision.streamlit.app/) (precomputed samples; cloud env is Streamlit-only via `app/requirements.txt`) | live |

**Known weaknesses that motivate this roadmap** (observed on the deployed
samples, especially `hudl_seg1`):

1. **ID switches** — ByteTrack has no appearance model; basketball's constant
   occlusion/crossing breaks motion-only association. Camera pan (Hudl
   auto-tracking) additionally violates the Kalman static-scene assumption.
2. **360p + domain gap** — the 235-image fine-tune set doesn't cover low-res
   gym footage; distant players flicker, ball coverage drops to 2%.
3. **Fixed-camera-only homography** — minimap/shot charts only work on
   verified static windows, which excludes typical amateur game film.

## 1. Phase overview

| Phase | Theme | One-line goal | Status |
|---|---|---|---|
| v1.1 | Tracking robustness | Measure MOT quality, then fix association (appearance + camera-motion compensation) | ◐ measurement harness done; labels + fixes next |
| v2 | Dynamic homography | Per-frame court registration so minimap/shot charts work on panning cameras | ☐ not started |
| v3 | Product: "Hudl-lite" | Auto game report for amateur teams (stats, shot charts, highlights) on the free stack | ☐ not started |

## 2. Working agreements (unchanged from v1 — do not relax)

- **$0 operations.** Free tiers only: Colab/Kaggle GPUs, M-series MPS locally,
  Streamlit Community Cloud, GitHub free. No paid APIs, no paid labeling.
- **Honesty rule.** A number appears in README only if a committed script
  reproduces it. No placeholder results. Report bad numbers plainly.
- **Git discipline.** Feature branch → `ruff check` + `ruff format` + `pytest`
  locally → PR → CI green → squash-merge. Never commit to `main` directly.
  No AI signatures in commits/PRs/branches. (Repo does not allow auto-merge:
  wait for checks, then `gh pr merge N --squash --delete-branch`.)
- **Data hygiene.** Raw video never committed (`data/` is gitignored except
  `README.md` and `labels/`); document every source in `data/README.md`;
  API keys only via env vars.
- **Cloud app stays thin.** `app/requirements.txt` (Streamlit-only) drives the
  deployed env — Community Cloud reads the entrypoint-dir dependency file
  before the root `uv.lock`. Never reintroduce a root `packages.txt` (its apt
  deps broke the deploy and nothing in the hosted path imports cv2).

---

## 3. v1.1 — Tracking robustness (measure first, then fix)

**Motivation.** The most visible quality problem in the demo. Also the
cheapest phase, and v2 inherits whatever tracker quality exists (a minimap dot
is only as stable as its track).

**Design stance.** Do not swap trackers on vibes. Build the measurement
harness first, get baseline numbers for ByteTrack, then land improvements one
at a time with before/after in the same table.

### 3.1 MOT ground truth + metrics harness

- ✅ **Unsupervised diagnostics** — `scripts/track_diagnostics.py` reports
  fragmentation ratio, track life, churn, and an ID-switch proxy with no
  labels. Baseline recorded in README (both clips fragment ~8–10×, median
  track life ~1.5 s). This is the "before" the fixes must beat.
- ✅ **Supervised harness** — `scripts/eval_tracking.py` computes IDF1/MOTA/
  IDsw/MT/ML via `motmetrics` (dev dep; `np.asfarray` shim for NumPy 2),
  unit-tested on synthetic sequences, self-eval verified on real MOT files.
  Reads MOTChallenge CSV from `data/labels/mot/gt/<clip>.txt`; predictions
  come from `track_diagnostics.py --dump-mot` (kept local — see `.gitignore`).
- ☐ **Ground-truth labels (next, needs a human review pass)** — hand-label
  player IDs on `pickup_seg3` (static 1080p) and `hudl_seg1` (panning 360p).
  Bootstrap boxes from the detector, correct IDs by hand (CVAT local via
  Docker is free, or a small OpenCV review tool). Commit under
  `data/labels/mot/gt/`. Note: HOTA needs `trackeval`; `motmetrics` covers
  IDF1/MOTA today — add HOTA when labels justify it.
- **Accept:** baseline ByteTrack IDF1/MOTA for both clips in README once
  labels land (diagnostics baseline is already there).

### 3.2 Association upgrades (one PR each, measured)

1. **Camera-motion compensation (GMC).** Estimate per-frame global motion
   (e.g. `cv2.calcOpticalFlowPyrLK` on background corners or ECC on a
   downscaled gray frame) and warp Kalman predictions before matching —
   BoT-SORT's trick. Directly targets the panning clip.
2. **Appearance embedding.** Add a lightweight ReID feature (start free:
   torso-crop color histogram / LAB stats already computed in `teams.py`;
   stretch: OSNet-lite ONNX) and combine IoU + appearance cost. Targets
   occlusion crossings.
3. **Team-aware association.** Forbid/penalize matches across team labels
   using the k-means assignment. Cheap and complementary.
- Keep the `Detector`-style pattern: tracker behind a small interface in
  `src/hoopvision/track.py` so ByteTrack remains selectable for comparison.
  (Note: `supervision` is pinned `<0.30`; its ByteTrack replacement package
  `trackers` shipped a broken wheel — re-check before touching the pin.)
- **Accept:** IDF1 improves on *both* clips with no shot-metric regression
  (`scripts/eval_shots.py` unchanged numbers or better); README gets a
  before/after tracking table.

### 3.3 Optional detection boost (only if 3.2 plateaus)

- Fine-tune on frames from the target domain: pseudo-label Hudl frames with
  the current model, hand-fix a few hundred boxes, retrain (Colab free tier).
- **Accept:** player AP50 on a small held-out Hudl-frame set improves and
  IDF1 moves with it.

**Deliverables:** `scripts/eval_tracking.py`, MOT labels, upgraded tracker
module, README "Tracking" results section, updated demo samples.
**Risks:** hand-labeling time (bound it: 2 clips only); free ReID model
weights license (check before bundling — histogram fallback always works).

---

## 4. v2 — Dynamic homography (drop the fixed-camera constraint)

**Motivation.** The single biggest capability unlock: minimap + shot charts on
*typical* amateur/broadcast film, not just verified static windows. Known in
the literature as *sports field registration* — recognizable, resume-worthy.

**Design stance.** Reuse what v1 already built. `scripts/auto_calibrate.py`
(paint segmentation + ICP against court lines) is a working single-frame
calibrator on lined courts; v2 turns it into (a) a pseudo-label factory and
(b) a per-frame runtime with temporal smoothing.

### 4.1 Court keypoint dataset (pseudo-labeled, $0)

- Run `auto_calibrate.py` over the static windows already cataloged in
  `data/README.md`; project a fixed landmark set (paint corners, FT-line
  ends, arc extremes, halfcourt intersections — extend `court.LANDMARKS`)
  into each frame → keypoint annotations for free.
- Augment with synthetic homography warps + color jitter of those frames to
  simulate pans/zooms. Target: a few thousand frames without hand labeling.
- **Accept:** dataset builder script committed; sample overlays visually
  correct (spot-check like the v1 corner-compare workflow).

### 4.2 Keypoint model + per-frame registration

- Small heatmap model predicting landmark locations + visibility. Two paths:
  fine-tune YOLO11n-pose, or extend the in-repo CenterNet-lite with K heatmap
  channels (nice continuity with the from-scratch chapter). Train on Colab
  free tier or MPS.
- Runtime: keypoints → RANSAC homography per frame → temporal smoothing
  (EMA on reprojected landmarks; fall back to last-good H when <4 confident
  points, exactly like the ball-coverage gate philosophy).
- **Accept (quantitative):** on held-out *static* clips with known
  calibrations (`calib_hudl_static2.json` + at least one new manual
  calibration), median landmark reprojection error and court-region IoU
  reported by a committed eval script; **on the panning clip**, minimap
  jitter (frame-to-frame player court-position variance during stands-still
  moments) reported before/after smoothing.
- **Accept (qualitative):** minimap PIP video on `hudl_seg1` (the panning
  clip that v1 could not calibrate) looks stable; committed as a new app
  sample + README GIF.

### 4.3 Pipeline + product integration

- `pipeline.py --calibration auto` mode: no JSON needed; per-frame H feeds
  minimap and `court.to_court()` for events/shot charts. Quality gate: if
  registration confidence is low for >X% of frames, report court analytics
  "unavailable" (same honesty pattern as ball coverage).
- **Accept:** shot-chart court coordinates now produced for at least one
  panning clip; W-style checkbox + results land in README.

**Deliverables:** dataset builder, keypoint trainer/eval, smoothed runtime,
new demo sample, README section with metrics.
**Risks:** pseudo-labels inherit the NBA-model-vs-real-court mismatch
documented in v1 (1.7 ft floor at 360p) — mitigate by adding one or two
higher-res, correctly-dimensioned sources; single-court overfit — augment
aggressively, and say so honestly in the write-up if generalization is thin.

---

## 5. v3 — Product framing: "Hudl-lite for amateur teams"

**Motivation.** Package v1.1+v2 into something a coach/player actually uses:
upload game film → get an automatic game report. Also completes the
portfolio arc with the sibling project (NBA Forecast Lab: video → stats →
prediction).

**Scope (thin slices, in order — each is independently shippable):**

1. **Game report v0.** For one uploaded (or sample) clip: team shot charts,
   made/missed timeline, per-track minutes-on-screen + distance/avg-speed in
   feet (homography makes these physical units). Rendered as a Streamlit
   report page + downloadable JSON/PNG. *Depends only on v1.1.*
2. **Auto-highlights.** Cut ±N seconds around each shot event with ffmpeg,
   stitch a highlight reel; "made shots only" toggle. Cheap, high demo value.
3. **Per-player identity.** Jersey-number OCR on track crops (small digit
   classifier or free OCR lib) + track merging across broken IDs →
   per-player box score (attempts, makes, distance). *Hardest; needs v1.1
   track stability; keep optional.*
4. **Batch/local runner.** CLI to process a full game and emit the report;
   the hosted app stays precomputed-samples-only (free-tier limits — same
   split as v1).

**Accept:** a stranger can open the live app, pick a sample game, and read a
coherent game report; README repositions the project as pipeline + product;
resume bullets updated with v2/v3 numbers.
**Non-goals (state them in README):** real-time/courtside processing, paid
hosting, cross-game player identity, referee/coach detection.

---

## 6. Sequencing, effort, and definition of done

```
v1.1 (2–3 wk)  ──▶  v2 (3–4 wk)  ──▶  v3 (2–3 wk, thin slices)
 measure→fix        register→smooth     report→highlights→OCR
```

- Rough effort assumes the v1 cadence (evenings/weekends, MPS + free GPUs).
- v3 slice 1–2 can start after v1.1 if v2 drags — only slice "court-accurate
  per-player physical stats on panning film" hard-depends on v2.
- **Roadmap DoD:** all three phase checkboxes in §1 ticked, every new number
  script-reproducible, demo app updated per phase, and a final write-up
  linking the arc: *detector → tracker → registration → product*.

## 7. Appendix — orientation for a fresh agent

- **Repo map:** `src/hoopvision/` (pipeline modules), `scratch_detector/`
  (from-scratch chapter), `scripts/` (calibrate, auto_calibrate,
  download_clips, eval_shots, benchmark, finetune_yolo), `app/` (Streamlit +
  precomputed samples), `tests/` (pure-logic units), `data/README.md`
  (clip provenance + labeling docs).
- **Verify environment:** `uv sync && uv run pytest -q && uv run ruff check .`
  — 40 tests green as of v1.
- **Local quirk (macOS + uv):** after any `uv sync`/`uv add`, `.pth` files in
  the venv can carry the `UF_HIDDEN` flag and Python skips them
  (`ModuleNotFoundError: hoopvision`). Fix:
  `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`. A
  `sitecustomize.py` fallback and `tests/conftest.py` already guard tests.
- **Clips:** `data/clips/` is local-only; re-download via
  `scripts/download_clips.py` using the URLs/timestamps in `data/README.md`.
  Static-window verification method (frame blending + ceiling-strip phase
  correlation) is documented there too.
- **Weights:** fine-tuned YOLO = release v0.2.0 (`hoopvision_best.pt` at repo
  root, gitignored); scratch = release v0.3.0.
- **Deploy:** pushes to `main` auto-redeploy the Streamlit app; entrypoint
  `app/streamlit_app.py`; keep the hosted dependency set inside
  `app/requirements.txt`.
