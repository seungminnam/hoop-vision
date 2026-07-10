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
>
> **See also.** [docs/reference-analysis.md](docs/reference-analysis.md)
> compares Hoop Vision to two reference basketball-CV projects and ranks what
> to adopt (appearance tracking, speed/distance stats, camera-motion
> compensation, jersey OCR) against this roadmap.
> [docs/decisions.md](docs/decisions.md) is the ADR log — the *why* behind
> notable design/plan forks (court profiles, the NCAA recalibration, the v2
> external-dataset adoption).

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
| v2 | Dynamic homography | Per-frame court registration so minimap/shot charts work on panning cameras | ◐ §4.2 done (detector 0.985 mAP; template 0.17 ft; registration 0.57 ft); §4.3 registered player stats done (100% on panning clip); shot charts + app integration remain |
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
- ✅ **Labeling tool** — `scripts/label_tracks.py` (OpenCV): bootstraps from
  the tracker's fragmented output, click a box + type a number to relabel a
  whole track (collapses a fragment) or a single box (fixes a swap); undo,
  resume, save to MOTChallenge. Pure edit logic unit-tested.
- ✅ **Ground-truth labels** — `data/labels/mot/gt/pickup_label.txt`: hand-labeled
  10 s / 300-frame window of `pickup_seg3` (9 players, static 1080p). Committed.
  (Crowded 360p footage with benches / refs-as-players was judged not worth
  hand-labeling — see the clip choice rationale in `data/README.md`.)
- ✅ **ByteTrack baseline** — IDF1 **0.730**, 1 ID switch, IDP/IDR 0.585/0.970,
  MOTA 0.341 (dragged by out-of-scope detector FPs) — recorded in README. The
  story: identity is stable on clean footage but fragments across many IDs, so
  the fix target is IDP. Note: HOTA needs `trackeval`; add it if labels justify.
- **Accept:** ✅ baseline in README. Next: an improvement PR that raises IDF1/IDP
  on this GT (§3.2), before/after in the same table.

### 3.2 Association upgrades (one PR each, measured)

- ✅ **Appearance track stitching** (`src/hoopvision/stitch.py`, done). Offline
  post-process: re-attach fragmented tracklets by temporal + spatial +
  torso-color-histogram gates (union-find, disjoint frame ranges). On by
  default. Measured on `pickup_label`: IDF1 0.730 → **0.752**, switches 1 → 0,
  median track life 1.8 s → **4.4 s**, unique IDs 19 → 14, no regression.
1. ◐ **Camera-motion compensation (GMC)** — `src/hoopvision/motion.py` estimates
   the global affine from background optical flow and can track in a stabilised
   reference frame (`analyze(compensate_camera=True)`). The estimator works
   (synthetic-verified; captures the 262 px pan on `hudl_seg1`), but **warping
   boxes did not improve tracking** on the auto-tracking clip (track life
   3.4→3.0 s) — pan correlated with play + cumulative drift; off by default.
   Honest negative result. A stretch is BoT-SORT-style per-frame Kalman
   compensation (needs a tracker that exposes its predict step). The estimator
   is reused for v2 below.
2. **Appearance embedding (in-tracker).** The stitching above is offline; a
   stretch is folding a ReID/OSNet-lite cost into association itself for the
   harder occlusion crossings.
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

### 4.1 Court keypoint dataset (pseudo-labeled, $0) — ✅ done

- ✅ **Keypoint schema** — `court.COURT_KEYPOINTS`: 16 ordered, geometrically
  unambiguous landmarks (baseline/paint/halfcourt corners, FT line + circle,
  3-pt arc top, corner threes, rim, center circle). The index is a permanent
  contract (heatmap channel / annotation column order), so append-only. A
  `FLIP_INDEX` for horizontal-mirror augmentation is derived from court
  geometry, not hand-typed, so it can't drift.
- ✅ **Pseudo-label factory** — `scripts/build_court_keypoints.py` projects the
  schema into each frame of a calibrated static clip via `to_image()` and
  writes a COCO-keypoints dataset. Pure geometry/augmentation lives in
  `src/hoopvision/keypoints.py` (unit-tested, no video I/O).
- ✅ **Augmentation** — random image→image homography warps (virtual
  pans/zooms/tilts), 50% horizontal flips with correct landmark relabeling,
  and brightness/contrast/hue jitter. Off-frame landmarks are dropped
  (visibility flag), matching the ≥4-point homography gate.
- ✅ **Court geometry profiles** — `court.PROFILES` (NBA/NCAA/HS): lane width,
  3-pt radius and corner-three geometry vary by level, so each clip declares
  its profile and the projected landmarks stay metrically correct across court
  types (undefined landmarks, e.g. the straight corner-three on a pure-arc
  court, are dropped). `auto_calibrate.py --profile` uses the same models.
  Bonus: re-fitting `hudl_static2` per profile identifies its court type —
  NBA 2.14 ft / **NCAA 0.87 ft** / HS 1.03 ft refined reprojection, so it is an
  NCAA court and v1's NBA-assumed 1.7 ft used the wrong geometry.
- ✅ **Spot-check** — `docs/court_keypoints_sample.jpg`: projected keypoints on
  a `hudl_static2` frame form a coherent court constellation. First run: 80
  samples from 20 frames (×3 aug), mean 15/16 landmarks visible.
- Dataset image bytes land under gitignored `data/court_kpts/` (regenerable;
  we never commit raw broadcast frames).
- **Next (4.2):** pool more static clips + higher-res sources, then train the
  heatmap model. The generation is $0 and one command per clip.

### 4.2 Keypoint model + per-frame registration

> **Phase 1 done (2026-07-10) — NBA court keypoint detector.** ✅ YOLO11n-pose
> fine-tuned on the 33-point dataset (see below); held-out **NBA test**: keypoint
> mAP50 **0.985** / mAP50-95 0.878 / P·R 0.98·0.98, court-box mAP50 0.995.
> Scripts: `convert_court_coco_to_yolo_pose.py`, `train_court_pose.py`,
> `predict_court_pose.py`; weights [release v0.4.0](https://github.com/seungminnam/hoop-vision/releases/tag/v0.4.0).
> **Phase 2 template done (2026-07-10, [ADR-005](docs/decisions.md)).** ✅ The
> 33-point schema has no published real-world template, so we derived one:
> `recover_court_template.py` places all 33 into one frame (0.73 px median),
> then a 15-point seed-fit anchors it to **exact NBA feet**
> (`hoopvision.court_template.NBA_FULLCOURT_FT`). Validated independently on all
> 1,220 labeled frames — image→feet reprojection **median 0.17 ft / p90 0.41 ft**
> over ~14k point observations (`scripts/anchor_court_template.py --validate`;
> diagram `docs/court_template_nba.png`). Basket points 6/26 are elevated
> (parallax) and excluded from the planar fit. **Remaining (◐):** the per-frame
> runtime — keypoints → RANSAC homography → smoothing → minimap/stats (feed the
> model 640×640-stretched frames to match training; the homography absorbs it).

> **Data strategy update (2026-07-09, [ADR-003](docs/decisions.md)).** Rather
> than train only on our single-court (NCAA) pseudo-labels — which overfit one
> gym — adopt a public **multi-venue court-keypoint dataset** as the training
> backbone (`roboflow-jvuqo/basketball-court-detection-2`, CC BY 4.0, used by
> the MIT `roboflow/sports` repo), keeping the §4.1 pseudo-label factory as a
> complement for our own clips. This directly attacks overfit and aligns with
> the NBA-stats goal (the same repo ships a jersey-number OCR dataset that
> revives the shelved "D" task). Trade-off: the external court-keypoint schema
> differs from our 16-pt `court.COURT_KEYPOINTS`, so adopting it means
> retargeting/mapping the schema — pending dataset inspection.

- Small heatmap model predicting landmark locations + visibility. Two paths:
  fine-tune YOLO11n-pose, or extend the in-repo CenterNet-lite with K heatmap
  channels (nice continuity with the from-scratch chapter). Train on Colab
  free tier or MPS.
- Runtime: keypoints → RANSAC homography per frame → temporal smoothing
  (EMA on reprojected landmarks; fall back to last-good H when <4 confident
  points, exactly like the ball-coverage gate philosophy). Between confident
  keypoint frames, the `motion.py` camera-motion estimator (built in C) can
  carry the homography forward cheaply (pan/zoom form of dynamic registration).
- ✅ **Runtime done (2026-07-10, [ADR-006](docs/decisions.md)).**
  `hoopvision.registration.CourtRegistrar`: predicted keypoints → RANSAC
  homography over the 31 planar points → EMA smoothing (reproject a canonical
  court basis to image, EMA there, refit) → last-good fallback and an "≥4
  points" gate. End-to-end on the held-out **NBA test split** (detector →
  homography, scored on GT pixels): **99% of 101 frames registered, court-
  position error median 0.57 ft / p90 1.61 ft** (`scripts/eval_registration.py`).
- ✅ **Accept (qualitative):** on the moving-camera Grizzlies–Magic broadcast the
  court model reprojects onto the floor as the camera pans, players map to a
  top-down minimap (`scripts/register_court.py`); committed README GIF
  `docs/court_registration_nba.gif`. The far half drifts where few keypoints are
  visible (honest extrapolation limit); the observed half tracks tightly.

### 4.3 Pipeline + product integration

- ✅ **Registered player stats done (2026-07-10, [ADR-007](docs/decisions.md)).**
  `scripts/registered_stats.py` runs detection + tracking + per-frame
  registration and maps each player's foot to full-court NBA feet, so a panning
  broadcast yields the physical stats v1 needed a static calibration for.
  Reuses v1's distance/speed math via a coordinate-frame-agnostic
  `stats.stats_from_paths` (Euclidean feet are the same full/half-court). Court
  coordinates are camera-invariant, so the pan needs no motion compensation —
  the honest win over v1.1's GMC. On the Grizzlies–Magic clip (30 s, 900
  frames): **100% registered** (above the 0.8 analytics gate), 100 tracks /
  50 with ≥15 frames, top track **202.6 ft / 6.0 mph avg / 16.7 mph top**.
  Artifacts: `docs/registered_stats_nba.json`, `docs/registered_occupancy_nba.png`.
- **Honest limits:** stats are per *track* not per player (panning + occlusion
  fragments ~10 players into ~50 tracks; no stitching applied), and **shot
  events are deferred** until 720p ball/rim coverage is measured. Naming players
  needs jersey OCR (shelved task D — the `basketball-jersey-numbers-ocr` dataset
  could revive it).
- Remaining (later): fold `auto` registration into `pipeline.py` / the Streamlit
  app; shot charts once ball coverage justifies them.

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
   - ✅ **Movement stats + demo tab done** (`src/hoopvision/stats.py`,
     `scripts/player_stats.py --json/--heatmap`): per-track distance, avg/top
     speed (mph), and a court occupancy heatmap, surfaced in the Streamlit app's
     "Player stats" tab with a committed `stats.json` + `heatmap.png` sample.
     Remaining slice-1 polish: made/missed timeline, downloadable report bundle.
2. **Auto-highlights.** Cut ±N seconds around each shot event with ffmpeg,
   stitch a highlight reel; "made shots only" toggle. Cheap, high demo value.
3. ◐ **Per-player identity (jersey OCR) — unblocked, in progress.** Was blocked
   by data 2026-07-08 (our footage was 360p or numberless). **Unblocked
   2026-07-10 ([ADR-008](docs/decisions.md))** by two public NBA-broadcast
   datasets (`basketball-jersey-numbers-ocr` 3,188 crops → digits;
   `basketball-player-detection-3-ycjdo` with a `number` class). Plan: detect
   number → crop → classify → vote → merge tracks, upgrading §4.3's per-track
   stats to per-player. Live risk is resolution (number boxes ~12–17 px), not
   data (see docs/reference-analysis.md §D).
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
