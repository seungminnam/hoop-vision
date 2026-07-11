# Design decisions (ADR log)

A running log of notable design/plan decisions and **why** they were made — not
just what changed. The goal is that any contributor (or agent) can reconstruct
the reasoning behind a direction, in the spirit of the honesty rule
([ROADMAP.md](../ROADMAP.md) §2).

**When to add an entry.** Any time a spec, roadmap, or plan changes, or a
non-obvious technical fork is taken. Add the entry in the same PR as the change.

**Format.** Context → options considered → decision → rationale → consequences.
Newest last. Status ∈ {accepted, superseded, pending}.

---

## ADR-001 — Court geometry profiles (NBA / NCAA / HS)

- **Date:** 2026-07-09 · **Status:** accepted (PR #22)
- **Context.** v2 needs a court-keypoint dataset. The user asked to broaden it
  across levels (high-school, college, NBA) to fight single-court overfit. But
  `court.py` hardcoded one NBA geometry (16 ft lane, 23.75 ft arc).
- **Options.**
  1. Keep one geometry, feed all courts through it. Rejected: a HS/NCAA court's
     landmarks (paint corners, arc top, corner threes) would map to physically
     wrong court-feet, silently corrupting pseudo-labels — a violation of the
     honesty rule.
  2. Per-court-type geometry profiles. Chosen.
- **Decision.** Add `court.CourtProfile` + `NBA`/`NCAA`/`HIGH_SCHOOL`. Universal
  dims (50×47 court, 19 ft FT line, 6 ft FT circle, 5.25 ft rim) stay module
  constants; only lane width, 3-pt radius, corner-three geometry vary.
  Landmarks a level lacks (the straight corner-three on a pure-arc court) are
  `np.nan` → dropped, rather than faked.
- **Rationale.** Keypoint *pixel detection* benefits from court diversity, but
  the *homography* must use each court's true coordinates. Profiles separate the
  two cleanly and keep the keypoint index contract stable across levels.
- **Consequences.** `auto_calibrate.py --profile` and `build_court_keypoints.py`
  per-source `PROFILE`. NBA reproduces prior values exactly (locked by a test),
  so backward compatible. Enabled ADR-002.

## ADR-002 — Recalibrate `hudl_static2` as an NCAA court

- **Date:** 2026-07-09 · **Status:** accepted (PR #23)
- **Context.** With profiles available, re-fitting `hudl_static2` under each
  profile (same frame/curves) gave refined paint-corner reprojection of
  NBA 2.14 ft / **NCAA 0.87 ft** / HS 1.03 ft. The clip is an NCAA-dimension
  court — v1 had assumed NBA and reported a 1.7 ft "360p limitation."
- **Options.** (1) Leave the NBA calibration and just note the finding.
  (2) Regenerate the calibration + all derived artifacts with the NCAA profile.
  Chosen (2).
- **Decision.** Regenerate `calib_hudl_static2.json` (`--profile ncaa`, 0.87 ft)
  and everything derived: player stats, occupancy heatmap, and the annotated
  minimap video; update README/data-README numbers.
- **Rationale.** The gap was wrong geometry, not resolution — a real, measurable
  accuracy win (reprojection halved; now under the 1 ft W3 target). Leaving the
  wrong-geometry calibration in place would contradict the honesty rule.
- **Consequences.** Player-stat distances dropped ~10–30 % (over-scaled
  homography corrected; longest track 96 → 86 ft over 21 s — more realistic).
  Known residual: `shotchart`/`viz` still *draw* an NBA-dimension court outline,
  so the heatmap has a ~2 ft lane-width cosmetic offset (deferred; profile-aware
  drawing is out of scope for now).

## ADR-003 — Adopt an external multi-venue court-keypoint dataset for v2 §4.2

- **Date:** 2026-07-09 · **Status:** accepted (dataset inspected; see ADR-004)
- **Context.** Broadening the dataset by sourcing our own fixed-camera clips hit
  a discovery bottleneck: headless tools can't watch video to confirm a clip is
  static with visible court lines, and our only calibrated court is one NCAA
  gym. Research (prompted by the user) surfaced public, pre-labeled court
  keypoint datasets under CC BY 4.0 / MIT.
- **Options.**
  - **A. Adopt an external dataset** (e.g. `roboflow-jvuqo/basketball-court-detection-2`,
    used by the MIT `roboflow/sports` repo) as the v2 training backbone; keep
    our §4.1 pseudo-label factory as a complement. Chosen (recommended).
  - **B. Keep our 16-pt schema and manually source + calibrate diverse clips.**
    More control/continuity but slower, and the overfit fix stays gated on
    finding good fixed-camera footage.
- **Decision (recommended, pending verification).** Path A. Download and inspect
  `basketball-court-detection-2` (schema, image count, venue variety, quality)
  before committing to retarget code.
- **Rationale.** Real multi-venue labels kill single-court overfit directly,
  reuse our existing Roboflow tooling (v1 detection dataset), and align with the
  NBA-stats north star (the repo also ships a `basketball-jersey-numbers-ocr`
  dataset that revives the shelved jersey-OCR "D" task → player identification).
- **Consequences / open questions.** The external court-keypoint schema differs
  from our 16-pt `court.COURT_KEYPOINTS`; adopting means retargeting or mapping
  schemas (the keypoint index is a permanent contract, so this is a deliberate
  fork). Download needs `ROBOFLOW_API_KEY` (user's free key, per-command export;
  never committed). Roboflow Universe pages 403 to automated fetch — inspect via
  the API after download.
- **Inspection result (accepted).** `basketball-court-detection-2` v13: **1,220
  images** (train 1006 / val 113 / test 101; ~610 source frames × brightness
  aug), **33 court keypoints** per image, single `court` class, CC BY 4.0. The
  frames are **real NBA playoff broadcast** across **18 games** (Nuggets–Clippers,
  Knicks–Pistons, Timberwolves–Thunder, …), mean 12.5/33 landmarks visible,
  every image usable (≥4). Labels are high quality (points sit on line
  intersections / arc / corners). Caveats: images are **stretched to 640×640**
  (aspect distorted — matters for homography, not detection), it is a
  **full-court** schema (our model was halfcourt), and **no real-world coordinate
  template is published** for the 33 points. This confirms Path A and drives
  ADR-004.

## ADR-004 — Pivot v2 to the 33-point full-court NBA keypoint schema

- **Date:** 2026-07-09 · **Status:** accepted (Phase 1 in progress)
- **Context.** Following ADR-003, the adopted dataset uses a 33-point full-court
  NBA schema that does not match our 16-point halfcourt `court.COURT_KEYPOINTS`.
  We must choose which schema the v2 registration model targets.
- **Options.** (1) Adopt the dataset's 33-point schema. (2) Keep our 16-point
  schema and map the dataset's 33 points onto it. Chosen (1).
- **Decision.** Adopt the 33-point full-court schema for the v2 (NBA) path. Our
  16-point halfcourt schema + §4.1 pseudo-label factory stay as a complement for
  our own amateur/NCAA clips and as the "from-scratch" story; the profile system
  and projection/augmentation utilities are reused, not discarded.
- **Rationale.** Mapping 33→16 would need the very real-world template we don't
  have and would throw away information; training directly on 1,220 labeled NBA
  frames is the shortest path to a detector that works on the north-star domain.
- **Consequences.** **Phase 1** (this PR): train YOLO11-pose on the 33-point
  data (MPS, `fliplr=0` because the 33-point left/right mirror map is unknown —
  a flip would scramble identities); report val/test keypoint metrics. **Phase 2**
  (later): reverse-engineer the 33-point real-world court template + a full-court
  model, then keypoints → RANSAC homography → per-frame NBA registration. The
  640×640 stretch must be undone (or folded into the homography) at inference.

## ADR-005 — Anchor the 33-point template to NBA feet by seed-fit + independent validation

- **Date:** 2026-07-10 · **Status:** accepted (Phase 2 template step)
- **Context.** The 33-point detector (ADR-004, release v0.4.0) is trained, but
  the dataset ships **no real-world coordinates**, so its keypoints can't yet
  drive a homography to court feet. We need a metric template. We confirmed
  `roboflow/sports` has only a *soccer* config (no basketball), so no published
  template exists to reuse — it must be derived.
- **Options.**
  1. **Hand-guess** each point's court coordinate from broadcast frames.
     Rejected: error-prone and unverifiable on a perspective view.
  2. **Recover relative geometry, then seed-fit to feet, then validate against
     all labels.** Chosen. `recover_court_template.py` already placed all 33
     points in one reference frame (chained homographies, 0.73 px median). Fit a
     homography from a confident 15-point seed (both baselines' corners / lane
     edges / 3-ft corner-3 marks + the halfcourt line) to exact NBA feet, let it
     **predict** the other 18, and check each prediction lands on a real court
     feature.
- **Decision.** Path 2. Every non-seed point landed on a real feature (FT
  elbows at 19 ft, arc tops at rim+23.75 ft, corner-3 elbows at 14 ft, the 28-ft
  coaching-box sideline hashes, baskets). The template is stored as **exact NBA
  geometry** (not the noisy fitted values) in
  `hoopvision.court_template.NBA_FULLCOURT_FT`, a new module — the 16-point
  halfcourt schema in `court.py` is untouched (separate permanent contract).
- **Rationale / validation (honesty rule).** The identity assignment is verified
  *independently of how it was derived*: for all **1,220 labeled frames**, fit
  image→feet from the visible points and measure reprojection error — **median
  0.17 ft, p90 0.41 ft** over ~14k observations. Real labels agree with the
  template to ~2 inches, so the coordinates are right, not just internally
  consistent. Reproducible: `scripts/anchor_court_template.py --validate`.
- **Consequences.** The two **basket points (idx 6, 26) are ~10 ft above the
  court plane**, so they break the planar homography (parallax; they validate
  ~1 ft worse). They are flagged `ELEVATED_KEYPOINTS` and excluded from fitting;
  `PLANAR_KEYPOINTS` (31 points) is the homography set. Committed reference:
  `docs/court_template_nba.png`. Remaining Phase-2 step: the per-frame runtime
  (detector keypoints → RANSAC homography over planar points → smoothing), which
  must feed the model 640×640-stretched frames to match training, then map to
  feet (the homography absorbs the anisotropic stretch).
- **Addendum (2026-07-10) — independently confirmed by the dataset authors.**
  After deriving the template we found the dataset's authors (Roboflow / SkalskiP)
  do publish an official NBA court config — in `roboflow/sports` on the
  **`feat/basketball` branch** (`sports/basketball/config.py`), not on `main`
  (soccer only), which is why the search at derivation time found nothing. Their
  vertex order, labels ("01".."41"), basket indices (6/26) and corner indices
  ([0,5,27,32]) match our schema exactly. Reproducing their centimeter presets and
  comparing (`scripts/compare_court_template.py`) gives **mean 0.089 ft, 29/33
  points within 0.1 ft** — our reverse-engineered identities are correct. The
  only real disagreements: the four **sideline coaching-box hashes** (we place
  them at 28 ft, they at 27.4 ft) and the **corner-3 straight elbows** (14.0 vs
  13.91 ft). We keep our values: the label reprojection residual (median ~0.19 ft
  at those hashes; a 0.6 ft error would show) and the NBA rulebook both put the
  coaching-box hash at 28 ft, and 14 ft is the rulebook corner-3 length. This is
  the strongest possible honesty-rule check short of an official template we could
  have reused.

## ADR-006 — Per-frame registration runtime: smooth in image space, gate on confidence

- **Date:** 2026-07-10 · **Status:** accepted (Phase 2 runtime)
- **Context.** With the metric template (ADR-005), each frame's detected
  keypoints can be turned into an image→feet homography. Naively re-fitting every
  frame independently makes the minimap jitter, and frames with few visible
  points produce wild homographies. We need temporal stability and a failure mode.
- **Options for smoothing.**
  1. **EMA the 3×3 homography matrix elements.** Rejected: the matrix entries are
     not commensurate (scale/translation/perspective mix), so element-wise
     averaging warps unpredictably.
  2. **EMA in image space via a canonical court basis.** Chosen. Each frame,
     project a fixed, well-spread set of court points (corners, center, arc tops)
     to image through the new homography, EMA those pixel positions against the
     running estimate, then refit feet↔image from the smoothed positions. Smooths
     a geometrically meaningful quantity (where the court lands on screen).
- **Decision.** `CourtRegistrar`: fit RANSAC over the 31 `PLANAR_KEYPOINTS`
  (baskets excluded — ADR-005), image-space EMA (α=0.35), a **≥4-point gate**,
  and a **last-good fallback** that coasts on the previous homography for up to
  `max_misses` frames before declaring registration "unavailable" — the same
  honesty pattern as v1's ball-coverage gate. Pure geometry, unit-tested;
  detection I/O lives in `scripts/register_court.py`.
- **Rationale / validation.** End-to-end on the held-out NBA test split
  (detector → homography, scored on independent GT pixels): **99% registered,
  median 0.57 ft / p90 1.61 ft** court-position error (`eval_registration.py`).
  The detector adds ~0.4 ft over the template's own 0.17 ft — an honest picture
  of the full pipeline. Inference resizes frames to 640×640 (matching the
  stretched training data); the homography absorbs the anisotropic scale, so no
  separate un-stretch step is needed.
- **Consequences.** Registration is accurate on the camera-facing half and drifts
  where few keypoints are visible (extrapolation) — stated plainly in the README
  and shown in `docs/court_registration_nba.gif`. Next (§4.3): feed these court
  coordinates into the existing shot-chart / stats pipeline (`court.to_court`).
- **Note vs the reference pipeline.** The dataset authors' own notebook
  (see ADR-005 addendum) re-fits an independent homography every frame with no
  smoothing, no elevated-point exclusion, and no low-confidence fallback. The
  `CourtRegistrar` (image-space EMA + planar-only fit + last-good gate) is our
  differentiator over that baseline.

## ADR-007 — §4.3: registration-based player stats in full-court feet (no halfcourt adapter)

- **Date:** 2026-07-10 · **Status:** accepted
- **Context.** §4.2 registers each frame to NBA feet. §4.3 turns that into
  court-coordinate player stats (distance/speed/occupancy) on a *panning*
  broadcast — the thing v1 could only do on a hand-calibrated static clip.
- **Options for the coordinate frame.**
  1. **Map full-court feet → v1's halfcourt frame** (x0..50, y0..47) to reuse
     `court.py` / `shotchart.py` / `stats.player_stats` directly. Rejected:
     a full-court possession spans both halves so a halfcourt frame doesn't fit,
     the mapping needs an ambiguous "which half" decision near center, and it
     would add an adapter with no other consumer this slice (YAGNI).
  2. **Work in full-court feet throughout.** Chosen. Distance/speed are Euclidean
     in feet, *identical* in either frame, so the v1 math is reused by extracting
     a coordinate-frame-agnostic `stats.stats_from_paths` (non-breaking refactor;
     `player_stats` now delegates to it, existing tests unchanged). Occupancy is
     drawn on the full court (`registration.court_polylines_ft`). v1's halfcourt
     modules stay untouched.
- **Decision.** `scripts/registered_stats.py`: per frame, register (§4.2) +
  detect players (v1 `hoopvision_best.pt`) + track (ByteTrack); map each tracked
  player's foot through the homography to full-court feet; accumulate per-track
  paths and feed `stats_from_paths`. Emit per-track distance/speed JSON + a
  full-court occupancy heatmap.
- **Rationale — camera-pan invariance for free.** Court coordinates are
  camera-invariant, so per-frame registration *is* the motion compensation. This
  is the honest win over v1.1's camera-motion estimator (ROADMAP §3.2 / the GMC
  work), which did **not** improve image-space tracking: we don't correct the
  camera, we leave the image-space tracker alone and only transform the foot
  point after association.
- **Validation.** On the panning Grizzlies–Magic clip (30 s, 900 frames):
  **100% of frames registered** (≥ the 0.8 quality gate → analytics reported,
  not "unavailable"), 100 tracks seen / 50 with ≥15 frames; top track **202.6 ft
  travelled, 6.0 mph avg, 16.7 mph top** — realistic for an NBA player over the
  window. Reproduce: `scripts/registered_stats.py`. Committed artifacts:
  `docs/registered_stats_nba.json`, `docs/registered_occupancy_nba.png`.
- **Consequences / honest limits.** Stats are **per track, not per player** —
  30 s of panning + occlusion fragments the ~10 on-court players into ~50 tracks
  (no appearance stitching applied here; that is a static-clip post-process).
  Distances are therefore per-fragment lower bounds. Naming players needs jersey
  OCR (shelved task D; the `basketball-jersey-numbers-ocr` dataset could revive
  it — next backlog item). **Shot events deferred**: 720p broadcast ball/rim
  detection quality is unmeasured, so this slice ships movement stats + occupancy
  and leaves shot charts to a later PR after ball-coverage is measured.

## ADR-009 — D-4: player identity by IoS match + temporal vote + non-overlap merge

- **Date:** 2026-07-10 · **Status:** accepted (D-4)
- **Context.** §4.3 (ADR-007) gives per-*track* court stats, but tracks are
  anonymous and fragmented. D-2 (number detector, AP50 0.970) and D-3 (number
  classifier, test acc 0.955) supply the pieces to name a track. D-4 wires them
  into the §4.3 runner and merges fragments of the same player.
- **Decision — three pure, unit-tested steps** (`src/hoopvision/identity.py`,
  16 tests):
  1. **Match** each number box to a player track by **Intersection over Smaller
     area (IoS) ≥ 0.9**, not IoU: a number box is tiny and sits inside the player
     box, so IoS ≈ 1 for the right pair while IoU stays near 0.
  2. **Vote** over time — a track's number is confirmed only with **≥ 3 reads and
     a ≥ 50% plurality**. Voting, not per-crop softmax confidence, is the noise
     filter, because the classifier labels even unreadable crops confidently
     (D-3 finding). Sparse/contradictory reads leave a track anonymous.
  3. **Merge** same-number tracks **only when their frame ranges are disjoint**
     (two concurrent tracks cannot be one player; an overlap means a misread, so
     they stay separate). Union of disjoint fragments, canonical id = smallest.
- **Coordinate handling.** Player detection + registration run on 640×640
  stretched frames, but number boxes vanish there, so number detection runs on
  the **native frame at imgsz=1280** and crops come from the native frame; the
  native number box is scaled into 640 space to match track boxes for IoS
  (`scripts/identify_players.py`).
- **Options rejected.** (1) *IoU matching* — fails on the tiny-inside-large box
  geometry. (2) *Per-crop confidence gate for "unknown"* — measured useless
  against unreadable crops (D-3). (3) *Merge overlapping same-number tracks* —
  would fuse two different players on a misread; rejected for the conservative
  disjoint-range rule. (4) *Appearance stitching first* (stitch.py) — deferred:
  it would lengthen tracks and lift the read rate, but it is a separate lever;
  D-4 measures the number path alone first (below).
- **Validation — the honest headline is the read rate** (`_nba_raw`, 30 s /
  900 frames, reads every 5th frame). Reproduce: `scripts/identify_players.py`;
  committed artifact `docs/player_identity_nba.json`.
  - Plumbing works: 100% registered, **339 number reads**, 107 tracks → 98 after
    merge, and named tracks carry realistic stats (e.g. #11: 107 ft over 374
    frames, 5.6 mph avg).
  - **But only ~9% of tracks get a confirmed number (8 of ~98)**, and one number
    (#22) is confirmed on **four concurrent tracks** — which cannot be one
    player. The read histogram exposes why: **"22" is read 113 of 339 times
    (33%)**, i.e. the classifier collapses many small, motion-blurred in-game
    numbers onto one class. So the bottleneck is **read *precision* on a 720p
    panning broadcast**, not the matching/voting/merge logic — the classifier was
    trained on curated close crops and over-commits on in-game ones.
- **Consequences.** Ship the pipeline as a **hybrid**: per-*player* where a
  number is confirmed, per-*track* otherwise — an honest, correct-by-construction
  result, not a claim of full box scores. The meta block reports
  `read_rate`, `number_read_histogram`, and `numbers_on_multiple_players` so the
  limitation is visible in the artifact itself. Levers to raise the rate (future
  work, not this slice): appearance stitching before reading, a stricter number
  detector confidence, a higher vote threshold, or an "unreadable" class in the
  classifier so it can abstain. Roster (number→name) mapping stays out of scope.

## ADR-008 — Unblock task D (jersey identity): adopt two NBA datasets, detect+classify

- **Date:** 2026-07-10 · **Status:** accepted (D-1)
- **Context.** §4.3 stats are per *track*, not per *player* — the last gap to the
  north star (named box scores). Jersey OCR was shelved 2026-07-08 as
  "blocked by data" (`reference-analysis.md` §D): our Hudl footage is 360p
  (digits ~15–20 px, illegible) and our pickup footage has no numbers. The
  dataset authors (ADR-005) publish NBA-broadcast datasets that lift exactly that
  constraint.
- **Inspection (reproducible via `scripts/inspect_jersey_datasets.py`).**
  - `basketball-jersey-numbers-ocr` v3 — **text-image-pairs** (a `.jsonl` maps
    each crop to a `suffix` number string, not a class). **3,188 crops**
    (train 2547 / valid 324 / test 317), all **224×224**, **40 number classes**
    (0–77, incl. "00") + 52 empty/unreadable, well balanced (max "8"=269, min=10,
    none < 5 samples), CC BY 4.0.
  - `basketball-player-detection-3-ycjdo` v1 — YOLOv11 detection, **10 classes**,
    411 imgs (285/63/63) at 1280×1280, CC BY 4.0. `number`=2469, `player`=3853,
    `referee`=1128, `ball`=373, `rim`=406; the **action classes are too rare to
    train** (player-in-possession 56, jump-shot 76, layup-dunk 11, shot-block 65,
    ball-in-basket 27). Both are the same NBA-playoff ecosystem as the court
    dataset (Nuggets–Clippers, Knicks–Pistons, …).
- **Decision.** Adopt both. Architecture = **detect the `number` box → crop →
  classify the number** (40-way closed set), mirroring the authors' validated
  path (their ResNet-32 beat a fine-tuned VLM, 93 vs 86%). D-2 trains a number
  detector, D-3 a 224² crop classifier, D-4 wires read → vote → track-merge.
- **Options rejected.** (1) *Classify torso crops directly* (skip detection):
  the number is visible in only a fraction of frames, so a whole-torso classifier
  drowns in numberless views — the authors localize the digit first for a reason.
  (2) *VLM OCR* (SmolVLM2): heavier, and lost to a small classifier on their own
  data; against the $0 / lightweight rule.
- **Consequences / risk (the key finding).** **Number boxes are tiny**:
  median **12.5 × 17.4 px at native 1280×720**, ~6 px wide at 640. The §4.3
  runtime processes 640×640 frames, so number detection must run at **native
  resolution (imgsz≈1280)**, not the 640 the court detector uses — a real
  speed/accuracy trade the D-2/D-4 work must carry. The empty/unreadable class
  maps to "unknown" (abstains from voting). `referee` is a bonus class v1 lacks
  (can exclude refs from player stats). Roster (number→name) mapping is out of
  scope — "player #12" is the target.

## ADR-010 — Court-space track stitching before number reading (task E)

- **Date:** 2026-07-11 · **Status:** accepted (E-1)
- **Context.** ADR-009 measured a ~9% read rate and named it the D-4 headline.
  Two causes: (1) **fragmentation** — ByteTrack splinters ~10 players into 107
  tracks, so per-track reads are too sparse to clear the vote; (2) **read
  precision** — the classifier collapses blurry in-game numbers onto "22". This
  slice attacks (1): stitch fragments into longer tracks *before* voting so a
  player's reads pool.
- **Decision — stitch in court feet, not image pixels.** `stitch.stitch_court`
  reuses the union-find of the v1.1 image-space `stitch()` but replaces the
  spatial gate: instead of "reappears within N box-heights of pixels", it is
  "reappears within `base_ft + max_speed_fps × gap_s` **court feet**". Runs in
  `identify_players.py` before `identify()`; `--no-stitch` gives an honest
  baseline on the same clip.
- **Options rejected.** *Reuse image-space `stitch()` as-is* — its pixel gate is
  meaningless under a panning camera (the same player reappears at an arbitrary
  pixel). Court coordinates are camera-invariant (ADR-007), so the physical
  speed bound is the correct frame — the same "registration is the motion
  compensation" idea, now applied to fragmentation. *Fold stitching into the
  in-frame tracker* — out of scope; the offline post-process is enough here.
- **Validation — `_nba_raw` (30 s), stitch off vs on** (reproduce:
  `identify_players.py --no-stitch` / `--stitch`):

  | metric | off | on |
  |---|---|---|
  | tracks after stitch | 107 | **53** |
  | median votes among read tracks | 4 | **9** |
  | read rate (identified / player-tracks) | 0.075 | **0.113** |
  | tracks identified | 8 | 6 |
  | `numbers_on_multiple_players` | `{22: 4}` | `{22: 3}` |

- **The honest read — a partial win with a clear ceiling.** The mechanism works:
  fragmentation halves (107 → 53) and per-track votes double (median 4 → 9), so
  reads pool exactly as intended, and a *correctly* read player's stats get more
  complete (#11: 374 → 502 frames, 107 → 140 ft). **But it does not unlock new
  distinct players**: both runs are dominated by the bogus "22" (still on 3
  concurrent tracks), and the distinct-number count barely moves. Stitching
  cannot fix a classifier that mislabels — **read *precision*, not fragmentation,
  is now the binding constraint** (sharpening ADR-009). `tracks_identified` even
  dips 8 → 6 as a short 80-frame "10" fragment falls below `min_frames` after the
  cleaner consolidation.
- **Decision on default.** Keep stitching **on**: it makes the *movement* stats
  per-player rather than per-fragment (the headline §4.3 product) and lifts the
  read rate, at no cost but the honest fact that it can't manufacture identities
  the classifier can't read. Raising precision is the separate backlog lever
  (classifier abstain class / in-game-crop fine-tune), not this slice.

## ADR-011 — Surface v2 in the deployed demo via a precomputed NBA sample

- **Date:** 2026-07-11 · **Status:** accepted (F-1)
- **Context.** v2 (registration §4.2, registered stats §4.3, identity §4.4/E) was
  all in the repo but **invisible in the deployed demo** — the live app served
  only v1 fixed-camera samples, and its "How it works" text still said jersey OCR
  needed higher-res footage (pre-task-D). For a portfolio the deployed app is what
  people see, so the strongest story (a *moving* broadcast → physical player stats
  → honest hybrid identity) was the biggest thing missing from view.
- **Decision.** Add a precomputed `app/samples/nba_broadcast/` sample (annotated
  registration GIF, identity `stats.json` with `meta`, occupancy heatmap) built by
  `scripts/build_nba_sample.py` from the already-committed v2 artifacts (each
  regenerable via its source script with `--regen`). Teach `streamlit_app.py` to
  detect the v2 format (`stats.json` has a `meta` block) and render registration +
  read-rate metrics, a per-player table with a **jersey-number column**, and an
  expander that states the ~11% read-rate limit plainly. Default the sample
  selector to `nba_broadcast` so the demo opens on the flagship v2 result.
- **Options rejected.** *Fold auto-registration into `pipeline.py` / the
  "run on my clip" path* — deferred to backlog: the deployed app can't run video
  inference anyway (free-tier limit), so it adds no demo value for real cost; it's
  a v3 local-runner concern. *Hide the low read rate / show only named players* —
  rejected outright: the honesty gate is this project's brand, so the demo shows
  the ~11% and the "#22 on several players" misread in the open. A confidently
  reported weak number reads as a stronger signal than a hidden one.
- **Constraint respected.** `app/streamlit_app.py` keeps its module-level imports
  to **streamlit + stdlib only** (Community Cloud builds from `app/requirements.txt`,
  not the root env) — verified, so the deploy can't break on a heavy import as the
  old `packages.txt` did (see the README deploy note).
- **Validation.** Ran the app locally (headless) on the sample: the video tab shows
  the registration GIF + minimap with 900 frames / 100% registered / 11% read
  rate; the stats tab shows tracks 107 → 53, the hybrid caption, the numbered table
  (#22 …, anonymous rows "—"), and the honest-limit expander. v1 samples still
  render unchanged. Reproduce: `uv run streamlit run app/streamlit_app.py`.
- **Consequences.** The deployed demo now leads with v2. No raw broadcast video is
  committed — only the annotated GIF/PNG/JSON (data-hygiene rule). The
  `pipeline.py` v2 integration and classifier-precision work remain backlog.
