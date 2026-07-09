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
