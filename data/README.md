# Data

Everything in `data/` except this file is gitignored — datasets and raw video
stay local.

## Detection fine-tuning dataset

Fill this in when the dataset is downloaded (`scripts/download_data.py`):

| Field | Value |
|---|---|
| Name | basketball-computer-vision v14 |
| URL | https://universe.roboflow.com/basketballcomputervision/basketball-computer-vision/dataset/14 |
| License | Public Domain |
| Classes used | player, ball, rim |
| Images (train/val/test) | 235 (165/46/24) |

## Evaluation clips

5–10 short clips (10–30 s): mix of fixed-camera amateur footage and broadcast
footage, sourced with `yt-dlp`. At least 2 clips are held out as a demo set and
never used for tuning decisions.

| Clip | Source URL | Camera | Held-out? | Ground-truth labels |
|---|---|---|---|---|
| hudl_seg1 (t=1115–1140s) | https://www.youtube.com/watch?v=fqlN0rmPpbE | auto-tracking (Hudl) — pans slowly; unsuitable for homography | no | — (ball coverage 2% at 360p; shot analytics gated off) |
| hudl_seg2 (t=300–325s) | https://www.youtube.com/watch?v=fqlN0rmPpbE | auto-tracking (Hudl) | no | — |
| pickup_seg1 (t=527–552s) | https://www.youtube.com/watch?v=2wpMTCfsGDc | fixed (verified: frame-blend over 92 s) | no | `labels/pickup_seg1.csv` |
| pickup_seg2 (t=555–580s) | https://www.youtube.com/watch?v=2wpMTCfsGDc | fixed | no | `labels/pickup_seg2.csv` |
| pickup_seg3 (t=585–615s) | https://www.youtube.com/watch?v=2wpMTCfsGDc | fixed | yes (held-out demo) | `labels/pickup_seg3.csv` |

Note: the source video repositions its camera a few times between plays; the
segments above were cut from a 96-second window (t=525–621s) verified static
by ceiling-strip phase correlation + first/last frame blending.

**Homography caveat (pickup clips):** this pickup game is played *cross-court*
on a side basket — the painted floor lines belong to the main court (rotated
90°), so the played halfcourt has no usable landmarks. These clips support
detection/tracking/teams/shot-events but not court calibration; the W3
minimap uses a clip on a properly lined court instead.

| Clip | Source URL | Camera | Held-out? | Notes |
|---|---|---|---|---|
| hudl_static1 (t=599–619s) | https://www.youtube.com/watch?v=fqlN0rmPpbE | static window (verified) | no | lined court; W3 minimap |
| hudl_static2 (t=1616–1637s) | https://www.youtube.com/watch?v=fqlN0rmPpbE | static window (verified) | no | lined court; calibration `calib_hudl_static2.json` |
| _nba_raw (t=1800–1840s) | https://www.youtube.com/watch?v=J8WABIinM64 (Grizzlies vs Magic full game) | broadcast — pans, one cut at 37 s | no | v1 out-of-domain detection test; **v2 §4.2 court-registration demo** (`scripts/register_court.py`) and **§4.3 registered player stats** (`scripts/registered_stats.py`; the 30 s window before the 37 s cut) |

The Hudl auto-tracking camera holds still in two ≥20 s windows (found by
1 fps ceiling-strip phase correlation over the full game, verified by frame
blending). Calibration was recovered from the paint region's segmented
corners, then refined against the visible 3-pt arc / center circle / halfcourt
line; with the correct NCAA court profile the paint-corner reprojection error
is **0.87 ft** (under the 1 ft target). v1 assumed NBA dimensions and got
1.7 ft — the gap was wrong court geometry, not resolution (see the court-profile
section below).

Ground-truth shot labels are a simple CSV per clip: `time_s,outcome` with
outcome ∈ {made, missed} — used by W4 precision/recall reporting.

## Tracking (MOT) labels — `labels/mot/`

For the v1.1 tracking work (see [../ROADMAP.md](../ROADMAP.md)):

- `labels/mot/gt/<clip>.txt` — hand-labeled player-ID ground truth,
  MOTChallenge CSV: `frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1`,
  `frame` = 1-based processed-frame ordinal. Committed (precious, hand-made).
- `labels/mot/pred/<clip>.txt` — tracker predictions, same format, produced by
  `scripts/track_diagnostics.py --dump-mot`. **Gitignored** (regenerable, tied
  to the current weights).

`scripts/eval_tracking.py` scores predictions against ground truth (IDF1,
MOTA, ID switches). Ground-truth labeling is pending — until then,
`scripts/track_diagnostics.py` reports unsupervised tracking-health proxies
that need no labels.

To make the `gt/` files, use `scripts/label_tracks.py` (OpenCV): it bootstraps
from the tracker's predictions and lets you relabel a whole track with one
keystroke (the fragmentation fix), a single box (a swap fix), or remove a
track entirely (a referee / bystander / spurious box, key `r`), then saves
MOTChallenge GT.

The recommended substrate is a short window of a **static, high-res** clip —
crowded 360p footage (benches, refs misdetected as players) is painful to
label and yields noisy GT. `pickup_label.mp4` is a 10 s / 300-frame window of
`pickup_seg3` (static 1080p pickup game, ~6 players, no refs); the tracker
fragments it into ~19 IDs, so it is ~13 track-merges of work:

```bash
uv run python scripts/label_tracks.py data/clips/pickup_label.mp4 \
    --boxes data/labels/mot/pred/pickup_label.txt
# regenerate that clip: ffmpeg -ss 5 -i data/clips/pickup_seg3.mp4 -t 10 \
#   -c:v libx264 -crf 18 -an data/clips/pickup_label.mp4
```

## Court keypoints (v2) — `court_kpts/`

For v2 dynamic homography (see [../ROADMAP.md](../ROADMAP.md) §4.1). A
calibrated static clip is a free keypoint annotator: `scripts/build_court_keypoints.py`
projects the fixed `court.COURT_KEYPOINTS` schema (16 landmarks) into each
frame via the homography and augments with random warps + flips + color jitter,
producing a COCO-keypoints dataset for training a per-frame registration model.

```bash
uv run python scripts/build_court_keypoints.py \
    --source data/clips/hudl_static2.mp4 calib_hudl_static2.json ncaa \
    --output data/court_kpts --stride 15 --augment 4 \
    --overlay docs/court_keypoints_sample.jpg
# pool clips of different levels by repeating --source CLIP CALIB PROFILE
```

**Court geometry profiles.** Court dimensions vary by level (NBA 16 ft lane /
23.75 ft arc, NCAA 12 ft / 22.15 ft, HS 12 ft / 19.75 ft), so each clip
declares its profile (`court.PROFILES`); the projected landmarks use that
level's coordinates and any landmark a level doesn't have (e.g. the straight
corner-three segment on a pure-arc court) is dropped. `auto_calibrate.py
--profile` uses the same models. Empirically this also identifies a clip's
court type: re-fitting `hudl_static2` (same frame/curves) gives NBA 2.14 ft /
**NCAA 0.87 ft** / HS 1.03 ft refined reprojection — it is an NCAA-dimension
court, so v1's NBA-assumed 1.7 ft was the wrong geometry. `calib_hudl_static2.json`
and its `app/samples` stats/heatmap have been regenerated with `--profile ncaa`
(the player-stat distances dropped ~10–30 % as the over-scaled homography was
corrected).

`court_kpts/` (images + `annotations.json`) is **gitignored** — regenerable,
tied to the calibrations, and we never commit raw broadcast frames. The only
committed artifact is `docs/court_keypoints_sample.jpg` (one annotated
excerpt, spot-check). Pseudo-labels are only as accurate as the seed
calibration (0.87 ft with the NCAA profile on `hudl_static2`); add more
correctly-lined sources of other levels before trusting the trained model off
this court.

## NBA court keypoints (v2 §4.2) — external dataset

The v2 registration model trains on a public, pre-labeled dataset instead of
only our single-court pseudo-labels (see [../docs/decisions.md](../docs/decisions.md)
ADR-003/004). This is real multi-venue NBA data — the north-star domain.

| Field | Value |
|---|---|
| Name | basketball-court-detection-2 (v13) |
| URL | https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2 |
| License | CC BY 4.0 |
| Images | 1,220 (train 1006 / val 113 / test 101; ~610 source frames ×2 brightness aug) |
| Content | real NBA playoff broadcast, 18 games; 33 court keypoints per image |
| Format | COCO keypoints, resized 640×640 (stretch) |

Download (needs a free `ROBOFLOW_API_KEY`, per-command export, never committed):

```bash
export ROBOFLOW_API_KEY=...
uv run --with roboflow python scripts/download_data.py \
    --workspace roboflow-jvuqo --project basketball-court-detection-2 \
    --version 13 --format coco
uv run python scripts/convert_court_coco_to_yolo_pose.py   # → data/court_pose/
uv run python scripts/train_court_pose.py --epochs 100 --device mps
# Phase 2 (registration): validate the derived NBA feet template, score the
# end-to-end registration, and render the moving-camera demo
uv run python scripts/anchor_court_template.py --validate    # template: median 0.17 ft
uv run python scripts/eval_registration.py                   # end-to-end: median 0.57 ft
uv run python scripts/register_court.py --players --gif docs/court_registration_nba.gif
```

The raw dataset (`data/basketball-court-detection-2-13/`) and the YOLO-pose
conversion (`data/court_pose/`) are **gitignored** — regenerable from the
commands above; we never commit the frames. Trained weights ship as a GitHub
release, not in git. Known caveats: images are stretched to 640×640 (aspect
distorted, but the homography absorbs it — matched at inference by resizing to
640×640); the 33-point schema had no published real-world template, so Phase 2
derived + validated one (`hoopvision.court_template`, [decisions ADR-005](../docs/decisions.md)).

## Ethics / legal

- Clips are used for research/demo only.
- Raw broadcast video is **never** committed or redistributed; the repo only
  contains annotated GIFs/screenshots of short excerpts (`app/samples/`).
- Every source is documented in the table above.
