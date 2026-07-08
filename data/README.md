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

The Hudl auto-tracking camera holds still in two ≥20 s windows (found by
1 fps ceiling-strip phase correlation over the full game, verified by frame
blending). Calibration was recovered from the paint region's segmented
corners, then refined against the visible 3-pt arc / center circle / halfcourt
line; at 360p the paint-corner reprojection error is 1.7 ft (target <1 ft
needs a higher-resolution source — documented limitation).

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

## Ethics / legal

- Clips are used for research/demo only.
- Raw broadcast video is **never** committed or redistributed; the repo only
  contains annotated GIFs/screenshots of short excerpts (`app/samples/`).
- Every source is documented in the table above.
