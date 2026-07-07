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

Ground-truth shot labels are a simple CSV per clip: `time_s,outcome` with
outcome ∈ {made, missed} — used by W4 precision/recall reporting.

## Ethics / legal

- Clips are used for research/demo only.
- Raw broadcast video is **never** committed or redistributed; the repo only
  contains annotated GIFs/screenshots of short excerpts (`app/samples/`).
- Every source is documented in the table above.
