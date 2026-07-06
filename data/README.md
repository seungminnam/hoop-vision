# Data

Everything in `data/` except this file is gitignored — datasets and raw video
stay local.

## Detection fine-tuning dataset

Fill this in when the dataset is downloaded (`scripts/download_data.py`):

| Field | Value |
|---|---|
| Name | TBD (Roboflow Universe, search "basketball players ball rim") |
| URL | TBD |
| License | TBD — record it *before* training |
| Classes used | player, ball, rim |
| Images (train/val/test) | TBD |

## Evaluation clips

5–10 short clips (10–30 s): mix of fixed-camera amateur footage and broadcast
footage, sourced with `yt-dlp`. At least 2 clips are held out as a demo set and
never used for tuning decisions.

| Clip | Source URL | Camera | Held-out? | Ground-truth labels |
|---|---|---|---|---|
| TBD | | fixed | yes | `labels/<clip>.csv` |

Ground-truth shot labels are a simple CSV per clip: `time_s,outcome` with
outcome ∈ {made, missed} — used by W4 precision/recall reporting.

## Ethics / legal

- Clips are used for research/demo only.
- Raw broadcast video is **never** committed or redistributed; the repo only
  contains annotated GIFs/screenshots of short excerpts (`app/samples/`).
- Every source is documented in the table above.
