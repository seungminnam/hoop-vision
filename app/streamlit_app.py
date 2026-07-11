"""Hoop Vision demo app (Streamlit).

Two modes:
  1. Explore precomputed sample results bundled in app/samples/<clip>/
     (this is what the free Streamlit Community Cloud deployment serves —
     running YOLO over video exceeds free-tier limits, see README).
  2. Run the pipeline locally on an uploaded clip (requires the full env).

Run locally:  uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SAMPLES_DIR = Path(__file__).parent / "samples"

st.set_page_config(page_title="Hoop Vision", page_icon="🏀", layout="wide")
st.title("🏀 Hoop Vision — basketball video analytics")
st.caption(
    "Player detection & tracking → team assignment → court homography minimap "
    "→ shot detection → shot chart. [Source](https://github.com/seungminnam/hoop-vision)"
)


def list_samples() -> list[Path]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir())


def find_one(folder: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        hits = sorted(folder.glob(pattern))
        if hits:
            return hits[0]
    return None


def _v2_player_stats(players: list[dict], meta: dict) -> None:
    """Render the v2 registered/identity stats: registration + read-rate metrics,
    a per-player table with jersey numbers, and an honest read-rate caveat."""
    cols = st.columns(3)
    cols[0].metric("Court registered", f"{meta.get('registration_rate', 0):.0%}")
    cols[1].metric(
        "Tracks (raw → stitched)",
        f"{meta.get('tracks_seen', '—')} → {meta.get('tracks_after_stitch', '—')}",
    )
    cols[2].metric("Jersey read rate", f"{meta.get('read_rate', 0):.0%}")

    st.caption(
        "Physical stats (feet) on a **panning broadcast** — court coordinates are "
        "camera-invariant, so per-frame registration handles the pan. Rows with a "
        "jersey number are per-**player** (fragments stitched, number voted); "
        "anonymous rows are still per-track."
    )
    rows = [
        {
            "number": f"#{p['number']}" if p.get("number") else "—",
            "track": p["track_id"],
            "frames": p["frames"],
            "distance_ft": p["distance_ft"],
            "avg_speed_mph": p["avg_speed_mph"],
            "top_speed_mph": p["top_speed_mph"],
        }
        for p in players
    ]
    st.dataframe(rows, width="stretch")

    dupes = meta.get("numbers_on_multiple_players") or {}
    with st.expander("Why is the read rate only ~10%? (honest limit)"):
        st.markdown(
            f"""
            Reading jersey numbers on a **720p broadcast** is hard: the number
            detector (AP50 0.97) and classifier (acc 0.96) are strong on curated
            close crops, but small, motion-blurred in-game numbers collapse onto a
            few classes. Here the classifier reads **{meta.get("number_reads", 0)}
            numbers** but confirms only **{meta.get("tracks_identified", 0)}**
            players{
                f", and #{next(iter(dupes))} lands on {next(iter(dupes.values()))} "
                "different players at once (a misread, so those tracks stay separate)"
                if dupes
                else ""
            }.

            The bottleneck is read **precision**, not the matching/voting/merge
            logic — so this ships as an honest **hybrid** (named where read,
            per-track otherwise) rather than a fake full box score.
            """
        )


def show_results(folder: Path) -> None:
    events_file = find_one(folder, ["*events.json"])
    payload = json.loads(events_file.read_text()) if events_file else {}
    stats_file = find_one(folder, ["*stats.json"])
    stats = json.loads(stats_file.read_text()) if stats_file else {}
    meta = stats.get("meta")  # present only on v2 (registered / identity) samples

    video_tab, chart_tab, stats_tab, events_tab, about_tab = st.tabs(
        ["🎬 Annotated video", "🗺️ Shot chart", "📊 Player stats", "📋 Events", "ℹ️ How it works"]
    )

    with video_tab:
        media = find_one(folder, ["*annotated*.gif", "*annotated*.mp4", "*.gif", "*.mp4"])
        if media is None:
            st.info("No annotated video in this sample folder.")
        elif media.suffix == ".gif":
            st.image(str(media))
        else:
            st.video(str(media))
        if payload:
            cols = st.columns(3)
            cols[0].metric("Frames processed", payload.get("frames_processed", "—"))
            cols[1].metric("Ball track coverage", f"{payload.get('ball_coverage', 0):.0%}")
            cols[2].metric(
                "Shot analytics",
                "available" if payload.get("shot_analytics_available") else "unavailable",
            )
        elif meta:  # v2 panning-broadcast sample: no shot events, show registration
            cols = st.columns(3)
            cols[0].metric("Frames processed", meta.get("frames_processed", "—"))
            cols[1].metric("Court registered", f"{meta.get('registration_rate', 0):.0%}")
            cols[2].metric("Jersey read rate", f"{meta.get('read_rate', 0):.0%}")

    with chart_tab:
        chart = find_one(folder, ["*shotchart*.png"])
        if chart is not None:
            st.image(str(chart), width=520)
        elif payload and not payload.get("shot_analytics_available"):
            st.warning(
                f"Shot analytics unavailable for this clip: "
                f"{payload.get('reason', 'unknown')} (quality gate — the pipeline "
                "refuses to emit low-confidence events)."
            )
        else:
            st.info("No shot chart for this sample.")

    with stats_tab:
        heatmap = find_one(folder, ["*heatmap*.png"])
        players = stats.get("players", [])
        if players and meta:
            _v2_player_stats(players, meta)
        elif players:
            st.caption(
                "Per-track distance and speed in physical units (homography → feet). "
                "Per track, not per named player — short tracks are still fragments."
            )
            st.dataframe(players, width="stretch")
            left, right = st.columns(2)
            top = max(players, key=lambda p: p["distance_ft"])
            fastest = max(players, key=lambda p: p["top_speed_mph"])
            left.metric("Most distance", f"{top['distance_ft']:.0f} ft", f"track {top['track_id']}")
            right.metric(
                "Top speed", f"{fastest['top_speed_mph']:.1f} mph", f"track {fastest['track_id']}"
            )
        else:
            st.info("No player stats for this sample (needs a calibrated fixed-camera clip).")
        if heatmap is not None:
            st.image(str(heatmap), caption="Court occupancy", width=460)

    with events_tab:
        events = payload.get("events", [])
        if events:
            st.dataframe(events, width="stretch")
            made = sum(e["outcome"] == "made" for e in events)
            st.caption(f"{len(events)} attempts · {made} made · {len(events) - made} missed")
        else:
            st.info("No shot events recorded for this clip.")

    with about_tab:
        st.markdown(
            """
            **v1 pipeline (fixed camera)** — YOLO (fine-tuned on player/ball/rim) →
            ByteTrack IDs + appearance track stitching → jersey-color k-means team
            assignment → homography → trajectory state machine for shot
            attempts/outcomes, plus per-player distance/speed and an occupancy heatmap.

            **v2 pipeline (panning broadcast)** — a YOLO11-pose model finds 33 court
            keypoints per frame → RANSAC homography registered to full-court NBA feet
            (smoothed, last-good fallback), so a moving camera still yields physical
            player stats. A jersey-number detector + classifier then reads numbers and
            merges track fragments into per-player identities. Try the **nba_broadcast**
            sample.

            **Honesty gate** — shot analytics are withheld when ball coverage is too
            low (<40%); jersey read rate is reported as-is (~10% on a 720p pan — read
            precision, not the logic, is the limit), so stats are an honest hybrid:
            per-player where a number is confirmed, per-track otherwise.

            **$0 stack** — Colab/Kaggle free GPUs (training), Roboflow Universe
            (dataset), Streamlit Community Cloud (this app), GitHub (repo + CI).
            """
        )


mode = st.sidebar.radio("Mode", ["Sample results", "Run on my clip (local only)"])

if mode == "Sample results":
    samples = list_samples()
    if not samples:
        st.info(
            "No precomputed samples yet. Generate them with\n\n"
            "`uv run python -m hoopvision.pipeline clip.mp4 --calibration calib.json "
            "--output app/samples/<clip-name>`\n\nthen commit small GIF/JSON/PNG "
            "outputs (never raw broadcast video — see data/README.md)."
        )
    else:
        _names = [p.name for p in samples]
        _default = _names.index("nba_broadcast") if "nba_broadcast" in _names else 0
        chosen = st.sidebar.selectbox(
            "Sample clip", samples, index=_default, format_func=lambda p: p.name
        )
        show_results(chosen)
else:
    st.sidebar.warning(
        "Video inference exceeds Streamlit's free-tier resources; run this mode "
        "on your own machine."
    )
    upload = st.file_uploader(
        "Upload a short clip (10–30 s, fixed camera works best)", type=["mp4", "mov"]
    )
    weights = st.text_input("YOLO weights", "yolo11n.pt")
    stride = st.slider("Frame stride (higher = faster, coarser)", 1, 6, 2)
    if upload is not None and st.button("Run pipeline"):
        try:
            from hoopvision.pipeline import run
        except ImportError as err:
            st.error(f"Pipeline dependencies missing: {err}")
            st.stop()
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / upload.name
            clip.write_bytes(upload.getvalue())
            out = Path(tmp) / "out"
            with st.spinner("Running detection + tracking…"):
                run(clip, weights=weights, output_dir=out, stride=stride)
            st.success("Done")
            show_results(out)
