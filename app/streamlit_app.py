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


def show_results(folder: Path) -> None:
    events_file = find_one(folder, ["*events.json"])
    payload = json.loads(events_file.read_text()) if events_file else {}

    video_tab, chart_tab, events_tab, about_tab = st.tabs(
        ["🎬 Annotated video", "🗺️ Shot chart", "📋 Events", "ℹ️ How it works"]
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
            **Pipeline** — YOLO (fine-tuned on player/ball/rim) → ByteTrack IDs →
            jersey-color k-means team assignment → manual 4-point homography →
            trajectory state machine for shot attempts/outcomes → shot chart.

            **Honesty gate** — if the ball track covers <40% of frames, the clip's
            shot analytics are reported *unavailable* instead of guessing.

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
        chosen = st.sidebar.selectbox("Sample clip", samples, format_func=lambda p: p.name)
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
