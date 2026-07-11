"""Assemble the NBA-broadcast demo sample for the Streamlit app (task F).

The deployed app serves precomputed samples from `app/samples/<name>/`. This
builds `app/samples/nba_broadcast/` from the v2 artifacts so the demo shows the
panning-broadcast story (per-frame court registration → full-court-feet player
stats → jersey-number hybrid identity) that until now lived only in the repo.

By default it **copies the already-committed, reproducible artifacts** (fast and
deterministic); every one of them is itself regenerable by the script named
below, so `--regen` re-runs those to rebuild from the clip (needs the full env
+ MPS, ~10 min, and the gitignored `data/clips/_nba_raw.webm`):

    annotated.gif  <- docs/court_registration_nba.gif   (scripts/register_court.py)
    stats.json     <- docs/player_identity_nba.json      (scripts/identify_players.py)
    heatmap.png    <- docs/registered_occupancy_nba.png  (scripts/registered_stats.py)

No `events.json`: shot analytics are honestly unsupported on this clip (720p
ball/rim coverage unmeasured, ADR-007), and the app already renders that
gracefully.

    uv run python scripts/build_nba_sample.py            # copy committed artifacts
    uv run python scripts/build_nba_sample.py --regen    # rebuild from the clip
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "app/samples/nba_broadcast"

# (destination name, committed source, regen command)
ARTIFACTS = [
    (
        "annotated.gif",
        ROOT / "docs/court_registration_nba.gif",
        [
            sys.executable,
            str(ROOT / "scripts/register_court.py"),
            "--players",
            "--gif",
            "docs/court_registration_nba.gif",
        ],
    ),
    (
        "stats.json",
        ROOT / "docs/player_identity_nba.json",
        [
            sys.executable,
            str(ROOT / "scripts/identify_players.py"),
            "--start",
            "2",
            "--seconds",
            "30",
            "--json",
            "docs/player_identity_nba.json",
        ],
    ),
    (
        "heatmap.png",
        ROOT / "docs/registered_occupancy_nba.png",
        [
            sys.executable,
            str(ROOT / "scripts/registered_stats.py"),
            "--start",
            "2",
            "--seconds",
            "30",
            "--heatmap",
            "docs/registered_occupancy_nba.png",
        ],
    ),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regen", action="store_true", help="rebuild artifacts from the clip first")
    args = p.parse_args()

    if args.regen:
        for _, _, cmd in ARTIFACTS:
            print(f"regen: {' '.join(cmd[1:])}")
            subprocess.run(cmd, cwd=ROOT, check=True)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, source, _ in ARTIFACTS:
        if not source.exists():
            raise SystemExit(f"missing artifact {source} (run with --regen or the source script)")
        shutil.copyfile(source, SAMPLE_DIR / name)
        print(f"{source.name} -> {SAMPLE_DIR / name}")
    print(f"\nnba_broadcast sample ready in {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
