"""Download evaluation clips with yt-dlp and cut them to short segments.

$0 tooling: yt-dlp is free; clips stay local (data/ is gitignored — see the
ethics note in data/README.md; never commit or redistribute raw footage).

Usage (one clip, cut 20 s starting at 1:30):
    uv run --with yt-dlp python scripts/download_clips.py \
        "https://www.youtube.com/watch?v=..." --start 90 --duration 20 --name fixed_cam_1

Then document the clip in data/README.md and, for shot-event ground truth,
create data/labels/<name>.csv with rows: time_s,outcome (made|missed).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--name", required=True, help="output name (data/clips/<name>.mp4)")
    parser.add_argument("--start", type=float, default=0.0, help="segment start (seconds)")
    parser.add_argument("--duration", type=float, default=20.0, help="segment length (seconds)")
    parser.add_argument("--max-height", type=int, default=720, help="cap resolution")
    args = parser.parse_args()

    out_dir = Path("data/clips")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.mp4"
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists — pick another --name")

    section = f"*{args.start}-{args.start + args.duration}"
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        f"bestvideo[height<={args.max_height}][ext=mp4]/best[ext=mp4]/best",
        "--download-sections",
        section,
        "--force-keyframes-at-cuts",
        "-o",
        str(out_path),
        args.url,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nSaved {out_path}")
    print("Now: 1) add the clip row to data/README.md (source URL, camera type, held-out?)")
    print("     2) for shot ground truth, create data/labels/" + args.name + ".csv")


if __name__ == "__main__":
    main()
