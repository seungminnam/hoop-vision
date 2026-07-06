"""Download a basketball detection dataset from Roboflow Universe (free tier).

1. Create a free account at https://universe.roboflow.com and pick a dataset
   with player/ball/rim classes (search "basketball players ball rim").
2. Get your API key from https://app.roboflow.com/settings/api.
3. Run (the `roboflow` package is only needed here, not by the pipeline):

    export ROBOFLOW_API_KEY=...
    uv run --with roboflow python scripts/download_data.py \
        --workspace <workspace> --project <project> --version <n>

The dataset lands in data/<project>-<version>/ in YOLO format. Record the
dataset name, URL, license, and image counts in data/README.md.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--format", default="yolov11", help="export format")
    parser.add_argument("--output", default="data")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY (free key: app.roboflow.com/settings/api)")

    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit(
            "roboflow package missing. Run via: uv run --with roboflow "
            "python scripts/download_data.py ..."
        ) from None

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    target = Path(args.output) / f"{args.project}-{args.version}"
    dataset = version.download(args.format, location=str(target))
    print(f"Downloaded to {dataset.location}")
    print("Now document name/URL/license/image counts in data/README.md")


if __name__ == "__main__":
    main()
