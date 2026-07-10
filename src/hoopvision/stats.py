"""Per-player movement stats from tracks + court homography.

Once tracks are stable and the court is calibrated, each player's foot point
projects to a court coordinate every frame, so a track becomes a path in feet.
From that we derive distance covered, average and top speed, and a court
occupancy heatmap — the first "advanced stats" and the bridge to game-flow
features (see docs/reference-analysis.md).

Physical units, so the numbers are only meaningful on a **calibrated
fixed-camera clip**. They are per *track*, not per named player (jersey-number
OCR is future work), so a track that is still fragmented is an under-count —
run after appearance stitching. Detection jitter is smoothed and impossible
speeds (detection jumps) are rejected before summing distance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import court as c
from .court import CourtCalibration

MPH_PER_FPS = 0.681818  # feet/second → miles/hour
MAX_HUMAN_FPS = 32.0  # ~22 mph; a segment faster than this is a detection jump


@dataclass
class PlayerStat:
    track_id: int
    team: int | None
    frames: int
    seconds: float
    distance_ft: float
    avg_speed_mph: float
    top_speed_mph: float


def _smooth(points: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average over (N, 2) points; preserves linear motion."""
    n = len(points)
    if n < 3 or window < 2:
        return points
    half = window // 2
    out = np.empty_like(points, dtype=float)
    for i in range(n):
        out[i] = points[max(0, i - half) : min(n, i + half + 1)].mean(axis=0)
    return out


TrackPath = list[tuple[float, tuple[float, float], int | None]]  # (time_s, court_xy_ft, team)


def stats_from_paths(
    paths: dict[int, TrackPath],
    min_frames: int = 15,
    smooth_window: int = 5,
    top_percentile: float = 95.0,
) -> list[PlayerStat]:
    """Distance + speed per track from court-space paths, sorted by distance.

    Coordinate-frame agnostic: `court_xy` may be v1 halfcourt feet (static
    calibration) or v2 full-court feet (per-frame registration) — Euclidean
    distances in feet are the same either way. Callers do their own in-bounds
    filtering before handing paths here.
    """
    out: list[PlayerStat] = []
    for track_id, obs in paths.items():
        if len(obs) < min_frames:
            continue
        times = np.array([o[0] for o in obs])
        court = np.array([o[1] for o in obs], dtype=float)

        smoothed = _smooth(court, smooth_window)
        seg = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
        dt = np.diff(times)
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dt > 0, seg / dt, 0.0)
        ok = (dt > 0) & (speed <= MAX_HUMAN_FPS)  # drop teleports from bad boxes
        if not ok.any():
            continue
        distance = float(seg[ok].sum())
        seconds = float(times[-1] - times[0])
        avg_fps = distance / seconds if seconds > 0 else 0.0
        top_fps = float(np.percentile(speed[ok], top_percentile))
        team = Counter(o[2] for o in obs).most_common(1)[0][0]
        out.append(
            PlayerStat(
                track_id=track_id,
                team=team,
                frames=len(obs),
                seconds=round(seconds, 1),
                distance_ft=round(distance, 1),
                avg_speed_mph=round(avg_fps * MPH_PER_FPS, 1),
                top_speed_mph=round(top_fps * MPH_PER_FPS, 1),
            )
        )
    return sorted(out, key=lambda s: -s.distance_ft)


def player_stats(
    analysis,
    calibration: CourtCalibration,
    min_frames: int = 15,
    smooth_window: int = 5,
    top_percentile: float = 95.0,
) -> list[PlayerStat]:
    """Distance + speed per track (static calibration), sorted by distance."""
    per: dict[int, TrackPath] = defaultdict(list)
    for record in analysis.records:
        t = record.index / analysis.fps
        for p in record.players:
            per[p.track_id].append((t, p.foot, p.team))

    paths: dict[int, TrackPath] = {}
    for track_id, obs in per.items():
        court = calibration.to_court(np.array([o[1] for o in obs]))
        inbounds = calibration.in_bounds(court, margin_ft=2.0)
        kept: TrackPath = [
            (obs[i][0], (float(court[i, 0]), float(court[i, 1])), obs[i][2])
            for i in np.nonzero(inbounds)[0]
        ]
        if kept:
            paths[track_id] = kept
    return stats_from_paths(paths, min_frames, smooth_window, top_percentile)


def court_heatmap(
    analysis,
    calibration: CourtCalibration,
    output_path: str | Path | None = None,
    bins: tuple[int, int] = (50, 47),
    title: str = "Player occupancy",
):
    """Render a court-occupancy heatmap of all in-bounds player foot positions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    from .shotchart import draw_halfcourt

    feet = [p.foot for record in analysis.records for p in record.players]
    court = calibration.to_court(np.array(feet)) if feet else np.empty((0, 2))
    court = court[calibration.in_bounds(court, margin_ft=0.0)] if len(court) else court

    hist, _, _ = np.histogram2d(
        court[:, 0] if len(court) else [],
        court[:, 1] if len(court) else [],
        bins=bins,
        range=[[0, c.COURT_WIDTH_FT], [0, c.COURT_LENGTH_FT]],
    )
    hist = gaussian_filter(hist, sigma=1.2)

    fig, ax = plt.subplots(figsize=(6, 5.8))
    if hist.max() > 0:
        ax.imshow(
            hist.T,
            extent=[0, c.COURT_WIDTH_FT, 0, c.COURT_LENGTH_FT],
            origin="lower",
            cmap="hot",
            alpha=0.7,
            interpolation="bilinear",
        )
    draw_halfcourt(ax)
    ax.set_title(title)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
