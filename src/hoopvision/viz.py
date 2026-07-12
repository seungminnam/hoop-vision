"""Annotated video rendering: boxes, IDs, team colors, minimap overlay."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import court as courtmod
from .detect import Detection
from .track import TrackedPlayer

# BGR palettes
TEAM_COLORS = {0: (244, 133, 66), 1: (66, 66, 244), None: (160, 160, 160)}
BALL_COLOR = (0, 220, 255)
RIM_COLOR = (0, 120, 255)


def annotate_frame(
    frame: np.ndarray,
    players: list[TrackedPlayer],
    ball: Detection | None,
    rim_box: tuple[float, float, float, float] | None,
) -> np.ndarray:
    out = frame.copy()
    for p in players:
        x1, y1, x2, y2 = map(int, p.xyxy)
        color = TEAM_COLORS.get(p.team, TEAM_COLORS[None])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{p.track_id}"
        cv2.putText(
            out, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
        )
    if ball is not None:
        bx, by = map(int, ball.center)
        cv2.circle(out, (bx, by), 7, BALL_COLOR, 2)
    if rim_box is not None:
        x1, y1, x2, y2 = map(int, rim_box)
        cv2.rectangle(out, (x1, y1), (x2, y2), RIM_COLOR, 2)
        cv2.putText(
            out,
            "rim",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            RIM_COLOR,
            1,
            cv2.LINE_AA,
        )
    return out


def render_minimap(
    positions: list[tuple[int, float, float, int | None]],  # (track_id, x_ft, y_ft, team)
    width: int = 250,
) -> np.ndarray:
    """Render a top-down halfcourt with player dots. Feet → minimap pixels."""
    scale = width / courtmod.COURT_WIDTH_FT
    height = int(courtmod.COURT_LENGTH_FT * scale)
    img = np.full((height, width, 3), (43, 33, 24), dtype=np.uint8)  # dark wood

    def px(x_ft: float, y_ft: float) -> tuple[int, int]:
        # y flipped so the baseline (y=0) is at the bottom of the minimap
        return int(x_ft * scale), int(height - y_ft * scale)

    white = (230, 230, 230)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), white, 1)
    # Paint
    cv2.rectangle(
        img,
        px(25 - courtmod.PAINT_HALF_WIDTH, courtmod.FT_LINE_Y),
        px(25 + courtmod.PAINT_HALF_WIDTH, 0),
        white,
        1,
    )
    cv2.circle(img, px(25, courtmod.FT_LINE_Y), int(courtmod.FT_CIRCLE_RADIUS * scale), white, 1)
    # Rim + backboard
    cv2.circle(
        img, px(*courtmod.RIM_CENTER), max(int(courtmod.RIM_RADIUS_FT * scale), 2), (0, 120, 255), 2
    )
    cv2.line(img, px(22, courtmod.BACKBOARD_Y), px(28, courtmod.BACKBOARD_Y), white, 2)
    # Three-point line
    theta = np.degrees(np.arccos((25 - courtmod.CORNER_THREE_X) / courtmod.THREE_PT_RADIUS))
    cv2.line(
        img,
        px(courtmod.CORNER_THREE_X, 0),
        px(courtmod.CORNER_THREE_X, courtmod.CORNER_THREE_Y),
        white,
        1,
    )
    cv2.line(
        img,
        px(courtmod.COURT_WIDTH_FT - courtmod.CORNER_THREE_X, 0),
        px(courtmod.COURT_WIDTH_FT - courtmod.CORNER_THREE_X, courtmod.CORNER_THREE_Y),
        white,
        1,
    )
    cv2.ellipse(
        img,
        px(*courtmod.RIM_CENTER),
        (int(courtmod.THREE_PT_RADIUS * scale), int(courtmod.THREE_PT_RADIUS * scale)),
        0,
        -theta,
        -(180 - theta),
        white,
        1,
    )

    for track_id, x_ft, y_ft, team in positions:
        color = TEAM_COLORS.get(team, TEAM_COLORS[None])
        cv2.circle(img, px(x_ft, y_ft), 5, color, -1)
        cv2.putText(
            img,
            str(track_id),
            px(x_ft + 0.8, y_ft + 0.8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


def overlay_minimap(
    frame: np.ndarray, minimap: np.ndarray, alpha: float = 0.85, margin: int = 12
) -> np.ndarray:
    """Picture-in-picture: minimap in the bottom-left corner of the frame."""
    out = frame.copy()
    mh, mw = minimap.shape[:2]
    fh = out.shape[0]
    y1, y2 = fh - margin - mh, fh - margin
    x1, x2 = margin, margin + mw
    if y1 < 0 or x2 > out.shape[1]:
        return out
    roi = out[y1:y2, x1:x2]
    out[y1:y2, x1:x2] = cv2.addWeighted(minimap, alpha, roi, 1 - alpha, 0)
    return out


class VideoSink:
    """cv2.VideoWriter wrapper (mp4v). Convert to H.264 with ffmpeg for the web."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()

    def __enter__(self) -> VideoSink:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def draw_court_ft(ax) -> None:
    """Draw the NBA full court (feet) onto a matplotlib axis."""
    from .court_template import COURT_LENGTH_FT, COURT_WIDTH_FT
    from .registration import court_polylines_ft

    for poly in court_polylines_ft():
        ax.plot(poly[:, 0], poly[:, 1], color="#555", lw=1, zorder=1)
    ax.set_xlim(-3, COURT_LENGTH_FT + 3)
    ax.set_ylim(-3, COURT_WIDTH_FT + 3)
    ax.set_aspect("equal")
    ax.axis("off")


def render_court_heatmap(points_ft: np.ndarray, out: str | Path, title: str) -> None:
    """Full-court occupancy heatmap from (N, 2) court-feet points.

    Shared by `scripts/registered_stats.py` (whole-clip occupancy) and
    `scripts/game_report.py` (per-player, task H-2). Matplotlib is imported
    lazily so importing `viz` stays cheap for video-only callers.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    from .court_template import COURT_LENGTH_FT, COURT_WIDTH_FT

    fig, ax = plt.subplots(figsize=(9, 5))
    if len(points_ft):
        hist, _, _ = np.histogram2d(
            points_ft[:, 0],
            points_ft[:, 1],
            bins=(94, 50),
            range=[[0, COURT_LENGTH_FT], [0, COURT_WIDTH_FT]],
        )
        hist = gaussian_filter(hist, sigma=1.5)
        ax.imshow(
            hist.T,
            extent=[0, COURT_LENGTH_FT, 0, COURT_WIDTH_FT],
            origin="lower",
            cmap="hot",
            alpha=0.75,
            interpolation="bilinear",
            zorder=0,
        )
    draw_court_ft(ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
