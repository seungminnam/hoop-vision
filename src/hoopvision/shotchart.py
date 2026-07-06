"""Shot chart rendering on an NBA halfcourt (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle

from . import court as c


def draw_halfcourt(ax: plt.Axes) -> plt.Axes:
    """Draw an NBA halfcourt in feet. Origin: left baseline corner."""
    ax.set_xlim(-1, c.COURT_WIDTH_FT + 1)
    ax.set_ylim(-1, c.COURT_LENGTH_FT + 1)
    ax.set_aspect("equal")
    ax.axis("off")

    line = dict(color="black", lw=1.5, fill=False)
    # Court boundary
    ax.add_patch(Rectangle((0, 0), c.COURT_WIDTH_FT, c.COURT_LENGTH_FT, **line))
    # Paint
    ax.add_patch(
        Rectangle((25 - c.PAINT_HALF_WIDTH, 0), 2 * c.PAINT_HALF_WIDTH, c.FT_LINE_Y, **line)
    )
    # Free-throw circle
    ax.add_patch(Circle((25, c.FT_LINE_Y), c.FT_CIRCLE_RADIUS, **line))
    # Backboard and rim
    ax.plot([22, 28], [c.BACKBOARD_Y, c.BACKBOARD_Y], color="black", lw=2)
    ax.add_patch(Circle(c.RIM_CENTER, c.RIM_RADIUS_FT, color="orange", fill=False, lw=2))
    # Three-point line: corners + arc
    ax.plot([c.CORNER_THREE_X] * 2, [0, c.CORNER_THREE_Y], color="black", lw=1.5)
    ax.plot(
        [c.COURT_WIDTH_FT - c.CORNER_THREE_X] * 2,
        [0, c.CORNER_THREE_Y],
        color="black",
        lw=1.5,
    )
    # Arc spans between the angles where its radius meets the corner-three lines
    theta = np.degrees(np.arccos((25 - c.CORNER_THREE_X) / c.THREE_PT_RADIUS))
    ax.add_patch(
        Arc(
            c.RIM_CENTER,
            2 * c.THREE_PT_RADIUS,
            2 * c.THREE_PT_RADIUS,
            theta1=theta,
            theta2=180 - theta,
            color="black",
            lw=1.5,
        )
    )
    return ax


def render_shot_chart(
    shots: list[dict],
    output_path: str | Path | None = None,
    title: str = "Shot chart",
) -> plt.Figure:
    """Render shots: [{"court_xy": (x_ft, y_ft), "outcome": "made"|"missed"}]."""
    fig, ax = plt.subplots(figsize=(6, 5.8))
    draw_halfcourt(ax)
    made = [s["court_xy"] for s in shots if s["outcome"] == "made"]
    missed = [s["court_xy"] for s in shots if s["outcome"] == "missed"]
    if made:
        xs, ys = zip(*made, strict=True)
        ax.scatter(xs, ys, marker="o", s=90, c="#2e9e4f", label=f"Made ({len(made)})")
    if missed:
        xs, ys = zip(*missed, strict=True)
        ax.scatter(xs, ys, marker="x", s=90, c="#d33c2e", label=f"Missed ({len(missed)})")
    ax.set_title(title)
    if made or missed:
        ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
