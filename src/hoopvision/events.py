"""Shot attempt & outcome detection: a heuristic state machine over ball + rim.

Inputs are the per-frame ball center (None when the ball was not detected) and
a static rim box for the clip. Ball gaps up to `max_interpolation_gap` frames
are filled linearly before the state machine runs.

State machine
    IDLE       — waiting for the ball to rise above the rim inside the
                 horizontal attempt window.
    RESOLVING  — an attempt was registered; watch for the ball center to cross
                 downward through the rim interior (MADE). If the timeout
                 expires or the ball leaves the window without crossing: MISS.

Quality gate: if the (interpolated) ball track covers less than
`min_ball_coverage` of frames, shot analytics are reported unavailable rather
than emitting low-confidence events.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShotConfig:
    horizontal_window: float = 1.8  # attempt window, in rim-widths from rim center
    outcome_timeout_frames: int = 60  # frames to resolve an attempt after trigger
    max_interpolation_gap: int = 8  # fill ball gaps up to this many frames
    min_ball_coverage: float = 0.40  # quality gate (fraction of frames with ball)
    cooldown_frames: int = 15  # min frames between attempts


@dataclass(frozen=True)
class ShotEvent:
    frame: int  # frame index of the attempt trigger
    time_s: float
    outcome: str  # "made" | "missed"
    ball_xy: tuple[float, float]  # image px of ball at attempt trigger
    resolved_frame: int  # frame where the outcome was decided


@dataclass
class ShotResult:
    events: list[ShotEvent] = field(default_factory=list)
    ball_coverage: float = 0.0
    available: bool = False
    reason: str = ""


Point = tuple[float, float]


def interpolate_track(centers: list[Point | None], max_gap: int) -> list[Point | None]:
    """Linearly fill None-gaps of length <= max_gap between two known points."""
    out: list[Point | None] = list(centers)
    known = [i for i, c in enumerate(out) if c is not None]
    for a, b in zip(known, known[1:], strict=False):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            (x0, y0), (x1, y1) = out[a], out[b]
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                out[a + step] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return out


def detect_shots(
    ball_centers: list[Point | None],
    rim_box: tuple[float, float, float, float] | None,
    fps: float,
    config: ShotConfig | None = None,
) -> ShotResult:
    config = config or ShotConfig()
    n = len(ball_centers)
    if rim_box is None:
        return ShotResult(available=False, reason="no rim detected")
    if n == 0:
        return ShotResult(available=False, reason="empty ball track")

    track = interpolate_track(ball_centers, config.max_interpolation_gap)
    coverage = sum(c is not None for c in track) / n
    if coverage < config.min_ball_coverage:
        return ShotResult(
            ball_coverage=coverage,
            available=False,
            reason=f"ball track coverage {coverage:.0%} < "
            f"{config.min_ball_coverage:.0%} quality gate",
        )

    rx1, ry1, rx2, ry2 = rim_box
    rim_cx = (rx1 + rx2) / 2
    rim_cy = (ry1 + ry2) / 2
    rim_w = max(rx2 - rx1, 1.0)
    window = config.horizontal_window * rim_w

    events: list[ShotEvent] = []
    state = "IDLE"
    attempt_frame = -1
    attempt_xy: Point = (0.0, 0.0)
    cooldown_until = -1
    prev: Point | None = None

    for i, cur in enumerate(track):
        if cur is None:
            prev = None
            continue
        bx, by = cur
        in_window = abs(bx - rim_cx) <= window
        above_rim_top = by < ry1  # image y grows downward

        if state == "IDLE":
            if in_window and above_rim_top and i >= cooldown_until:
                state = "RESOLVING"
                attempt_frame = i
                attempt_xy = cur
        elif state == "RESOLVING":
            made = (
                prev is not None
                and prev[1] <= rim_cy < by  # crossed rim center plane downward
                and rx1 <= bx <= rx2  # inside rim interior at the crossing
            )
            timed_out = i - attempt_frame > config.outcome_timeout_frames
            left_window = not in_window and by > rim_cy  # descended outside rim
            if made or timed_out or left_window:
                events.append(
                    ShotEvent(
                        frame=attempt_frame,
                        time_s=attempt_frame / fps,
                        outcome="made" if made else "missed",
                        ball_xy=attempt_xy,
                        resolved_frame=i,
                    )
                )
                state = "IDLE"
                cooldown_until = i + config.cooldown_frames
        prev = cur

    # An attempt still unresolved at end-of-clip counts as a miss (ball never
    # passed through the rim interior on camera).
    if state == "RESOLVING":
        events.append(
            ShotEvent(
                frame=attempt_frame,
                time_s=attempt_frame / fps,
                outcome="missed",
                ball_xy=attempt_xy,
                resolved_frame=n - 1,
            )
        )

    return ShotResult(events=events, ball_coverage=coverage, available=True)
