"""Split a full broadcast into analysable game-camera segments — task H.

A full NBA broadcast is not one continuous game camera: it cuts to close-ups,
replays, ads, the crowd, and the halftime desk every few seconds. Those frames
have no visible court, so the §4.2 court registration *fails* on them — which
makes registration a free scene classifier: **a frame registers iff it is a
game-camera frame worth analysing.**

The runner does a cheap coarse pass (sample ~1 fps, record whether each sample
registers), then this pure logic groups the registered samples into segments,
bridging brief dropouts (a player occluding the keypoints for a sample) and
discarding fragments too short to yield stable tracks. The expensive per-frame
pipeline then runs only inside these segments, with the tracker/registrar reset
at each boundary (identity cannot survive a cut — only a jersey number can).

Pure and unit-tested; the video/model I/O lives in `scripts/game_report.py`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start_frame: int  # first registered sample (original frame index)
    end_frame: int  # last registered sample
    duration_s: float


def segment_registered(
    samples: list[tuple[int, bool]],
    fps: float,
    min_len_s: float = 8.0,
    max_gap_s: float = 2.0,
) -> list[Segment]:
    """Group registered coarse samples into game-camera segments.

    `samples` is `(frame_index, registered)` from the coarse pass (any order).
    A run of registered samples stays open across dropouts shorter than
    `max_gap_s`; a run is emitted as a Segment only if it spans at least
    `min_len_s`. Boundaries are the first/last *registered* sample frames.
    """
    start: int | None = None  # first registered frame of the open run
    last: int | None = None  # most recent registered frame
    out: list[Segment] = []

    def emit() -> None:
        if start is None or last is None:
            return
        dur = (last - start) / fps
        if dur >= min_len_s:
            out.append(Segment(start, last, round(dur, 1)))

    for frame, ok in sorted(samples):
        if ok:
            if start is None:
                start = frame
            last = frame
        elif start is not None and last is not None and (frame - last) / fps > max_gap_s:
            emit()
            start = last = None
    emit()
    return out


def coverage(samples: list[tuple[int, bool]]) -> float:
    """Fraction of coarse samples that registered (game-camera share of broadcast)."""
    if not samples:
        return 0.0
    return sum(ok for _, ok in samples) / len(samples)


def analysable_seconds(segments: list[Segment]) -> float:
    """Total game-camera time captured by the kept segments."""
    return round(sum(s.duration_s for s in segments), 1)
