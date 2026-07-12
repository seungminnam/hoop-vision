"""Cross-segment per-number aggregation — task H design ②.

Inside a broadcast segment the pipeline stitches and votes as usual; *between*
segments appearance and coordinates are void (camera cut, substitutions, time
passing), so **the only key that crosses a cut is a confirmed jersey number**.
This module folds every segment's player rows into one tracking box score per
number: game-camera seconds, distance, average speed (recomputed from the
totals, not averaged averages), and top speed (max).

Tracks that never got a number stay an anonymous residual — reported, not
hidden, because the cumulative-identification hypothesis (a player only needs
to be read once anywhere in the game for that segment to attach) is exactly
what task H measures.

Pure logic, unit-tested; `scripts/game_report.py` supplies the rows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentPlayerRow:
    """One (possibly anonymous) player in one analysed segment."""

    segment: int
    track_id: int
    number: str | None
    seconds: float
    distance_ft: float
    avg_speed_mph: float
    top_speed_mph: float


@dataclass(frozen=True)
class NumberTotals:
    """One player's tracking box score, accumulated across segments by number."""

    number: str
    segments: int  # segments this number was confirmed in
    seconds: float  # total game-camera time
    distance_ft: float
    avg_speed_mph: float  # total distance / total time (not a mean of means)
    top_speed_mph: float  # max across segments


MPH_PER_FPS = 0.681818  # feet/second -> miles/hour (matches stats.py)


def aggregate_by_number(
    rows: list[SegmentPlayerRow],
) -> tuple[list[NumberTotals], dict]:
    """Fold segment rows into per-number totals + an honesty meta block.

    Returns (totals sorted by seconds desc, meta). meta reports the anonymous
    residual: how many rows/seconds/feet never attached to a number, and the
    identified fraction of tracked time — the headline honesty number for the
    cumulative-identification hypothesis.
    """
    per: dict[str, list[SegmentPlayerRow]] = {}
    anon_rows = 0
    anon_seconds = 0.0
    anon_distance = 0.0
    for r in rows:
        if r.number is None:
            anon_rows += 1
            anon_seconds += r.seconds
            anon_distance += r.distance_ft
        else:
            per.setdefault(r.number, []).append(r)

    totals: list[NumberTotals] = []
    for number, group in per.items():
        seconds = sum(g.seconds for g in group)
        distance = sum(g.distance_ft for g in group)
        avg_mph = (distance / seconds) * MPH_PER_FPS if seconds > 0 else 0.0
        totals.append(
            NumberTotals(
                number=number,
                segments=len({g.segment for g in group}),
                seconds=round(seconds, 1),
                distance_ft=round(distance, 1),
                avg_speed_mph=round(avg_mph, 1),
                top_speed_mph=round(max(g.top_speed_mph for g in group), 1),
            )
        )
    totals.sort(key=lambda t: -t.seconds)

    ided_seconds = sum(t.seconds for t in totals)
    total_seconds = ided_seconds + anon_seconds
    meta = {
        "identified_numbers": len(totals),
        "identified_rows": len(rows) - anon_rows,
        "anonymous_rows": anon_rows,
        "anonymous_seconds": round(anon_seconds, 1),
        "anonymous_distance_ft": round(anon_distance, 1),
        "identified_time_fraction": round(ided_seconds / total_seconds, 3)
        if total_seconds > 0
        else 0.0,
    }
    return totals, meta
