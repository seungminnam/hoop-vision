"""Per-player identity from jersey numbers — task D (D-4).

v2 §4.3 produces per-*track* court stats, but tracks are anonymous and
fragmented (ByteTrack splinters each player into many ids). This module turns
a read jersey number into a stable identity so fragmented tracks of the same
player can be merged into one, giving a per-*player* box score.

The design follows the reference pipeline (ADR-008, decisions ADR-009):

1. **Match** each detected number box to a player track box by
   *Intersection over Smaller area* (IoS): a number box is tiny and sits
   inside the player box, so IoS ≈ 1 when they belong together, whereas IoU
   would be near 0. Gate at `min_ios` (0.9).
2. **Vote** over time: a track collects (frame, number) reads; a number is
   confirmed only with enough votes and a clear plurality. Voting — not
   per-crop softmax confidence — is what rejects noise, because the classifier
   labels even unreadable crops confidently (see scripts/train_number_classifier).
3. **Merge** tracks that share a confirmed number *and never overlap in time*
   (two tracks on court in the same frame cannot be one player, so an overlap
   means a misread — we keep them separate, conservatively).

Everything here is pure and unit-tested; the video loop, number detection at
native resolution, and cropping/classification live in the runner
(scripts/identify_players.py).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass(frozen=True)
class NumberRead:
    """One number detection in one frame, already classified.

    `xyxy` is in the SAME coordinate space as the track boxes it is matched
    against (the runner is responsible for keeping both in native-resolution
    pixels). `number` is the classifier's string label (e.g. "23", "00").
    """

    frame: int
    xyxy: Box
    number: str


@dataclass(frozen=True)
class TrackBox:
    """One player track box in one frame (native-resolution pixels)."""

    frame: int
    track_id: int
    xyxy: Box


def ios(a: Box, b: Box) -> float:
    """Intersection over the smaller box's area (0..1).

    Preferred over IoU here because a number box is far smaller than a player
    box; when the number sits inside the player, intersection ≈ number area, so
    IoS ≈ 1 while IoU stays tiny.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def match_reads_to_tracks(
    reads: list[NumberRead],
    track_boxes: list[TrackBox],
    min_ios: float = 0.9,
) -> dict[int, list[tuple[int, str]]]:
    """Attribute each number read to the best-overlapping track in its frame.

    Returns per track_id a list of (frame, number) votes. A read that overlaps
    no track box at IoS ≥ `min_ios` is dropped (it belongs to no tracked
    player — e.g. a bench number).
    """
    by_frame: dict[int, list[TrackBox]] = defaultdict(list)
    for tb in track_boxes:
        by_frame[tb.frame].append(tb)

    votes: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for read in reads:
        best_tid, best_ios = None, min_ios
        for tb in by_frame.get(read.frame, ()):
            score = ios(read.xyxy, tb.xyxy)
            if score >= best_ios:
                best_tid, best_ios = tb.track_id, score
        if best_tid is not None:
            votes[best_tid].append((read.frame, read.number))
    return dict(votes)


def confirm_numbers(
    votes: dict[int, list[tuple[int, str]]],
    min_votes: int = 3,
    min_fraction: float = 0.5,
) -> dict[int, str]:
    """Confirm a track's number by plurality vote.

    A track's number is confirmed only if the winning number has at least
    `min_votes` reads and holds at least `min_fraction` of that track's reads.
    Sparse or contradictory reads leave the track unconfirmed (anonymous),
    which is the honest outcome for a track whose number never faces the camera.
    """
    confirmed: dict[int, str] = {}
    for tid, obs in votes.items():
        if len(obs) < min_votes:
            continue
        counts = Counter(num for _, num in obs)
        number, count = counts.most_common(1)[0]
        if count >= min_votes and count / len(obs) >= min_fraction:
            confirmed[tid] = number
    return confirmed


def merge_by_number(
    confirmed: dict[int, str],
    spans: dict[int, tuple[int, int]],
) -> dict[int, int]:
    """Remap {track_id: canonical_id}, merging same-number disjoint tracks.

    Two tracks may be the same player only if they carry the same confirmed
    number and their frame ranges do not overlap (an overlap means two players
    on court at once, so the shared number is a misread — kept separate).
    Within a number, tracks are joined greedily in start order into groups with
    disjoint spans; the canonical id is the smallest member id. Tracks without
    a confirmed number (or absent from `spans`) keep their own id.
    """
    remap = {tid: tid for tid in spans}

    by_number: dict[str, list[int]] = defaultdict(list)
    for tid, number in confirmed.items():
        if tid in spans:
            by_number[number].append(tid)

    for tids in by_number.values():
        # groups: list of [members, group_start, group_end]
        groups: list[list] = []
        for tid in sorted(tids, key=lambda t: spans[t][0]):
            start, end = spans[tid]
            target = None
            for g in groups:
                if start > g[2]:  # disjoint: starts after this group's last end
                    target = g
                    break
            if target is None:
                groups.append([[tid], start, end])
            else:
                target[0].append(tid)
                target[2] = max(target[2], end)
        for members, _, _ in groups:
            canonical = min(members)
            for m in members:
                remap[m] = canonical
    return remap


def identify(
    reads: list[NumberRead],
    track_boxes: list[TrackBox],
    spans: dict[int, tuple[int, int]],
    min_ios: float = 0.9,
    min_votes: int = 3,
    min_fraction: float = 0.5,
) -> tuple[dict[int, int], dict[int, str]]:
    """Full identity pass: match → vote → merge.

    Returns `(remap, canonical_number)` where `remap` sends every track id to
    its canonical id and `canonical_number` gives the confirmed number of each
    canonical id (only for ids that carry one).
    """
    votes = match_reads_to_tracks(reads, track_boxes, min_ios=min_ios)
    confirmed = confirm_numbers(votes, min_votes=min_votes, min_fraction=min_fraction)
    remap = merge_by_number(confirmed, spans)

    canonical_number: dict[int, str] = {}
    for tid, number in confirmed.items():
        canonical_number[remap[tid]] = number
    return remap, canonical_number
