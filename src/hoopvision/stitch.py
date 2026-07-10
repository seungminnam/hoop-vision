"""Offline tracklet stitching: re-attach fragmented tracks by appearance.

ByteTrack is motion-only, so an occlusion or a missed detection ends a track
and the player reappears under a new id — the fragmentation measured on the
demo clips (IDP 0.585 on `pickup_label`: identity recall is high but the
tracker splinters each player across many short ids).

Because the pipeline is already two-pass and offline, we can stitch after the
fact: a tracklet that *ends* and another that *begins* a few frames later,
near the same place, with a similar torso color, are almost certainly the same
player. We link them with three gates:

- **temporal** — the successor starts after the predecessor ends, within a
  small frame gap (they must not overlap in time; overlapping = two players);
- **spatial** — the reappearance is within a few body-heights of the
  disappearance (scale-aware, since near players are bigger);
- **appearance** — the torso color histograms are similar (cosine).

Union-find merges chains (A→B→C) while keeping every merged group's frame
range disjoint, so two players who are on court at the same time can never
collapse into one id.

This is a tracker-agnostic post-process; it consumes tracklets and returns an
id remap, so it is pure and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .teams import color_histogram, torso_crop


@dataclass
class Tracklet:
    track_id: int
    start_frame: int  # first processed-frame ordinal the track appears
    end_frame: int  # last
    start_xy: tuple[float, float]  # box center at start
    end_xy: tuple[float, float]  # box center at end
    start_h: float  # box height at start (px, for scale)
    end_h: float
    feature: np.ndarray  # L2-normalized appearance histogram


def _center(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class TrackletBuilder:
    """Accumulate per-track geometry + appearance during the pipeline's pass 1."""

    _first: dict[int, int] = field(default_factory=dict)
    _last: dict[int, int] = field(default_factory=dict)
    _start_xy: dict[int, tuple[float, float]] = field(default_factory=dict)
    _end_xy: dict[int, tuple[float, float]] = field(default_factory=dict)
    _start_h: dict[int, float] = field(default_factory=dict)
    _end_h: dict[int, float] = field(default_factory=dict)
    _feats: dict[int, list[np.ndarray]] = field(default_factory=dict)

    def observe(
        self, track_id: int, ordinal: int, xyxy: tuple[float, float, float, float], frame
    ) -> None:
        center = _center(xyxy)
        height = xyxy[3] - xyxy[1]
        if track_id not in self._first:
            self._first[track_id] = ordinal
            self._start_xy[track_id] = center
            self._start_h[track_id] = height
            self._feats[track_id] = []
        self._last[track_id] = ordinal
        self._end_xy[track_id] = center
        self._end_h[track_id] = height
        crop = torso_crop(frame, xyxy)
        if crop is not None:
            self._feats[track_id].append(color_histogram(crop))

    def build(self) -> list[Tracklet]:
        tracklets = []
        for tid in self._first:
            feats = self._feats[tid]
            if feats:
                mean = np.mean(feats, axis=0)
                norm = float(np.linalg.norm(mean))
                feature = mean / norm if norm > 0 else mean
            else:
                feature = np.zeros(1, dtype=np.float32)
            tracklets.append(
                Tracklet(
                    tid,
                    self._first[tid],
                    self._last[tid],
                    self._start_xy[tid],
                    self._end_xy[tid],
                    self._start_h[tid],
                    self._end_h[tid],
                    feature,
                )
            )
        return tracklets


@dataclass
class CourtTracklet:
    """A tracklet in court space (feet) for stitching on a *panning* clip.

    Unlike `Tracklet` (image pixels), the positions are camera-invariant court
    coordinates recovered by §4.2 registration, so the spatial gate becomes a
    physical speed bound instead of a box-scale factor — the right frame for a
    moving broadcast camera, where a player reappears at an arbitrary pixel but
    a bounded number of feet away.
    """

    track_id: int
    start_frame: int
    end_frame: int
    start_ft: tuple[float, float]  # court position (feet) at start
    end_ft: tuple[float, float]  # court position (feet) at end
    feature: np.ndarray  # L2-normalized torso-colour histogram


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def stitch(
    tracklets: list[Tracklet],
    max_gap: int = 45,
    max_dist_factor: float = 2.5,
    min_similarity: float = 0.5,
) -> dict[int, int]:
    """Return a remap {track_id: canonical_id} merging fragmented tracklets.

    `max_gap` is in processed frames; `max_dist_factor` multiplies the mean box
    height to bound the reappearance distance; `min_similarity` is the minimum
    torso-histogram cosine. Merged groups keep disjoint frame ranges.
    """
    order = sorted(tracklets, key=lambda t: t.start_frame)
    parent = {t.track_id: t.track_id for t in order}
    # per-group tail state, keyed by group root
    g_end = {t.track_id: t.end_frame for t in order}
    g_end_xy = {t.track_id: t.end_xy for t in order}
    g_end_h = {t.track_id: t.end_h for t in order}
    g_feat = {t.track_id: t.feature for t in order}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for b in order:
        best_root, best_score = None, -np.inf
        for a in order:
            if a.start_frame >= b.start_frame:
                continue
            root = find(a.track_id)
            if root == find(b.track_id):
                continue
            gap = b.start_frame - g_end[root]
            if gap <= 0 or gap > max_gap:  # must be disjoint and close in time
                continue
            scale = 0.5 * (g_end_h[root] + b.start_h)
            if scale <= 0:
                continue
            ex, ey = g_end_xy[root]
            dist = float(np.hypot(ex - b.start_xy[0], ey - b.start_xy[1]))
            if dist > max_dist_factor * scale:
                continue
            sim = _cosine(g_feat[root], b.feature)
            if sim < min_similarity:
                continue
            score = sim - 0.5 * (dist / (max_dist_factor * scale)) - 0.5 * (gap / max_gap)
            if score > best_score:
                best_root, best_score = root, score
        if best_root is not None:
            parent[find(b.track_id)] = best_root
            g_end[best_root] = b.end_frame
            g_end_xy[best_root] = b.end_xy
            g_end_h[best_root] = b.end_h

    # canonical id = smallest original id in each group (stable, readable)
    groups: dict[int, list[int]] = {}
    for t in order:
        groups.setdefault(find(t.track_id), []).append(t.track_id)
    remap = {}
    for members in groups.values():
        canonical = min(members)
        for m in members:
            remap[m] = canonical
    return remap


def stitch_court(
    tracklets: list[CourtTracklet],
    fps: float,
    max_gap_s: float = 1.5,
    base_ft: float = 3.0,
    max_speed_fps: float = 25.0,
    min_similarity: float = 0.5,
) -> dict[int, int]:
    """Return a remap {track_id: canonical_id} stitching fragments in court feet.

    A successor tracklet joins a predecessor group when it (a) starts after the
    group ends within `max_gap_s` seconds (disjoint in time), (b) reappears
    within a *speed-bounded* court distance `base_ft + max_speed_fps * gap_s`
    (so a longer gap forgives a longer move, matching real running), and (c) has
    a similar torso-colour histogram (cosine ≥ `min_similarity`). Merged groups
    keep disjoint frame ranges, so two players on court at once never collapse.

    This is the panning-clip analogue of `stitch()`: same union-find, but the
    spatial gate is physical feet (camera-invariant) rather than image pixels,
    which drift under the pan. `max_speed_fps` is deliberately below the sprint
    ceiling used elsewhere — a conservative gate spreads less contamination.
    """
    order = sorted(tracklets, key=lambda t: t.start_frame)
    parent = {t.track_id: t.track_id for t in order}
    g_end = {t.track_id: t.end_frame for t in order}
    g_end_ft = {t.track_id: t.end_ft for t in order}
    g_feat = {t.track_id: t.feature for t in order}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for b in order:
        best_root, best_score = None, -np.inf
        for a in order:
            if a.start_frame >= b.start_frame:
                continue
            root = find(a.track_id)
            if root == find(b.track_id):
                continue
            gap_frames = b.start_frame - g_end[root]
            if gap_frames <= 0:  # must be disjoint in time
                continue
            gap_s = gap_frames / fps
            if gap_s > max_gap_s:
                continue
            ex, ey = g_end_ft[root]
            dist = float(np.hypot(ex - b.start_ft[0], ey - b.start_ft[1]))
            max_dist = base_ft + max_speed_fps * gap_s
            if dist > max_dist:
                continue
            sim = _cosine(g_feat[root], b.feature)
            if sim < min_similarity:
                continue
            score = sim - 0.5 * (dist / max_dist) - 0.5 * (gap_s / max_gap_s)
            if score > best_score:
                best_root, best_score = root, score
        if best_root is not None:
            parent[find(b.track_id)] = best_root
            g_end[best_root] = b.end_frame
            g_end_ft[best_root] = b.end_ft

    groups: dict[int, list[int]] = {}
    for t in order:
        groups.setdefault(find(t.track_id), []).append(t.track_id)
    remap = {}
    for members in groups.values():
        canonical = min(members)
        for m in members:
            remap[m] = canonical
    return remap
