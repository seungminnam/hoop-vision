# Reference-project analysis: where Hoop Vision stands and what to adopt

**Purpose.** Compare Hoop Vision against two well-known basketball-CV reference
projects, decide what is worth adopting, and connect it to the product vision
(automatic stats + game-flow prediction). Feeds [ROADMAP.md](../ROADMAP.md).

**Honesty note.** The two references are identified by title only — this
analysis reasons about the *standard* pipeline each represents (and my
knowledge of the domain), not frame-by-frame video content:

- **R1** — "Basketball AI: Player Tracking, Team Detection, and Number
  Recognition with Python" (`youtube.com/watch?v=yGQb9KkvQ1Q`). Distinctive
  feature: **jersey-number recognition** → per-player identity.
- **R2** — "Build an AI/ML NBA Basketball Analysis system with YOLO, OpenCV,
  and Python" (`youtube.com/watch?v=QqVahw9tBfw`). The canonical end-to-end
  tutorial: detection → tracking → teams → **camera-motion estimation
  (optical flow)** → **perspective transform** → **speed & distance stats**.

## 1. What the references do (standard pipeline)

Both are single-broadcast-clip pipelines. Combined, the techniques they cover:

| Step | R1 | R2 | Notes |
|---|---|---|---|
| Fine-tuned YOLO detection | ✓ | ✓ | players / ball / refs |
| Multi-object tracking | ✓ (ByteTrack) | ✓ (ByteTrack) | motion-based |
| Team assignment | ✓ | ✓ (KMeans on jersey pixels) | R1 may use embeddings |
| **Jersey-number OCR** | ✓ | — | enables per-player box scores |
| Ball-position interpolation | — | ✓ | fill detection gaps |
| **Camera-motion estimation** | — | ✓ (Lucas–Kanade optical flow) | subtract camera pan before analysis |
| Perspective transform (homography) | — | ✓ | pixels → court metres |
| **Speed & distance per player** | — | ✓ | the headline "advanced stat" |

## 2. Side-by-side with Hoop Vision (measured state)

| Capability | References | Hoop Vision today | Gap |
|---|---|---|---|
| Detection | fine-tuned YOLO | YOLO11n fine-tuned, **mAP50 0.919** | at parity |
| Tracking | ByteTrack | ByteTrack, **IDF1 0.730, IDP 0.585** | same tool, **fragmentation is our weak point** |
| Teams | color / embeddings | LAB + KMeans(2), majority vote | at parity (simple) |
| Ball gaps | interpolation | interpolation in `events.py` | at parity |
| Camera motion | optical-flow comp. | **none** | missing — needed for panning clips |
| Court mapping | perspective transform | manual + **auto** (`auto_calibrate.py`) | ahead (we auto-recover it), fixed-camera only |
| Shot events | (not shown) | state machine, **80/80 P/R** | ahead |
| Speed / distance stats | ✓ | **none** | missing — but homography is already there |
| Jersey-number OCR | R1 ✓ | **none** | missing — blocks per-player stats |
| Game-flow / prediction | — | — | greenfield (the vision) |

Read: we are **at or ahead** on detection, court mapping, and shot analytics;
we are **behind** on tracking robustness, camera-motion handling, and the
downstream stats (speed/distance, per-player identity) that turn tracks into a
box score. Notably, the references mostly *don't measure* their tracking — we
do (IDF1), which is itself a differentiator.

## 3. The occlusion observation (why boxes vanish on overlaps)

Observed while labeling: detections disappear when players overlap, worst at
this camera angle. Root cause, honestly:

- **This is mainly a detection-recall problem, not a tracking bug.** A single
  low-angle amateur camera projects overlapping players onto nearly the same
  pixels; the detector can't separate two bodies that occlude, so it emits one
  box or none. No tracker can associate a detection that was never made.
- **What's a hard limit:** the camera angle. Elevated/multi-camera rigs
  (SportVU, Second Spectrum, broadcast high-angle) exist precisely to reduce
  occlusion. We can't change the angle of existing clips.
- **What's fixable (mitigation, not cure):**
  1. *Track-level gap-filling* — a longer lost-track buffer + appearance
     re-ID lets an ID survive a few occluded frames and re-attach when the
     player re-emerges. Directly targets the "box flickers off then a new ID
     appears" pattern (our IDP 0.585).
  2. *Detector recall* — lower confidence threshold / higher-res inference /
     more occlusion examples in fine-tuning trade precision for recall under
     overlap.
  3. *Interpolation for analytics* — for stats (not display), linearly fill
     short gaps, exactly as the ball track already does.

So part of what you saw is a genuine ceiling of the footage; the rest is the
fragmentation that v1.1's appearance work is meant to fix.

## 4. What to adopt, prioritized

Ranked by (value to the vision) × (fit with what we already have) ÷ effort.

### A. Appearance-based tracking — **do first** (v1.1 §3.2)
Add an appearance cost to association (start free: the torso LAB features
already in `teams.py`; stretch: a small ReID/SigLIP embedding) plus a longer
lost-track buffer. Directly attacks the fragmentation you felt and our low IDP.
Measured before/after on the committed GT (`eval_tracking.py`). *Effort: ~1
PR. Risk: low.*

### B. Speed & distance stats — **highest vision payoff** (v3 slice 1)
R2's headline feature, and we already have the missing piece they build from
scratch: homography. On a fixed-camera clip, project each track's foot point to
court feet per frame → per-player distance covered, average/max speed, a
court-occupancy heatmap. This is the first real "advanced stat" and it's
buildable now. *Effort: ~1–2 PRs. Risk: low (fixed-camera clips).* Depends on A
for clean tracks.

### C. Camera-motion compensation — **unlocks panning clips** (v1.1 / v2 bridge)
R2's optical-flow trick (Lucas–Kanade on background points, subtract global
motion before association and before homography). Lets tracking + minimap work
on the Hudl-style panning clips that v1 excluded, and is a stepping stone to v2
dynamic homography. *Effort: ~1 PR. Risk: medium (tuning).*

### D. Jersey-number OCR — **blocked by available data** (verdict 2026-07-08)
R1's differentiator and the key to a per-*named*-player box score. Probed for
feasibility by cropping jersey regions from both sources:
- **Hudl (numbered HS jerseys) is 360p**: a near player's box is ~120 px tall,
  so the digits are ~15–20 px and motion-blurred — barely human-legible, well
  below reliable OCR. Expected recall would be very low.
- **The 1080p pickup footage has no numbers at all** (plain athletic tanks /
  t-shirts) — OCR is moot there.

So per-player identity was **not achievable from the current clips**; it needs
footage that is *both* numbered *and* ≥720p (a broadcast or a higher-res HS
recording). Not worth adding a heavy OCR dependency to chase ~0% recall.
Documented as a data constraint, not attempted. Depends on A/B *and* better data.

**Unblocked 2026-07-10 ([ADR-008](decisions.md)).** The exact "numbered + ≥720p"
footage now exists as public data: the dataset authors (ADR-005) publish
`basketball-jersey-numbers-ocr` (3,188 NBA-broadcast number crops → digit
strings) and `basketball-player-detection-3-ycjdo` (a `number` detection class).
Task D resumes as detect-number → crop → classify → vote → merge tracks. The
live risk is no longer data but resolution: number boxes are ~12–17 px even at
native 720p, so detection runs at full res and read-rate on our panning clip is
the honest unknown to measure.

**Built + measured 2026-07-10 ([ADR-009](decisions.md)).** Detector `number`
AP50 **0.970**, classifier acc **0.955** (release v0.5.0); the IoS-match / vote /
merge logic is pure and unit-tested. The measured unknown came back as feared:
on the 30 s panning clip **only ~9% of tracks get a confirmed number**, and the
classifier collapses many blurry in-game numbers onto "22" (113/339 reads). So
the resolution risk was real — read *precision*, not the plumbing, is the wall.
Ships as a hybrid (per-player where read, per-track otherwise); R1-style
per-named-player box scores need either higher-res close footage or the levers
in ADR-009 (appearance stitching before reading, an abstain class).

### E. Not worth copying
Ball interpolation (already have it) and basic KMeans teams (already have it).

## 5. The vision: from tracks to stats to game flow

The through-line that makes this more than a tutorial:

```
detection ─▶ robust tracking (A) ─▶ homography (have) ─▶ player stats (B): distance, speed, spacing, heatmaps
                                    │                                        │
                              jersey OCR (D) ─────▶ per-player box score ────┤
                                                                             ▼
                                              game-flow features (possession, pace, lead changes)
                                                                             │
                                                                             ▼
                                              NBA Forecast Lab  ─▶  win-probability / momentum
```

The sibling project (NBA Forecast Lab) predicts games from tabular stats;
Hoop Vision can *produce* those stats from video. Speed/distance/spacing and a
per-player box score are exactly the inputs a game-flow model consumes — so B
and D are not side quests, they are the bridge to the prediction vision.

## 6. Recommended sequence

1. **A — appearance tracking** (fix fragmentation; measured on GT). Closes v1.1.
2. **B — speed & distance stats** on a fixed-camera clip (first advanced stat;
   demo-visible; feeds the vision). Opens v3.
3. **C — camera-motion compensation** (widen to panning clips; bridge to v2).
4. **D — jersey OCR** on 1080p (per-player box score) once A/B are solid.

A and B together turn the honest weakness you found (fragmented tracks, no
downstream stats) into the project's next visible win, without changing the
$0 / fixed-camera-first constraints.
