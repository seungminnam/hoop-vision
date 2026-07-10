"""Cross-validate our derived 33-point template against the dataset authors' config.

The `basketball-court-detection-2` dataset (v2 §4.2) ships no real-world
coordinates, so we derived them (ADR-005). It later turned out the dataset's
authors (Roboflow / SkalskiP) publish an *official* court config in the
`roboflow/sports` repo on its **`feat/basketball` branch** — not on `main`
(which has only soccer), which is why ADR-005 concluded "no published template"
at derivation time.

This script reproduces that config's NBA vertices from its published
centimeter presets and compares them, point by point, to our independently
derived `court_template.NBA_FULLCOURT_FT`. Agreement is a strong honesty-rule
check that our reverse-engineered identities are right.

Source (verbatim NBA preset + vertex logic, transcribed so this stays
reproducible even if the branch is rebased/deleted):
  https://github.com/roboflow/sports/blob/feat/basketball/sports/basketball/config.py

    uv run python scripts/compare_court_template.py

Their frame differs from ours only by a sideline convention (their y=0 is our
y=50 far sideline), so we mirror y before comparing.
"""

from __future__ import annotations

import numpy as np

from hoopvision.court_template import NBA_FULLCOURT_FT, NUM_KEYPOINTS

CENTIMETERS_PER_FOOT = 30.48

# roboflow/sports @ feat/basketball — PRESETS_CENTIMETERS[League.NBA], verbatim
ROBOFLOW_NBA_CM: dict[str, int] = {
    "court_width": 1524,
    "court_length": 2865,
    "three_point_arc_radius": 724,
    "straight_section_three_point_line": 424,
    "sideline_to_three_point_line": 91,
    "paint_width": 488,
    "paint_length": 579,
    "free_throw_line_distance": 457,
    "center_circle_radius": 183,
    "baseline_to_rim_center": 160,
    "baseline_to_throw_line": 835,
}


def roboflow_vertices_ft() -> np.ndarray:
    """Reproduce `CourtConfiguration(NBA).vertices`, in feet, our schema order.

    This is `_raw_vertices_centimeters()` from the source config, transcribed.
    """
    p = ROBOFLOW_NBA_CM
    paint_start = (p["court_width"] - p["paint_width"]) // 2
    mid = p["court_width"] // 2
    W = p["court_width"]
    L = p["court_length"]
    b2r = p["baseline_to_rim_center"]
    arc = p["three_point_arc_radius"]
    s3 = p["straight_section_three_point_line"]
    s2t = p["sideline_to_three_point_line"]
    pl = p["paint_length"]
    b2t = p["baseline_to_throw_line"]

    v_cm = [
        (0, 0),  # 0
        (0, s2t),  # 1
        (0, paint_start),  # 2
        (0, paint_start + p["paint_width"]),  # 3
        (0, W - s2t),  # 4
        (0, W),  # 5
        (b2r, mid),  # 6 basket
        (s3, s2t),  # 7
        (s3, W - s2t),  # 8
        (pl, paint_start),  # 9
        (pl, paint_start + p["paint_width"] // 2),  # 10
        (pl, paint_start + p["paint_width"]),  # 11
        (b2t, 0),  # 12
        (b2r + arc, mid),  # 13 arc top
        (b2t, W),  # 14
        (L // 2, 0),  # 15
        (L // 2, mid),  # 16
        (L // 2, W),  # 17
        (L - b2t, 0),  # 18
        (L - b2r - arc, mid),  # 19 arc top
        (L - b2t, W),  # 20
        (L - pl, paint_start),  # 21
        (L - pl, paint_start + p["paint_width"] // 2),  # 22
        (L - pl, paint_start + p["paint_width"]),  # 23
        (L - s3, s2t),  # 24
        (L - s3, W - s2t),  # 25
        (L - b2r, mid),  # 26 basket
        (L, 0),  # 27
        (L, s2t),  # 28
        (L, paint_start),  # 29
        (L, paint_start + p["paint_width"]),  # 30
        (L, W - s2t),  # 31
        (L, W),  # 32
    ]
    return np.array(v_cm, dtype=float) / CENTIMETERS_PER_FOOT


def comparison() -> dict:
    """Point-by-point |ours - roboflow(y-mirrored)| in feet."""
    ours = np.array([NBA_FULLCOURT_FT[i] for i in range(NUM_KEYPOINTS)], float)
    rf = roboflow_vertices_ft()
    rf_mirrored = rf.copy()
    rf_mirrored[:, 1] = 50.0 - rf_mirrored[:, 1]  # their y=0 == our far sideline
    diff = np.linalg.norm(ours - rf_mirrored, axis=1)
    return {
        "ours": ours,
        "roboflow": rf_mirrored,
        "diff": diff,
        "mean_ft": float(diff.mean()),
        "max_ft": float(diff.max()),
        "within_0_1_ft": int((diff <= 0.1).sum()),
    }


def main() -> None:
    c = comparison()
    print(f"{'idx':>3} {'ours':>16} {'roboflow(y-mir)':>18} {'|diff| ft':>10}")
    for i in range(NUM_KEYPOINTS):
        o = f"({c['ours'][i][0]:.2f}, {c['ours'][i][1]:.2f})"
        r = f"({c['roboflow'][i][0]:.2f}, {c['roboflow'][i][1]:.2f})"
        flag = "  <- differs" if c["diff"][i] > 0.15 else ""
        print(f"{i:>3} {o:>16} {r:>18} {c['diff'][i]:>10.3f}{flag}")
    print(
        f"\nmean {c['mean_ft']:.3f} ft, max {c['max_ft']:.3f} ft, "
        f"{c['within_0_1_ft']}/{NUM_KEYPOINTS} points within 0.1 ft"
    )
    print(
        "Differences: sideline hashes (12/14/18/20) 28 ft ours vs 27.4 ft theirs "
        "(our label residual + NBA rulebook support 28 ft); corner-3 straight "
        "(7/8/24/25) 14.0 ft ours vs 13.91 ft theirs."
    )


if __name__ == "__main__":
    main()
