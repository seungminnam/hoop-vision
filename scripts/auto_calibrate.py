"""Automatic court calibration from a static-camera frame.

Recovers the image→court homography without manual clicking:

1. Segment the painted key ("the paint") by HSV color and keep the connected
   component nearest the basket (rim found with the fine-tuned weights, or
   given via --basket-px / --probe), reduced to a quadrilateral.
2. Order the quad corners against the court model (the paint is 16 ft wide
   and 19 ft deep, baseline → free-throw line) using the basket position,
   then compute the initial homography from those 4 correspondences.
3. Refine ICP-style against the visible court lines (3-pt arc, center
   circle, halfcourt line): project model points into the image, match each
   to the darkest pixel along the local normal, and re-fit the homography
   with the paint corners as strong anchors (Huber least squares). Repeat
   until the fit stops moving.

Outputs the pipeline-compatible calibration JSON, an overlay JPEG with the
projected court model for visual inspection, and the paint-corner
reprojection error in feet.

Usage:
    uv run python scripts/auto_calibrate.py data/clips/hudl_static2.mp4 \
        --weights hoopvision_best.pt --output calib.json --overlay overlay.jpg

Assumes the camera is on the spectator/halfcourt side (image-left ==
court-left); pass --flip for footage shot from behind the baseline or
mirrored footage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hoopvision.court import (  # noqa: E402
    COURT_LENGTH_FT,
    FT_CIRCLE_RADIUS,
    FT_LINE_Y,
    PAINT_HALF_WIDTH,
    RIM_CENTER,
    THREE_PT_RADIUS,
    CourtCalibration,
)
from hoopvision.ingest import frames  # noqa: E402

# Court-model corners of the paint, in the order this script produces image
# corners: baseline-left, baseline-right, ft-left, ft-right.
PAINT_CORNERS_FT = np.array(
    [
        [25.0 - PAINT_HALF_WIDTH, 0.0],
        [25.0 + PAINT_HALF_WIDTH, 0.0],
        [25.0 - PAINT_HALF_WIDTH, FT_LINE_Y],
        [25.0 + PAINT_HALF_WIDTH, FT_LINE_Y],
    ]
)

CURVE_CHOICES = ("three", "center", "halfcourt", "ft-circle")


# ---------------------------------------------------------------------------
# Step 1 — paint segmentation
# ---------------------------------------------------------------------------


def paint_mask(frame: np.ndarray, s_min: int, v_min: int) -> np.ndarray:
    """Binary mask of saturated, bright pixels — the painted key.

    Morphology is a single gentle open (3x3): closing — even 3x3 — bridges
    the thin baseline/sideline lines separating the paint from
    similarly-colored aprons or lettering bands, merging them into one
    component. Holes inside the paint don't matter because the quad comes
    from the outer contour's convex hull.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, s_min, v_min), (179, 255, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)


def pick_component(
    mask: np.ndarray,
    min_area: int | None,
    basket_px: tuple[float, float] | None,
    probe_px: tuple[float, float] | None,
) -> np.ndarray:
    """Select one connected component: the one containing --probe if given,
    otherwise the sufficiently-large one whose centroid is nearest the basket.

    ``min_area`` defaults to 0.5% of the frame so stickers/jerseys near the
    basket don't outcompete the paint on distance alone.
    """
    if min_area is None:
        min_area = int(0.005 * mask.shape[0] * mask.shape[1])
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if probe_px is not None:
        label = int(labels[int(round(probe_px[1])), int(round(probe_px[0]))])
        if label == 0:
            raise SystemExit(
                f"--probe {probe_px} is not inside any paint-colored region; "
                "adjust --s-min/--v-min or re-pick the probe pixel"
            )
        return (labels == label).astype(np.uint8)
    if basket_px is None:
        raise SystemExit("Need a basket location: pass --weights, --basket-px, or --probe")
    anchor = np.asarray(basket_px, dtype=np.float64)
    best: tuple[float, int] | None = None
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        distance = float(np.linalg.norm(centroids[i] - anchor))
        if best is None or distance < best[0]:
            best = (distance, i)
    if best is None:
        raise SystemExit(
            f"No paint-colored component with area >= {min_area} px; "
            "lower --min-area or adjust --s-min/--v-min"
        )
    return _detach_appendages((labels == best[1]).astype(np.uint8))


def _detach_appendages(component: np.ndarray) -> np.ndarray:
    """Open with a kernel scaled to the component and keep the largest piece.

    Players or logos touching the paint survive the gentle global morphology
    and drag the convex hull away from the true corners; a stronger open cuts
    those thin bridges while the paint's straight edges (erode + dilate)
    return to their original position.
    """
    xs, ys = np.nonzero(component.T)
    size = min(xs.max() - xs.min(), ys.max() - ys.min())
    kernel = max(3, int(round(size / 10)) | 1)
    opened = cv2.morphologyEx(component, cv2.MORPH_OPEN, np.ones((kernel, kernel), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if count <= 1:
        return component  # open erased everything — fall back
    largest = max(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def quad_from_mask(component: np.ndarray) -> np.ndarray:
    """Reduce a component mask to a 4-corner convex quadrilateral (px)."""
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise SystemExit("Paint component has no contour (empty mask?)")
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for epsilon in np.linspace(0.02, 0.15, 14):
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(approx) == 4:
            rough = approx.reshape(4, 2).astype(np.float64)
            return _refine_corners_by_line_fit(contour.reshape(-1, 2).astype(np.float64), rough)
    raise SystemExit("Could not reduce the paint region to 4 corners")


def _refine_corners_by_line_fit(contour: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Re-derive quad corners as intersections of lines fit to each edge.

    approxPolyDP vertices sit on the (morphology-rounded) contour, so they are
    biased a few px inward at sharp corners; lines fit to the straight middle
    of each edge are unaffected and intersect at the true corners.
    """
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        along = (contour - p0) @ unit
        offset = np.abs((contour - p0) @ normal)
        selected = contour[(offset < 3.0) & (along > 0.15 * length) & (along < 0.85 * length)]
        if len(selected) >= 2:
            # Huber: players standing on a line bulge the mask boundary locally
            fit = cv2.fitLine(selected.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            vx, vy, x0, y0 = (float(v) for v in fit)
            lines.append((np.array([x0, y0]), np.array([vx, vy])))
        else:
            lines.append((p0, unit))
    refined = np.empty((4, 2))
    for i in range(4):
        point = _intersect_lines(*lines[(i - 1) % 4], *lines[i])
        refined[i] = quad[i] if point is None else point
    return refined


def _intersect_lines(
    q1: np.ndarray, v1: np.ndarray, q2: np.ndarray, v2: np.ndarray
) -> np.ndarray | None:
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if abs(cross) < 1e-9:
        return None
    t = ((q2[0] - q1[0]) * v2[1] - (q2[1] - q1[1]) * v2[0]) / cross
    return q1 + t * v1


# ---------------------------------------------------------------------------
# Step 2 — corner ordering + initial homography
# ---------------------------------------------------------------------------


def order_paint_corners(
    quad: np.ndarray, basket_px: tuple[float, float], flip: bool = False
) -> np.ndarray:
    """Order 4 paint corners as (baseline-left, baseline-right, ft-left, ft-right).

    The baseline is the edge minimizing the *sum of endpoint distances* to the
    basket pixel: its two corners flank the basket, so the cue survives both
    oblique views (where the nearest-edge heuristic latches onto a lane line
    passing close to the rim) and the rim being detected 10 ft above the
    floor (where reprojection scoring breaks down near the horizon). The rim
    sits on the court's symmetry axis so it cannot resolve left vs right:
    that follows image x (camera on the spectator side; ``flip`` swaps).
    Matches PAINT_CORNERS_FT row order.
    """
    quad = np.asarray(quad, dtype=np.float64)
    if quad.shape != (4, 2):
        raise ValueError(f"Expected a (4, 2) quad, got {quad.shape}")
    center = quad.mean(axis=0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0]))]

    distances = np.linalg.norm(quad - np.asarray(basket_px, dtype=np.float64), axis=1)
    i = int(np.argmin(distances + np.roll(distances, -1)))
    baseline_left, baseline_right = quad[i], quad[(i + 1) % 4]
    # ft corners share a lane line with their baseline corner (hull adjacency)
    ft_left, ft_right = quad[(i + 3) % 4], quad[(i + 2) % 4]

    if (baseline_left[0] > baseline_right[0]) != flip:
        baseline_left, baseline_right = baseline_right, baseline_left
        ft_left, ft_right = ft_right, ft_left
    return np.array([baseline_left, baseline_right, ft_left, ft_right])


# ---------------------------------------------------------------------------
# Step 3 — ICP-style refinement against court lines
# ---------------------------------------------------------------------------


def court_curves(names: list[str]) -> list[np.ndarray]:
    """Sampled court-model lines (each an ordered (N, 2) array in feet)."""
    rim_x, rim_y = RIM_CENTER
    curves: list[np.ndarray] = []
    if "three" in names:
        theta = np.linspace(0.39, np.pi - 0.39, 80)
        arc = np.stack(
            [rim_x + THREE_PT_RADIUS * np.cos(theta), rim_y + THREE_PT_RADIUS * np.sin(theta)],
            axis=1,
        )
        curves.append(arc[arc[:, 1] <= COURT_LENGTH_FT])
    if "center" in names:
        theta = np.linspace(np.pi, 2 * np.pi, 40)
        curves.append(
            np.stack(
                [
                    rim_x + FT_CIRCLE_RADIUS * np.cos(theta),
                    COURT_LENGTH_FT + FT_CIRCLE_RADIUS * np.sin(theta),
                ],
                axis=1,
            )
        )
    if "halfcourt" in names:
        x = np.linspace(0.5, 49.5, 50)
        curves.append(np.stack([x, np.full_like(x, COURT_LENGTH_FT)], axis=1))
    if "ft-circle" in names:
        theta = np.linspace(0.0, 2 * np.pi, 40, endpoint=False)
        xs = rim_x + FT_CIRCLE_RADIUS * np.cos(theta)
        ys = FT_LINE_Y + FT_CIRCLE_RADIUS * np.sin(theta)
        curves.append(np.stack([xs, ys], axis=1))
    return curves


def _to_matrix(params: np.ndarray) -> np.ndarray:
    return np.append(params, 1.0).reshape(3, 3)


def _apply(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float64), h).reshape(-1, 2)


def _image_normals(projected: np.ndarray) -> np.ndarray:
    """Unit normals of a projected curve (perpendicular to the local tangent)."""
    tangent = np.gradient(projected, axis=0)
    normals = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return normals / lengths


def match_along_normals(
    gray: np.ndarray,
    projected: np.ndarray,
    normals: np.ndarray,
    search_px: int = 12,
    min_contrast: int = 15,
) -> np.ndarray:
    """For each projected point, the darkest pixel along its normal (NaN = no match)."""
    height, width = gray.shape
    offsets = np.arange(-search_px, search_px + 1, dtype=np.float64)
    matched = np.full_like(projected, np.nan)
    for j, (point, normal) in enumerate(zip(projected, normals, strict=True)):
        samples = point[None, :] + offsets[:, None] * normal[None, :]
        xs = np.round(samples[:, 0]).astype(int)
        ys = np.round(samples[:, 1]).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if valid.sum() < 0.6 * len(offsets):
            continue  # mostly outside the frame
        values = gray[ys[valid], xs[valid]].astype(np.float64)
        if values.max() - values.min() < min_contrast:
            continue  # no line evidence here
        matched[j] = samples[valid][int(np.argmin(values))]
    return matched


def refine(
    gray: np.ndarray,
    corners_px: np.ndarray,
    initial: CourtCalibration,
    curve_names: list[str],
    iterations: int = 5,
    corner_weight: float = 12.0,
    search_px: int = 12,
) -> CourtCalibration:
    """Joint fit: court lines matched to dark pixels + paint corners as anchors."""
    curves = court_curves(curve_names)
    g_init = np.linalg.inv(initial.homography)
    params = (g_init / g_init[2, 2]).flatten()[:8]

    for _ in range(iterations):
        g = _to_matrix(params)
        model_points, matched_points, normals = [], [], []
        for curve in curves:
            projected = _apply(g, curve)
            curve_normals = _image_normals(projected)
            matched = match_along_normals(gray, projected, curve_normals, search_px=search_px)
            keep = ~np.isnan(matched[:, 0])
            model_points.append(curve[keep])
            matched_points.append(matched[keep])
            normals.append(curve_normals[keep])
        model = np.concatenate(model_points)
        target = np.concatenate(matched_points)
        normal = np.concatenate(normals)
        if len(model) < 8:
            print("refine: too few line matches — keeping the corner-only homography")
            break

        def residuals(p: np.ndarray, model=model, target=target, normal=normal) -> np.ndarray:
            g_p = _to_matrix(p)
            line = ((_apply(g_p, model) - target) * normal).sum(axis=1)
            anchors = (_apply(g_p, PAINT_CORNERS_FT) - corners_px).ravel() * corner_weight
            return np.concatenate([line, anchors])

        result = least_squares(residuals, params, loss="huber", f_scale=3.0, x_scale="jac")
        if np.allclose(result.x, params, rtol=0, atol=1e-9):
            params = result.x
            break
        params = result.x

    g = _to_matrix(params)
    h = np.linalg.inv(g)
    return CourtCalibration(h / h[2, 2], corners_px, PAINT_CORNERS_FT.copy())


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def grab_frame(video: str, frame_index: int) -> np.ndarray:
    for index, frame in frames(video):
        if index >= frame_index:
            return frame
    raise SystemExit(f"Video has fewer than {frame_index} frames.")


def detect_basket_px(frame: np.ndarray, weights: str) -> tuple[float, float]:
    from hoopvision.detect import RIM, YoloDetector

    detector = YoloDetector(weights)
    rims = [d for d in detector.detect(frame) if d.class_name == RIM]
    if not rims:
        raise SystemExit("No rim detected in the reference frame; pass --basket-px X,Y instead")
    return max(rims, key=lambda d: d.confidence).center


def draw_overlay(frame: np.ndarray, calib: CourtCalibration, curve_names: list[str]) -> np.ndarray:
    canvas = frame.copy()
    height, width = frame.shape[:2]
    left, right = 25.0 - PAINT_HALF_WIDTH, 25.0 + PAINT_HALF_WIDTH
    paint_outline = np.array(
        [[left, 0.0], [right, 0.0], [right, FT_LINE_Y], [left, FT_LINE_Y], [left, 0.0]]
    )
    for points_ft in [paint_outline, *court_curves(curve_names)]:
        points_px = calib.to_image(points_ft)
        inside = (
            (points_px[:, 0] >= -width)
            & (points_px[:, 0] <= 2 * width)
            & (points_px[:, 1] >= -height)
            & (points_px[:, 1] <= 2 * height)
        )
        for segment in _runs(inside):
            polyline = points_px[segment].round().astype(np.int32)
            cv2.polylines(canvas, [polyline], False, (0, 220, 255), 1, cv2.LINE_AA)
    for x, y in calib.image_points:
        cv2.circle(canvas, (int(round(x)), int(round(y))), 4, (0, 80, 255), -1)
    return canvas


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    """Indices of consecutive True runs (len >= 2) in a boolean mask."""
    runs, current = [], []
    for i, ok in enumerate(mask):
        if ok:
            current.append(i)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [np.array(r) for r in runs if len(r) >= 2]


def _parse_xy(spec: str | None) -> tuple[float, float] | None:
    if spec is None:
        return None
    x, y = spec.split(",")
    return float(x), float(y)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video")
    parser.add_argument("--frame", type=int, default=0, help="reference frame index")
    parser.add_argument("--output", default="calib.json")
    parser.add_argument("--overlay", default=None, help="write an overlay JPEG for inspection")
    parser.add_argument("--weights", default=None, help="fine-tuned weights, used to find the rim")
    parser.add_argument("--basket-px", default=None, help="rim location X,Y in pixels")
    parser.add_argument("--probe", default=None, help="X,Y of any pixel inside the paint")
    parser.add_argument("--s-min", type=int, default=120, help="HSV saturation threshold")
    parser.add_argument("--v-min", type=int, default=165, help="HSV value threshold")
    parser.add_argument(
        "--min-area",
        type=int,
        default=None,
        help="min paint component area in px (default: 0.5%% of the frame)",
    )
    parser.add_argument(
        "--curves",
        default="three,center,halfcourt",
        help=f"refinement lines, comma-separated from {CURVE_CHOICES}",
    )
    parser.add_argument("--no-refine", action="store_true", help="stop at the 4-corner homography")
    parser.add_argument("--flip", action="store_true", help="camera behind the baseline / mirrored")
    args = parser.parse_args()

    curve_names = [c.strip() for c in args.curves.split(",") if c.strip()]
    unknown = set(curve_names) - set(CURVE_CHOICES)
    if unknown:
        raise SystemExit(f"Unknown curves {sorted(unknown)}; choose from {CURVE_CHOICES}")

    frame = grab_frame(args.video, args.frame)
    basket_px = _parse_xy(args.basket_px)
    probe_px = _parse_xy(args.probe)
    if basket_px is None and args.weights:
        basket_px = detect_basket_px(frame, args.weights)
        print(f"rim detected at ({basket_px[0]:.0f}, {basket_px[1]:.0f}) px")
    if basket_px is None:
        raise SystemExit("Need the basket location: pass --weights or --basket-px X,Y")

    mask = paint_mask(frame, args.s_min, args.v_min)
    component = pick_component(mask, args.min_area, basket_px, probe_px)
    corners_px = order_paint_corners(quad_from_mask(component), basket_px, flip=args.flip)
    print("paint corners (px):", corners_px.round(1).tolist())

    calib = CourtCalibration.from_points(corners_px, PAINT_CORNERS_FT)
    print(f"initial corner reprojection error: {calib.reprojection_error_ft():.2f} ft")

    if not args.no_refine:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        calib = refine(gray, corners_px, calib, curve_names)
        print(f"refined corner reprojection error: {calib.reprojection_error_ft():.2f} ft")

    calib.save(args.output)
    print(f"saved {args.output}")
    if args.overlay:
        cv2.imwrite(args.overlay, draw_overlay(frame, calib, curve_names))
        print(f"saved {args.overlay}")


if __name__ == "__main__":
    main()
