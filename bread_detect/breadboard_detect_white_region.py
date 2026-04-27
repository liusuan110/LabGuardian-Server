from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from breadboard_detect import (
    annotate_connectivity,
    build_connections,
    build_holes,
    build_regions,
    detect_board_quad,
    draw_holes,
    draw_quad,
    draw_region_and_connection_overlay,
    fit_grid,
    odd_size,
    order_points,
    perspective_points,
    score_image,
    warp_board,
    write_csv,
    write_connections_csv,
)


def _quad_area(quad: np.ndarray) -> float:
    pts = order_points(quad).astype(np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _expand_quad(quad: np.ndarray, image_shape: tuple[int, int], padding_ratio: float, padding_px: float) -> np.ndarray:
    h, w = image_shape[:2]
    pts = order_points(quad).astype(np.float32)
    center = np.mean(pts, axis=0)
    width = max(float(np.linalg.norm(pts[1] - pts[0])), float(np.linalg.norm(pts[2] - pts[3])))
    height = max(float(np.linalg.norm(pts[3] - pts[0])), float(np.linalg.norm(pts[2] - pts[1])))
    diag = max(float(np.hypot(width, height)), 1.0)
    scale = 1.0 + (float(padding_px) + diag * float(padding_ratio)) / diag
    expanded = center + (pts - center) * scale
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
    return order_points(expanded)


def _line_from_points(points: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 8:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    direction = np.array([float(vx), float(vy)], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction /= norm
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    point = np.array([float(x0), float(y0)], dtype=np.float32)
    return np.array([normal[0], normal[1], -float(np.dot(normal, point))], dtype=np.float32)


def _line_from_segment(start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    p0 = np.asarray(start, dtype=np.float32).reshape(2)
    p1 = np.asarray(end, dtype=np.float32).reshape(2)
    direction = p1 - p0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    normal = np.array([-direction[1] / norm, direction[0] / norm], dtype=np.float32)
    return np.array([normal[0], normal[1], -float(np.dot(normal, p0))], dtype=np.float32)


def _intersect_lines(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    a1, b1, c1 = [float(v) for v in line_a]
    a2, b2, c2 = [float(v) for v in line_b]
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    return np.array([(b1 * c2 - b2 * c1) / det, (c1 * a2 - c2 * a1) / det], dtype=np.float32)


def _fit_envelope_line(
    pts: np.ndarray,
    axis_values: np.ndarray,
    edge_values: np.ndarray,
    *,
    axis_min: float,
    axis_max: float,
    bins: int,
    percentile: float,
    min_points: int,
) -> np.ndarray | None:
    if axis_max <= axis_min:
        return None
    envelope: list[np.ndarray] = []
    edges = np.linspace(axis_min, axis_max, int(bins) + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        inside = (axis_values >= float(left)) & (axis_values < float(right))
        if int(np.count_nonzero(inside)) < int(min_points):
            continue
        selected_edge = edge_values[inside]
        selected_pts = pts[inside]
        target = float(np.percentile(selected_edge, float(percentile)))
        keep = np.abs(selected_edge - target) <= max(4.0, 0.010 * (axis_max - axis_min))
        if int(np.count_nonzero(keep)) < 2:
            idx = int(np.argmin(np.abs(selected_edge - target)))
            envelope.append(selected_pts[idx])
        else:
            envelope.append(np.mean(selected_pts[keep], axis=0))
    if len(envelope) < 8:
        return None
    return _line_from_points(np.asarray(envelope, dtype=np.float32))


def _refine_quad_from_contour_edges(
    contour: np.ndarray,
    preliminary_quad: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = image_shape[:2]
    base = order_points(preliminary_quad).astype(np.float32)
    tl, tr, br, bl = base
    ux = tr - tl
    vy = bl - tl
    width = float(np.linalg.norm(ux))
    height = float(np.linalg.norm(vy))
    if width < 80.0 or height < 40.0:
        return base, {"edge_refined": False, "edge_reason": "small_base_quad"}
    ux /= width
    vy /= height

    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 40:
        return base, {"edge_refined": False, "edge_reason": "too_few_contour_points"}

    rel = pts - tl
    u = rel @ ux
    v = rel @ vy
    margin_u = width * 0.035
    margin_v = height * 0.035

    top_band = (u >= margin_u) & (u <= width - margin_u) & (v <= height * 0.45)
    bottom_band = (u >= margin_u) & (u <= width - margin_u) & (v >= height * 0.55)
    left_band = (v >= margin_v) & (v <= height - margin_v) & (u <= width * 0.35)
    right_band = (v >= margin_v) & (v <= height - margin_v) & (u >= width * 0.65)

    bins_u = int(np.clip(width / 45.0, 32, 90))
    bins_v = int(np.clip(height / 32.0, 18, 60))
    min_bin_points = 2
    top = _fit_envelope_line(pts[top_band], u[top_band], v[top_band], axis_min=margin_u, axis_max=width - margin_u, bins=bins_u, percentile=8, min_points=min_bin_points)
    bottom = _fit_envelope_line(pts[bottom_band], u[bottom_band], v[bottom_band], axis_min=margin_u, axis_max=width - margin_u, bins=bins_u, percentile=92, min_points=min_bin_points)
    left = _fit_envelope_line(pts[left_band], v[left_band], u[left_band], axis_min=margin_v, axis_max=height - margin_v, bins=bins_v, percentile=8, min_points=min_bin_points)
    right = _fit_envelope_line(pts[right_band], v[right_band], u[right_band], axis_min=margin_v, axis_max=height - margin_v, bins=bins_v, percentile=92, min_points=min_bin_points)
    lines = [top, right, bottom, left]
    if any(line is None for line in lines):
        return base, {
            "edge_refined": False,
            "edge_reason": "missing_side_line",
            "edge_points_top": int(np.count_nonzero(top_band)),
            "edge_points_bottom": int(np.count_nonzero(bottom_band)),
            "edge_points_left": int(np.count_nonzero(left_band)),
            "edge_points_right": int(np.count_nonzero(right_band)),
        }

    intersections = [
        _intersect_lines(top, left),  # type: ignore[arg-type]
        _intersect_lines(top, right),  # type: ignore[arg-type]
        _intersect_lines(bottom, right),  # type: ignore[arg-type]
        _intersect_lines(bottom, left),  # type: ignore[arg-type]
    ]
    if any(point is None for point in intersections):
        return base, {"edge_refined": False, "edge_reason": "parallel_side_lines"}
    refined = order_points(np.asarray(intersections, dtype=np.float32).reshape(4, 2))
    refined[:, 0] = np.clip(refined[:, 0], 0, w - 1)
    refined[:, 1] = np.clip(refined[:, 1], 0, h - 1)

    base_area = _quad_area(base)
    refined_area = _quad_area(refined)
    if refined_area < base_area * 0.65 or refined_area > base_area * 1.35:
        return base, {
            "edge_refined": False,
            "edge_reason": "area_jump",
            "edge_base_area": base_area,
            "edge_refined_area": refined_area,
        }

    refined_rel = refined - tl
    refined_u = refined_rel @ ux
    refined_v = refined_rel @ vy
    br_right_inset = max(0.0, float(width - refined_u[2]))
    br_bottom_inset = max(0.0, float(height - refined_v[2]))
    right_inset_limit = max(32.0, width * 0.018)
    area_shrink_limit = 0.94
    guard_reason: str | None = None
    if br_right_inset > right_inset_limit:
        guard_reason = "br_right_inset"
    elif refined_area < base_area * area_shrink_limit:
        guard_reason = "area_shrink"

    if guard_reason is not None:
        base_right = _line_from_segment(tr, br)
        base_bottom = _line_from_segment(br, bl)
        if base_right is not None and base_bottom is not None:
            guarded_intersections = [
                _intersect_lines(top, left),  # type: ignore[arg-type]
                _intersect_lines(top, base_right),  # type: ignore[arg-type]
                _intersect_lines(base_bottom, base_right),
                _intersect_lines(base_bottom, left),  # type: ignore[arg-type]
            ]
            if not any(point is None for point in guarded_intersections):
                guarded = order_points(np.asarray(guarded_intersections, dtype=np.float32).reshape(4, 2))
                guarded[:, 0] = np.clip(guarded[:, 0], 0, w - 1)
                guarded[:, 1] = np.clip(guarded[:, 1], 0, h - 1)
                guarded_area = _quad_area(guarded)
                if base_area * 0.65 <= guarded_area <= base_area * 1.35:
                    refined = guarded
                    refined_area = guarded_area

    return refined, {
        "edge_refined": True,
        "edge_base_area": base_area,
        "edge_refined_area": refined_area,
        "edge_guard_reason": guard_reason or "",
        "edge_br_right_inset": br_right_inset,
        "edge_br_bottom_inset": br_bottom_inset,
        "edge_right_inset_limit": right_inset_limit,
        "edge_points_top": int(np.count_nonzero(top_band)),
        "edge_points_bottom": int(np.count_nonzero(bottom_band)),
        "edge_points_left": int(np.count_nonzero(left_band)),
        "edge_points_right": int(np.count_nonzero(right_band)),
    }


def _candidate_mask(
    image: np.ndarray,
    *,
    white_threshold: int,
    max_saturation: int,
    min_value: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    strict_hsv = ((val > 165) & (sat < 95)).astype(np.uint8) * 255
    relaxed_hsv = ((val > int(min_value)) & (sat < int(max_saturation))).astype(np.uint8) * 255
    fixed_l = ((l_channel > int(white_threshold)) & (sat < int(max_saturation) + 25)).astype(np.uint8) * 255

    bright_seed = (((l_channel > int(white_threshold) - 10) | (val > int(min_value) + 35)) & (sat < int(max_saturation) + 35)).astype(np.uint8)
    density_kernel = (odd_size(w * 0.018, 35, 91), odd_size(h * 0.018, 35, 91))
    density = cv2.boxFilter(bright_seed.astype(np.float32), -1, density_kernel, normalize=True)
    dense_white = (density > 0.48).astype(np.uint8) * 255

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (odd_size(w * 0.012, 21, 71), odd_size(h * 0.018, 31, 111)),
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (odd_size(w * 0.004, 7, 19), odd_size(h * 0.004, 7, 19)),
    )
    fill_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (odd_size(w * 0.022, 35, 121), odd_size(h * 0.024, 45, 151)),
    )

    masks: dict[str, np.ndarray] = {}
    for name, raw in {
        "strict_hsv": strict_hsv,
        "relaxed_hsv": relaxed_hsv,
        "fixed_l": fixed_l,
        "dense_white": dense_white,
        "strict_or_dense": cv2.bitwise_or(strict_hsv, dense_white),
    }.items():
        cleaned = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fill_kernel, iterations=1)
        masks[name] = cleaned
    return masks, l_channel, hsv


def _score_contour(
    contour: np.ndarray,
    image_shape: tuple[int, int],
    l_eq: np.ndarray,
    hsv: np.ndarray,
) -> tuple[float, dict[str, Any]] | None:
    h, w = image_shape[:2]
    image_area = float(max(h * w, 1))
    area = float(cv2.contourArea(contour))
    if area < image_area * 0.015 or area > image_area * 0.62:
        return None

    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), _ = rect
    short = float(min(rw, rh))
    long = float(max(rw, rh))
    if short < 35.0 or long < 180.0:
        return None
    aspect = long / max(short, 1e-6)
    if aspect < 2.0 or aspect > 9.5:
        return None

    rect_area = max(float(rw * rh), 1.0)
    extent = area / rect_area
    if extent < 0.42:
        return None

    x, y, bw, bh = cv2.boundingRect(contour)
    border = max(6, int(round(min(h, w) * 0.003)))
    touches_border = x <= border or y <= border or x + bw >= w - border or y + bh >= h - border

    candidate_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(candidate_mask, [contour], -1, 255, -1)
    inside = candidate_mask > 0
    if not np.any(inside):
        return None

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    dark_density = float(np.mean(l_eq[inside] < 118))
    color_density = float(np.mean((sat[inside] > 70) & (val[inside] > 80)))
    mean_lightness = float(np.mean(l_eq[inside]))
    y_center = float(y + bh * 0.5) / float(max(h, 1))

    aspect_score = min(aspect, 6.0) / 6.0
    score = area * (0.55 + aspect_score * 0.35 + extent * 0.20)
    score += area * min(dark_density, 0.35) * 2.20
    score += area * min(color_density, 0.20) * 2.80
    score += area * max(0.0, y_center - 0.25) * 0.10
    if touches_border:
        score *= 0.45

    metrics = {
        "area": area,
        "aspect": aspect,
        "extent": extent,
        "dark_density": dark_density,
        "color_density": color_density,
        "mean_lightness": mean_lightness,
        "touches_border": float(touches_border),
        "score": float(score),
    }
    return float(score), metrics


def detect_white_region_quad(
    image: np.ndarray,
    *,
    debug_dir: Path | None = None,
    debug_prefix: str = "debug",
    min_board_area_ratio: float = 0.015,
    white_threshold: int = 185,
    max_saturation: int = 125,
    min_value: int = 135,
    padding_ratio: float = 0.006,
    padding_px: float = 4.0,
    fallback_auto: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h, w = image.shape[:2]
    candidate_masks, l_channel, hsv = _candidate_mask(
        image,
        white_threshold=white_threshold,
        max_saturation=max_saturation,
        min_value=min_value,
    )

    scored: list[tuple[float, np.ndarray, dict[str, float]]] = []
    for mask_name, boardish_mask in candidate_masks.items():
        contours, _ = cv2.findContours(boardish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            scored_result = _score_contour(contour, (h, w), l_channel, hsv)
            if scored_result is None:
                continue
            score, metrics = scored_result
            if metrics["area"] < float(min_board_area_ratio) * float(h * w):
                continue
            metrics["mask_name"] = mask_name
            scored.append((score, contour, metrics))

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        for mask_name, mask in candidate_masks.items():
            cv2.imwrite(str(debug_dir / f"{debug_prefix}_white_{mask_name}.png"), mask)
        boardish_mask_all = np.zeros((h, w), dtype=np.uint8)
        for mask in candidate_masks.values():
            boardish_mask_all = cv2.bitwise_or(boardish_mask_all, mask)
        cv2.imwrite(str(debug_dir / f"{debug_prefix}_white_board_mask_all.png"), boardish_mask_all)
        candidates = image.copy()
        for rank, (score, contour, metrics) in enumerate(sorted(scored, key=lambda item: item[0], reverse=True)[:8], start=1):
            box = order_points(cv2.boxPoints(cv2.minAreaRect(contour))).astype(np.int32)
            color = (0, 255, 0) if rank == 1 else (0, 180, 255)
            cv2.polylines(candidates, [box], True, color, 4, cv2.LINE_AA)
            pt = tuple(box[0])
            cv2.putText(
                candidates,
                f"{rank}:{score/1000000:.2f} a{metrics['aspect']:.1f} d{metrics['dark_density']:.2f}",
                (int(pt[0]), max(24, int(pt[1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(debug_dir / f"{debug_prefix}_white_candidates.png"), candidates)

    if not scored:
        if not fallback_auto:
            raise RuntimeError("Could not find a white breadboard-shaped region.")
        rough = order_points(detect_board_quad(image)).astype(np.float32)
        rough_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(rough_mask, rough.astype(np.int32), 255)
        return rough, rough_mask, {"mode": 0.0, "score": 0.0}

    _, contour, metrics = max(scored, key=lambda item: item[0])
    board_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(board_mask, [contour], -1, 255, -1)

    hull = cv2.convexHull(contour)
    rect = cv2.minAreaRect(hull)
    preliminary_quad = order_points(cv2.boxPoints(rect))
    quad, edge_metrics = _refine_quad_from_contour_edges(contour, preliminary_quad, (h, w))
    quad = _expand_quad(quad, (h, w), padding_ratio=padding_ratio, padding_px=padding_px)

    if _quad_area(quad) < float(h * w) * float(min_board_area_ratio):
        if not fallback_auto:
            raise RuntimeError("Detected white region is too small.")
        quad = order_points(detect_board_quad(image)).astype(np.float32)

    if debug_dir is not None:
        cv2.imwrite(str(debug_dir / f"{debug_prefix}_white_board_mask.png"), board_mask)
        cv2.imwrite(str(debug_dir / f"{debug_prefix}_white_corners.png"), draw_quad(image, quad))

    metrics = dict(metrics)
    metrics.update(edge_metrics)
    metrics["mode"] = 1.0
    return quad, board_mask, metrics


def _locate_score_peak(
    score_map: np.ndarray,
    x: float,
    y: float,
    *,
    radius: int,
    min_peak: float,
) -> tuple[np.ndarray, float] | None:
    h, w = score_map.shape[:2]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    x1 = max(0, xi - int(radius))
    x2 = min(w, xi + int(radius) + 1)
    y1 = max(0, yi - int(radius))
    y2 = min(h, yi + int(radius) + 1)
    patch = score_map[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    _, peak, _, _ = cv2.minMaxLoc(patch)
    peak = float(peak)
    if peak < float(min_peak):
        return None

    threshold = max(peak * 0.72, peak - 18.0)
    ys, xs = np.nonzero(patch >= threshold)
    if xs.size == 0:
        return None

    weights = patch[ys, xs].astype(np.float32) - float(threshold) + 1e-3
    px = float(x1) + float(np.average(xs, weights=weights))
    py = float(y1) + float(np.average(ys, weights=weights))
    return np.array([px, py], dtype=np.float32), peak


def _refine_quad_from_hole_lattice(
    image: np.ndarray,
    quad: np.ndarray,
    *,
    main_columns: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    coarse = order_points(quad).astype(np.float32)
    warped, _, inverse_matrix = warp_board(image, coarse)
    model = fit_grid(warped, main_columns)
    warped_score = score_image(warped)
    holes = build_holes(model, inverse_matrix, warped_score, main_columns)
    if not holes:
        return coarse, {"hole_lattice_refined": False, "hole_lattice_reason": "no_holes"}

    original_score = score_image(image)
    visibility_scores = np.asarray([float(hole["visible_score"]) for hole in holes], dtype=np.float32)
    visibility_cut = max(8.0, float(np.percentile(visibility_scores, 30)))
    search_radius = int(np.clip(round(min(image.shape[:2]) * 0.010), 8, 14))

    source_points: list[list[float]] = []
    target_points: list[list[float]] = []
    initial_shifts: list[float] = []
    peak_values: list[float] = []
    for hole in holes:
        if float(hole["visible_score"]) < visibility_cut:
            continue

        found = _locate_score_peak(
            original_score,
            float(hole["x_image"]),
            float(hole["y_image"]),
            radius=search_radius,
            min_peak=12.0,
        )
        if found is None:
            continue

        refined_center, peak_value = found
        shift = float(np.linalg.norm(refined_center - np.array([float(hole["x_image"]), float(hole["y_image"])], dtype=np.float32)))
        if shift > search_radius * 0.95:
            continue

        source_points.append([float(hole["x_warp"]), float(hole["y_warp"])])
        target_points.append([float(refined_center[0]), float(refined_center[1])])
        initial_shifts.append(shift)
        peak_values.append(float(peak_value))

    match_count = len(source_points)
    if match_count < 40:
        return coarse, {
            "hole_lattice_refined": False,
            "hole_lattice_reason": "too_few_matches",
            "hole_lattice_match_count": match_count,
            "hole_lattice_visibility_cut": visibility_cut,
        }

    src_pts = np.asarray(source_points, dtype=np.float32)
    dst_pts = np.asarray(target_points, dtype=np.float32)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    if homography is None or mask is None:
        return coarse, {
            "hole_lattice_refined": False,
            "hole_lattice_reason": "homography_failed",
            "hole_lattice_match_count": match_count,
        }

    inliers = mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count) / float(max(match_count, 1))
    min_inliers = max(60, int(round(match_count * 0.35)))
    if inlier_count < min_inliers:
        return coarse, {
            "hole_lattice_refined": False,
            "hole_lattice_reason": "too_few_inliers",
            "hole_lattice_match_count": match_count,
            "hole_lattice_inliers": inlier_count,
            "hole_lattice_inlier_ratio": inlier_ratio,
        }

    projected_inliers = perspective_points(src_pts[inliers], homography)
    reproj_error = float(np.mean(np.linalg.norm(projected_inliers - dst_pts[inliers], axis=1)))

    warp_corners = np.array(
        [[0, 0], [warped.shape[1] - 1, 0], [warped.shape[1] - 1, warped.shape[0] - 1], [0, warped.shape[0] - 1]],
        dtype=np.float32,
    )
    refined = order_points(perspective_points(warp_corners, homography)).astype(np.float32)
    refined[:, 0] = np.clip(refined[:, 0], 0, image.shape[1] - 1)
    refined[:, 1] = np.clip(refined[:, 1], 0, image.shape[0] - 1)

    coarse_area = max(_quad_area(coarse), 1.0)
    refined_area = _quad_area(refined)
    area_ratio = refined_area / coarse_area
    corner_shifts = np.linalg.norm(refined - coarse, axis=1)
    max_corner_shift = float(np.max(corner_shifts))
    mean_corner_shift = float(np.mean(corner_shifts))
    image_diag = float(np.hypot(image.shape[1], image.shape[0]))
    max_shift_limit = max(85.0, 0.045 * image_diag)

    accept = (
        np.all(np.isfinite(refined))
        and 0.85 <= area_ratio <= 1.10
        and reproj_error <= 4.0
        and max_corner_shift <= max_shift_limit
    )
    if not accept:
        return coarse, {
            "hole_lattice_refined": False,
            "hole_lattice_reason": "guard_rejected",
            "hole_lattice_match_count": match_count,
            "hole_lattice_inliers": inlier_count,
            "hole_lattice_inlier_ratio": inlier_ratio,
            "hole_lattice_reproj_error": reproj_error,
            "hole_lattice_area_ratio": area_ratio,
            "hole_lattice_max_corner_shift": max_corner_shift,
        }

    return refined, {
        "hole_lattice_refined": True,
        "hole_lattice_reason": "",
        "hole_lattice_match_count": match_count,
        "hole_lattice_inliers": inlier_count,
        "hole_lattice_inlier_ratio": inlier_ratio,
        "hole_lattice_reproj_error": reproj_error,
        "hole_lattice_visibility_cut": visibility_cut,
        "hole_lattice_search_radius": float(search_radius),
        "hole_lattice_mean_initial_shift": float(np.mean(initial_shifts)),
        "hole_lattice_mean_peak": float(np.mean(peak_values)),
        "hole_lattice_area_ratio": area_ratio,
        "hole_lattice_max_corner_shift": max_corner_shift,
        "hole_lattice_mean_corner_shift": mean_corner_shift,
    }


def process_image(
    *,
    image_path: Path,
    out_dir: Path,
    prefix: str,
    main_columns: int,
    debug: bool,
    white_threshold: int,
    max_saturation: int,
    min_value: int,
    padding_ratio: float,
    padding_px: float,
    fallback_auto: bool,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir if debug else None
    quad, _, detection_metrics = detect_white_region_quad(
        image,
        debug_dir=debug_dir,
        debug_prefix=prefix,
        white_threshold=white_threshold,
        max_saturation=max_saturation,
        min_value=min_value,
        padding_ratio=padding_ratio,
        padding_px=padding_px,
        fallback_auto=fallback_auto,
    )

    quad = order_points(quad)
    quad, lattice_metrics = _refine_quad_from_hole_lattice(image, quad, main_columns=main_columns)
    detection_metrics = dict(detection_metrics)
    detection_metrics.update(lattice_metrics)
    warped, _, inverse_matrix = warp_board(image, quad)
    model = fit_grid(warped, main_columns)
    score_map = score_image(warped)
    holes = build_holes(model, inverse_matrix, score_map, main_columns)
    regions = build_regions(model, inverse_matrix, (warped.shape[1], warped.shape[0]), main_columns)
    connections = build_connections(model, inverse_matrix, main_columns)
    holes, connections = annotate_connectivity(holes, connections)

    paths = {
        "corners": out_dir / f"{prefix}_corners.png",
        "warped": out_dir / f"{prefix}_warped.png",
        "annotated_warped": out_dir / f"{prefix}_annotated_warped.png",
        "annotated_original": out_dir / f"{prefix}_annotated_original.png",
        "connectivity_warped": out_dir / f"{prefix}_connectivity_warped.png",
        "connectivity_original": out_dir / f"{prefix}_connectivity_original.png",
        "csv": out_dir / f"{prefix}_holes.csv",
        "connections_csv": out_dir / f"{prefix}_connections.csv",
        "json": out_dir / f"{prefix}_holes.json",
    }

    cv2.imwrite(str(paths["corners"]), draw_quad(image, quad))
    cv2.imwrite(str(paths["warped"]), warped)
    cv2.imwrite(str(paths["annotated_warped"]), draw_holes(warped, holes, "warp", draw_labels=False))
    cv2.imwrite(str(paths["annotated_original"]), draw_holes(image, holes, "image", draw_labels=False))
    cv2.imwrite(str(paths["connectivity_warped"]), draw_region_and_connection_overlay(warped, holes, regions, connections, "warp", draw_labels=False))
    cv2.imwrite(str(paths["connectivity_original"]), draw_region_and_connection_overlay(image, holes, regions, connections, "image", draw_labels=False))
    write_csv(paths["csv"], holes)
    write_connections_csv(paths["connections_csv"], connections)

    metadata: dict[str, Any] = {
        "image": str(image_path),
        "hole_count": len(holes),
        "main_columns": main_columns,
        "corner_mode": "white_region",
        "detection_metrics": detection_metrics,
        "quad_tl_tr_br_bl": [[round(float(x), 3), round(float(y), 3)] for x, y in order_points(quad)],
        "warped_size": {"width": warped.shape[1], "height": warped.shape[0]},
        "grid": model.__dict__,
        "regions": [asdict(region) for region in regions],
        "connections": [asdict(connection) for connection in connections],
        "holes": holes,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    with paths["json"].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breadboard detection using white-region segmentation instead of black corner dots.")
    parser.add_argument("--image", default="bread_black_point.jpg", type=Path, help="Input breadboard image.")
    parser.add_argument("--out-dir", default=Path("outputs"), type=Path, help="Directory for generated files.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to '<image stem>_white'.")
    parser.add_argument("--main-columns", default=63, type=int, help="Terminal-strip columns. Standard 830-point boards use 63.")
    parser.add_argument("--debug", action="store_true", help="Write intermediate masks and candidate overlays.")
    parser.add_argument("--white-threshold", type=int, default=185, help="LAB lightness threshold for white plastic.")
    parser.add_argument("--max-saturation", type=int, default=125, help="Maximum HSV saturation accepted as pale board material.")
    parser.add_argument("--min-value", type=int, default=135, help="Minimum HSV value accepted as pale board material.")
    parser.add_argument("--padding-ratio", type=float, default=0.006, help="Small outward expansion of the final quad.")
    parser.add_argument("--padding-px", type=float, default=4.0, help="Additional pixel padding for the final quad.")
    parser.add_argument("--no-fallback-auto", action="store_true", help="Fail instead of falling back to the old auto detector.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.prefix or f"{args.image.stem}_white"
    result = process_image(
        image_path=args.image,
        out_dir=args.out_dir,
        prefix=prefix,
        main_columns=args.main_columns,
        debug=bool(args.debug),
        white_threshold=int(args.white_threshold),
        max_saturation=int(args.max_saturation),
        min_value=int(args.min_value),
        padding_ratio=float(args.padding_ratio),
        padding_px=float(args.padding_px),
        fallback_auto=not bool(args.no_fallback_auto),
    )
    print(f"Detected {result['hole_count']} holes.")
    print(f"corner_mode: {result['corner_mode']}")
    print(f"Board corners TL/TR/BR/BL: {result['quad_tl_tr_br_bl']}")
    print(f"detection_metrics: {result['detection_metrics']}")
    for name, path in (result.get("paths") or {}).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
