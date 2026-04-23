#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.vision.calibrator import BreadboardCalibrator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize breadboard 2D calibration results.")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Absolute image paths to visualize.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/labguardian_calibrator_debug"),
        help="Directory to write visualizations into.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=63,
        help="Expected breadboard rows.",
    )
    parser.add_argument(
        "--cols-per-side",
        type=int,
        default=5,
        help="Expected breadboard columns per side.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for image_path_str in args.images:
        image_path = Path(image_path_str)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"skip unreadable image: {image_path}")
            continue

        calibrator = BreadboardCalibrator(rows=args.rows, cols_per_side=args.cols_per_side)
        success = calibrator.ensure_calibrated(image)
        warped = calibrator.warp(image)

        out_dir = args.output_root / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        orig_overlay = image.copy()
        warped_overlay = warped.copy()
        warped_points_only = warped.copy()
        orig_demo = image.copy()
        warped_demo = warped.copy()

        draw_board_candidates(orig_overlay, calibrator, image)
        draw_original_grid_overlay(orig_overlay, calibrator)
        draw_warped_grid_overlay(warped_overlay, calibrator)
        draw_warped_points_alignment(warped_points_only, calibrator)
        draw_anchor_grid_demo(warped_demo, calibrator)
        draw_original_anchor_grid_demo(orig_demo, calibrator)

        cv2.imwrite(str(out_dir / "original_with_grid.png"), orig_overlay)
        cv2.imwrite(str(out_dir / "warped_grid.png"), warped_overlay)
        cv2.imwrite(str(out_dir / "warped_grid_points.png"), warped_points_only)
        cv2.imwrite(str(out_dir / "warped_anchor_grid_demo.png"), warped_demo)
        cv2.imwrite(str(out_dir / "original_anchor_grid_demo.png"), orig_demo)
        cv2.imwrite(str(out_dir / "warped_raw.png"), warped)

        summary = build_summary(image_path, calibrator, success)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"saved: {out_dir}")
    return 0


def build_summary(image_path: Path, calibrator: BreadboardCalibrator, success: bool) -> dict:
    alignment = compute_alignment_metrics(calibrator)
    observed = summarize_indexed_holes(calibrator)
    return {
        "image_path": str(image_path),
        "success": bool(success),
        "is_grid_ready": bool(calibrator.is_grid_ready),
        "mode": "synthetic_fallback" if getattr(calibrator, "_synthetic_grid", False) else "visual",
        "has_perspective_matrix": calibrator._perspective_matrix is not None,  # type: ignore[attr-defined]
        "has_inverse_perspective": calibrator._inv_perspective is not None,  # type: ignore[attr-defined]
        "row_count": int(len(calibrator.row_coords)) if calibrator.row_coords is not None else 0,
        "col_count": int(len(calibrator.col_coords)) if calibrator.col_coords is not None else 0,
        "landscape": bool(calibrator.landscape),
        "top_rails": [float(v) for v in getattr(calibrator, "_top_rails", [])],
        "bot_rails": [float(v) for v in getattr(calibrator, "_bot_rails", [])],
        "roi_rect": list(map(int, calibrator.get_roi_rect((calibrator._img_h or 0, calibrator._img_w or 0, 3))))  # type: ignore[attr-defined]
        if getattr(calibrator, "_img_h", 0) and getattr(calibrator, "_img_w", 0)
        else None,
        "grid_origin": list(getattr(calibrator, "_grid_origin", []) or []),
        "grid_spacing": list(getattr(calibrator, "_grid_spacing", []) or []),
        "rail_tolerance": float(getattr(calibrator, "_rail_tolerance", 0.0)),
        "indexed_holes": observed,
        "alignment": alignment,
    }


def draw_board_candidates(canvas: np.ndarray, calibrator: BreadboardCalibrator, image: np.ndarray) -> None:
    try:
        candidates = calibrator._detect_board_region_candidates(image)  # type: ignore[attr-defined]
    except Exception:
        candidates = []
    for idx, quad in enumerate(candidates[:3]):
        pts = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
        color = [(255, 255, 0), (0, 255, 255), (255, 128, 0)][idx % 3]
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        first = tuple(int(v) for v in pts[0, 0])
        cv2.putText(canvas, f"cand{idx+1}", first, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_original_grid_overlay(canvas: np.ndarray, calibrator: BreadboardCalibrator) -> None:
    points = collect_grid_points(calibrator)
    if not points:
        return
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    inv = getattr(calibrator, "_inv_perspective", None)
    if inv is not None:
        pts = cv2.perspectiveTransform(pts, inv)
    for (x, y), meta in zip(pts.reshape(-1, 2), collect_grid_meta(calibrator)):
        cv2.circle(canvas, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
        if meta["draw_label"]:
            cv2.putText(
                canvas,
                meta["label"],
                (int(round(x)) + 4, int(round(y)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )


def draw_warped_grid_overlay(canvas: np.ndarray, calibrator: BreadboardCalibrator) -> None:
    if calibrator.row_coords is None or calibrator.col_coords is None:
        return
    row_coords = calibrator.row_coords
    col_coords = calibrator.col_coords

    for ri, row_val in enumerate(row_coords):
        color = (60, 180, 60) if ri % 5 else (0, 220, 0)
        if calibrator.landscape:
            pt1 = (int(round(row_val)), int(round(col_coords[0])))
            pt2 = (int(round(row_val)), int(round(col_coords[-1])))
        else:
            pt1 = (int(round(col_coords[0])), int(round(row_val)))
            pt2 = (int(round(col_coords[-1])), int(round(row_val)))
        cv2.line(canvas, pt1, pt2, color, 1)

    for ci, col_val in enumerate(col_coords):
        color = (180, 60, 60) if ci not in (4, 5) else (0, 165, 255)
        if calibrator.landscape:
            pt1 = (int(round(row_coords[0])), int(round(col_val)))
            pt2 = (int(round(row_coords[-1])), int(round(col_val)))
        else:
            pt1 = (int(round(col_val)), int(round(row_coords[0])))
            pt2 = (int(round(col_val)), int(round(row_coords[-1])))
        cv2.line(canvas, pt1, pt2, color, 1)

    for (x, y), meta in zip(collect_grid_points(calibrator), collect_grid_meta(calibrator)):
        cv2.circle(canvas, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
        if meta["draw_label"]:
            cv2.putText(
                canvas,
                meta["label"],
                (int(round(x)) + 4, int(round(y)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )


def draw_warped_points_alignment(canvas: np.ndarray, calibrator: BreadboardCalibrator) -> None:
    hole_centers = getattr(calibrator, "hole_centers", []) or []
    for hx, hy in hole_centers:
        cv2.circle(canvas, (int(round(hx)), int(round(hy))), 2, (255, 180, 0), -1)

    points = collect_grid_points(calibrator)
    metas = collect_grid_meta(calibrator)
    hole_arr = np.asarray(hole_centers, dtype=np.float32) if hole_centers else None

    for (x, y), meta in zip(points, metas):
        anchor = bool(meta.get("anchor", meta.get("observed", False)))
        color = (255, 0, 255) if not anchor else (0, 0, 255)
        radius = 3
        if hole_arr is not None and len(hole_arr) > 0:
            dists = np.sqrt(((hole_arr[:, 0] - x) ** 2) + ((hole_arr[:, 1] - y) ** 2))
            min_dist = float(np.min(dists))
            if not anchor:
                color = (255, 0, 255)
            elif min_dist <= 3.0:
                color = (0, 220, 0)
            elif min_dist <= 6.0:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            radius = 4 if min_dist > 6.0 else 3

        cv2.circle(canvas, (int(round(x)), int(round(y))), radius, color, -1)
        if meta["draw_label"]:
            cv2.putText(
                canvas,
                meta["label"],
                (int(round(x)) + 4, int(round(y)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )


def _reliable_line_masks(calibrator: BreadboardCalibrator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed_main = getattr(calibrator, "_observed_main_mask", None)
    observed_top = getattr(calibrator, "_observed_top_mask", None)
    observed_bot = getattr(calibrator, "_observed_bot_mask", None)
    valid_main = getattr(calibrator, "_valid_main_mask", None)

    if observed_main is None or valid_main is None:
        return (
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=bool),
        )

    row_mask = (observed_main.sum(axis=1) >= 4)
    col_mask = (observed_main.sum(axis=0) >= max(8, observed_main.shape[0] // 6))
    valid_point_mask = valid_main.astype(bool)
    top_mask = (observed_top.sum(axis=0) >= max(10, observed_top.shape[0] // 4)) if observed_top is not None else np.zeros(2, dtype=bool)
    bot_mask = (observed_bot.sum(axis=0) >= max(10, observed_bot.shape[0] // 4)) if observed_bot is not None else np.zeros(2, dtype=bool)
    return row_mask, col_mask, valid_point_mask, top_mask, bot_mask


def draw_anchor_grid_demo(canvas: np.ndarray, calibrator: BreadboardCalibrator) -> None:
    if calibrator.row_coords is None or calibrator.col_coords is None:
        return
    row_mask, col_mask, valid_main_mask, top_mask, bot_mask = _reliable_line_masks(calibrator)
    row_coords = calibrator.row_coords
    col_coords = calibrator.col_coords

    x_min = int(round(row_coords[0]))
    x_max = int(round(row_coords[-1]))
    y_min = int(round(col_coords[0]))
    y_max = int(round(col_coords[-1]))

    for ri, row_val in enumerate(row_coords):
        if ri < len(row_mask) and not row_mask[ri]:
            continue
        cv2.line(canvas, (int(round(row_val)), y_min), (int(round(row_val)), y_max), (80, 200, 255), 1)

    for ci, col_val in enumerate(col_coords):
        if ci < len(col_mask) and not col_mask[ci]:
            continue
        cv2.line(canvas, (x_min, int(round(col_val))), (x_max, int(round(col_val))), (60, 220, 120), 1)

    for rail_idx, rail_val in enumerate(getattr(calibrator, "_top_rails", []) or []):
        if rail_idx < len(top_mask) and top_mask[rail_idx]:
            cv2.line(canvas, (x_min, int(round(rail_val))), (x_max, int(round(rail_val))), (255, 170, 0), 1)
    for rail_idx, rail_val in enumerate(getattr(calibrator, "_bot_rails", []) or []):
        if rail_idx < len(bot_mask) and bot_mask[rail_idx]:
            cv2.line(canvas, (x_min, int(round(rail_val))), (x_max, int(round(rail_val))), (255, 170, 0), 1)

    if getattr(calibrator, "_grid_matrix", None) is not None:
        grid = calibrator._grid_matrix  # type: ignore[attr-defined]
        observed_main = getattr(calibrator, "_observed_main_mask", None)
        for r in range(grid.shape[0]):
            if r < len(row_mask) and not row_mask[r]:
                continue
            for c in range(grid.shape[1]):
                if c < len(col_mask) and not col_mask[c]:
                    continue
                if not valid_main_mask[r, c]:
                    continue
                x, y = grid[r, c]
                color = (0, 255, 0) if bool(observed_main[r, c]) else (255, 255, 255)
                radius = 3 if bool(observed_main[r, c]) else 2
                cv2.circle(canvas, (int(round(x)), int(round(y))), radius, color, -1)

    for matrix_name, observed_name in [("_top_rail_matrix", "_observed_top_mask"), ("_bot_rail_matrix", "_observed_bot_mask")]:
        matrix = getattr(calibrator, matrix_name, None)
        observed = getattr(calibrator, observed_name, None)
        valid = getattr(calibrator, "_valid_top_mask" if matrix_name == "_top_rail_matrix" else "_valid_bot_mask", None)
        mask = top_mask if matrix_name == "_top_rail_matrix" else bot_mask
        if matrix is None or observed is None or valid is None:
            continue
        for r in range(matrix.shape[0]):
            for rail_idx in range(matrix.shape[1]):
                if rail_idx < len(mask) and not mask[rail_idx]:
                    continue
                if not valid[r, rail_idx]:
                    continue
                x, y = matrix[r, rail_idx]
                color = (0, 255, 0) if bool(observed[r, rail_idx]) else (255, 255, 255)
                radius = 3 if bool(observed[r, rail_idx]) else 2
                cv2.circle(canvas, (int(round(x)), int(round(y))), radius, color, -1)


def draw_original_anchor_grid_demo(canvas: np.ndarray, calibrator: BreadboardCalibrator) -> None:
    inv = getattr(calibrator, "_inv_perspective", None)
    if inv is None:
        return
    indexed = [item for item in calibrator.iter_indexed_holes() if bool(item.get("anchor", item.get("observed", False)))]
    if not indexed:
        return

    def project_points(points: np.ndarray) -> np.ndarray:
        pts = points.astype(np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, inv).reshape(-1, 2)

    line_segments: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int], int]] = []

    def add_segment(
        p1: tuple[float, float],
        p2: tuple[float, float],
        color: tuple[int, int, int],
        thickness: int,
        max_dist: float,
    ) -> None:
        dist = float(np.linalg.norm(np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)))
        if dist <= 0.0 or dist > max_dist:
            return
        p1i = tuple(int(round(v)) for v in p1)
        p2i = tuple(int(round(v)) for v in p2)
        line_segments.append((p1i, p2i, color, thickness))

    projected_map: list[dict[str, object]] = []
    warped_points = np.asarray([item["pixel"] for item in indexed], dtype=np.float32)
    original_points = project_points(warped_points)
    for item, (ox, oy) in zip(indexed, original_points):
        projected_map.append(
            {
                "group": str(item.get("group") or "main"),
                "logic_loc": tuple(item["logic_loc"]),
                "point": (float(ox), float(oy)),
            }
        )

    main_points: dict[tuple[int, str], tuple[float, float]] = {}
    top_rail_points: dict[tuple[int, str], tuple[float, float]] = {}
    bot_rail_points: dict[tuple[int, str], tuple[float, float]] = {}

    for item in projected_map:
        group = str(item["group"])
        logic_loc = tuple(item["logic_loc"])
        pt = tuple(item["point"])
        if group == "main":
            main_points[(int(str(logic_loc[0])), str(logic_loc[1]))] = pt
        elif group == "top_rail":
            top_rail_points[(int(str(logic_loc[0])), str(logic_loc[1]))] = pt
        elif group == "bot_rail":
            bot_rail_points[(int(str(logic_loc[0])), str(logic_loc[1]))] = pt

    upper_cols = ["a", "b", "c", "d", "e"]
    lower_cols = ["f", "g", "h", "i", "j"]
    all_cols = upper_cols + lower_cols

    # Horizontal short segments inside the upper and lower main regions.
    for row in range(1, 64):
        for left, right in zip(upper_cols, upper_cols[1:]):
            p1 = main_points.get((row, left))
            p2 = main_points.get((row, right))
            if p1 is not None and p2 is not None:
                add_segment(p1, p2, (0, 200, 255), 2, max_dist=32.0)
        for left, right in zip(lower_cols, lower_cols[1:]):
            p1 = main_points.get((row, left))
            p2 = main_points.get((row, right))
            if p1 is not None and p2 is not None:
                add_segment(p1, p2, (0, 200, 255), 2, max_dist=32.0)

    # Vertical short segments within each column, never across adjacent columns.
    for col in all_cols:
        for row in range(1, 63):
            p1 = main_points.get((row, col))
            p2 = main_points.get((row + 1, col))
            if p1 is not None and p2 is not None:
                add_segment(p1, p2, (60, 220, 120), 1, max_dist=24.0)

    for rail_name in ("rail_top+", "rail_top-"):
        for row in range(1, 50):
            p1 = top_rail_points.get((row, rail_name))
            p2 = top_rail_points.get((row + 1, rail_name))
            if p1 is not None and p2 is not None:
                add_segment(p1, p2, (255, 170, 0), 2, max_dist=34.0)
    for rail_name in ("rail_bot+", "rail_bot-"):
        for row in range(1, 50):
            p1 = bot_rail_points.get((row, rail_name))
            p2 = bot_rail_points.get((row + 1, rail_name))
            if p1 is not None and p2 is not None:
                add_segment(p1, p2, (255, 170, 0), 2, max_dist=34.0)

    h, w = canvas.shape[:2]
    for item in projected_map:
        ox, oy = item["point"]
        xi = int(round(ox))
        yi = int(round(oy))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(canvas, (xi, yi), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    for p1i, p2i, color, thickness in line_segments:
        cv2.line(canvas, p1i, p2i, color, thickness, lineType=cv2.LINE_AA)


def compute_alignment_metrics(calibrator: BreadboardCalibrator) -> dict:
    points = collect_grid_points(calibrator)
    hole_centers = getattr(calibrator, "hole_centers", []) or []
    if not points or not hole_centers:
        return {
            "grid_point_count": len(points),
            "hole_center_count": len(hole_centers),
            "mean_nearest_hole_distance_px": None,
            "median_nearest_hole_distance_px": None,
            "p90_nearest_hole_distance_px": None,
            "within_3px_ratio": None,
            "within_6px_ratio": None,
        }
    hole_arr = np.asarray(hole_centers, dtype=np.float32)
    nearest = []
    for x, y in points:
        dists = np.sqrt(((hole_arr[:, 0] - x) ** 2) + ((hole_arr[:, 1] - y) ** 2))
        nearest.append(float(np.min(dists)))
    nearest_arr = np.asarray(nearest, dtype=np.float32)
    return {
        "grid_point_count": len(points),
        "hole_center_count": len(hole_centers),
        "mean_nearest_hole_distance_px": float(np.mean(nearest_arr)),
        "median_nearest_hole_distance_px": float(np.median(nearest_arr)),
        "p90_nearest_hole_distance_px": float(np.percentile(nearest_arr, 90)),
        "within_3px_ratio": float(np.mean(nearest_arr <= 3.0)),
        "within_6px_ratio": float(np.mean(nearest_arr <= 6.0)),
    }


def summarize_indexed_holes(calibrator: BreadboardCalibrator) -> dict:
    indexed = calibrator.iter_indexed_holes()
    by_group: dict[str, dict[str, int]] = {}
    expected_by_group = {
        "main": 63 * 10,
        "top_rail": 50 * 2,
        "bot_rail": 50 * 2,
    }
    for item in indexed:
        group = str(item.get("group") or "unknown")
        observed = bool(item.get("observed", False))
        anchor = bool(item.get("anchor", observed))
        slot = by_group.setdefault(group, {"observed": 0, "interpolated": 0, "anchors": 0, "total": 0})
        slot["total"] += 1
        slot["observed" if observed else "interpolated"] += 1
        if anchor:
            slot["anchors"] += 1
    for group, expected in expected_by_group.items():
        slot = by_group.setdefault(group, {"observed": 0, "interpolated": 0, "anchors": 0, "total": 0})
        slot["expected"] = expected
        slot["coverage_ratio"] = float(slot["observed"] / expected) if expected > 0 else 0.0
        slot["anchor_ratio"] = float(slot["anchors"] / expected) if expected > 0 else 0.0
    total_observed = sum(v["observed"] for v in by_group.values())
    total_interpolated = sum(v["interpolated"] for v in by_group.values())
    total_anchors = sum(v["anchors"] for v in by_group.values())
    return {
        "total_observed": total_observed,
        "total_interpolated": total_interpolated,
        "total_anchors": total_anchors,
        "expected_total": sum(expected_by_group.values()),
        "observed_ratio": float(total_observed / sum(expected_by_group.values())),
        "anchor_ratio": float(total_anchors / sum(expected_by_group.values())),
        "by_group": by_group,
    }


def collect_grid_points(calibrator: BreadboardCalibrator) -> list[tuple[float, float]]:
    indexed = calibrator.iter_indexed_holes()
    if indexed:
        return [tuple(item["pixel"]) for item in indexed]
    if calibrator.row_coords is None or calibrator.col_coords is None:
        return []
    points: list[tuple[float, float]] = []
    for ri, row_val in enumerate(calibrator.row_coords):
        for ci, col_val in enumerate(calibrator.col_coords):
            if calibrator.landscape:
                points.append((float(row_val), float(col_val)))
            else:
                points.append((float(col_val), float(row_val)))
    return points


def collect_grid_meta(calibrator: BreadboardCalibrator) -> list[dict[str, object]]:
    indexed = calibrator.iter_indexed_holes()
    if indexed:
        metas = []
        for item in indexed:
            hole_id = str(item["hole_id"])
            group = str(item.get("group") or "main")
            row_num = int(str(item["logic_loc"][0]))
            col_logic = str(item["logic_loc"][1])
            draw_label = group == "main" and (row_num % 8 == 1) and (col_logic in ("a", "e", "f", "j"))
            metas.append(
                {
                    "label": hole_id.lower(),
                    "draw_label": draw_label,
                    "observed": bool(item.get("observed", False)),
                    "anchor": bool(item.get("anchor", item.get("observed", False))),
                    "source": str(item.get("source", "anchor" if item.get("observed", False) else "inferred_from_anchor")),
                }
            )
        return metas
    if calibrator.row_coords is None or calibrator.col_coords is None:
        return []
    metas = []
    col_names = list("abcde") + list("fghij")
    for ri, _ in enumerate(calibrator.row_coords):
        for ci, _ in enumerate(calibrator.col_coords):
            draw_label = (ri % 8 == 0) and (ci in (0, 4, 5, 9))
            metas.append(
                {
                    "label": f"{col_names[ci]}{ri+1}",
                    "draw_label": draw_label,
                }
            )
    return metas


if __name__ == "__main__":
    raise SystemExit(main())
