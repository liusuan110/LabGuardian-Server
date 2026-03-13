"""
Stage 2: 坐标映射

将 S1 的像素级检测结果映射到面包板逻辑坐标，
确定每个元件占用的 (row, col) 引脚对。
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.pipeline.vision.calibrator import BreadboardCalibrator
from app.pipeline.vision.pin_utils import select_best_pin_pair
from app.pipeline.vision.pin_hole_detector import compensate_occluded_pins

logger = logging.getLogger(__name__)


def run_mapping(
    detections: List[dict],
    calibrator: BreadboardCalibrator,
    image_shape: Tuple[int, int],
    images_b64: List[str] | None = None,
    pinned_hints: List[dict] | None = None,
) -> Dict[str, Any]:
    """把像素坐标 → 面包板逻辑坐标

    Args:
        detections: S1 检测结果 (仅元件)
        calibrator: 面包板校准器
        image_shape: 主图 (h, w)
        images_b64: 原始图片 (用于校准)
        pinned_hints: S1 检测到的引脚孔洞辅助信息 [{center, bbox, confidence}]
    """
    t0 = time.time()

    # ── 确保校准器已校准 ──
    primary_image = None
    if images_b64:
        primary_image = _decode_primary_image(images_b64[0])
        _ensure_calibrated(calibrator, images_b64[0], image_shape)
    elif image_shape[0] > 0 and image_shape[1] > 0:
        calibrator.build_synthetic_grid(image_shape)

    # ── 遮挡补偿: 用视觉搜索精炼被遮挡的引脚像素位置 ──
    if primary_image is not None and calibrator.is_grid_ready:
        compensate_occluded_pins(primary_image, detections, calibrator)

    mapped: List[dict] = []

    # ── 用 pinned_hints 精确化引脚像素坐标 ──
    pin_centers = []
    if pinned_hints:
        pin_centers = [tuple(p["center"]) for p in pinned_hints if p.get("center")]

    for det in detections:
        comp = dict(det)  # shallow copy

        p1_px = tuple(det["pin1_pixel"]) if det.get("pin1_pixel") else None
        p2_px = tuple(det["pin2_pixel"]) if det.get("pin2_pixel") else None
        bbox = tuple(det["bbox"])
        class_name = det["class_name"]

        if pin_centers:
            p1_px, p2_px = _refine_pins_with_pinned(
                p1_px, p2_px, bbox, pin_centers,
            )

        p1_candidates = _get_candidates(p1_px, calibrator)
        p2_candidates = _get_candidates(p2_px, calibrator)

        if not p1_candidates or not p2_candidates:
            bp1, bp2 = _infer_pixels_from_bbox(bbox)
            if not p1_candidates:
                p1_candidates = _get_candidates(bp1, calibrator)
            if not p2_candidates:
                p2_candidates = _get_candidates(bp2, calibrator)

        pin1_logic = None
        pin2_logic = None
        if p1_candidates and p2_candidates:
            pin1_logic, pin2_logic = select_best_pin_pair(
                p1_candidates, p2_candidates, class_name,
            )
        elif p1_candidates:
            pin1_logic = p1_candidates[0]
        elif p2_candidates:
            pin2_logic = p2_candidates[0]

        comp["pin1_logic"] = list(pin1_logic) if pin1_logic else None
        comp["pin2_logic"] = list(pin2_logic) if pin2_logic else None
        mapped.append(comp)

    # 后处理: Wire 双轨修正 → 行合并/LED桥接 → Wire端点吸附
    _fix_wire_dual_rail(mapped, calibrator)
    _refine_row_connectivity(mapped)
    _snap_wire_to_components(mapped)

    duration_ms = (time.time() - t0) * 1000
    return {"components": mapped, "duration_ms": duration_ms}


def _decode_primary_image(image_b64: str) -> Optional[np.ndarray]:
    """解码第一张 base64 图片, 供遮挡补偿使用"""
    try:
        data = base64.b64decode(image_b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _ensure_calibrated(
    calibrator: BreadboardCalibrator,
    image_b64: str,
    image_shape: Tuple[int, int],
):
    """确保校准器已校准"""
    if calibrator.is_grid_ready:
        return
    try:
        data = base64.b64decode(image_b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            calibrator.ensure_calibrated(img)
            if calibrator.is_grid_ready:
                return
    except Exception as e:
        logger.warning("Calibration from image failed: %s", e)
    logger.info("Falling back to synthetic grid")
    calibrator.build_synthetic_grid(image_shape)


def _get_candidates(
    pixel: Optional[Tuple[float, float]],
    calibrator: BreadboardCalibrator,
    k: int = 5,
) -> List[Tuple[str, str]]:
    """像素坐标 → Top-K 逻辑坐标候选"""
    if pixel is None:
        return []
    try:
        candidates = calibrator.frame_pixel_to_logic_candidates(pixel[0], pixel[1], k=k)
        return candidates
    except Exception:
        return []


def _infer_pixels_from_bbox(
    bbox: tuple,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """从 bbox 的两端推断引脚像素位置"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    if w >= h:
        return ((float(x1), float(cy)), (float(x2), float(cy)))
    else:
        return ((float(cx), float(y1)), (float(cx), float(y2)))


def _refine_pins_with_pinned(
    p1_px: Optional[Tuple[float, float]],
    p2_px: Optional[Tuple[float, float]],
    bbox: tuple,
    pin_centers: List[Tuple[float, float]],
    search_expand: int = 40,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """用 pinned 检测的孔洞中心来精确化元件的引脚像素位置。"""
    x1, y1, x2, y2 = bbox
    nearby = []
    for pc in pin_centers:
        px, py = pc
        if (x1 - search_expand <= px <= x2 + search_expand and
            y1 - search_expand <= py <= y2 + search_expand):
            nearby.append(pc)

    if len(nearby) < 1:
        return p1_px, p2_px

    if len(nearby) == 1:
        nc = nearby[0]
        d1 = _pixel_dist(p1_px, nc) if p1_px else float("inf")
        d2 = _pixel_dist(p2_px, nc) if p2_px else float("inf")
        if d1 <= d2:
            return nc, p2_px
        else:
            return p1_px, nc

    if p1_px and p2_px:
        sorted_by_p1 = sorted(nearby, key=lambda c: _pixel_dist(p1_px, c))
        sorted_by_p2 = sorted(nearby, key=lambda c: _pixel_dist(p2_px, c))
        best_p1 = sorted_by_p1[0]
        best_p2 = sorted_by_p2[0]
        if best_p2 == best_p1 and len(sorted_by_p2) > 1:
            best_p2 = sorted_by_p2[1]
        return best_p1, best_p2
    elif p1_px:
        best = min(nearby, key=lambda c: _pixel_dist(p1_px, c))
        return best, p2_px
    elif p2_px:
        best = min(nearby, key=lambda c: _pixel_dist(p2_px, c))
        return p1_px, best
    return p1_px, p2_px


def _pixel_dist(
    a: Optional[Tuple[float, float]],
    b: Tuple[float, float],
) -> float:
    if a is None:
        return float("inf")
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _fix_wire_dual_rail(mapped: List[dict], calibrator: BreadboardCalibrator):
    """Wire 双轨修正: 如果 Wire 两端都在电轨上, 修正距主 grid 更近的那端"""
    col_names = list("abcde") + list("fghij")
    for comp in mapped:
        if comp.get("class_name", "").lower() != "wire":
            continue
        p1 = comp.get("pin1_logic")
        p2 = comp.get("pin2_logic")
        if not p1 or not p2:
            continue
        p1_is_rail = p1[1].startswith("rail_")
        p2_is_rail = p2[1].startswith("rail_")
        if not (p1_is_rail and p2_is_rail):
            continue

        best_pn = None
        best_dist = float("inf")
        for pn in [1, 2]:
            pp = comp.get(f"pin{pn}_pixel")
            if pp is None:
                continue
            px, py = pp
            col_val = py if calibrator.landscape else px
            if calibrator.col_coords is not None:
                dist = float(np.min(np.abs(calibrator.col_coords - col_val)))
                if dist < best_dist:
                    best_dist = dist
                    best_pn = pn

        if best_pn is None:
            continue

        pp = comp.get(f"pin{best_pn}_pixel")
        px, py = pp
        if calibrator.landscape:
            col_val = py
            row_val = px
        else:
            col_val = px
            row_val = py

        if calibrator.col_coords is not None:
            col_dists = np.abs(calibrator.col_coords - col_val)
            col_idx = int(np.argmin(col_dists))
            col_name = col_names[col_idx] if col_idx < len(col_names) else str(col_idx)
            row_idx = int(np.argmin(np.abs(calibrator.row_coords - row_val)))
            old_col = comp[f"pin{best_pn}_logic"][1]
            new_logic = [str(row_idx + 1), col_name]
            comp[f"pin{best_pn}_logic"] = new_logic
            logger.warning(
                "[S2 Fix] Wire pin%d: %s → %s (rail→grid, px=%s, row_val=%.1f, col_val=%.1f, row_idx=%d)",
                best_pn, old_col, new_logic, pp, row_val, col_val, row_idx,
            )


def _snap_wire_to_components(mapped: List[dict], max_snap_gap: int = 20):
    """Wire 端点吸附: 将孤立的 Wire 非电轨端点吸附到同侧最近的非 Wire 元件引脚行."""
    comp_pins = {}
    for idx, comp in enumerate(mapped):
        ctype = comp.get("class_name", "")
        if ctype.lower() == "wire":
            continue
        for pn in [1, 2]:
            logic = comp.get(f"pin{pn}_logic")
            if not logic:
                continue
            row_str, col_str = logic[0], logic[1]
            if col_str.startswith("rail_"):
                continue
            try:
                row_int = int(row_str)
            except (ValueError, TypeError):
                continue
            side = "L" if col_str in "abcde" else "R"
            comp_pins.setdefault((row_int, side), []).append((idx, pn))

    if not comp_pins:
        return

    for wire_idx, comp in enumerate(mapped):
        if comp.get("class_name", "").lower() != "wire":
            continue
        for pn in [1, 2]:
            logic = comp.get(f"pin{pn}_logic")
            if not logic:
                continue
            row_str, col_str = logic[0], logic[1]
            if col_str.startswith("rail_"):
                continue
            try:
                wire_row = int(row_str)
            except (ValueError, TypeError):
                continue
            side = "L" if col_str in "abcde" else "R"

            if (wire_row, side) in comp_pins:
                continue

            best_row = None
            best_gap = max_snap_gap + 1
            for (cr, cs), _ in comp_pins.items():
                if cs != side:
                    continue
                gap = abs(wire_row - cr)
                if gap < best_gap:
                    best_gap = gap
                    best_row = cr

            if best_row is not None and best_gap <= max_snap_gap:
                old_row = logic[0]
                logic[0] = str(best_row)
                logger.info(
                    "[S2 Snap] Wire pin%d: Row %s → Row %d (snap to component, gap=%d, side=%s)",
                    pn, old_row, best_row, best_gap, side,
                )


def _refine_row_connectivity(mapped: List[dict], max_row_gap: int = 2):
    """后处理: 如果两个元件的引脚行号相差 ≤ max_row_gap, 将它们合并到同一行"""
    priority = {"Resistor": 3, "Capacitor": 3, "LED": 2, "Wire": 1}
    wire_max_row_gap = 20

    def _get_pin_info(comp_idx: int, pin_num: int):
        logic = mapped[comp_idx].get(f"pin{pin_num}_logic")
        if not logic:
            return None
        row_str, col_str = logic[0], logic[1]
        if col_str.startswith("rail_"):
            return None
        try:
            return (int(row_str), "L" if col_str in "abcde" else "R", col_str)
        except (ValueError, TypeError):
            return None

    adjusted = set()
    anchor_order = sorted(range(len(mapped)),
                          key=lambda i: priority.get(mapped[i].get("class_name", ""), 2),
                          reverse=True)

    for anchor_idx in anchor_order:
        anchor_comp = mapped[anchor_idx]
        anchor_type = anchor_comp.get("class_name", "")
        anchor_pri = priority.get(anchor_type, 2)

        for anchor_pn in [1, 2]:
            anchor_info = _get_pin_info(anchor_idx, anchor_pn)
            if anchor_info is None:
                continue
            anchor_row, anchor_side, _ = anchor_info

            for target_idx, target_comp in enumerate(mapped):
                if target_idx == anchor_idx:
                    continue
                target_type = target_comp.get("class_name", "")
                target_pri = priority.get(target_type, 2)
                if target_pri >= anchor_pri:
                    continue

                for target_pn in [1, 2]:
                    if (target_idx, target_pn) in adjusted:
                        continue
                    target_info = _get_pin_info(target_idx, target_pn)
                    if target_info is None:
                        continue
                    target_row, target_side, _ = target_info

                    if target_side != anchor_side:
                        continue
                    gap = abs(target_row - anchor_row)
                    effective_gap = wire_max_row_gap if target_type.lower() == "wire" else max_row_gap
                    if gap == 0 or gap > effective_gap:
                        continue

                    logic = mapped[target_idx][f"pin{target_pn}_logic"]
                    old_row = logic[0]
                    logic[0] = str(anchor_row)
                    adjusted.add((target_idx, target_pn))
                    logger.info(
                        "[S2 Refine] %s pin%d: Row %s → Row %d (align with %s)",
                        target_type, target_pn, old_row, anchor_row, anchor_type,
                    )

    for comp in mapped:
        if comp.get("class_name", "").lower() != "led":
            continue
        p1 = comp.get("pin1_logic")
        p2 = comp.get("pin2_logic")
        if not p1 or not p2:
            continue
        r1, c1 = p1[0], p1[1]
        r2, c2 = p2[0], p2[1]
        if c1.startswith("rail_") or c2.startswith("rail_"):
            continue
        s1 = "L" if c1 in "abcde" else "R"
        s2 = "L" if c2 in "abcde" else "R"
        if s1 == s2:
            continue
        try:
            ri1, ri2 = int(r1), int(r2)
        except (ValueError, TypeError):
            continue
        if ri1 != ri2:
            p2[0] = r1
            logger.info(
                "[S2 Refine] LED bridge: pin2 Row %s → Row %s (align with pin1)",
                r2, r1,
            )
