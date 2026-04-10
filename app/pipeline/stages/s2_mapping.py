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

from app.domain.board_schema import BoardSchema
from app.pipeline.vision.calibrator import BreadboardCalibrator
from app.pipeline.vision.pin_utils import select_best_pin_pair
from app.pipeline.vision.pin_hole_detector import compensate_occluded_pins

logger = logging.getLogger(__name__)

_TYPE_PREFIX = {
    "resistor": "R",
    "capacitor": "C",
    "wire": "W",
    "led": "LED",
    "diode": "D",
    "ic": "IC",
    "potentiometer": "POT",
}


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
    board_schema = BoardSchema.default_breadboard()
    component_counters: Dict[str, int] = {}
    view_ids = _view_ids_from_images(images_b64)

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
        comp["pin1_logic_candidates"] = [list(item) for item in p1_candidates]
        comp["pin2_logic_candidates"] = [list(item) for item in p2_candidates]
        mapped.append(comp)

    # 后处理仍然允许修正旧链路字段, 但新的结构化 `pins[]`
    # 必须在所有修正完成之后再统一生成, 否则 hole_id/node_id 会失配。
    _fix_wire_dual_rail(mapped, calibrator)
    _refine_row_connectivity(mapped)
    _snap_wire_to_components(mapped)
    _attach_structured_pins(
        mapped,
        board_schema=board_schema,
        counters=component_counters,
        view_ids=view_ids,
    )

    duration_ms = (time.time() - t0) * 1000
    return {"components": mapped, "duration_ms": duration_ms}


def _attach_structured_pins(
    mapped: List[dict],
    board_schema: BoardSchema,
    counters: Dict[str, int],
    view_ids: List[str],
):
    """把迁移期的 `pin1_logic/pin2_logic` 提升成新链路的 `components[].pins[]`。

    这里的目标不是一次性变成最终视觉格式, 而是先把后端真正依赖的主语义
    稳定下来:

    - component_id
    - pin_name
    - hole_id
    - electrical_node_id
    - observations / candidate_hole_ids / ambiguity
    """
    for comp in mapped:
        class_name = str(comp.get("class_name") or "UNKNOWN")
        component_id = comp.get("component_id") or _next_component_id(class_name, counters)
        comp["component_id"] = component_id
        comp["component_type"] = class_name
        comp["package_type"] = _default_package_type(class_name)
        comp["part_subtype"] = comp.get("part_subtype") or ""
        comp["symmetry_group"] = _default_symmetry_group(class_name)

        pins: List[Dict[str, Any]] = []
        pin_schema_id = "fixed_pins"
        if class_name == "IC":
            pin_schema_id = "dip8_anchor_pair"

        for pin_idx in [1, 2]:
            logic_loc = comp.get(f"pin{pin_idx}_logic")
            if not logic_loc:
                continue
            logic_tuple = (str(logic_loc[0]), str(logic_loc[1]))
            hole_id = board_schema.logic_loc_to_hole_id(logic_tuple)
            electrical_node_id = board_schema.resolve_hole_to_node(hole_id)
            pin_name = _default_pin_name(class_name, pin_idx)
            pin_pixel = comp.get(f"pin{pin_idx}_pixel")
            pin_candidates = _candidate_hole_ids(comp, pin_idx, board_schema)
            observations = _build_pin_observations(
                pin_pixel=pin_pixel,
                logic_loc=logic_loc,
                view_ids=view_ids,
                confidence=float(comp.get("confidence", 1.0)),
            )
            candidate_node_ids = _candidate_node_ids(pin_candidates, board_schema)
            ambiguity_reasons = _pin_ambiguity_reasons(pin_candidates, observations)
            pins.append(
                {
                    "pin_id": pin_idx,
                    "pin_name": pin_name,
                    "logic_loc": [logic_tuple[0], logic_tuple[1]],
                    "hole_id": hole_id,
                    "electrical_node_id": electrical_node_id,
                    "confidence": float(comp.get("confidence", 1.0)),
                    "observations": observations,
                    "candidate_hole_ids": pin_candidates,
                    "candidate_node_ids": candidate_node_ids,
                    "candidate_count": len(pin_candidates),
                    "primary_visibility": max((obs["visibility"] for obs in observations), default=0),
                    "visible_view_ids": [obs["view_id"] for obs in observations if obs["visibility"] > 0],
                    "observation_count": len(observations),
                    "is_ambiguous": bool(ambiguity_reasons),
                    "ambiguity_reasons": ambiguity_reasons,
                    "is_anchor_pin": class_name == "IC",
                }
            )

        comp["pin_schema_id"] = pin_schema_id
        comp["pins"] = pins


def _next_component_id(class_name: str, counters: Dict[str, int]) -> str:
    key = class_name.lower()
    prefix = _TYPE_PREFIX.get(key, key[:3].upper() or "CMP")
    counters[key] = counters.get(key, 0) + 1
    return f"{prefix}{counters[key]}"


def _default_package_type(class_name: str) -> str:
    key = class_name.lower()
    if key == "resistor":
        return "axial_2pin"
    if key == "wire":
        return "jumper_wire_2pin"
    if key == "led":
        return "led_2pin"
    if key == "diode":
        return "diode_2pin"
    if key == "capacitor":
        return "capacitor_2pin"
    if key == "potentiometer":
        return "potentiometer_3pin"
    if key == "ic":
        return "dip8"
    return "generic"


def _default_symmetry_group(class_name: str) -> List[List[str]]:
    key = class_name.lower()
    if key in ("resistor", "wire", "capacitor"):
        return [["pin1", "pin2"]]
    return []


def _default_pin_name(class_name: str, pin_idx: int) -> str:
    key = class_name.lower()
    if key == "ic":
        return f"anchor_pin{pin_idx}"
    return f"pin{pin_idx}"


def _pin_visibility(
    pin_pixel: Optional[Tuple[float, float] | List[float]],
    logic_loc: Optional[List[str]],
) -> int:
    if logic_loc and pin_pixel:
        return 2
    if logic_loc:
        return 1
    return 0


def _candidate_hole_ids(comp: dict, pin_idx: int, board_schema: BoardSchema) -> List[str]:
    """暴露当前 pin 的 Top-K 孔位候选, 并确保最终选中孔位排在首位。

    当前阶段的候选来自校准器给出的逻辑坐标候选, 仍然是轻量版本。
    后续如果视觉侧输出真实 pin heatmap / hole distribution, 可以直接在
    这里替换而不影响下游 netlist / validator 结构。
    """
    ordered: List[str] = []
    logic_loc = comp.get(f"pin{pin_idx}_logic")
    if logic_loc:
        logic_tuple = (str(logic_loc[0]), str(logic_loc[1]))
        ordered.append(board_schema.logic_loc_to_hole_id(logic_tuple))

    for item in comp.get(f"pin{pin_idx}_logic_candidates", []):
        if not item:
            continue
        logic_tuple = (str(item[0]), str(item[1]))
        ordered.append(board_schema.logic_loc_to_hole_id(logic_tuple))

    deduped: List[str] = []
    seen = set()
    for hole_id in ordered:
        normalized = board_schema.normalize_hole_id(hole_id)
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _candidate_node_ids(candidate_holes: List[str], board_schema: BoardSchema) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for hole_id in candidate_holes:
        node_id = board_schema.resolve_hole_to_node(hole_id)
        if node_id not in seen:
            seen.add(node_id)
            ordered.append(node_id)
    return ordered


def _view_ids_from_images(images_b64: List[str] | None) -> List[str]:
    if not images_b64:
        return ["top"]
    defaults = ["top", "left_front", "right_front"]
    view_ids = defaults[: len(images_b64)]
    if len(images_b64) > len(defaults):
        for idx in range(len(defaults), len(images_b64)):
            view_ids.append(f"aux_view_{idx - len(defaults) + 1}")
    return view_ids


def _build_pin_observations(
    pin_pixel: Optional[Tuple[float, float] | List[float]],
    logic_loc: Optional[List[str]],
    view_ids: List[str],
    confidence: float,
) -> List[Dict[str, Any]]:
    """构造多视图 pin 观测。

    当前实现里:
    - top 视图尽量保留真实 keypoint
    - 侧视图先保留 visibility 占位, 便于后续视觉模型直接补全
    """
    observations: List[Dict[str, Any]] = []
    top_visibility = _pin_visibility(pin_pixel, logic_loc)
    for view_id in view_ids:
        if view_id == "top":
            observations.append(
                {
                    "view_id": view_id,
                    "keypoint": [float(pin_pixel[0]), float(pin_pixel[1])] if pin_pixel else None,
                    "visibility": top_visibility,
                    "confidence": confidence,
                }
            )
            continue

        observations.append(
            {
                "view_id": view_id,
                "keypoint": None,
                "visibility": 1 if logic_loc else 0,
                "confidence": confidence * 0.8 if logic_loc else 0.0,
            }
        )
    return observations


def _pin_ambiguity_reasons(
    candidate_hole_ids: List[str],
    observations: List[Dict[str, Any]],
) -> List[str]:
    """把“为什么这个 pin 不够确定”显式编码出来, 供 validator / agent 复用。"""
    reasons: List[str] = []
    if len(candidate_hole_ids) > 1:
        reasons.append("multiple_candidate_holes")
    top_obs = next((obs for obs in observations if obs["view_id"] == "top"), None)
    if top_obs and int(top_obs.get("visibility", 0)) < 2:
        reasons.append("top_view_not_fully_visible")
    visible_views = [obs for obs in observations if int(obs.get("visibility", 0)) > 0]
    if len(visible_views) <= 1 and len(observations) > 1:
        reasons.append("limited_multi_view_support")
    return reasons


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
