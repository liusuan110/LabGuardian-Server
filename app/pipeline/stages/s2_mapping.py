"""
Stage 2: hole mapping.

将 S1.5 的 ordered pin 预测映射到面包板 hole_id / electrical node。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.domain.board_schema import BoardSchema
from app.pipeline.vision.image_io import decode_images_b64, decode_summary
from app.pipeline.vision.calibrator import BreadboardCalibrator
from app.pipeline.vision.label_mapping import (
    default_package_type,
    default_symmetry_group,
    normalize_component_type,
)
from app.pipeline.vision.projection import BoardProjection, resolve_pin_board_projection

logger = logging.getLogger(__name__)


def run_mapping(
    components: List[dict],
    calibrator: BreadboardCalibrator,
    image_shape: Tuple[int, int],
    images_b64: List[str] | None = None,
) -> Dict[str, Any]:
    """把 pin keypoint / schema 输出吸附到 hole_id。"""
    t0 = time.time()

    calibration_mode = "uninitialized"
    decode_meta: Dict[str, Any] = {
        "decoded_view_count": 0,
        "available_view_ids": [],
        "dropped_view_ids": [],
        "decode_errors": {},
    }
    if images_b64:
        decoded = decode_images_b64(images_b64, logger=logger, stage_name="S2")
        decode_meta = decode_summary(decoded)
        _ensure_calibrated(calibrator, decoded, image_shape)
        calibration_mode = _calibration_mode(calibrator)
    elif calibrator.is_grid_ready:
        calibration_mode = _calibration_mode(calibrator)
    elif image_shape[0] > 0 and image_shape[1] > 0:
        calibrator.build_synthetic_grid(image_shape)
        calibration_mode = _calibration_mode(calibrator)

    board_schema = BoardSchema.default_breadboard()
    view_ids = _view_ids_from_images(images_b64)
    mapped: List[dict] = []
    for item in components:
        comp = dict(item)
        component_type = normalize_component_type(str(comp.get("component_type") or comp.get("class_name") or "UNKNOWN"))
        comp["component_type"] = component_type
        comp["class_name"] = component_type
        comp["package_type"] = comp.get("package_type") or default_package_type(component_type)
        comp["input_pin_detect_interface_version"] = comp.get("input_pin_detect_interface_version") or "component_pin_detect_v1"
        comp["part_subtype"] = comp.get("part_subtype") or ""
        comp["symmetry_group"] = comp.get("symmetry_group") or default_symmetry_group(component_type)
        comp["pins"] = _map_component_pins(
            comp=comp,
            calibrator=calibrator,
            board_schema=board_schema,
            view_ids=view_ids,
        )
        mapped.append(comp)

    return {
        "interface_version": "hole_mapping_v1",
        "board_schema_id": board_schema.schema_id,
        "calibration": {
            "mode": calibration_mode,
            "grid_ready": calibrator.is_grid_ready,
        },
        **decode_meta,
        "components": mapped,
        "duration_ms": (time.time() - t0) * 1000,
    }


def _map_component_pins(
    *,
    comp: dict,
    calibrator: BreadboardCalibrator,
    board_schema: BoardSchema,
    view_ids: List[str],
) -> List[Dict[str, Any]]:
    mapped_pins: List[Dict[str, Any]] = []
    for idx, pin in enumerate(comp.get("pins") or [], start=1):
        keypoints_by_view = dict(pin.get("keypoints_by_view") or {})
        visibility_by_view = dict(pin.get("visibility_by_view") or {})
        pin_metadata = dict(pin.get("metadata") or {})
        observations = _build_pin_observations_from_predictions(
            keypoints_by_view=keypoints_by_view,
            visibility_by_view=visibility_by_view,
            score_by_view=dict(pin.get("score_by_view") or {}),
            source_by_view=dict(pin.get("source_by_view") or {}),
            per_view_metadata=dict(pin_metadata.get("per_view") or {}),
            pin_metadata=pin_metadata,
            view_ids=view_ids,
            confidence=float(pin.get("confidence", comp.get("confidence", 1.0))),
            calibrator=calibrator,
            board_schema=board_schema,
        )
        vote_result = _vote_hole_from_observations(
            observations=observations,
            board_schema=board_schema,
            explicit_hole_id=pin.get("hole_id"),
            fallback_candidates=pin.get("candidate_hole_ids") or [],
        )
        hole_id = vote_result["selected_hole_id"]
        if not hole_id:
            continue

        selected_logic = _first_logic_for_hole(observations, hole_id)
        # P0.B1 (audit 2026-05-19) — hole_id is the source of truth.
        # Previously this code preferred the upstream's stale
        # electrical_node_id (set from a prior pipeline run before the
        # hole was moved), which made manual corrections silently drop.
        electrical_node_id = (
            board_schema.resolve_hole_to_node(hole_id)
            if hole_id
            else pin.get("electrical_node_id")
        )
        candidate_hole_ids = vote_result["candidate_hole_ids"]
        candidate_node_ids = _candidate_node_ids(candidate_hole_ids, board_schema)
        ambiguity_reasons = _pin_ambiguity_reasons(
            candidate_hole_ids,
            observations,
            vote_scores=vote_result["vote_scores"],
            selected_hole_id=hole_id,
        )
        selected_board_point = _selected_board_point(
            selected_logic=selected_logic,
            observations=observations,
            hole_id=hole_id,
            calibrator=calibrator,
        )
        mapped_pins.append(
            {
                "pin_id": int(pin.get("pin_id") or idx),
                "pin_name": str(pin.get("pin_name") or f"pin{idx}"),
                "pin_display_name": str(pin.get("pin_display_name") or pin.get("pin_name") or f"pin{idx}"),
                "polarity_role": str(pin.get("polarity_role") or "UNKNOWN"),
                "polarity_candidate_role": str(pin.get("polarity_candidate_role") or "UNKNOWN"),
                "logic_loc": list(selected_logic) if selected_logic else None,
                "hole_id": hole_id,
                "board_2d_point": (
                    [float(selected_board_point[0]), float(selected_board_point[1])]
                    if selected_board_point is not None
                    else None
                ),
                "electrical_node_id": electrical_node_id,
                "confidence": float(pin.get("confidence", comp.get("confidence", 1.0))),
                "observations": observations,
                "candidate_hole_ids": candidate_hole_ids,
                "candidate_node_ids": candidate_node_ids,
                "candidate_count": len(candidate_hole_ids),
                "primary_visibility": max((obs["visibility"] for obs in observations), default=0),
                "visible_view_ids": [obs["view_id"] for obs in observations if obs["visibility"] > 0],
                "observation_count": len(observations),
                "is_ambiguous": bool(ambiguity_reasons),
                "ambiguity_reasons": ambiguity_reasons,
                "is_anchor_pin": bool(pin.get("is_anchor_pin", False)),
                "source": str(pin.get("source") or "unknown"),
                "evidence_source": vote_result.get("evidence_source", "unknown"),
                "decisive_view_id": vote_result.get("decisive_view_id", ""),
                "fusion_confidence": vote_result.get("normalized_confidence", 0.0),
                "fusion_margin": vote_result.get("margin", 0.0),
                "cross_view_agreement": vote_result.get("cross_view_agreement", 0.0),
                "snap_distance_px": _best_snap_distance(observations, hole_id),
                "snap_confidence": _best_snap_confidence(observations, hole_id),
                "metadata": {
                    **pin_metadata,
                    "mapping_interface_version": "hole_mapping_v1",
                    "vote_scores": vote_result["vote_scores"],
                    "selected_by": vote_result["selected_by"],
                    "selected_board_2d_point": (
                        [float(selected_board_point[0]), float(selected_board_point[1])]
                        if selected_board_point is not None
                        else None
                    ),
                    "fusion": {
                        "evidence_source": vote_result.get("evidence_source", "unknown"),
                        "decisive_view_id": vote_result.get("decisive_view_id", ""),
                        "normalized_confidence": vote_result.get("normalized_confidence", 0.0),
                        "margin": vote_result.get("margin", 0.0),
                        "cross_view_agreement": vote_result.get("cross_view_agreement", 0.0),
                        "per_view_contribution": vote_result.get("per_view_contribution", {}),
                        "per_view_top1": vote_result.get("per_view_top1", {}),
                        "occlusion_boost": vote_result.get("occlusion_boost", {}),
                    },
                },
            }
        )
    return _apply_component_pair_selector(
        comp=comp,
        mapped_pins=mapped_pins,
        calibrator=calibrator,
        board_schema=board_schema,
    )


# Tier 2A: 把 axial 2-pin 元件也纳入 pair selector
# - TWO_PIN_AXIAL：刚性轴向，受 bbox 长度强约束、限制同 row band
# - TWO_PIN_FREE：跳线 / 电解电容，方向松、不限 row band
TWO_PIN_AXIAL = {"Resistor", "Diode", "CapacitorCeramic", "Inductor"}
TWO_PIN_FREE = {"Wire", "CapacitorElectrolytic"}
TWO_PIN_PAIRED = TWO_PIN_AXIAL | TWO_PIN_FREE


def _apply_component_pair_selector(
    *,
    comp: dict,
    mapped_pins: List[Dict[str, Any]],
    calibrator: BreadboardCalibrator,
    board_schema: BoardSchema,
) -> List[Dict[str, Any]]:
    component_type = normalize_component_type(str(comp.get("component_type") or comp.get("class_name") or "UNKNOWN"))
    if component_type == "Potentiometer":
        return _apply_potentiometer_pin_selector(
            mapped_pins=mapped_pins,
            calibrator=calibrator,
            board_schema=board_schema,
        )
    if component_type not in TWO_PIN_PAIRED:
        return mapped_pins
    if len(mapped_pins) < 2:
        return mapped_pins

    pair = mapped_pins[:2]
    bbox_prior = _bbox_layout_prior(comp, component_type, calibrator)
    resolved = _resolve_two_pin_hole_pair(
        pins=pair,
        component_type=component_type,
        calibrator=calibrator,
        board_schema=board_schema,
        bbox_prior=bbox_prior,
    )
    if resolved is None:
        return mapped_pins

    for idx, resolved_pin in enumerate(resolved):
        _refresh_selected_pin_evidence(resolved_pin, board_schema=board_schema)
        mapped_pins[idx] = resolved_pin
    return mapped_pins


def _apply_potentiometer_pin_selector(
    *,
    mapped_pins: List[Dict[str, Any]],
    calibrator: BreadboardCalibrator,
    board_schema: BoardSchema,
) -> List[Dict[str, Any]]:
    by_name = {str(pin.get("pin_name") or ""): dict(pin) for pin in mapped_pins}
    if not all(name in by_name for name in ("terminal_a", "wiper", "terminal_b")):
        return mapped_pins

    terminal_a = by_name["terminal_a"]
    wiper = by_name["wiper"]
    terminal_b = by_name["terminal_b"]
    wiper_hole = wiper.get("hole_id")
    if not wiper_hole:
        return [terminal_a, wiper, terminal_b]
    wiper_hole = board_schema.normalize_hole_id(str(wiper_hole))
    wiper_point = _hole_id_to_board_point(wiper_hole, board_schema, calibrator)
    wiper_seed = _pin_seed_board_point(wiper)
    pitch = _mapping_pitch(calibrator)

    candidates_a = _ranked_candidates_for_pin(terminal_a, board_schema=board_schema, calibrator=calibrator)
    candidates_b = _ranked_candidates_for_pin(terminal_b, board_schema=board_schema, calibrator=calibrator)
    if not candidates_a or not candidates_b:
        return [terminal_a, wiper, terminal_b]

    seed_a = _pin_seed_board_point(terminal_a)
    seed_b = _pin_seed_board_point(terminal_b)
    expected_a_wiper = _pair_seed_distance(seed_a, wiper_seed)
    expected_b_wiper = _pair_seed_distance(seed_b, wiper_seed)
    expected_terminal_span = _pair_seed_distance(seed_a, seed_b)

    best: Optional[
        Tuple[
            float,
            Tuple[str, Optional[Tuple[str, str]], Optional[Tuple[float, float]], int],
            Tuple[str, Optional[Tuple[str, str]], Optional[Tuple[float, float]], int],
        ]
    ] = None
    for cand_a in candidates_a:
        for cand_b in candidates_b:
            hole_a, _logic_a, point_a, rank_a = cand_a
            hole_b, _logic_b, point_b, rank_b = cand_b
            if hole_a == hole_b or hole_a == wiper_hole or hole_b == wiper_hole:
                continue
            score = (rank_a + rank_b) * 0.55
            if wiper_point is not None:
                if point_a is not None and expected_a_wiper is not None:
                    score += abs(_euclidean(point_a, wiper_point) - expected_a_wiper) / max(pitch, 1.0) * 0.30
                if point_b is not None and expected_b_wiper is not None:
                    score += abs(_euclidean(point_b, wiper_point) - expected_b_wiper) / max(pitch, 1.0) * 0.30
            if point_a is not None and point_b is not None and expected_terminal_span is not None:
                score += abs(_euclidean(point_a, point_b) - expected_terminal_span) / max(pitch, 1.0) * 0.18
            if best is None or score < best[0]:
                best = (score, cand_a, cand_b)

    if best is None:
        return [terminal_a, wiper, terminal_b]

    _, chosen_a, chosen_b = best
    terminal_a = _apply_joint_selected_candidate(
        terminal_a,
        chosen_a,
        board_schema=board_schema,
        selector_name="potentiometer_terminal_selector",
    )
    terminal_b = _apply_joint_selected_candidate(
        terminal_b,
        chosen_b,
        board_schema=board_schema,
        selector_name="potentiometer_terminal_selector",
    )
    wiper_meta = dict(wiper.get("metadata") or {})
    wiper_meta["potentiometer_selector"] = {
        "strategy": "wiper_independent_terminal_joint_selection",
        "role": "wiper",
        "selected_hole_id": wiper_hole,
    }
    wiper["metadata"] = wiper_meta
    _refresh_selected_pin_evidence(wiper, board_schema=board_schema)
    _refresh_selected_pin_evidence(terminal_a, board_schema=board_schema)
    _refresh_selected_pin_evidence(terminal_b, board_schema=board_schema)
    return [terminal_a, wiper, terminal_b]


def _ranked_candidates_for_pin(
    pin: Dict[str, Any],
    *,
    board_schema: BoardSchema,
    calibrator: BreadboardCalibrator,
) -> List[Tuple[str, Optional[Tuple[str, str]], Optional[Tuple[float, float]], int]]:
    ranked: List[Tuple[str, Optional[Tuple[str, str]], Optional[Tuple[float, float]], int]] = []
    seen = set()
    for rank, hole_id in enumerate(pin.get("candidate_hole_ids") or []):
        normalized = board_schema.normalize_hole_id(str(hole_id))
        if normalized in seen:
            continue
        seen.add(normalized)
        logic_loc = board_schema.hole_id_to_logic_loc(normalized)
        board_point = _hole_id_to_board_point(normalized, board_schema, calibrator)
        ranked.append((normalized, logic_loc, board_point, rank))
    if pin.get("hole_id"):
        normalized = board_schema.normalize_hole_id(str(pin["hole_id"]))
        if normalized not in seen:
            logic_loc = board_schema.hole_id_to_logic_loc(normalized)
            board_point = _hole_id_to_board_point(normalized, board_schema, calibrator)
            ranked.insert(0, (normalized, logic_loc, board_point, 0))
    return ranked[:6]


def _apply_joint_selected_candidate(
    pin: Dict[str, Any],
    chosen: Tuple[str, Optional[Tuple[str, str]], Optional[Tuple[float, float]], int],
    *,
    board_schema: BoardSchema,
    selector_name: str,
) -> Dict[str, Any]:
    updated = dict(pin)
    hole_id, logic_loc, board_point, _rank = chosen
    candidate_hole_ids = [hole_id] + [hid for hid in updated.get("candidate_hole_ids") or [] if hid != hole_id]
    updated["hole_id"] = hole_id
    updated["logic_loc"] = list(logic_loc) if logic_loc else updated.get("logic_loc")
    updated["board_2d_point"] = [float(board_point[0]), float(board_point[1])] if board_point is not None else updated.get("board_2d_point")
    updated["electrical_node_id"] = board_schema.resolve_hole_to_node(hole_id)
    updated["candidate_hole_ids"] = candidate_hole_ids
    updated["candidate_node_ids"] = _candidate_node_ids(candidate_hole_ids, board_schema)
    updated["candidate_count"] = len(candidate_hole_ids)
    metadata = dict(updated.get("metadata") or {})
    metadata["selected_by"] = f"{metadata.get('selected_by', 'multi_view_weighted_vote')}+{selector_name}"
    metadata[selector_name] = {
        "strategy": "joint_hole_selection",
        "selected_hole_id": hole_id,
    }
    if board_point is not None:
        metadata["selected_board_2d_point"] = [float(board_point[0]), float(board_point[1])]
    updated["metadata"] = metadata
    return updated
def _ensure_calibrated(
    calibrator: BreadboardCalibrator,
    decoded_images: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
):
    if calibrator.is_grid_ready:
        return
    top_item = next((item for item in decoded_images if item["view_id"] == "top" and item.get("decoded")), None)
    img = top_item["image"] if top_item else None
    if img is not None:
        try:
            calibrator.ensure_calibrated(img)
            if calibrator.is_grid_ready:
                return
        except Exception as exc:
            logger.warning("Calibration from image failed: %s", exc)
    else:
        logger.warning("S2 top view unavailable for calibration; using synthetic fallback")
    logger.info("Falling back to synthetic grid")
    calibrator.build_synthetic_grid(image_shape)


def _get_candidates(
    pixel: Optional[Tuple[float, float]],
    calibrator: BreadboardCalibrator,
    k: int = 5,
) -> List[Tuple[str, str]]:
    if pixel is None:
        return []
    try:
        return calibrator.frame_pixel_to_logic_candidates(pixel[0], pixel[1], k=k)
    except Exception as exc:
        logger.warning("S2 candidate lookup failed for pixel %s: %s", pixel, exc)
        return []


def _get_board_candidates(
    board_point: Optional[Tuple[float, float]],
    calibrator: BreadboardCalibrator,
    k: int = 5,
) -> List[Tuple[str, str]]:
    if board_point is None:
        return []
    try:
        return calibrator.board_point_to_logic_candidates(board_point[0], board_point[1], k=k)
    except Exception as exc:
        logger.warning("S2 board candidate lookup failed for point %s: %s", board_point, exc)
        return []


def _scored_candidates_for_projection(
    *,
    pixel: Optional[Tuple[float, float]],
    projection: BoardProjection,
    calibrator: BreadboardCalibrator,
    k: int = 5,
) -> List[Tuple[Tuple[str, str], float]]:
    """Return `(logic_loc, snap_distance_px)` pairs for the active projection.

    Falls back gracefully when calibrator does not yet expose the scored API.
    """
    try:
        if projection.should_use_board_point_for_mapping:
            if projection.board_point is None:
                return []
            return calibrator.board_point_to_logic_candidates_scored(
                projection.board_point[0], projection.board_point[1], k=k,
            )
        if pixel is None:
            return []
        return calibrator.frame_pixel_to_logic_candidates_scored(pixel[0], pixel[1], k=k)
    except AttributeError:
        # Older calibrator without scored API: synthesize unscored fallback.
        return [(loc, float("nan")) for loc in _candidates_for_projection(
            pixel=pixel, projection=projection, calibrator=calibrator, k=k,
        )]
    except Exception as exc:
        logger.warning("S2 scored candidate lookup failed: %s", exc)
        return []


def _candidates_for_projection(
    *,
    pixel: Optional[Tuple[float, float]],
    projection: BoardProjection,
    calibrator: BreadboardCalibrator,
    k: int = 5,
) -> List[Tuple[str, str]]:
    if projection.should_use_board_point_for_mapping:
        return _get_board_candidates(projection.board_point, calibrator, k=k)
    return _get_candidates(pixel, calibrator, k=k)


def _filter_candidates_by_pin_constraints(
    *,
    scored_candidates: List[Tuple[Tuple[str, str], float]],
    pin_metadata: Dict[str, Any],
) -> List[Tuple[Tuple[str, str], float]]:
    """Drop logic candidates that violate physical constraints stamped by S1.5.

    Constraints honored (set by ``s1b_pin_detect``):
      * ``row_lock`` (str letter, e.g. ``"e"``): IC pin must snap to that letter row;
        used both for DIP IC e/f bridge and for horizontally-inserted potentiometers.
      * ``column_lock`` (str digit): vertically-inserted potentiometer must keep
        the same digit column across all three pins.
      * ``pot_logic_slots`` (``List[(digit, letter)]``): the exact 3-collinear hole
        triplet the POT pins were snapped to; any candidate outside that triplet
        would cross to a different physical line and must be rejected.

    Returns the filtered list. If filtering removes every candidate, returns an
    empty list — downstream then leaves the pin unmapped instead of falling back
    to the geometrically wrong nearest hole.
    """
    if not scored_candidates or not pin_metadata:
        return scored_candidates

    row_lock_raw = pin_metadata.get("row_lock")
    column_lock_raw = pin_metadata.get("column_lock")
    slots_raw = pin_metadata.get("pot_logic_slots")

    row_lock = str(row_lock_raw).strip().lower() if row_lock_raw else None
    column_lock = str(column_lock_raw).strip() if column_lock_raw else None
    slot_set: Optional[set[Tuple[str, str]]] = None
    if slots_raw:
        slot_set = set()
        for entry in slots_raw:
            if entry is None or len(entry) < 2:
                continue
            slot_set.add((str(entry[0]).strip(), str(entry[1]).strip().lower()))

    if row_lock is None and column_lock is None and not slot_set:
        return scored_candidates

    filtered: List[Tuple[Tuple[str, str], float]] = []
    for logic_loc, distance in scored_candidates:
        if logic_loc is None or len(logic_loc) < 2:
            continue
        digit = str(logic_loc[0]).strip()
        letter = str(logic_loc[1]).strip().lower()
        if row_lock is not None and letter != row_lock:
            continue
        if column_lock is not None and digit != column_lock:
            continue
        if slot_set is not None and (digit, letter) not in slot_set:
            continue
        filtered.append((logic_loc, distance))
    return filtered


def _snap_confidence_from_distance(distance_px: float, pitch_px: float) -> float:
    """Convert a snap distance to a [0, 1] confidence score.

    Uses a soft cosine-like falloff so that confidence ≈ 1 at d=0,
    drops to ≈ 0.5 at half a pitch, and floors at 0 once the predicted
    point is more than one pitch away from the nearest hole.
    """
    if not (distance_px == distance_px):  # NaN
        return 0.5  # unknown — neutral, do not punish
    if distance_px == float("inf"):
        return 0.0
    if pitch_px <= 1e-3:
        return 0.0 if distance_px > 1.0 else 1.0
    ratio = max(0.0, distance_px / pitch_px)
    if ratio >= 1.0:
        return 0.0
    # Smooth falloff: 1 - ratio^2, so quadratic penalty for far snaps.
    return float(max(0.0, 1.0 - ratio * ratio))


def _candidate_hole_ids_from_logic(
    *,
    selected_hole_id: str,
    logic_candidates: List[Tuple[str, str]],
    board_schema: BoardSchema,
    fallback_candidates: List[str],
) -> List[str]:
    ordered = [selected_hole_id]
    for logic_loc in logic_candidates:
        ordered.append(board_schema.logic_loc_to_hole_id(logic_loc))
    ordered.extend(str(item) for item in fallback_candidates)

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


# 不同类型元件的 bbox→pin_spacing 关系不同：
#   * 刚性轴向 (Resistor / Diode / Inductor)：
#       YOLO bbox 紧贴 body，leads 在 bbox 边缘 90° 弯下入孔
#       → pin_spacing ≈ bbox 长边 (lead margin ≈ 0)
#   * 瓷片电容 (CapacitorCeramic)：
#       bbox 覆盖整个元件，pin 在 bbox 内偏底部 1 pitch
#       → pin_spacing ≈ bbox 长边
#   * 电解电容 (CapacitorElectrolytic)：
#       bbox 覆盖圆柱，axial leads 略超 bbox
#       → pin_spacing ≈ bbox 长边 + 0~0.4 pitch  (放宽容差搞定)
#   * 跳线 (Wire)：
#       bbox 包住弯曲的整段线，pin_spacing < bbox 长边，差值不可预测
#       → 不用 bbox prior，回退 seed 距离
LEAD_MARGIN_PITCHES = {
    "Resistor": 0.0,
    "Diode": 0.0,
    "Inductor": 0.0,
    "CapacitorCeramic": 0.0,
    "CapacitorElectrolytic": 0.0,
}


def _bbox_layout_prior(
    comp: dict,
    component_type: str,
    calibrator: BreadboardCalibrator,
) -> Optional[Dict[str, Any]]:
    """从 component bbox 推导 axial / paired 元件的几何 prior。

    返回 None 表示不适用 (跳线 / bbox 缺失 / 校准未就绪)，调用方退回 seed_points 逻辑。
    返回字典含：
      expected_distance_px : 两 pin 在 board-plane 上的期望欧氏距离 (≈ bbox 长边)
      axis                 : "horizontal" / "vertical" — bbox 在 board-plane 上的长边方向
      tolerance_px         : 允许误差 (≈ 半个 pitch)
      pitch_px             : 当前 pitch，用于评分归一
    """
    # 跳线不适用 bbox prior：bbox 包住弯线，pin_spacing 远小于 bbox 长度
    if component_type not in LEAD_MARGIN_PITCHES:
        return None
    bbox = comp.get("bbox")
    if not bbox or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None
    if not hasattr(calibrator, "frame_pixel_to_board_point"):
        return None
    try:
        b_top_left = calibrator.frame_pixel_to_board_point(x1, y1)
        b_bot_right = calibrator.frame_pixel_to_board_point(x2, y2)
    except Exception as exc:
        logger.debug("S2 bbox prior projection failed (%s): %s", component_type, exc)
        return None
    if b_top_left is None or b_bot_right is None:
        return None
    dx = abs(float(b_bot_right[0]) - float(b_top_left[0]))
    dy = abs(float(b_bot_right[1]) - float(b_top_left[1]))
    long_side = max(dx, dy)
    if long_side <= 1e-3:
        return None
    pitch = _mapping_pitch(calibrator)
    if pitch <= 1e-3:
        return None
    # 关键修正：刚性轴向元件 pin_spacing 直接等于 bbox 长边（leads 在 bbox 边缘弯下）
    # 不再扣除 lead margin —— 之前 max(long_side - pitch, pitch) 把期望间距拉小了 1 col，
    # 导致 R2 这类 case 持续偏 1 列。
    margin_pitches = LEAD_MARGIN_PITCHES.get(component_type, 0.0)
    expected_distance_px = max(long_side - 2 * margin_pitches * pitch, pitch)
    axis = "horizontal" if dx >= dy else "vertical"
    tolerance_px = pitch * 0.5
    return {
        "expected_distance_px": expected_distance_px,
        "axis": axis,
        "tolerance_px": tolerance_px,
        "pitch_px": pitch,
    }


def _row_band(logic_loc: Optional[Tuple[str, str]]) -> str:
    """把 logic_loc 的行字母分到 TOP/BOT/RAIL_*/OTHER 几个语义带。

    logic_loc 形如 ("5", "b") — 第二项是行字母 (a-j) 或电源轨记号 (+/-)。
    """
    if not logic_loc or len(logic_loc) < 2:
        return "UNKNOWN"
    row_letter = (str(logic_loc[1]) or "").lower().strip()
    if not row_letter:
        return "UNKNOWN"
    if row_letter in {"a", "b", "c", "d", "e"}:
        return "TOP"
    if row_letter in {"f", "g", "h", "i", "j"}:
        return "BOT"
    if row_letter in {"+", "-"}:
        return f"RAIL_{row_letter}"
    if row_letter.startswith("rail_"):
        return row_letter.upper()
    return "OTHER"


def _resolve_two_pin_hole_pair(
    *,
    pins: List[Dict[str, Any]],
    component_type: str,
    calibrator: BreadboardCalibrator,
    board_schema: BoardSchema,
    bbox_prior: Optional[Dict[str, Any]] = None,
) -> Optional[List[Dict[str, Any]]]:
    if len(pins) < 2:
        return None

    candidates_per_pin: List[List[Tuple[str, Optional[Tuple[str, str]], Optional[Tuple[float, float]], int]]] = []
    seed_points: List[Optional[Tuple[float, float]]] = []
    for pin in pins[:2]:
        seed_points.append(_pin_seed_board_point(pin))
        ranked_candidates: List[Tuple[str, Optional[Tuple[str, str]], Optional[Tuple[float, float]], int]] = []
        seen = set()
        for rank, hole_id in enumerate(pin.get("candidate_hole_ids") or []):
            normalized = board_schema.normalize_hole_id(str(hole_id))
            if normalized in seen:
                continue
            seen.add(normalized)
            logic_loc = board_schema.hole_id_to_logic_loc(normalized)
            board_point = _hole_id_to_board_point(normalized, board_schema, calibrator)
            ranked_candidates.append((normalized, logic_loc, board_point, rank))
        if pin.get("hole_id"):
            normalized = board_schema.normalize_hole_id(str(pin["hole_id"]))
            if normalized not in seen:
                logic_loc = board_schema.hole_id_to_logic_loc(normalized)
                board_point = _hole_id_to_board_point(normalized, board_schema, calibrator)
                ranked_candidates.insert(0, (normalized, logic_loc, board_point, 0))
        candidates_per_pin.append(ranked_candidates[:6])

    if not candidates_per_pin[0] or not candidates_per_pin[1]:
        return None

    best: Optional[Tuple[float, Tuple[str, Optional[Tuple[str, str]], Optional[Tuple[float, float]], int], Tuple[str, Optional[Tuple[str, str]], Optional[Tuple[float, float]], int]]] = None
    desired_distance = _pair_seed_distance(seed_points[0], seed_points[1])
    pair_orientation = _pair_seed_orientation(seed_points[0], seed_points[1])
    pitch = _mapping_pitch(calibrator)
    is_axial = component_type in TWO_PIN_AXIAL
    prior_axis = (bbox_prior or {}).get("axis")
    prior_distance = (bbox_prior or {}).get("expected_distance_px")
    prior_tolerance = (bbox_prior or {}).get("tolerance_px")
    for cand1 in candidates_per_pin[0]:
        for cand2 in candidates_per_pin[1]:
            hole1, logic1, point1, rank1 = cand1
            hole2, logic2, point2, rank2 = cand2
            if hole1 == hole2:
                continue
            score = (rank1 + rank2) * 0.55
            if point1 is not None and point2 is not None:
                actual_distance = _euclidean(point1, point2)

                # 1. 距离约束：bbox prior 优先，否则退回 seed_points
                if prior_distance is not None and prior_tolerance is not None:
                    deviation = abs(actual_distance - prior_distance)
                    norm_dev = deviation / max(pitch, 1.0)
                    if deviation > prior_tolerance:
                        # 超容差，惩罚平方陡增 → 1 列偏差被淘汰
                        score += (norm_dev ** 2) * 1.5
                    else:
                        score += norm_dev * 0.30
                elif desired_distance is not None:
                    score += abs(actual_distance - desired_distance) / max(pitch, 1.0) * 0.28

                # 2. 方向约束：bbox axis 优先
                axis = prior_axis or pair_orientation
                if axis == "horizontal" and point1[0] >= point2[0]:
                    score += 6.0
                elif axis == "vertical" and point1[1] >= point2[1]:
                    score += 6.0

                # 3. 轴向元件：同 row band 硬约束 (Resistor / Diode / Inductor 不允许跨 A-E↔F-J)
                if is_axial and component_type != "CapacitorCeramic":
                    band1 = _row_band(logic1)
                    band2 = _row_band(logic2)
                    if band1 != "UNKNOWN" and band2 != "UNKNOWN" and band1 != band2:
                        score += 50.0

                # 4. 电解电容极性偏置
                if component_type == "CapacitorElectrolytic":
                    score += abs(point1[1] - point2[1]) / max(pitch, 1.0) * 0.12
            elif desired_distance is not None:
                score += 3.0
            if best is None or score < best[0]:
                best = (score, cand1, cand2)

    if best is None:
        return None

    _, left, right = best
    resolved = [dict(pin) for pin in pins[:2]]
    for pin, chosen in zip(resolved, [left, right]):
        hole_id, logic_loc, board_point, _rank = chosen
        candidate_hole_ids = [hole_id] + [hid for hid in pin.get("candidate_hole_ids") or [] if hid != hole_id]
        pin["hole_id"] = hole_id
        pin["logic_loc"] = list(logic_loc) if logic_loc else pin.get("logic_loc")
        pin["board_2d_point"] = [float(board_point[0]), float(board_point[1])] if board_point is not None else pin.get("board_2d_point")
        pin["electrical_node_id"] = board_schema.resolve_hole_to_node(hole_id)
        pin["candidate_hole_ids"] = candidate_hole_ids
        pin["candidate_node_ids"] = _candidate_node_ids(candidate_hole_ids, board_schema)
        pin["candidate_count"] = len(candidate_hole_ids)
        metadata = dict(pin.get("metadata") or {})
        vote_scores = dict(metadata.get("vote_scores") or {})
        metadata["selected_by"] = f"{metadata.get('selected_by', 'multi_view_weighted_vote')}+pair_selector"
        metadata["pair_selector"] = {
            "component_type": component_type,
            "strategy": "two_pin_joint_hole_selection",
            "selected_hole_id": hole_id,
            "vote_scores": vote_scores,
        }
        if board_point is not None:
            metadata["selected_board_2d_point"] = [float(board_point[0]), float(board_point[1])]
        pin["metadata"] = metadata
    return resolved


def _pin_seed_board_point(pin: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    for obs in pin.get("observations") or []:
        board_point = obs.get("board_2d_point")
        if board_point and len(board_point) >= 2:
            return (float(board_point[0]), float(board_point[1]))
    point = pin.get("board_2d_point")
    if point and len(point) >= 2:
        return (float(point[0]), float(point[1]))
    return None


def _hole_id_to_board_point(
    hole_id: str,
    board_schema: BoardSchema,
    calibrator: BreadboardCalibrator,
) -> Optional[Tuple[float, float]]:
    logic_loc = board_schema.hole_id_to_logic_loc(hole_id)
    if logic_loc is None or not hasattr(calibrator, "logic_to_board_point"):
        return None
    try:
        point = calibrator.logic_to_board_point(logic_loc)
    except Exception as exc:
        logger.warning("S2 pair selector board point lookup failed for %s: %s", hole_id, exc)
        return None
    if point is None:
        return None
    return (float(point[0]), float(point[1]))


def _pair_seed_distance(
    point1: Optional[Tuple[float, float]],
    point2: Optional[Tuple[float, float]],
) -> Optional[float]:
    if point1 is None or point2 is None:
        return None
    return _euclidean(point1, point2)


def _pair_seed_orientation(
    point1: Optional[Tuple[float, float]],
    point2: Optional[Tuple[float, float]],
) -> Optional[str]:
    if point1 is None or point2 is None:
        return None
    if abs(point2[0] - point1[0]) >= abs(point2[1] - point1[1]):
        return "horizontal"
    return "vertical"


def _mapping_pitch(calibrator: BreadboardCalibrator) -> float:
    row_pitch = calibrator._median_pitch(calibrator._row_coords) if getattr(calibrator, "_row_coords", None) is not None else 10.0
    col_pitch = calibrator._median_pitch(calibrator._col_coords) if getattr(calibrator, "_col_coords", None) is not None else 10.0
    return max(1.0, min(row_pitch, col_pitch))


def _euclidean(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return float(((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5)


def _selected_board_point(
    *,
    selected_logic: Optional[Tuple[str, str]],
    observations: List[Dict[str, Any]],
    hole_id: str,
    calibrator: BreadboardCalibrator,
) -> Optional[Tuple[float, float]]:
    if selected_logic is not None and hasattr(calibrator, "logic_to_board_point"):
        try:
            board_point = calibrator.logic_to_board_point(selected_logic)
            if board_point is not None:
                return (float(board_point[0]), float(board_point[1]))
        except Exception as exc:
            logger.warning("S2 selected board point lookup failed for %s: %s", selected_logic, exc)

    for obs in observations:
        if hole_id not in (obs.get("candidate_hole_ids") or []):
            continue
        point = obs.get("board_2d_point")
        if point and len(point) >= 2:
            return (float(point[0]), float(point[1]))
    return None


def _view_ids_from_images(images_b64: List[str] | None) -> List[str]:
    if not images_b64:
        return ["top"]
    defaults = ["top", "left_front", "right_front"]
    view_ids = defaults[: len(images_b64)]
    if len(images_b64) > len(defaults):
        for idx in range(len(defaults), len(images_b64)):
            view_ids.append(f"aux_view_{idx - len(defaults) + 1}")
    return view_ids


def _build_pin_observations_from_predictions(
    keypoints_by_view: Dict[str, Any],
    visibility_by_view: Dict[str, Any],
    score_by_view: Dict[str, Any],
    source_by_view: Dict[str, Any],
    per_view_metadata: Dict[str, Any],
    pin_metadata: Dict[str, Any],
    view_ids: List[str],
    confidence: float,
    calibrator: BreadboardCalibrator,
    board_schema: BoardSchema,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for view_id in view_ids:
        keypoint = keypoints_by_view.get(view_id)
        visibility = int(visibility_by_view.get(view_id, 0))
        pixel = (float(keypoint[0]), float(keypoint[1])) if keypoint else None
        projection = resolve_pin_board_projection(
            view_id=view_id,
            keypoint=keypoint,
            per_view_metadata=per_view_metadata.get(view_id) or {},
            pin_metadata=pin_metadata,
            calibrator=calibrator,
        )
        scored_candidates = (
            _scored_candidates_for_projection(
                pixel=pixel,
                projection=projection,
                calibrator=calibrator,
            )
            if pixel is not None or projection.should_use_board_point_for_mapping
            else []
        )
        scored_candidates = _filter_candidates_by_pin_constraints(
            scored_candidates=scored_candidates,
            pin_metadata=pin_metadata,
        )
        logic_candidates = [item[0] for item in scored_candidates]
        candidate_hole_ids = [
            board_schema.normalize_hole_id(board_schema.logic_loc_to_hole_id(logic_loc))
            for logic_loc in logic_candidates
        ]
        candidate_node_ids = _candidate_node_ids(candidate_hole_ids, board_schema)
        board_2d_point = (
            [float(projection.board_point[0]), float(projection.board_point[1])]
            if projection.board_point is not None
            else None
        )
        if visibility <= 0 and projection.should_use_board_point_for_mapping and candidate_hole_ids:
            visibility = 1

        pitch_px = _safe_pitch(calibrator)
        candidate_distances_px = [float(item[1]) for item in scored_candidates]
        snap_distance_px = candidate_distances_px[0] if candidate_distances_px else float("inf")
        snap_normalized = (
            snap_distance_px / pitch_px
            if pitch_px > 0 and snap_distance_px not in (float("inf"),) and snap_distance_px == snap_distance_px
            else float("inf")
        )
        snap_confidence = _snap_confidence_from_distance(snap_distance_px, pitch_px)

        observations.append(
            {
                "view_id": view_id,
                "keypoint": [float(keypoint[0]), float(keypoint[1])] if keypoint else None,
                "board_2d_point": board_2d_point,
                "projection": projection.to_metadata(),
                "visibility": visibility,
                "confidence": float(score_by_view.get(view_id, confidence if visibility > 0 else 0.0)),
                "source": str(source_by_view.get(view_id, "unknown")),
                "roi_source": str((per_view_metadata.get(view_id) or {}).get("roi_source", "unknown")),
                "candidate_logic_locs": [list(item) for item in logic_candidates],
                "candidate_hole_ids": candidate_hole_ids,
                "candidate_node_ids": candidate_node_ids,
                "candidate_distances_px": [round(d, 4) if d != float("inf") else None for d in candidate_distances_px],
                "snap_distance_px": (
                    round(snap_distance_px, 4) if snap_distance_px not in (float("inf"),) and snap_distance_px == snap_distance_px else None
                ),
                "snap_normalized": (
                    round(snap_normalized, 4) if snap_normalized != float("inf") else None
                ),
                "snap_confidence": round(snap_confidence, 4),
                "pitch_px": round(pitch_px, 4),
            }
        )
    return observations


def _best_snap_distance(observations: List[Dict[str, Any]], hole_id: str) -> Optional[float]:
    """Distance of the visible observation that voted for the selected hole."""
    best: Optional[float] = None
    for obs in observations:
        if int(obs.get("visibility", 0)) <= 0:
            continue
        dist = _observation_hole_snap_distance(obs, hole_id)
        if dist is None:
            continue
        if best is None or dist < best:
            best = dist
    return best


def _best_snap_confidence(observations: List[Dict[str, Any]], hole_id: str) -> float:
    """Highest snap confidence among visible observations that voted for the hole."""
    best = 0.0
    found = False
    for obs in observations:
        if int(obs.get("visibility", 0)) <= 0:
            continue
        distance = _observation_hole_snap_distance(obs, hole_id)
        if distance is None:
            continue
        pitch_px = _observation_pitch_px(obs)
        value = _snap_confidence_from_distance(distance, pitch_px)
        found = True
        if value > best:
            best = value
    return float(best) if found else 0.0


def _observation_hole_snap_distance(obs: Dict[str, Any], hole_id: str) -> Optional[float]:
    """Return this observation's distance to a specific candidate hole."""
    candidates = list(obs.get("candidate_hole_ids") or [])
    try:
        index = candidates.index(hole_id)
    except ValueError:
        return None
    distances = list(obs.get("candidate_distances_px") or [])
    if index >= len(distances):
        return None
    distance = distances[index]
    if distance is None:
        return None
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def _observation_pitch_px(obs: Dict[str, Any]) -> float:
    try:
        pitch = float(obs.get("pitch_px", 10.0))
    except (TypeError, ValueError):
        return 10.0
    return pitch if pitch > 1e-3 else 10.0


def _safe_pitch(calibrator: BreadboardCalibrator) -> float:
    """Read calibrator's representative pitch with a safe default."""
    try:
        pitch = float(calibrator.representative_pitch_px())
    except AttributeError:
        pitch = 10.0
    except Exception as exc:
        logger.warning("S2 pitch lookup failed: %s", exc)
        pitch = 10.0
    return pitch if pitch > 1e-3 else 10.0


def _pin_ambiguity_reasons(
    candidate_hole_ids: List[str],
    observations: List[Dict[str, Any]],
    *,
    vote_scores: Dict[str, float],
    selected_hole_id: Optional[str],
) -> List[str]:
    reasons: List[str] = []
    if len(candidate_hole_ids) > 1:
        reasons.append("multiple_candidate_holes")
    top_obs = next((obs for obs in observations if obs["view_id"] == "top"), None)
    if top_obs and int(top_obs.get("visibility", 0)) < 2:
        reasons.append("top_view_not_fully_visible")
    visible_views = [obs for obs in observations if int(obs.get("visibility", 0)) > 0]
    if len(visible_views) <= 1 and len(observations) > 1:
        reasons.append("limited_multi_view_support")
    preferred = [
        tuple(obs.get("candidate_hole_ids", [])[:1])
        for obs in visible_views
        if obs.get("candidate_hole_ids")
    ]
    if len(set(preferred)) > 1:
        reasons.append("multi_view_vote_conflict")
    if len(vote_scores) >= 2:
        ordered = sorted(vote_scores.values(), reverse=True)
        # 票数差阈值 — 由 settings.MAPPING_AMBIGUOUS_MARGIN 调参 (Tier 1)
        if ordered[0] - ordered[1] < float(settings.MAPPING_AMBIGUOUS_MARGIN):
            reasons.append("close_vote_margin")
    if selected_hole_id and _best_snap_confidence(observations, selected_hole_id) < 0.5:
        reasons.append("low_snap_confidence")
    return reasons


def _vote_hole_from_observations(
    *,
    observations: List[Dict[str, Any]],
    board_schema: BoardSchema,
    explicit_hole_id: str | None,
    fallback_candidates: List[str],
) -> Dict[str, Any]:
    """多视图加权投票：选择最终 hole 并产出可解释的融合元信息。

    输出 vote_scores 同时附带：
    - normalized_confidence: 归一化到 [0,1] 的赢家得分
    - margin: 赢家与第二名的差值（同样归一化）
    - cross_view_agreement: 各可见视图 top-1 与赢家一致的比例
    - per_view_contribution: 每个视图贡献给赢家的分数
    - decisive_view_id / evidence_source: top|left_front|right_front|fused
    """
    occlusion_boost = _compute_occlusion_boost(observations)

    vote_scores: Dict[str, float] = {}
    for obs in observations:
        visibility = int(obs.get("visibility", 0))
        if visibility <= 0:
            continue
        confidence = float(obs.get("confidence", 0.0))
        if confidence <= 0.0:
            continue
        base = _observation_vote_base(obs, occlusion_boost)
        # 多视图加权投票 rank 衰减 — 由 settings.MAPPING_VOTE_RANK_DECAY 调参 (Tier 1)
        rank_decay = float(settings.MAPPING_VOTE_RANK_DECAY)
        for rank, hole_id in enumerate(obs.get("candidate_hole_ids") or []):
            normalized = board_schema.normalize_hole_id(str(hole_id))
            contribution = base * (rank_decay ** rank)
            vote_scores[normalized] = vote_scores.get(normalized, 0.0) + contribution

    if explicit_hole_id:
        normalized = board_schema.normalize_hole_id(str(explicit_hole_id))
        vote_scores[normalized] = vote_scores.get(normalized, 0.0) + 0.15

    for rank, hole_id in enumerate(fallback_candidates):
        normalized = board_schema.normalize_hole_id(str(hole_id))
        vote_scores[normalized] = vote_scores.get(normalized, 0.0) + 0.05 * (0.8 ** rank)

    ordered = [item[0] for item in sorted(vote_scores.items(), key=lambda item: item[1], reverse=True)]
    selected = ordered[0] if ordered else (board_schema.normalize_hole_id(str(explicit_hole_id)) if explicit_hole_id else None)
    support = _summarize_selected_hole_support(
        observations=observations,
        board_schema=board_schema,
        vote_scores=vote_scores,
        selected_hole_id=selected,
        occlusion_boost=occlusion_boost,
    )

    return {
        "selected_hole_id": selected,
        "candidate_hole_ids": ordered,
        "vote_scores": {key: round(val, 6) for key, val in sorted(vote_scores.items(), key=lambda item: item[1], reverse=True)},
        "selected_by": "multi_view_weighted_vote" if ordered else "explicit_or_empty",
        **support,
    }


def _compute_occlusion_boost(observations: List[Dict[str, Any]]) -> Dict[str, float]:
    """当 top 视图被遮挡或缺失时，把侧视图权重动态抬高。

    规则:
    - top visibility >= 2 且置信度合理 -> 不调整 (boost=1.0)
    - top visibility == 1 -> side 视图 *= 1.25
    - top visibility == 0 或 confidence <= 0 -> side 视图 *= 1.6, top *= 0.4
    这样 side 在 top 完全失效时能压倒 top 的低权重残留。
    """
    boost: Dict[str, float] = {}
    top_obs = next((obs for obs in observations if str(obs.get("view_id", "")) == "top"), None)
    if top_obs is None:
        return boost
    top_visibility = int(top_obs.get("visibility", 0))
    top_confidence = float(top_obs.get("confidence", 0.0))
    if top_visibility >= 2 and top_confidence > 0.3:
        return boost
    if top_visibility >= 1 and top_confidence > 0.0:
        side_factor = 1.25
        top_factor = 1.0
    else:
        side_factor = 1.6
        top_factor = 0.4
    for obs in observations:
        vid = str(obs.get("view_id", ""))
        if not vid:
            continue
        if vid == "top":
            boost[vid] = top_factor
        else:
            boost[vid] = side_factor
    return boost


def _observation_vote_base(obs: Dict[str, Any], occlusion_boost: Dict[str, float]) -> float:
    visibility = int(obs.get("visibility", 0))
    confidence = float(obs.get("confidence", 0.0))
    view_id = str(obs.get("view_id", ""))
    view_weight = _view_weight(view_id) * occlusion_boost.get(view_id, 1.0)
    source_weight = _prediction_source_weight(str(obs.get("source", "")))
    roi_weight = _roi_source_weight(str(obs.get("roi_source", "")))
    snap_weight = _snap_weight(obs.get("snap_confidence"))
    return confidence * _visibility_weight(visibility) * view_weight * source_weight * roi_weight * snap_weight


def _summarize_selected_hole_support(
    *,
    observations: List[Dict[str, Any]],
    board_schema: BoardSchema,
    vote_scores: Dict[str, float],
    selected_hole_id: Optional[str],
    occlusion_boost: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    occlusion_boost = dict(occlusion_boost or _compute_occlusion_boost(observations))
    per_view_top1: Dict[str, str] = {}
    per_view_contribution: Dict[str, float] = {}
    visible_views: List[str] = []

    for obs in observations:
        visibility = int(obs.get("visibility", 0))
        if visibility <= 0:
            continue
        view_id = str(obs.get("view_id", ""))
        if not view_id:
            continue
        visible_views.append(view_id)
        raw_candidates = [board_schema.normalize_hole_id(str(hole_id)) for hole_id in (obs.get("candidate_hole_ids") or [])]
        if raw_candidates:
            per_view_top1[view_id] = raw_candidates[0]
        if not selected_hole_id or selected_hole_id not in raw_candidates:
            continue
        rank = raw_candidates.index(selected_hole_id)
        contribution = _observation_vote_base(obs, occlusion_boost) * (0.72 ** rank)
        per_view_contribution[view_id] = per_view_contribution.get(view_id, 0.0) + contribution

    total_score = sum(vote_scores.values()) or 0.0
    selected_score = vote_scores.get(selected_hole_id, 0.0) if selected_hole_id else 0.0
    best_other = max((score for hole_id, score in vote_scores.items() if hole_id != selected_hole_id), default=0.0)
    normalized_confidence = (selected_score / total_score) if total_score > 0 else 0.0
    margin = ((selected_score - best_other) / total_score) if total_score > 0 else 0.0

    decisive_view_id = ""
    if per_view_contribution:
        decisive_view_id = max(per_view_contribution.items(), key=lambda kv: kv[1])[0]

    if selected_hole_id and visible_views:
        agreeing = sum(1 for vid in visible_views if per_view_top1.get(vid) == selected_hole_id)
        cross_view_agreement = agreeing / len(visible_views)
    else:
        cross_view_agreement = 0.0

    if not selected_hole_id:
        evidence_source = "none"
    elif len([vid for vid in visible_views if per_view_top1.get(vid) == selected_hole_id]) >= 2:
        evidence_source = "fused"
    elif decisive_view_id:
        evidence_source = decisive_view_id
    else:
        evidence_source = "explicit_or_fallback"

    return {
        "normalized_confidence": round(normalized_confidence, 6),
        "margin": round(margin, 6),
        "cross_view_agreement": round(cross_view_agreement, 6),
        "decisive_view_id": decisive_view_id,
        "evidence_source": evidence_source,
        "per_view_contribution": {vid: round(val, 6) for vid, val in per_view_contribution.items()},
        "per_view_top1": dict(per_view_top1),
        "occlusion_boost": {vid: round(val, 6) for vid, val in occlusion_boost.items()},
    }


def _refresh_selected_pin_evidence(
    pin: Dict[str, Any],
    *,
    board_schema: BoardSchema,
) -> None:
    hole_id = str(pin.get("hole_id") or "")
    observations = list(pin.get("observations") or [])
    metadata = dict(pin.get("metadata") or {})
    vote_scores_raw = dict(metadata.get("vote_scores") or {})
    vote_scores = {
        board_schema.normalize_hole_id(str(candidate_hole)): float(score)
        for candidate_hole, score in vote_scores_raw.items()
    }
    fusion = _summarize_selected_hole_support(
        observations=observations,
        board_schema=board_schema,
        vote_scores=vote_scores,
        selected_hole_id=hole_id or None,
    )

    pin["evidence_source"] = fusion["evidence_source"]
    pin["decisive_view_id"] = fusion["decisive_view_id"]
    pin["fusion_confidence"] = fusion["normalized_confidence"]
    pin["fusion_margin"] = fusion["margin"]
    pin["cross_view_agreement"] = fusion["cross_view_agreement"]
    pin["snap_distance_px"] = _best_snap_distance(observations, hole_id) if hole_id else None
    pin["snap_confidence"] = _best_snap_confidence(observations, hole_id) if hole_id else 0.0
    ambiguity_reasons = _pin_ambiguity_reasons(
        list(pin.get("candidate_hole_ids") or []),
        observations,
        vote_scores=vote_scores,
        selected_hole_id=hole_id or None,
    )
    pin["ambiguity_reasons"] = ambiguity_reasons
    pin["is_ambiguous"] = bool(ambiguity_reasons)
    metadata["fusion"] = {
        "evidence_source": fusion["evidence_source"],
        "decisive_view_id": fusion["decisive_view_id"],
        "normalized_confidence": fusion["normalized_confidence"],
        "margin": fusion["margin"],
        "cross_view_agreement": fusion["cross_view_agreement"],
        "per_view_contribution": fusion["per_view_contribution"],
        "per_view_top1": fusion["per_view_top1"],
        "occlusion_boost": fusion["occlusion_boost"],
    }
    pin["metadata"] = metadata


def _first_logic_for_hole(
    observations: List[Dict[str, Any]],
    hole_id: str,
) -> Optional[Tuple[str, str]]:
    for obs in observations:
        for logic_loc, candidate_hole in zip(obs.get("candidate_logic_locs") or [], obs.get("candidate_hole_ids") or []):
            if candidate_hole == hole_id and len(logic_loc) >= 2:
                return (str(logic_loc[0]), str(logic_loc[1]))
    return None


def _view_weight(view_id: str) -> float:
    if view_id == "top":
        return 1.0
    if view_id in {"left_front", "right_front"}:
        return 0.72
    return 0.6


def _visibility_weight(visibility: int) -> float:
    if visibility >= 2:
        return 1.0
    if visibility == 1:
        return 0.55
    return 0.0


def _prediction_source_weight(source: str) -> float:
    return {
        "model": 1.0,
        "mock_model": 1.0,
    }.get(source, 0.65)


def _snap_weight(snap_confidence: Any) -> float:
    """Map snap confidence to a vote weight in [0.4, 1.0].

    Snap confidence is in [0, 1]; we floor at 0.4 so a poorly snapped but
    otherwise high-confidence prediction still contributes (it just can't
    dominate). Unknown snap (None) is treated as neutral 0.85.
    """
    if snap_confidence is None:
        return 0.85
    try:
        value = float(snap_confidence)
    except (TypeError, ValueError):
        return 0.85
    if not (value == value):  # NaN
        return 0.85
    return float(max(0.4, min(1.0, value)))


def _roi_source_weight(source: str) -> float:
    return {
        "detected_bbox": 1.0,
        "associated_bbox_candidate": 0.9,
        "shared_bbox_fallback": 0.62,
        "unavailable": 0.0,
    }.get(source, 0.8)


def _calibration_mode(calibrator: BreadboardCalibrator) -> str:
    if getattr(calibrator, "_detected_hole_map", False):
        return "detected_hole_map"
    if getattr(calibrator, "_synthetic_grid", False):
        return "synthetic_fallback"
    if calibrator.is_grid_ready:
        return "visual"
    return "uninitialized"
