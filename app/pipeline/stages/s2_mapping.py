"""
Stage 2: hole mapping.

将 S1.5 的 ordered pin 预测映射到面包板 hole_id / electrical node。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

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
        electrical_node_id = pin.get("electrical_node_id") or board_schema.resolve_hole_to_node(hole_id)
        candidate_hole_ids = vote_result["candidate_hole_ids"]
        candidate_node_ids = _candidate_node_ids(candidate_hole_ids, board_schema)
        ambiguity_reasons = _pin_ambiguity_reasons(
            candidate_hole_ids,
            observations,
            vote_scores=vote_result["vote_scores"],
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
                },
            }
        )
    return _apply_component_pair_selector(
        comp=comp,
        mapped_pins=mapped_pins,
        calibrator=calibrator,
        board_schema=board_schema,
    )


def _apply_component_pair_selector(
    *,
    comp: dict,
    mapped_pins: List[Dict[str, Any]],
    calibrator: BreadboardCalibrator,
    board_schema: BoardSchema,
) -> List[Dict[str, Any]]:
    component_type = normalize_component_type(str(comp.get("component_type") or comp.get("class_name") or "UNKNOWN"))
    if component_type not in {"Wire", "CapacitorElectrolytic"}:
        return mapped_pins
    if len(mapped_pins) < 2:
        return mapped_pins

    pair = mapped_pins[:2]
    resolved = _resolve_two_pin_hole_pair(
        pins=pair,
        component_type=component_type,
        calibrator=calibrator,
        board_schema=board_schema,
    )
    if resolved is None:
        return mapped_pins

    for idx, resolved_pin in enumerate(resolved):
        mapped_pins[idx] = resolved_pin
    return mapped_pins
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


def _resolve_two_pin_hole_pair(
    *,
    pins: List[Dict[str, Any]],
    component_type: str,
    calibrator: BreadboardCalibrator,
    board_schema: BoardSchema,
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
        candidates_per_pin.append(ranked_candidates[:4])

    if not candidates_per_pin[0] or not candidates_per_pin[1]:
        return None

    best: Optional[Tuple[float, Tuple[str, Optional[Tuple[str, str]], Optional[Tuple[float, float]], int], Tuple[str, Optional[Tuple[str, str]], Optional[Tuple[float, float]], int]]] = None
    desired_distance = _pair_seed_distance(seed_points[0], seed_points[1])
    pair_orientation = _pair_seed_orientation(seed_points[0], seed_points[1])
    pitch = _mapping_pitch(calibrator)
    for cand1 in candidates_per_pin[0]:
        for cand2 in candidates_per_pin[1]:
            hole1, logic1, point1, rank1 = cand1
            hole2, logic2, point2, rank2 = cand2
            if hole1 == hole2:
                continue
            score = (rank1 + rank2) * 0.55
            if point1 is not None and point2 is not None and desired_distance is not None:
                actual_distance = _euclidean(point1, point2)
                score += abs(actual_distance - desired_distance) / max(pitch, 1.0) * 0.28
                if pair_orientation == "horizontal" and point1[0] >= point2[0]:
                    score += 6.0
                elif pair_orientation == "vertical" and point1[1] >= point2[1]:
                    score += 6.0
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
        logic_candidates = (
            _candidates_for_projection(
                pixel=pixel,
                projection=projection,
                calibrator=calibrator,
            )
            if pixel is not None or projection.should_use_board_point_for_mapping
            else []
        )
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
            }
        )
    return observations


def _pin_ambiguity_reasons(
    candidate_hole_ids: List[str],
    observations: List[Dict[str, Any]],
    *,
    vote_scores: Dict[str, float],
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
        if ordered[0] - ordered[1] < 0.2:
            reasons.append("close_vote_margin")
    return reasons


def _vote_hole_from_observations(
    *,
    observations: List[Dict[str, Any]],
    board_schema: BoardSchema,
    explicit_hole_id: str | None,
    fallback_candidates: List[str],
) -> Dict[str, Any]:
    vote_scores: Dict[str, float] = {}
    for obs in observations:
        visibility = int(obs.get("visibility", 0))
        if visibility <= 0:
            continue
        confidence = float(obs.get("confidence", 0.0))
        if confidence <= 0.0:
            continue
        view_weight = _view_weight(str(obs.get("view_id", "")))
        source_weight = _prediction_source_weight(str(obs.get("source", "")))
        roi_weight = _roi_source_weight(str(obs.get("roi_source", "")))
        base = confidence * _visibility_weight(visibility) * view_weight * source_weight * roi_weight
        for rank, hole_id in enumerate(obs.get("candidate_hole_ids") or []):
            normalized = board_schema.normalize_hole_id(str(hole_id))
            vote_scores[normalized] = vote_scores.get(normalized, 0.0) + base * (0.72 ** rank)

    if explicit_hole_id:
        normalized = board_schema.normalize_hole_id(str(explicit_hole_id))
        vote_scores[normalized] = vote_scores.get(normalized, 0.0) + 0.15

    for rank, hole_id in enumerate(fallback_candidates):
        normalized = board_schema.normalize_hole_id(str(hole_id))
        vote_scores[normalized] = vote_scores.get(normalized, 0.0) + 0.05 * (0.8 ** rank)

    ordered = [item[0] for item in sorted(vote_scores.items(), key=lambda item: item[1], reverse=True)]
    selected = ordered[0] if ordered else (board_schema.normalize_hole_id(str(explicit_hole_id)) if explicit_hole_id else None)
    return {
        "selected_hole_id": selected,
        "candidate_hole_ids": ordered,
        "vote_scores": {key: round(val, 6) for key, val in sorted(vote_scores.items(), key=lambda item: item[1], reverse=True)},
        "selected_by": "multi_view_weighted_vote" if ordered else "explicit_or_empty",
    }


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
        "heuristic_fallback": 0.72,
    }.get(source, 0.65)


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
