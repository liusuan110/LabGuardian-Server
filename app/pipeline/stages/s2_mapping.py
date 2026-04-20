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
from app.pipeline.vision.pin_schema import default_package_type, default_symmetry_group

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
    elif image_shape[0] > 0 and image_shape[1] > 0:
        calibrator.build_synthetic_grid(image_shape)
        calibration_mode = _calibration_mode(calibrator)

    board_schema = BoardSchema.default_breadboard()
    view_ids = _view_ids_from_images(images_b64)
    mapped: List[dict] = []
    for item in components:
        comp = dict(item)
        component_type = str(comp.get("component_type") or comp.get("class_name") or "UNKNOWN")
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
        mapped_pins.append(
            {
                "pin_id": int(pin.get("pin_id") or idx),
                "pin_name": str(pin.get("pin_name") or f"pin{idx}"),
                "logic_loc": list(selected_logic) if selected_logic else None,
                "hole_id": hole_id,
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
                },
            }
        )
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
        logic_candidates = _get_candidates(pixel, calibrator) if pixel else []
        candidate_hole_ids = [
            board_schema.normalize_hole_id(board_schema.logic_loc_to_hole_id(logic_loc))
            for logic_loc in logic_candidates
        ]
        candidate_node_ids = _candidate_node_ids(candidate_hole_ids, board_schema)
        observations.append(
            {
                "view_id": view_id,
                "keypoint": [float(keypoint[0]), float(keypoint[1])] if keypoint else None,
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
    if getattr(calibrator, "_synthetic_grid", False):
        return "synthetic_fallback"
    if calibrator.is_grid_ready:
        return "visual"
    return "uninitialized"
