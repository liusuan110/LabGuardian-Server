"""
Stage 2: hole mapping.

将 S1.5 的 ordered pin 预测映射到面包板 hole_id / electrical node。
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

    if images_b64:
        _ensure_calibrated(calibrator, images_b64[0], image_shape)
    elif image_shape[0] > 0 and image_shape[1] > 0:
        calibrator.build_synthetic_grid(image_shape)

    board_schema = BoardSchema.default_breadboard()
    view_ids = _view_ids_from_images(images_b64)
    mapped: List[dict] = []
    for item in components:
        comp = dict(item)
        component_type = str(comp.get("component_type") or comp.get("class_name") or "UNKNOWN")
        comp["component_type"] = component_type
        comp["class_name"] = component_type
        comp["package_type"] = comp.get("package_type") or default_package_type(component_type)
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
        top_keypoint = _extract_top_keypoint(keypoints_by_view)
        logic_candidates = _get_candidates(top_keypoint, calibrator) if top_keypoint else []

        selected_logic = logic_candidates[0] if logic_candidates else None
        hole_id = pin.get("hole_id")
        if not hole_id and selected_logic:
            hole_id = board_schema.logic_loc_to_hole_id(selected_logic)
        if not hole_id:
            continue

        hole_id = board_schema.normalize_hole_id(str(hole_id))
        electrical_node_id = pin.get("electrical_node_id") or board_schema.resolve_hole_to_node(hole_id)
        candidate_hole_ids = _candidate_hole_ids_from_logic(
            selected_hole_id=hole_id,
            logic_candidates=logic_candidates,
            board_schema=board_schema,
            fallback_candidates=pin.get("candidate_hole_ids") or [],
        )
        candidate_node_ids = _candidate_node_ids(candidate_hole_ids, board_schema)
        observations = _build_pin_observations_from_predictions(
            keypoints_by_view=keypoints_by_view,
            visibility_by_view=visibility_by_view,
            view_ids=view_ids,
            confidence=float(pin.get("confidence", comp.get("confidence", 1.0))),
        )
        ambiguity_reasons = _pin_ambiguity_reasons(candidate_hole_ids, observations)
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
                "metadata": dict(pin.get("metadata") or {}),
            }
        )
    return mapped_pins


def _extract_top_keypoint(
    keypoints_by_view: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    top = keypoints_by_view.get("top")
    if top and len(top) >= 2:
        return (float(top[0]), float(top[1]))
    for value in keypoints_by_view.values():
        if value and len(value) >= 2:
            return (float(value[0]), float(value[1]))
    return None


def _decode_primary_image(image_b64: str) -> Optional[np.ndarray]:
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
    if calibrator.is_grid_ready:
        return
    img = _decode_primary_image(image_b64)
    if img is not None:
        try:
            calibrator.ensure_calibrated(img)
            if calibrator.is_grid_ready:
                return
        except Exception as exc:
            logger.warning("Calibration from image failed: %s", exc)
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
    except Exception:
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
    view_ids: List[str],
    confidence: float,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for view_id in view_ids:
        keypoint = keypoints_by_view.get(view_id)
        visibility = int(visibility_by_view.get(view_id, 0))
        observations.append(
            {
                "view_id": view_id,
                "keypoint": [float(keypoint[0]), float(keypoint[1])] if keypoint else None,
                "visibility": visibility,
                "confidence": confidence if visibility > 0 else 0.0,
            }
        )
    return observations


def _pin_ambiguity_reasons(
    candidate_hole_ids: List[str],
    observations: List[Dict[str, Any]],
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
    return reasons
