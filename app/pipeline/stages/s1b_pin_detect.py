"""
Stage 1.5: Component ROI pin detection.

这一阶段承接组件检测结果，为每个 component 建立 ROI，并输出有序 pin 预测。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from app.pipeline.vision.pin_model import PinRoiDetector
from app.pipeline.vision.image_io import decode_images_b64, decode_summary
from app.pipeline.vision.label_mapping import component_id_prefix, normalize_component_type
from app.pipeline.vision.pin_schema import (
    default_package_type,
    default_pin_schema_id,
    default_symmetry_group,
)
from app.pipeline.vision.roi_cropper import crop_component_roi
from app.pipeline.vision.view_association import SideViewRoiResolver

logger = logging.getLogger(__name__)


ROI_RETRY_SCALES = {
    "top": [1.0, 1.4, 1.85, 2.35],
    "left_front": [1.0, 1.3, 1.7],
    "right_front": [1.0, 1.3, 1.7],
}

def run_pin_detect(
    detections: List[dict],
    images_b64: List[str],
    pin_detector: PinRoiDetector,
    supplemental_detections: List[dict] | None = None,
) -> Dict[str, Any]:
    """为每个组件 ROI 生成 ordered pin predictions。"""
    t0 = time.time()
    decoded = decode_images_b64(images_b64, logger=logger, stage_name="S1.5")
    summary = decode_summary(decoded)
    view_ids = _view_ids_from_images(images_b64)
    roi_resolver = SideViewRoiResolver()

    counters: Dict[str, int] = {}
    components: List[dict] = []
    for det in detections:
        component_type = normalize_component_type(str(det.get("component_type") or det.get("class_name") or "UNKNOWN"))
        component_id = det.get("component_id") or _next_component_id(component_type, counters)
        package_type = str(det.get("package_type") or default_package_type(component_type))
        bbox = tuple(det.get("bbox") or (0, 0, 0, 0))
        orientation = float(det.get("orientation", 0.0))
        obb_corners = det.get("obb_corners")
        rois_by_view = _build_rois_by_view(
            decoded,
            bbox,
            component_type=component_type,
            package_type=package_type,
            orientation=orientation,
            obb_corners=obb_corners,
            supplemental_detections=supplemental_detections,
            component_detection=det,
            roi_resolver=roi_resolver,
        )

        pin_schema_id = default_pin_schema_id(component_type, package_type)
        component = {
            "component_id": component_id,
            "component_type": component_type,
            "class_name": component_type,
            "package_type": package_type,
            "pin_schema_id": pin_schema_id,
            "input_pin_detect_interface_version": "component_pin_detect_v1",
            "input_detection_interface_version": det.get("input_detection_interface_version") or "component_detect_v1",
            "part_subtype": det.get("part_subtype") or "",
            "symmetry_group": det.get("symmetry_group") or default_symmetry_group(component_type),
            "bbox": list(bbox),
            "confidence": float(det.get("confidence", 1.0)),
            "orientation": orientation,
        }

        predictions_by_view: Dict[str, List[dict]] = {}
        for view_id in view_ids:
            roi_spec = rois_by_view.get(view_id) or {}
            predictions, roi_spec = _predict_with_adaptive_roi(
                pin_detector=pin_detector,
                component_id=component_id,
                component_type=component_type,
                package_type=package_type,
                pin_schema_id=pin_schema_id,
                component_confidence=float(det.get("confidence", 1.0)),
                view_id=view_id,
                base_roi_spec=roi_spec,
                image_item=next((item for item in decoded if item["view_id"] == view_id), None),
                component_detection=det,
                bbox=bbox,
                orientation=orientation,
                obb_corners=obb_corners,
            )
            rois_by_view[view_id] = roi_spec
            roi_available = bool(roi_spec.get("image") is not None)
            predictions_by_view[view_id] = [
                {
                    "pin_id": pred.pin_id,
                    "pin_name": pred.pin_name,
                    "keypoint": [float(pred.keypoint[0]), float(pred.keypoint[1])] if pred.keypoint else None,
                    "visibility": pred.visibility if roi_available else 0,
                    "confidence": float(pred.confidence) if roi_available else 0.0,
                    "source": pred.source if roi_available else "unavailable",
                    "metadata": {
                        **dict(pred.metadata),
                        "roi_source": roi_spec.get("source", "unavailable"),
                    },
                }
                for pred in predictions
            ]

        component["pins"] = _merge_predictions_by_view(
            predictions_by_view=predictions_by_view,
            view_ids=view_ids,
        )
        top_roi = rois_by_view.get("top") or {}
        component["roi"] = {
            "offset": list(top_roi.get("offset") or [0, 0]),
            "shape": list(top_roi.get("shape") or [0, 0]),
            "source": top_roi.get("source", "unavailable"),
            "crop_source": top_roi.get("crop_source", "unavailable"),
            "crop_profile": top_roi.get("crop_profile", "none"),
            "crop_bounds": top_roi.get("crop_bounds"),
            "body_bbox": top_roi.get("body_bbox"),
            "body_size": top_roi.get("body_size"),
            "roi_size": top_roi.get("roi_size"),
            "scale_multiplier": top_roi.get("scale_multiplier", 1.0),
            "retry_attempts": top_roi.get("retry_attempts", 1),
        }
        component["roi_by_view"] = {
            view_id: {
                "offset": list((rois_by_view.get(view_id) or {}).get("offset") or [0, 0]),
                "shape": list((rois_by_view.get(view_id) or {}).get("shape") or [0, 0]),
                "source": (rois_by_view.get(view_id) or {}).get("source", "unavailable"),
                "crop_source": (rois_by_view.get(view_id) or {}).get("crop_source", "unavailable"),
                "crop_profile": (rois_by_view.get(view_id) or {}).get("crop_profile", "none"),
                "crop_bounds": (rois_by_view.get(view_id) or {}).get("crop_bounds"),
                "body_bbox": (rois_by_view.get(view_id) or {}).get("body_bbox"),
                "body_size": (rois_by_view.get(view_id) or {}).get("body_size"),
                "roi_size": (rois_by_view.get(view_id) or {}).get("roi_size"),
                "scale_multiplier": (rois_by_view.get(view_id) or {}).get("scale_multiplier", 1.0),
                "retry_attempts": (rois_by_view.get(view_id) or {}).get("retry_attempts", 1),
                "association": (rois_by_view.get(view_id) or {}).get("association") or {},
                "available": bool((rois_by_view.get(view_id) or {}).get("image") is not None),
            }
            for view_id in view_ids
        }
        component["pin_detector"] = {
            "interface_version": pin_detector.interface_version,
            "backend_type": pin_detector.backend_type,
            "backend_mode": pin_detector.backend_mode,
        }
        components.append(component)

    return {
        "interface_version": "component_pin_detect_v1",
        "pin_detector_backend": pin_detector.backend_type,
        "pin_detector_mode": pin_detector.backend_mode,
        "pin_detector_contract": dict(getattr(pin_detector, "model_contract", {}) or {}),
        "side_roi_assoc_backend": roi_resolver.interface_version,
        "components": components,
        **summary,
        "duration_ms": (time.time() - t0) * 1000,
    }


def _view_ids_from_images(images_b64: List[str]) -> List[str]:
    defaults = ["top", "left_front", "right_front"]
    if not images_b64:
        return ["top"]
    view_ids = defaults[: len(images_b64)]
    if len(images_b64) > len(defaults):
        for idx in range(len(defaults), len(images_b64)):
            view_ids.append(f"aux_view_{idx - len(defaults) + 1}")
    return view_ids


def _build_rois_by_view(
    decoded_images: List[dict],
    bbox: tuple[int, int, int, int],
    *,
    component_type: str,
    package_type: str,
    orientation: float,
    obb_corners: Any,
    supplemental_detections: List[dict] | None,
    component_detection: dict,
    roi_resolver: SideViewRoiResolver,
) -> Dict[str, Dict[str, Any]]:
    rois: Dict[str, Dict[str, Any]] = {}
    for item in decoded_images:
        view_id = item["view_id"]
        image = item["image"]
        if image is None:
            rois[view_id] = {
                "image": None,
                "offset": (0, 0),
                "shape": [0, 0],
                "source": "unavailable",
                "crop_source": "unavailable",
                "crop_profile": "none",
                "crop_bounds": None,
            }
            continue
        view_bbox = bbox
        assoc_source = "detected_bbox" if view_id == "top" else "shared_bbox_fallback"
        assoc_meta: dict[str, Any] = {}
        if view_id != "top":
            association = roi_resolver.resolve(
                component_detection=component_detection,
                view_id=view_id,
                supplemental_detections=supplemental_detections,
            )
            if association is not None:
                view_bbox = association.bbox
                assoc_source = association.source
                assoc_meta = dict(association.metadata or {})
                assoc_meta["matched"] = association.matched
                assoc_meta["candidate_id"] = association.candidate_id
                assoc_meta["association_confidence"] = association.confidence
        roi_image, roi_offset, roi_meta = crop_component_roi(
            image,
            view_bbox,
            component_type=component_type,
            package_type=package_type,
            orientation=orientation,
            obb_corners=obb_corners if view_id == "top" else None,
            view_id=view_id,
        )
        rois[view_id] = {
            "image": roi_image,
            "offset": roi_offset,
            "shape": list(roi_image.shape[:2]) if roi_image is not None else [0, 0],
            "bbox": list(view_bbox),
            "source": assoc_source,
            "crop_source": roi_meta.get("source", "package_profile_crop"),
            "crop_profile": roi_meta.get("profile_name", "generic"),
            "crop_bounds": roi_meta.get("bounds"),
            "body_bbox": roi_meta.get("body_bbox"),
            "body_size": roi_meta.get("body_size"),
            "roi_size": roi_meta.get("roi_size"),
            "association": assoc_meta,
        }
    return rois


def _predict_with_adaptive_roi(
    *,
    pin_detector: PinRoiDetector,
    component_id: str,
    component_type: str,
    package_type: str,
    pin_schema_id: str,
    component_confidence: float,
    view_id: str,
    base_roi_spec: dict[str, Any],
    image_item: dict | None,
    component_detection: dict,
    bbox: tuple[int, int, int, int],
    orientation: float,
    obb_corners: Any,
) -> tuple[list, dict[str, Any]]:
    image = (image_item or {}).get("image")
    if image is None:
        predictions = pin_detector.predict_component_pins(
            component_id=component_id,
            component_type=component_type,
            package_type=package_type,
            pin_schema_id=pin_schema_id,
            roi_image=None,
            roi_offset=(0, 0),
            view_id=view_id,
            confidence=component_confidence,
        )
        base_roi_spec = dict(base_roi_spec)
        base_roi_spec["retry_scale"] = None
        base_roi_spec["retry_attempts"] = 0
        return predictions, base_roi_spec

    retry_scales = ROI_RETRY_SCALES.get(view_id, [1.0, 1.2, 1.45])
    assoc_source = str(base_roi_spec.get("source", "detected_bbox"))
    crop_source = str(base_roi_spec.get("crop_source", "package_profile_crop"))
    best_predictions = None
    best_roi_spec = dict(base_roi_spec)

    view_bbox = bbox
    if base_roi_spec.get("bbox") and len(base_roi_spec.get("bbox")) == 4:
        view_bbox = tuple(int(v) for v in base_roi_spec.get("bbox"))
    if view_id != "top" and assoc_source != "shared_bbox_fallback":
        association = (base_roi_spec.get("association") or {})
        candidate_bbox = association.get("candidate_bbox")
        if candidate_bbox and len(candidate_bbox) == 4:
            view_bbox = tuple(int(v) for v in candidate_bbox)

    for attempt, scale in enumerate(retry_scales, start=1):
        roi_image, roi_offset, roi_meta = crop_component_roi(
            image,
            view_bbox,
            component_type=component_type,
            package_type=package_type,
            orientation=orientation,
            obb_corners=obb_corners if view_id == "top" else None,
            view_id=view_id,
            scale_multiplier=scale,
        )
        predictions = pin_detector.predict_component_pins(
            component_id=component_id,
            component_type=component_type,
            package_type=package_type,
            pin_schema_id=pin_schema_id,
            roi_image=roi_image,
            roi_offset=tuple(roi_offset or (0, 0)),
            view_id=view_id,
            confidence=component_confidence,
        )
        roi_spec = dict(base_roi_spec)
        roi_spec.update(
            {
                "image": roi_image,
                "offset": roi_offset,
                "shape": list(roi_image.shape[:2]) if roi_image is not None else [0, 0],
                "source": assoc_source,
                "crop_source": roi_meta.get("source", crop_source),
                "crop_profile": roi_meta.get("profile_name", "generic"),
                "crop_bounds": roi_meta.get("bounds"),
                "body_bbox": roi_meta.get("body_bbox"),
                "body_size": roi_meta.get("body_size"),
                "roi_size": roi_meta.get("roi_size"),
                "scale_multiplier": float(scale),
                "retry_scale": float(scale),
                "retry_attempts": attempt,
            }
        )
        best_predictions = predictions
        best_roi_spec = roi_spec
        if _predictions_are_usable(predictions):
            break

    return best_predictions or [], best_roi_spec


def _predictions_are_usable(predictions: list) -> bool:
    if not predictions:
        return False
    visible = 0
    model_visible = 0
    for pred in predictions:
        if getattr(pred, "keypoint", None) is not None:
            visible += 1
        if getattr(pred, "source", "") == "model" and getattr(pred, "keypoint", None) is not None:
            model_visible += 1
    return visible >= 2 and model_visible >= max(1, visible // 2)


def _merge_predictions_by_view(
    *,
    predictions_by_view: Dict[str, List[dict]],
    view_ids: List[str],
) -> List[dict]:
    merged: Dict[int, dict] = {}
    for view_id in view_ids:
        for pred in predictions_by_view.get(view_id, []):
            pin_id = int(pred["pin_id"])
            pin_entry = merged.setdefault(
                pin_id,
                {
                    "pin_id": pin_id,
                    "pin_name": pred["pin_name"],
                    "keypoints_by_view": {vid: None for vid in view_ids},
                    "visibility_by_view": {vid: 0 for vid in view_ids},
                    "score_by_view": {vid: 0.0 for vid in view_ids},
                    "source_by_view": {vid: "unavailable" for vid in view_ids},
                    "confidence": 0.0,
                    "source": "unavailable",
                    "metadata": {"per_view": {}},
                },
            )
            pin_entry["keypoints_by_view"][view_id] = pred["keypoint"]
            pin_entry["visibility_by_view"][view_id] = int(pred["visibility"])
            pin_entry["score_by_view"][view_id] = float(pred["confidence"])
            pin_entry["source_by_view"][view_id] = str(pred["source"])
            pin_entry["metadata"]["per_view"][view_id] = dict(pred.get("metadata") or {})

    ordered = []
    for pin_id in sorted(merged):
        item = merged[pin_id]
        scores = [score for score in item["score_by_view"].values() if score > 0]
        item["confidence"] = max(scores) if scores else 0.0
        if any(source == "model" for source in item["source_by_view"].values()):
            item["source"] = "model"
        elif any(source == "heuristic_fallback" for source in item["source_by_view"].values()):
            item["source"] = "heuristic_fallback"
        ordered.append(item)
    return ordered


def _next_component_id(component_type: str, counters: Dict[str, int]) -> str:
    normalized = normalize_component_type(component_type)
    prefix = component_id_prefix(normalized)
    counters[normalized] = counters.get(normalized, 0) + 1
    return f"{prefix}{counters[normalized]}"
