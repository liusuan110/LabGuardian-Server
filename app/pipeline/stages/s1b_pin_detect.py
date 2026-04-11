"""
Stage 1.5: Component ROI pin detection.

这一阶段承接组件检测结果，为每个 component 建立 ROI，并输出有序 pin 预测。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from app.pipeline.vision.pin_model import PinRoiDetector
from app.pipeline.vision.image_io import decode_images_b64, decode_summary, view_id_for_index
from app.pipeline.vision.pin_schema import (
    default_package_type,
    default_pin_schema_id,
    default_symmetry_group,
)
from app.pipeline.vision.roi_cropper import crop_component_roi

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


def run_pin_detect(
    detections: List[dict],
    images_b64: List[str],
    pin_detector: PinRoiDetector,
) -> Dict[str, Any]:
    """为每个组件 ROI 生成 ordered pin predictions。"""
    t0 = time.time()
    decoded = decode_images_b64(images_b64, logger=logger, stage_name="S1.5")
    summary = decode_summary(decoded)
    view_ids = _view_ids_from_images(images_b64)

    counters: Dict[str, int] = {}
    components: List[dict] = []
    for det in detections:
        component_type = str(det.get("class_name") or "UNKNOWN")
        component_id = det.get("component_id") or _next_component_id(component_type, counters)
        package_type = str(det.get("package_type") or default_package_type(component_type))
        bbox = tuple(det.get("bbox") or (0, 0, 0, 0))
        rois_by_view = _build_rois_by_view(decoded, bbox)

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
            "orientation": float(det.get("orientation", 0.0)),
        }

        predictions_by_view: Dict[str, List[dict]] = {}
        for view_id in view_ids:
            roi_spec = rois_by_view.get(view_id) or {}
            roi_image = roi_spec.get("image")
            roi_offset = tuple(roi_spec.get("offset") or (0, 0))
            predictions = pin_detector.predict_component_pins(
                component_id=component_id,
                component_type=component_type,
                package_type=package_type,
                pin_schema_id=pin_schema_id,
                roi_image=roi_image,
                roi_offset=roi_offset,
                view_id=view_id,
                confidence=float(det.get("confidence", 1.0)),
            )
            predictions_by_view[view_id] = [
                {
                    "pin_id": pred.pin_id,
                    "pin_name": pred.pin_name,
                    "keypoint": [float(pred.keypoint[0]), float(pred.keypoint[1])] if pred.keypoint else None,
                    "visibility": pred.visibility if roi_image is not None else 0,
                    "confidence": float(pred.confidence) if roi_image is not None else 0.0,
                    "source": pred.source if roi_image is not None else "unavailable",
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
        }
        component["roi_by_view"] = {
            view_id: {
                "offset": list((rois_by_view.get(view_id) or {}).get("offset") or [0, 0]),
                "shape": list((rois_by_view.get(view_id) or {}).get("shape") or [0, 0]),
                "source": (rois_by_view.get(view_id) or {}).get("source", "unavailable"),
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


def _build_rois_by_view(decoded_images: List[dict], bbox: tuple[int, int, int, int]) -> Dict[str, Dict[str, Any]]:
    rois: Dict[str, Dict[str, Any]] = {}
    for item in decoded_images:
        view_id = item["view_id"]
        image = item["image"]
        if image is None:
            rois[view_id] = {"image": None, "offset": (0, 0), "shape": [0, 0], "source": "unavailable"}
            continue
        roi_image, roi_offset = crop_component_roi(image, bbox)
        source = "detected_bbox" if view_id == "top" else "shared_bbox_fallback"
        rois[view_id] = {
            "image": roi_image,
            "offset": roi_offset,
            "shape": list(roi_image.shape[:2]) if roi_image is not None else [0, 0],
            "source": source,
        }
    return rois


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
    key = component_type.lower()
    prefix = _TYPE_PREFIX.get(key, key[:3].upper() or "CMP")
    counters[key] = counters.get(key, 0) + 1
    return f"{prefix}{counters[key]}"
