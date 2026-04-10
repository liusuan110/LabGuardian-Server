"""
Stage 1.5: Component ROI pin detection.

这一阶段承接组件检测结果，为每个 component 建立 ROI，并输出有序 pin 预测。
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Dict, List

import cv2
import numpy as np

from app.pipeline.vision.pin_model import PinRoiDetector
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
    images = [_decode_image(b64) for b64 in images_b64]
    images = [img for img in images if img is not None]
    view_ids = _view_ids_from_images(images_b64)

    counters: Dict[str, int] = {}
    components: List[dict] = []
    for det in detections:
        component_type = str(det.get("class_name") or "UNKNOWN")
        component_id = det.get("component_id") or _next_component_id(component_type, counters)
        package_type = str(det.get("package_type") or default_package_type(component_type))
        bbox = tuple(det.get("bbox") or (0, 0, 0, 0))

        top_image = images[0] if images else None
        roi_image = None
        roi_offset = (0, 0)
        if top_image is not None:
            roi_image, roi_offset = crop_component_roi(top_image, bbox)

        pin_schema_id = default_pin_schema_id(component_type, package_type)
        component = {
            "component_id": component_id,
            "component_type": component_type,
            "class_name": component_type,
            "package_type": package_type,
            "pin_schema_id": pin_schema_id,
            "part_subtype": det.get("part_subtype") or "",
            "symmetry_group": det.get("symmetry_group") or default_symmetry_group(component_type),
            "bbox": list(bbox),
            "confidence": float(det.get("confidence", 1.0)),
            "orientation": float(det.get("orientation", 0.0)),
        }

        predictions = pin_detector.predict_component_pins(
            component_id=component_id,
            component_type=component_type,
            package_type=package_type,
            pin_schema_id=pin_schema_id,
            roi_image=roi_image,
            roi_offset=roi_offset,
            confidence=float(det.get("confidence", 1.0)),
        )
        component["pins"] = [
            {
                "pin_id": pred.pin_id,
                "pin_name": pred.pin_name,
                "keypoints_by_view": _keypoints_by_view(pred.keypoint, view_ids),
                "visibility_by_view": _visibility_by_view(pred.visibility, view_ids),
                "confidence": pred.confidence,
            }
            for pred in predictions
        ]
        component["roi"] = {
            "offset": [roi_offset[0], roi_offset[1]],
            "shape": list(roi_image.shape[:2]) if roi_image is not None else [0, 0],
        }
        components.append(component)

    return {
        "components": components,
        "duration_ms": (time.time() - t0) * 1000,
    }


def _decode_image(b64: str) -> np.ndarray | None:
    try:
        data = base64.b64decode(b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _view_ids_from_images(images_b64: List[str]) -> List[str]:
    defaults = ["top", "left_front", "right_front"]
    if not images_b64:
        return ["top"]
    view_ids = defaults[: len(images_b64)]
    if len(images_b64) > len(defaults):
        for idx in range(len(defaults), len(images_b64)):
            view_ids.append(f"aux_view_{idx - len(defaults) + 1}")
    return view_ids


def _keypoints_by_view(
    top_keypoint: tuple[float, float] | None,
    view_ids: List[str],
) -> dict[str, list[float] | None]:
    payload: dict[str, list[float] | None] = {}
    for view_id in view_ids:
        if view_id == "top" and top_keypoint is not None:
            payload[view_id] = [float(top_keypoint[0]), float(top_keypoint[1])]
        else:
            payload[view_id] = None
    return payload


def _visibility_by_view(
    top_visibility: int,
    view_ids: List[str],
) -> dict[str, int]:
    payload: dict[str, int] = {}
    for view_id in view_ids:
        payload[view_id] = top_visibility if view_id == "top" else 0
    return payload


def _next_component_id(component_type: str, counters: Dict[str, int]) -> str:
    key = component_type.lower()
    prefix = _TYPE_PREFIX.get(key, key[:3].upper() or "CMP")
    counters[key] = counters.get(key, 0) + 1
    return f"{prefix}{counters[key]}"
