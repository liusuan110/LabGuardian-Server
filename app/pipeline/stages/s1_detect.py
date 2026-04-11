"""
Stage 1: YOLO 检测

输入: 1-3 张 base64 JPEG 图片
输出: top 主实例 + side 补召回候选
"""

from __future__ import annotations

import math
import logging
import time
from typing import Any, Dict, List, Tuple

from app.pipeline.vision.detector import ComponentDetector, Detection
from app.pipeline.vision.image_io import decode_images_b64, decode_summary
from app.pipeline.vision.pin_schema import default_package_type, default_pin_schema_id

logger = logging.getLogger(__name__)

# ── 类名标准化映射: YOLO 输出 → Pipeline 标准类名 ──
# 确保不同模型训练出的类名统一
CLASS_NAME_MAP = {
    "resistor": "Resistor",
    "Resistor": "Resistor",
    "capacitor": "Capacitor",
    "Capacitor": "Capacitor",
    "wire": "Wire",
    "Wire": "Wire",
    "led": "LED",
    "LED": "LED",
    "Led": "LED",
    "diode": "Diode",
    "Diode": "Diode",
    "IC": "IC",
    "ic": "IC",
    "potentiometer": "Potentiometer",
    "Potentiometer": "Potentiometer",
}

# 作为元件参与拓扑构建的类别 (标准化后的名称)
COMPONENT_CLASSES = {
    "Resistor", "Capacitor", "Wire", "LED",
    "Diode", "IC", "Potentiometer",
}

# 过滤掉的背景类 (Breadboard, Line_area 等)
IGNORED_CLASSES = {"Breadboard", "Line_area", "breadboard", "line_area", "pinned", "Pinned"}

_TYPE_PREFIX = {
    "resistor": "R",
    "capacitor": "C",
    "wire": "W",
    "led": "LED",
    "diode": "D",
    "ic": "IC",
    "potentiometer": "POT",
}


def run_detect(
    images_b64: List[str],
    detector: ComponentDetector,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 1280,
    roi_rect: tuple | None = None,
) -> Dict[str, Any]:
    """执行组件检测 + 多视图补召回。

    当前策略:
    - top 视图负责建立全局 component_id
    - side 视图负责补召回候选, 但不直接进入主实例列表
    - side recall 候选显式输出到 supplemental_detections

    Returns:
        {
            "detections": [detection_dicts],
            "supplemental_detections": [...],
            "primary_image_shape": (h, w),
            "duration_ms": float,
        }
    """
    t0 = time.time()

    decoded = decode_images_b64(images_b64, logger=logger, stage_name="S1")
    summary = decode_summary(decoded)
    top_item = next((item for item in decoded if item["view_id"] == "top" and item["decoded"]), None)
    top_image = top_item["image"] if top_item else None
    supplemental_detections: List[dict] = []

    for item in decoded:
        if not item["decoded"] or item["view_id"] == "top":
            continue
        side_detections = _detect_components_for_view(
            image=item["image"],
            detector=detector,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            roi_rect=roi_rect,
        )
        supplemental_detections.extend(
            _candidate_detection_to_dict(det, view_id=item["view_id"], candidate_index=index + 1)
            for index, det in enumerate(side_detections)
        )

    if top_image is None:
        return {
            "interface_version": "component_detect_v1",
            "detector_backend": "yolo_obb_component",
            "detections": [],
            "supplemental_detections": supplemental_detections,
            "recall_mode": "side_candidates_only",
            "primary_image_shape": (0, 0),
            **summary,
            "duration_ms": 0,
        }

    component_dets = _detect_components_for_view(
        image=top_image,
        detector=detector,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        roi_rect=roi_rect,
    )
    _assign_component_ids(component_dets)

    duration_ms = (time.time() - t0) * 1000

    return {
        "interface_version": "component_detect_v1",
        "detector_backend": "yolo_obb_component",
        "detections": [_detection_to_dict(d) for d in component_dets],
        "supplemental_detections": supplemental_detections,
        "recall_mode": "top_primary_plus_side_candidates",
        "primary_image_shape": top_image.shape[:2],
        **summary,
        "duration_ms": duration_ms,
    }


def _assign_component_ids(detections: List[Detection]) -> None:
    counters: Dict[str, int] = {}
    for det in detections:
        key = det.class_name.lower()
        prefix = _TYPE_PREFIX.get(key, key[:3].upper() or "CMP")
        counters[key] = counters.get(key, 0) + 1
        setattr(det, "component_id", f"{prefix}{counters[key]}")


def _compute_orientation(det: Detection) -> float:
    if det.obb_corners is not None:
        p0 = det.obb_corners[0]
        p1 = det.obb_corners[1]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        return float(math.degrees(math.atan2(dy, dx)))
    x1, y1, x2, y2 = det.bbox
    return 0.0 if (x2 - x1) >= (y2 - y1) else 90.0


def _detect_components_for_view(
    *,
    image,
    detector: ComponentDetector,
    conf: float,
    iou: float,
    imgsz: int,
    roi_rect: tuple | None,
) -> List[Detection]:
    if roi_rect is not None:
        rx1, ry1, rx2, ry2 = roi_rect
        cropped = image[ry1:ry2, rx1:rx2]
        detections = detector.detect(cropped, conf=conf, iou=iou, imgsz=imgsz)
        detector.offset_detections(detections, rx1, ry1)
    else:
        detections = detector.detect(image, conf=conf, iou=iou, imgsz=imgsz)

    for det in detections:
        det.class_name = CLASS_NAME_MAP.get(det.class_name, det.class_name)
    return [det for det in detections if det.class_name in COMPONENT_CLASSES and det.class_name not in IGNORED_CLASSES]


def _detection_to_dict(det: Detection) -> dict:
    component_type = det.class_name
    package_type = default_package_type(component_type)
    pin_schema_id = default_pin_schema_id(component_type, package_type)
    return {
        "component_id": getattr(det, "component_id", ""),
        "input_detection_interface_version": "component_detect_v1",
        "class_name": det.class_name,
        "component_type": component_type,
        "package_type": package_type,
        "pin_schema_id": pin_schema_id,
        "confidence": det.confidence,
        "bbox": list(det.bbox),
        "is_obb": det.is_obb,
        "orientation": _compute_orientation(det),
        "view_id": "top",
        "source": "component_detector",
        "source_model_type": "yolo_obb_component",
        "wire_color": det.wire_color,
        "obb_corners": det.obb_corners.tolist() if det.obb_corners is not None else None,
    }


def _candidate_detection_to_dict(
    det: Detection,
    *,
    view_id: str,
    candidate_index: int,
) -> dict:
    component_type = det.class_name
    package_type = default_package_type(component_type)
    pin_schema_id = default_pin_schema_id(component_type, package_type)
    return {
        "candidate_id": f"{view_id}_{component_type.lower()}_{candidate_index}",
        "class_name": det.class_name,
        "component_type": component_type,
        "package_type": package_type,
        "pin_schema_id": pin_schema_id,
        "confidence": det.confidence,
        "bbox": list(det.bbox),
        "is_obb": det.is_obb,
        "orientation": _compute_orientation(det),
        "view_id": view_id,
        "source": "side_recall_candidate",
        "source_model_type": "yolo_obb_component",
        "instance_status": "candidate",
        "wire_color": det.wire_color,
        "obb_corners": det.obb_corners.tolist() if det.obb_corners is not None else None,
    }
