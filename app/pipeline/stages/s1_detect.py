"""
Stage 1: YOLO 检测

输入: 1-3 张 base64 JPEG 图片
输出: 融合后的 Detection 列表
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from app.pipeline.vision.detector import ComponentDetector, Detection
from app.pipeline.vision.wire_analyzer import WireAnalyzer

logger = logging.getLogger(__name__)

# 支持的元件类型
ACTIVE_CLASSES = {"Resistor", "Wire", "LED", "resistor", "wire", "led"}

IOU_MERGE_THRESHOLD = 0.3


def run_detect(
    images_b64: List[str],
    detector: ComponentDetector,
    conf: float = 0.25,
    imgsz: int = 1280,
    roi_rect: tuple | None = None,
) -> Dict[str, Any]:
    """执行 YOLO 检测 + Wire 端点精炼 + 多图融合

    Returns:
        {
            "detections": [detection_dicts],
            "primary_image_shape": (h, w),
            "duration_ms": float,
        }
    """
    t0 = time.time()

    images = [_decode_image(b64) for b64 in images_b64]
    images = [img for img in images if img is not None]

    if not images:
        return {"detections": [], "primary_image_shape": (0, 0), "duration_ms": 0}

    wire_analyzer = WireAnalyzer()
    all_det_lists: List[List[Detection]] = []

    for img in images:
        if roi_rect is not None:
            rx1, ry1, rx2, ry2 = roi_rect
            cropped = img[ry1:ry2, rx1:rx2]
            dets = detector.detect(cropped, conf=conf, imgsz=imgsz)
            detector.offset_detections(dets, rx1, ry1)
        else:
            dets = detector.detect(img, conf=conf, imgsz=imgsz)

        # Wire 端点精炼
        for det in dets:
            if det.class_name.lower() == "wire":
                try:
                    endpoints, color = wire_analyzer.analyze_wire(img, det.bbox)
                    if endpoints is not None:
                        det.pin1_pixel, det.pin2_pixel = endpoints
                    det.wire_color = color
                except Exception:
                    pass

        dets = [d for d in dets if d.class_name in ACTIVE_CLASSES]
        all_det_lists.append(dets)

    # 多图融合
    if len(all_det_lists) == 1:
        merged = all_det_lists[0]
    else:
        merged = _fuse_detections(all_det_lists)

    duration_ms = (time.time() - t0) * 1000

    return {
        "detections": [_detection_to_dict(d) for d in merged],
        "primary_image_shape": images[0].shape[:2],
        "duration_ms": duration_ms,
    }


def _decode_image(b64: str) -> np.ndarray | None:
    try:
        data = base64.b64decode(b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _fuse_detections(det_lists: List[List[Detection]]) -> List[Detection]:
    """多图 IoU 融合"""
    if not det_lists:
        return []
    base = list(det_lists[0])
    for extra_dets in det_lists[1:]:
        for det in extra_dets:
            best_iou = 0.0
            best_idx = -1
            for i, bd in enumerate(base):
                iou = _compute_iou(det.bbox, bd.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= IOU_MERGE_THRESHOLD and best_idx >= 0:
                bd = base[best_idx]
                if det.confidence > bd.confidence:
                    bd.confidence = det.confidence
                    bd.bbox = det.bbox
                if det.pin1_pixel and bd.pin1_pixel:
                    bd.pin1_pixel = (
                        (det.pin1_pixel[0] + bd.pin1_pixel[0]) / 2,
                        (det.pin1_pixel[1] + bd.pin1_pixel[1]) / 2,
                    )
                if det.pin2_pixel and bd.pin2_pixel:
                    bd.pin2_pixel = (
                        (det.pin2_pixel[0] + bd.pin2_pixel[0]) / 2,
                        (det.pin2_pixel[1] + bd.pin2_pixel[1]) / 2,
                    )
            else:
                base.append(det)
    return base


def _compute_iou(box1: tuple, box2: tuple) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _detection_to_dict(det: Detection) -> dict:
    return {
        "class_name": det.class_name,
        "confidence": det.confidence,
        "bbox": list(det.bbox),
        "pin1_pixel": list(det.pin1_pixel) if det.pin1_pixel else None,
        "pin2_pixel": list(det.pin2_pixel) if det.pin2_pixel else None,
        "is_obb": det.is_obb,
        "wire_color": det.wire_color,
    }
