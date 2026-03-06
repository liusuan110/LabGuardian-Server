"""
Stage 2: 坐标映射

将 S1 的像素级检测结果映射到面包板逻辑坐标，
确定每个元件占用的 (row, col) 引脚对。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.pipeline.vision.calibrator import BreadboardCalibrator
from app.pipeline.vision.pin_hole_detector import PinHoleVerifier
from app.pipeline.vision.pin_utils import select_best_pin_pair

logger = logging.getLogger(__name__)


def run_mapping(
    detections: List[dict],
    calibrator: BreadboardCalibrator,
    image_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """把像素坐标 → 面包板逻辑坐标

    Returns:
        {
            "components": [
                {
                    "class_name": str,
                    "confidence": float,
                    "bbox": [x1,y1,x2,y2],
                    "pin1_pixel": [x,y] | None,
                    "pin2_pixel": [x,y] | None,
                    "pin1_logic": (row, col) | None,
                    "pin2_logic": (row, col) | None,
                    "is_obb": bool,
                    "wire_color": str | None,
                },
                ...
            ],
            "duration_ms": float,
        }
    """
    t0 = time.time()

    pin_verifier = PinHoleVerifier()
    mapped: List[dict] = []

    for det in detections:
        comp = dict(det)  # shallow copy

        p1_px = tuple(det["pin1_pixel"]) if det.get("pin1_pixel") else None
        p2_px = tuple(det["pin2_pixel"]) if det.get("pin2_pixel") else None
        bbox = tuple(det["bbox"])
        class_name = det["class_name"]

        # 尝试映射 pin1
        pin1_logic = _map_pin(p1_px, calibrator)
        # 尝试映射 pin2
        pin2_logic = _map_pin(p2_px, calibrator)

        # 如果映射失败，使用 bbox 端点推理
        if pin1_logic is None or pin2_logic is None:
            inferred_pins = _infer_pins_from_bbox(bbox, calibrator)
            if pin1_logic is None and len(inferred_pins) >= 1:
                pin1_logic = inferred_pins[0]
            if pin2_logic is None and len(inferred_pins) >= 2:
                pin2_logic = inferred_pins[1]

        # 选择最佳引脚对（考虑电气约束）
        if pin1_logic is not None and pin2_logic is not None:
            pin1_logic, pin2_logic = select_best_pin_pair(
                class_name, pin1_logic, pin2_logic
            )

        comp["pin1_logic"] = list(pin1_logic) if pin1_logic else None
        comp["pin2_logic"] = list(pin2_logic) if pin2_logic else None
        mapped.append(comp)

    duration_ms = (time.time() - t0) * 1000
    return {"components": mapped, "duration_ms": duration_ms}


def _map_pin(
    pixel: Optional[Tuple[float, float]],
    calibrator: BreadboardCalibrator,
) -> Optional[Tuple[int, int]]:
    """像素坐标 → (row, col)，失败返回 None"""
    if pixel is None:
        return None
    try:
        result = calibrator.pixel_to_logic(pixel[0], pixel[1])
        if result is not None:
            return result
    except Exception:
        pass
    return None


def _infer_pins_from_bbox(
    bbox: tuple,
    calibrator: BreadboardCalibrator,
) -> List[Tuple[int, int]]:
    """从 bbox 的两端推断引脚位置"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    # 根据 bbox 长宽比确定方向
    if w >= h:
        # 水平元件 → 取左右两端中点
        left_px = (x1, cy)
        right_px = (x2, cy)
    else:
        # 垂直元件 → 取上下两端中点
        left_px = (cx, y1)
        right_px = (cx, y2)

    pins = []
    for px in (left_px, right_px):
        mapped = _map_pin(px, calibrator)
        if mapped is not None:
            pins.append(mapped)
    return pins
