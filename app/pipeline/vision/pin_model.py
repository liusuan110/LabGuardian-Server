"""
Component ROI pin detector.

第一版为可运行骨架:
- 输入组件 ROI + schema
- 输出 ordered pins
- 为后续真实第二个模型预留统一接口
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.pipeline.vision.pin_schema import default_pin_names


@dataclass
class PinPrediction:
    pin_id: int
    pin_name: str
    keypoint: tuple[float, float] | None
    confidence: float
    visibility: int


class PinRoiDetector:
    """ROI pin detector skeleton.

    当前实现是模型占位版:
    - 保留正式输入/输出接口
    - 只依赖 ROI + schema 生成几何先验 keypoint
    - 后续真实 pin 模型接入时，只需要替换 `predict_component_pins`
    """

    def predict_component_pins(
        self,
        *,
        component_id: str,
        component_type: str,
        package_type: str,
        pin_schema_id: str,
        roi_image: np.ndarray | None,
        roi_offset: tuple[int, int],
        confidence: float = 1.0,
    ) -> list[PinPrediction]:
        pin_count = 2
        if component_type == "IC" and package_type == "dip8":
            pin_count = 2  # anchor pair first; S2/S3 later expand to full DIP pins
        elif component_type == "Potentiometer":
            pin_count = 3

        pin_names = default_pin_names(component_type, pin_count)
        keypoints = self._stub_keypoints(
            component_type=component_type,
            package_type=package_type,
            pin_schema_id=pin_schema_id,
            roi_image=roi_image,
            roi_offset=roi_offset,
            pin_count=pin_count,
        )

        predictions: list[PinPrediction] = []
        for idx in range(pin_count):
            keypoint = keypoints[idx] if idx < len(keypoints) else None
            predictions.append(
                PinPrediction(
                    pin_id=idx + 1,
                    pin_name=pin_names[idx],
                    keypoint=keypoint,
                    confidence=confidence,
                    visibility=2 if keypoint is not None else 0,
                )
            )
        return predictions

    def _stub_keypoints(
        self,
        *,
        component_type: str,
        package_type: str,
        pin_schema_id: str,
        roi_image: np.ndarray | None,
        roi_offset: tuple[int, int],
        pin_count: int,
    ) -> list[tuple[float, float] | None]:
        """过渡 stub: 仅用 ROI 几何生成有序 pin keypoint。"""
        if roi_image is None or roi_image.size == 0:
            return [None] * pin_count

        h, w = roi_image.shape[:2]
        ox, oy = roi_offset

        def to_global(local_x: float, local_y: float) -> tuple[float, float]:
            return (float(local_x + ox), float(local_y + oy))

        if component_type == "IC" and package_type == "dip8":
            if h >= w:
                return [
                    to_global(w * 0.25, h * 0.2),
                    to_global(w * 0.25, h * 0.8),
                ]
            return [
                to_global(w * 0.2, h * 0.25),
                to_global(w * 0.8, h * 0.25),
            ]

        if pin_count == 3:
            if w >= h:
                return [
                    to_global(w * 0.2, h * 0.5),
                    to_global(w * 0.5, h * 0.5),
                    to_global(w * 0.8, h * 0.5),
                ]
            return [
                to_global(w * 0.5, h * 0.2),
                to_global(w * 0.5, h * 0.5),
                to_global(w * 0.5, h * 0.8),
            ]

        if w >= h:
            return [
                to_global(w * 0.15, h * 0.5),
                to_global(w * 0.85, h * 0.5),
            ]
        return [
            to_global(w * 0.5, h * 0.15),
            to_global(w * 0.5, h * 0.85),
        ]
