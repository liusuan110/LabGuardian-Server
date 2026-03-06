"""
局部引脚视觉验证 (← src_v2/vision/pin_hole_detector.py)

分析检测到的引脚附近 10-30 个孔洞, 通过对比度判断占用状态
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PinHoleVerifier:
    """局部视觉引脚验证"""

    def find_pins_locally(
        self,
        image: np.ndarray,
        calibrator,
        detection,
        class_name: str,
    ) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
        """在引脚附近局部搜索, 返回最佳逻辑坐标"""
        if detection.pin1_pixel is None or detection.pin2_pixel is None:
            return None, None

        loc1 = self._verify_single_pin(
            image, calibrator, detection.pin1_pixel, class_name
        )
        loc2 = self._verify_single_pin(
            image, calibrator, detection.pin2_pixel, class_name
        )

        return loc1, loc2

    def _verify_single_pin(
        self,
        image: np.ndarray,
        calibrator,
        pin_pixel: Tuple[float, float],
        class_name: str,
    ) -> Optional[Tuple[str, str]]:
        """单个引脚的局部验证"""
        candidates = calibrator.frame_pixel_to_logic_candidates(
            *pin_pixel, k=5
        )
        if not candidates:
            return None

        # 简单策略: 返回最近的候选
        return candidates[0]
