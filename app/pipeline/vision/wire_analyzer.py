"""
Wire 骨架分析 (← src_v2/vision/wire_analyzer.py)

HSV 颜色分割 + Zhang-Suen 骨架化 + 端点检测 + 颜色分类
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 颜色分类 HSV 范围
_COLOR_RANGES = {
    "red": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
    "blue": [((100, 70, 50), (130, 255, 255))],
    "green": [((35, 50, 50), (85, 255, 255))],
    "yellow": [((20, 70, 50), (35, 255, 255))],
    "orange": [((10, 70, 50), (20, 255, 255))],
    "black": [((0, 0, 0), (180, 255, 50))],
    "white": [((0, 0, 200), (180, 30, 255))],
}


class WireAnalyzer:
    """Wire 骨架端点精炼 + 颜色分类"""

    def analyze_wire(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[Optional[Tuple[Tuple[float, float], Tuple[float, float]]], str]:
        """分析单根 wire 的端点和颜色

        Returns:
            (endpoints, color) 其中 endpoints = ((x1,y1), (x2,y2)) or None
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # 扩展 bbox 边界, 避免导线端点贴近边缘被截断
        pad = max(10, int(0.1 * max(x2 - x1, y2 - y1)))
        x1_pad, y1_pad = max(0, x1 - pad), max(0, y1 - pad)
        x2_pad, y2_pad = min(w, x2 + pad), min(h, y2 + pad)

        crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop.size == 0:
            return None, ""

        # 颜色分类
        color = self._classify_color(crop)

        # 骨架端点
        endpoints = self._skeleton_endpoints(crop)
        if endpoints is not None:
            # 转回全图坐标 (注意使用 pad 后的偏移)
            (ex1, ey1), (ex2, ey2) = endpoints
            endpoints = ((ex1 + x1_pad, ey1 + y1_pad), (ex2 + x1_pad, ey2 + y1_pad))

        return endpoints, color

    def _skeleton_endpoints(
        self, crop: np.ndarray,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Zhang-Suen 骨架化 + 端点检测"""
        try:
            from skimage.morphology import skeletonize
        except ImportError:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 过滤背景 (白色/浅灰色面包板)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(white_mask))

        # 形态学开运算去除小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        if cv2.countNonZero(binary) < 10:
            return None

        skeleton = skeletonize(binary > 0).astype(np.uint8)

        # 端点: 8 邻域仅 1 个连接
        endpoints = []
        h, w = skeleton.shape
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skeleton[y, x] == 0:
                    continue
                neighbors = (
                    skeleton[y - 1, x - 1] + skeleton[y - 1, x] + skeleton[y - 1, x + 1]
                    + skeleton[y, x - 1] + skeleton[y, x + 1]
                    + skeleton[y + 1, x - 1] + skeleton[y + 1, x] + skeleton[y + 1, x + 1]
                )
                if neighbors == 1:
                    endpoints.append((float(x), float(y)))

        if len(endpoints) >= 2:
            # 取距离最远的两个端点
            max_dist = 0
            best_pair = (endpoints[0], endpoints[1])
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    dx = endpoints[i][0] - endpoints[j][0]
                    dy = endpoints[i][1] - endpoints[j][1]
                    d = dx * dx + dy * dy
                    if d > max_dist:
                        max_dist = d
                        best_pair = (endpoints[i], endpoints[j])
            return best_pair

        return None

    def _classify_color(self, crop: np.ndarray) -> str:
        """HSV 颜色分类"""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        best_color = ""
        best_count = 0

        for color_name, ranges in _COLOR_RANGES.items():
            total = 0
            for lo, hi in ranges:
                mask = cv2.inRange(hsv, lo, hi)
                total += cv2.countNonZero(mask)
            if total > best_count:
                best_count = total
                best_color = color_name

        return best_color
