"""
局部引脚视觉验证 + 遮挡补偿

当元件引脚完全插入面包板孔洞时, YOLO bbox 的端点并不精确指向孔位,
本模块通过以下策略进行补偿:

1. 局部对比度分析: 在引脚附近裁剪小区域, 通过灰度方差检测被占用的孔洞
2. 形态学孔洞搜索: 在 bbox 两端扩展区域搜索暗色圆形 (孔洞特征)
3. 几何约束选择: 结合面包板网格先验, 从视觉候选中选出最合理的引脚孔位
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_OCCUPIED_HOLE_MIN_AREA = 6
_OCCUPIED_HOLE_MAX_AREA = 150
_OCCUPIED_HOLE_MIN_CIRCULARITY = 0.3
_SEARCH_PAD = 35


class PinHoleVerifier:
    """局部视觉引脚验证 + 遮挡补偿"""

    def find_pins_locally(
        self,
        image: np.ndarray,
        calibrator,
        detection,
        class_name: str,
    ) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
        """在引脚附近局部搜索, 返回最佳逻辑坐标

        优先使用视觉验证 (被占用孔洞检测), 如果视觉检测失败则
        降级到校准器的 Top-K 候选 + 电气约束选择。
        """
        if detection.pin1_pixel is None or detection.pin2_pixel is None:
            return None, None

        loc1 = self._verify_single_pin(
            image, calibrator, detection.pin1_pixel, detection.bbox, class_name, pin_idx=1,
        )
        loc2 = self._verify_single_pin(
            image, calibrator, detection.pin2_pixel, detection.bbox, class_name, pin_idx=2,
        )
        return loc1, loc2

    def _verify_single_pin(
        self,
        image: np.ndarray,
        calibrator,
        pin_pixel: Tuple[float, float],
        bbox: Tuple[int, int, int, int],
        class_name: str,
        pin_idx: int,
    ) -> Optional[Tuple[str, str]]:
        """单个引脚的视觉验证 + 遮挡补偿"""
        px, py = pin_pixel
        h, w = image.shape[:2]

        x1c = max(0, int(px - _SEARCH_PAD))
        y1c = max(0, int(py - _SEARCH_PAD))
        x2c = min(w, int(px + _SEARCH_PAD))
        y2c = min(h, int(py + _SEARCH_PAD))
        crop = image[y1c:y2c, x1c:x2c]

        if crop.size == 0:
            return self._fallback_candidate(calibrator, pin_pixel)

        occupied_holes = self._detect_occupied_holes(crop)

        if occupied_holes:
            global_holes = [(hx + x1c, hy + y1c) for (hx, hy) in occupied_holes]
            best_logic = self._select_best_hole(global_holes, pin_pixel, calibrator)
            if best_logic is not None:
                logger.debug(
                    "[PinHole] pin%d visual match: (%s, %s) from %d holes near (%.0f,%.0f)",
                    pin_idx, best_logic[0], best_logic[1], len(occupied_holes), px, py,
                )
                return best_logic

        return self._fallback_candidate(calibrator, pin_pixel)

    def _detect_occupied_holes(self, crop: np.ndarray) -> List[Tuple[float, float]]:
        """在局部裁剪区域检测被占用的孔洞

        被占用孔洞特征: 与空孔洞相比颜色更深 (被引脚/焊锡遮挡),
        但仍保持近圆形轮廓。同时也检测空孔洞 (暗色圆形)。
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        holes: List[Tuple[float, float]] = []

        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 8,
        )
        holes.extend(self._extract_hole_centers(adaptive))

        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        holes.extend(self._extract_hole_centers(otsu))

        return self._merge_nearby(holes, merge_dist=5.0)

    @staticmethod
    def _extract_hole_centers(binary: np.ndarray) -> List[Tuple[float, float]]:
        """从二值图中提取近圆形区域的中心"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers: List[Tuple[float, float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < _OCCUPIED_HOLE_MIN_AREA or area > _OCCUPIED_HOLE_MAX_AREA:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-6:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < _OCCUPIED_HOLE_MIN_CIRCULARITY:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centers.append((cx, cy))
        return centers

    @staticmethod
    def _merge_nearby(
        points: List[Tuple[float, float]], merge_dist: float = 5.0,
    ) -> List[Tuple[float, float]]:
        """合并距离小于 merge_dist 的重复点"""
        if not points:
            return []
        merged: List[Tuple[float, float]] = []
        used = [False] * len(points)
        for i in range(len(points)):
            if used[i]:
                continue
            cluster = [points[i]]
            used[i] = True
            for j in range(i + 1, len(points)):
                if used[j]:
                    continue
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                if dx * dx + dy * dy < merge_dist * merge_dist:
                    cluster.append(points[j])
                    used[j] = True
            avg_x = sum(p[0] for p in cluster) / len(cluster)
            avg_y = sum(p[1] for p in cluster) / len(cluster)
            merged.append((avg_x, avg_y))
        return merged

    def _select_best_hole(
        self,
        holes: List[Tuple[float, float]],
        pin_pixel: Tuple[float, float],
        calibrator,
    ) -> Optional[Tuple[str, str]]:
        """从视觉检测到的孔洞中选择最佳: 距引脚像素最近 + 在有效网格上"""
        px, py = pin_pixel
        scored: List[Tuple[float, Tuple[str, str]]] = []

        for hx, hy in holes:
            logic = calibrator.pixel_to_logic(hx, hy)
            if logic is None:
                continue
            dist = ((hx - px) ** 2 + (hy - py) ** 2) ** 0.5
            scored.append((dist, logic))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    @staticmethod
    def _fallback_candidate(
        calibrator, pin_pixel: Tuple[float, float],
    ) -> Optional[Tuple[str, str]]:
        """降级: 直接返回校准器的最近候选"""
        candidates = calibrator.frame_pixel_to_logic_candidates(
            pin_pixel[0], pin_pixel[1], k=5,
        )
        return candidates[0] if candidates else None


def compensate_occluded_pins(
    image: np.ndarray,
    detections: list,
    calibrator,
) -> list:
    """批量遮挡补偿入口 — 供 S2 mapping 调用

    对每个非 Wire 检测结果, 在引脚附近进行视觉搜索,
    用更精确的孔洞中心替代 YOLO bbox 估计的引脚像素位置。
    """
    verifier = PinHoleVerifier()

    for det in detections:
        class_name = det.get("class_name", "")
        if class_name.lower() == "wire":
            continue

        p1 = det.get("pin1_pixel")
        p2 = det.get("pin2_pixel")
        if p1 is None or p2 is None:
            continue

        p1 = tuple(p1)
        p2 = tuple(p2)

        refined_p1 = _refine_pin_pixel(image, p1, verifier)
        refined_p2 = _refine_pin_pixel(image, p2, verifier)

        if refined_p1 is not None:
            det["pin1_pixel"] = list(refined_p1)
        if refined_p2 is not None:
            det["pin2_pixel"] = list(refined_p2)

    return detections


def _refine_pin_pixel(
    image: np.ndarray,
    pin_pixel: Tuple[float, float],
    verifier: PinHoleVerifier,
) -> Optional[Tuple[float, float]]:
    """用视觉检测精炼单个引脚的像素位置"""
    px, py = pin_pixel
    h, w = image.shape[:2]

    x1c = max(0, int(px - _SEARCH_PAD))
    y1c = max(0, int(py - _SEARCH_PAD))
    x2c = min(w, int(px + _SEARCH_PAD))
    y2c = min(h, int(py + _SEARCH_PAD))
    crop = image[y1c:y2c, x1c:x2c]

    if crop.size == 0:
        return None

    occupied = verifier._detect_occupied_holes(crop)
    if not occupied:
        return None

    global_holes = [(hx + x1c, hy + y1c) for (hx, hy) in occupied]

    best_hole = None
    best_dist = float("inf")
    for hx, hy in global_holes:
        d = ((hx - px) ** 2 + (hy - py) ** 2) ** 0.5
        if d < best_dist and d < _SEARCH_PAD:
            best_dist = d
            best_hole = (hx, hy)

    return best_hole
