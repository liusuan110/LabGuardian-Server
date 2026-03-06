"""
面包板校准器 (← src_v2/vision/calibrator.py)

提供像素坐标 ↔ 逻辑坐标 (Row×Col) 映射
服务端简化版: 保留核心校准 + 坐标映射, 去掉 GUI 交互
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BreadboardCalibrator:
    """面包板校准器 — 孔洞检测 + 坐标映射"""

    def __init__(self, rows: int = 63, cols_per_side: int = 5):
        self.rows = rows
        self.cols_per_side = cols_per_side
        self.total_cols = cols_per_side * 2  # a-e + f-j = 10

        # 校准状态
        self.is_calibrated = False
        self.hole_centers: List[Tuple[float, float]] = []
        self._perspective_matrix: Optional[np.ndarray] = None
        self._inv_perspective: Optional[np.ndarray] = None
        self._grid: Optional[np.ndarray] = None  # (rows, cols, 2) 孔洞像素坐标
        self._row_coords: Optional[np.ndarray] = None  # 行 y 坐标
        self._col_coords: Optional[np.ndarray] = None  # 列 x 坐标

        # 列名映射
        self._col_names = list("abcde") + list("fghij")

    def calibrate(self, corners: np.ndarray):
        """用四角坐标进行透视变换校准"""
        if corners.shape != (4, 2):
            raise ValueError("Need exactly 4 corner points")

        dst_w, dst_h = 800, 600
        dst_corners = np.array([
            [0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h],
        ], dtype=np.float32)

        self._perspective_matrix = cv2.getPerspectiveTransform(
            corners.astype(np.float32), dst_corners
        )
        self._inv_perspective = cv2.getPerspectiveTransform(
            dst_corners, corners.astype(np.float32)
        )
        self.is_calibrated = True

    def auto_calibrate(self, image: np.ndarray) -> bool:
        """自动校准: 检测面包板区域 + 孔洞"""
        try:
            corners = self._detect_board_region(image)
            if corners is None:
                return False
            self.calibrate(corners)
            warped = self.warp(image)
            self.detect_holes(warped)
            return len(self.hole_centers) >= 50
        except Exception as e:
            logger.warning(f"[Calibrator] Auto-calibrate failed: {e}")
            return False

    def _detect_board_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """检测面包板白色区域轮廓, 返回四角坐标"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 白色区域掩码
        mask = cv2.inRange(hsv, (0, 0, 160), (180, 50, 255))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mask, iterations=3)
        kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, kernel_e, iterations=3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < image.shape[0] * image.shape[1] * 0.05:
            return None

        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        if len(approx) != 4:
            return None

        pts = approx.reshape(4, 2).astype(np.float32)
        # 排序: 左上 → 右上 → 右下 → 左下
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        ordered = np.array([
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)],
        ], dtype=np.float32)

        return ordered

    def warp(self, image: np.ndarray) -> np.ndarray:
        """透视变换"""
        if self._perspective_matrix is None:
            return image
        return cv2.warpPerspective(image, self._perspective_matrix, (800, 600))

    def detect_holes(self, warped_image: np.ndarray):
        """在校正后的图像中检测孔洞"""
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        params = cv2.SimpleBlobDetector_Params()
        params.minArea = 15
        params.maxArea = 400
        params.filterByCircularity = True
        params.minCircularity = 0.2
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)
        inv = cv2.bitwise_not(gray)
        keypoints = detector.detect(inv)

        self.hole_centers = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        logger.info(f"[Calibrator] Detected {len(self.hole_centers)} holes")

        if len(self.hole_centers) >= 50:
            self._build_grid()

    def _build_grid(self):
        """从散点孔洞构建行列网格"""
        if not self.hole_centers:
            return

        pts = np.array(self.hole_centers)
        ys = np.sort(np.unique(np.round(pts[:, 1], 0)))
        xs = np.sort(np.unique(np.round(pts[:, 0], 0)))

        # 聚类行列
        self._row_coords = self._cluster_1d(pts[:, 1], self.rows)
        self._col_coords = self._cluster_1d(pts[:, 0], self.total_cols)

    @staticmethod
    def _cluster_1d(values: np.ndarray, expected_count: int) -> np.ndarray:
        """一维聚类: 将检测到的坐标聚类到预期数量的组"""
        sorted_vals = np.sort(values)
        if len(sorted_vals) < expected_count:
            return sorted_vals

        # 简单均匀采样
        indices = np.linspace(0, len(sorted_vals) - 1, expected_count, dtype=int)
        return sorted_vals[indices]

    def frame_pixel_to_logic(
        self, px: float, py: float,
    ) -> Optional[Tuple[str, str]]:
        """像素坐标 → 逻辑坐标 (行号, 列名)"""
        if self._row_coords is None or self._col_coords is None:
            return None

        # 如有透视矩阵, 先变换到校正坐标
        if self._perspective_matrix is not None:
            pt = np.array([[[px, py]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self._perspective_matrix)
            px, py = transformed[0, 0]

        row_idx = int(np.argmin(np.abs(self._row_coords - py)))
        col_idx = int(np.argmin(np.abs(self._col_coords - px)))

        row_name = str(row_idx + 1)
        col_name = self._col_names[col_idx] if col_idx < len(self._col_names) else str(col_idx)

        return (row_name, col_name)

    def frame_pixel_to_logic_candidates(
        self, px: float, py: float, k: int = 5,
    ) -> List[Tuple[str, str]]:
        """返回最近的 k 个逻辑坐标候选"""
        if self._row_coords is None or self._col_coords is None:
            return []

        if self._perspective_matrix is not None:
            pt = np.array([[[px, py]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self._perspective_matrix)
            px, py = transformed[0, 0]

        row_dists = np.abs(self._row_coords - py)
        col_dists = np.abs(self._col_coords - px)

        top_rows = np.argsort(row_dists)[:k]
        top_cols = np.argsort(col_dists)[:k]

        candidates = []
        for ri in top_rows:
            for ci in top_cols:
                row_name = str(ri + 1)
                col_name = self._col_names[ci] if ci < len(self._col_names) else str(ci)
                candidates.append((row_name, col_name))
                if len(candidates) >= k:
                    break
            if len(candidates) >= k:
                break

        return candidates

    def get_roi_rect(
        self, image_shape: tuple, padding: int = 30,
    ) -> Tuple[int, int, int, int]:
        """获取面包板 ROI 区域 (x1, y1, x2, y2)"""
        h, w = image_shape[:2]
        if not self.hole_centers:
            return (0, 0, w, h)

        pts = np.array(self.hole_centers)

        if self._inv_perspective is not None:
            pts_3d = np.hstack([pts, np.ones((len(pts), 1))]).astype(np.float32)
            src_pts = cv2.perspectiveTransform(
                pts.reshape(-1, 1, 2).astype(np.float32),
                self._inv_perspective,
            ).reshape(-1, 2)
        else:
            src_pts = pts

        x1 = max(0, int(src_pts[:, 0].min()) - padding)
        y1 = max(0, int(src_pts[:, 1].min()) - padding)
        x2 = min(w, int(src_pts[:, 0].max()) + padding)
        y2 = min(h, int(src_pts[:, 1].max()) + padding)

        return (x1, y1, x2, y2)
