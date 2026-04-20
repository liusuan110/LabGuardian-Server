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


# 标准面包板布局常量 (相对比例)
# 行方向: 上方 power rail (2行) + 主区 rows + 下方 power rail (2行)
_POWER_RAIL_TOP_NAMES = ("+_top", "-_top")
_POWER_RAIL_BOT_NAMES = ("+_bot", "-_bot")


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

        # 合成网格模式 (当视觉校准失败时使用)
        self._synthetic_grid = False
        self._img_h: int = 0
        self._img_w: int = 0

        # 朝向标记 & 电轨
        self._landscape: bool = False  # True = 行沿X轴, 列沿Y轴
        self._top_rails: List[float] = []
        self._bot_rails: List[float] = []

        # 列名映射
        self._col_names = list("abcde") + list("fghij")

        # 空间哈希参数 (参考 Spatial Hashing 算法, O(1) 坐标映射)
        self._grid_origin: Optional[Tuple[float, float]] = None   # (row_0, col_0)
        self._grid_spacing: Optional[Tuple[float, float]] = None  # (d_row, d_col)
        self._rail_tolerance: float = 15.0  # 自适应电轨容差
        self._grid_matrix: Optional[np.ndarray] = None  # grid[row][col] 二维矩阵

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
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
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
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            # RANSAC 兜底: 多边形近似非 4 点时, 用最小面积矩形拟合
            rect = cv2.minAreaRect(largest)
            pts = cv2.boxPoints(rect).astype(np.float32)
            logger.info("[Calibrator] approxPolyDP gave %d pts, falling back to minAreaRect", len(approx))
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

        # 计算空间哈希参数 + 2D 网格矩阵
        self._compute_grid_params()

    @staticmethod
    def _cluster_1d(values: np.ndarray, expected_count: int) -> np.ndarray:
        """一维聚类: 将检测到的坐标聚类到预期数量的组"""
        sorted_vals = np.sort(values)
        if len(sorted_vals) < expected_count:
            return sorted_vals

        # 简单均匀采样
        indices = np.linspace(0, len(sorted_vals) - 1, expected_count, dtype=int)
        return sorted_vals[indices]

    # ============================================================
    # 空间哈希 & RANSAC 单应性 (Spatial Hashing & Homography)
    # 参考: OpenCV Document Scanner, ArUco Marker Homography
    # ============================================================

    def _spatial_hash(self, row_val: float, col_val: float) -> Tuple[int, int]:
        """空间哈希: 连续坐标 → 离散孔位索引 O(1)

        利用面包板标准 2.54mm 等间距先验, 通过网格原点 + 间距直接取整,
        等价于 hash(x,y) = (round((x-x0)/dx), round((y-y0)/dy)),
        无需遍历坐标数组。参考 Spatial Hashing 算法。
        """
        if self._grid_origin is not None and self._grid_spacing is not None:
            r0, c0 = self._grid_origin
            dr, dc = self._grid_spacing
            if dr > 0 and dc > 0:
                row_idx = int(round((row_val - r0) / dr))
                col_idx = int(round((col_val - c0) / dc))
                row_idx = max(0, min(row_idx, len(self._row_coords) - 1))
                col_idx = max(0, min(col_idx, len(self._col_coords) - 1))
                return row_idx, col_idx
        # 降级: O(N) argmin 兜底
        row_idx = int(np.argmin(np.abs(self._row_coords - row_val)))
        col_idx = int(np.argmin(np.abs(self._col_coords - col_val)))
        return row_idx, col_idx

    def _compute_grid_params(self):
        """从已建立的行列坐标计算空间哈希参数 + grid[row][col] 矩阵 + 自适应电轨容差"""
        if self._row_coords is None or self._col_coords is None:
            return
        if len(self._row_coords) < 2 or len(self._col_coords) < 2:
            return

        self._grid_origin = (float(self._row_coords[0]), float(self._col_coords[0]))

        row_diffs = np.diff(self._row_coords)
        col_diffs = np.diff(self._col_coords)
        self._grid_spacing = (float(np.median(row_diffs)), float(np.median(col_diffs)))

        # 自适应电轨容差 = 列间距的 60% (替代硬编码 15px)
        self._rail_tolerance = float(np.median(col_diffs) * 0.6)

        # grid[row][col] = (row_coord, col_coord) 二维索引直查表
        nr, nc = len(self._row_coords), len(self._col_coords)
        self._grid_matrix = np.zeros((nr, nc, 2), dtype=np.float32)
        for r in range(nr):
            for c in range(nc):
                self._grid_matrix[r, c] = [self._row_coords[r], self._col_coords[c]]

        logger.info(
            "[Calibrator] Spatial hash: origin=(%.1f,%.1f), spacing=(%.1f,%.1f), "
            "rail_tol=%.1f, grid=%dx%d",
            *self._grid_origin, *self._grid_spacing,
            self._rail_tolerance, nr, nc,
        )

    def _calibrate_from_holes_ransac(self, image: np.ndarray) -> bool:
        """RANSAC 多点单应性估计 — 从孔洞对应关系直接计算 H 矩阵

        相比 4 点 getPerspectiveTransform, cv2.findHomography + RANSAC
        可容忍误检/遮挡/圆角, 是 Document Scanner 的标准做法。
        """
        holes = self._detect_holes_raw(image)
        if len(holes) < 30:
            return False

        src_pts = np.array(holes, dtype=np.float32)
        x_clusters = self._quick_cluster_1d(src_pts[:, 0])
        y_clusters = self._quick_cluster_1d(src_pts[:, 1])

        if len(x_clusters) < 8 or len(y_clusters) < 8:
            return False

        # 列方向有更少聚类 (≈10 列 vs 30+ 行)
        if len(x_clusters) < len(y_clusters):
            col_centers, row_centers = x_clusters, y_clusters
            col_axis = 0
        else:
            col_centers, row_centers = y_clusters, x_clusters
            col_axis = 1

        n_rows, n_cols = len(row_centers), len(col_centers)
        dst_w, dst_h = 800, 600
        ideal_col_sp = dst_w / (n_cols + 1)
        ideal_row_sp = dst_h / (n_rows + 1)

        matched_src, matched_dst = [], []
        for (cx, cy) in holes:
            col_val = cx if col_axis == 0 else cy
            row_val = cy if col_axis == 0 else cx
            ci = int(np.argmin([abs(c - col_val) for c in col_centers]))
            ri = int(np.argmin([abs(r - row_val) for r in row_centers]))
            matched_src.append([cx, cy])
            matched_dst.append([(ci + 1) * ideal_col_sp, (ri + 1) * ideal_row_sp])

        src = np.array(matched_src, dtype=np.float32)
        dst = np.array(matched_dst, dtype=np.float32)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return False

        inlier_ratio = float(np.sum(mask)) / len(mask)
        logger.info("[Calibrator] RANSAC inlier: %.1f%% (%d/%d)",
                    inlier_ratio * 100, int(np.sum(mask)), len(mask))
        if inlier_ratio < 0.3:
            return False

        self._perspective_matrix = H.astype(np.float32)
        try:
            self._inv_perspective = np.linalg.inv(H).astype(np.float32)
        except np.linalg.LinAlgError:
            self._inv_perspective = None
        self.is_calibrated = True

        warped = cv2.warpPerspective(image, H, (dst_w, dst_h))
        self.detect_holes(warped)
        return len(self.hole_centers) >= 50

    @staticmethod
    def _quick_cluster_1d(values: np.ndarray, min_gap_ratio: float = 1.8) -> List[float]:
        """快速一维聚类 (Spatial Hashing 预处理): 按间距自动分组, 返回组中心"""
        sorted_v = np.sort(values)
        if len(sorted_v) < 2:
            return [float(sorted_v[0])] if len(sorted_v) == 1 else []
        diffs = np.diff(sorted_v)
        positive_diffs = diffs[diffs > 0]
        if len(positive_diffs) == 0:
            return [float(np.median(sorted_v))]
        threshold = float(np.median(positive_diffs)) * min_gap_ratio
        clusters: List[float] = []
        current: List[float] = [float(sorted_v[0])]
        for i in range(1, len(sorted_v)):
            if sorted_v[i] - sorted_v[i - 1] > threshold:
                clusters.append(float(np.median(current)))
                current = []
            current.append(float(sorted_v[i]))
        if current:
            clusters.append(float(np.median(current)))
        return clusters

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

        # 根据朝向确定行/列映射轴
        if self._landscape:
            row_val, col_val = px, py  # 行沿X, 列沿Y
        else:
            row_val, col_val = py, px  # 行沿Y, 列沿X (默认)

        # 检查是否落入电轨区域 (包括超出面包板范围的引脚)
        rail_tolerance = self._rail_tolerance
        grid_min = float(self._col_coords[0]) if len(self._col_coords) > 0 else 0
        grid_max = float(self._col_coords[-1]) if len(self._col_coords) > 0 else 0

        # 超出主 grid 范围的引脚 → 比较到电轨和到主 grid 的距离
        if self._top_rails and col_val < grid_min:
            dist_to_grid = grid_min - col_val
            closest_rail_dist = min(abs(col_val - r) for r in self._top_rails)
            # 如果到 grid 的距离小于主 grid 间距 → 映射到 grid (可能是 bbox 估计偏移)
            grid_spacing = float(self._col_coords[1] - self._col_coords[0]) if len(self._col_coords) > 1 else 20
            if dist_to_grid < grid_spacing:
                # 离 grid 够近, 映射到 grid 而非 rail
                pass
            else:
                closest_rail_idx = int(np.argmin([abs(col_val - r) for r in self._top_rails]))
                row_idx = int(np.argmin(np.abs(self._row_coords - row_val)))
                rail_name = "+" if closest_rail_idx == 0 else "-"
                return (str(row_idx + 1), f"rail_top{rail_name}")

        if self._bot_rails and col_val > grid_max:
            dist_to_grid = col_val - grid_max
            grid_spacing = float(self._col_coords[-1] - self._col_coords[-2]) if len(self._col_coords) > 1 else 20
            if dist_to_grid < grid_spacing:
                pass  # 离 grid 够近
            else:
                closest_rail_idx = int(np.argmin([abs(col_val - r) for r in self._bot_rails]))
                row_idx = int(np.argmin(np.abs(self._row_coords - row_val)))
                rail_name = "+" if closest_rail_idx == 0 else "-"
                return (str(row_idx + 1), f"rail_bot{rail_name}")

        for i, rail_pos in enumerate(self._top_rails):
            if abs(col_val - rail_pos) < rail_tolerance:
                row_idx = int(np.argmin(np.abs(self._row_coords - row_val)))
                rail_name = "+" if i == 0 else "-"
                return (str(row_idx + 1), f"rail_top{rail_name}")

        for i, rail_pos in enumerate(self._bot_rails):
            if abs(col_val - rail_pos) < rail_tolerance:
                row_idx = int(np.argmin(np.abs(self._row_coords - row_val)))
                rail_name = "+" if i == 0 else "-"
                return (str(row_idx + 1), f"rail_bot{rail_name}")

        # 主 grid 区域 — 空间哈希 O(1) 查找
        row_idx, col_idx = self._spatial_hash(row_val, col_val)

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

        if self._landscape:
            row_val, col_val = px, py
        else:
            row_val, col_val = py, px

        # 先检查电轨 (包括超出 grid 范围的引脚)
        rail_tolerance = self._rail_tolerance
        grid_min = float(self._col_coords[0]) if len(self._col_coords) > 0 else 0
        grid_max = float(self._col_coords[-1]) if len(self._col_coords) > 0 else 0
        grid_spacing = float(self._col_coords[1] - self._col_coords[0]) if len(self._col_coords) > 1 else 20

        # 超出范围但离 grid 近的 → 映射到 grid, 离 grid 远的 → 映射到 rail
        if self._top_rails and col_val < grid_min:
            dist_to_grid = grid_min - col_val
            if dist_to_grid >= grid_spacing:
                closest_idx = int(np.argmin([abs(col_val - r) for r in self._top_rails]))
                row_dists = np.abs(self._row_coords - row_val)
                top_rows = np.argsort(row_dists)[:k]
                rail_name = "+" if closest_idx == 0 else "-"
                return [(str(ri + 1), f"rail_top{rail_name}") for ri in top_rows]

        if self._bot_rails and col_val > grid_max:
            dist_to_grid = col_val - grid_max
            if dist_to_grid >= grid_spacing:
                closest_idx = int(np.argmin([abs(col_val - r) for r in self._bot_rails]))
                row_dists = np.abs(self._row_coords - row_val)
                top_rows = np.argsort(row_dists)[:k]
                rail_name = "+" if closest_idx == 0 else "-"
                return [(str(ri + 1), f"rail_bot{rail_name}") for ri in top_rows]

        for rails, prefix in [(self._top_rails, "rail_top"), (self._bot_rails, "rail_bot")]:
            for i, rail_pos in enumerate(rails):
                if abs(col_val - rail_pos) < rail_tolerance:
                    row_dists = np.abs(self._row_coords - row_val)
                    top_rows = np.argsort(row_dists)[:k]
                    rail_name = "+" if i == 0 else "-"
                    return [(str(ri + 1), f"{prefix}{rail_name}") for ri in top_rows]

        # 空间哈希定位中心 + 邻域展开 (O(k²) 代替 O(N·k²))
        center_r, center_c = self._spatial_hash(row_val, col_val)
        scored = []
        radius = min(k, 3)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                ri = center_r + dr
                ci = center_c + dc
                if ri < 0 or ri >= len(self._row_coords):
                    continue
                if ci < 0 or ci >= len(self._col_coords):
                    continue
                dist = float((self._row_coords[ri] - row_val) ** 2 +
                             (self._col_coords[ci] - col_val) ** 2)
                row_name = str(ri + 1)
                col_name = self._col_names[ci] if ci < len(self._col_names) else str(ci)
                scored.append((dist, row_name, col_name))
        scored.sort(key=lambda x: x[0])

        candidates = [(r, c) for _, r, c in scored[:k]]
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

    # ---- 公共 API (Pipeline 调用入口) ----

    @property
    def is_grid_ready(self) -> bool:
        """校准器的行/列坐标是否已建立 (可用于坐标映射)"""
        return self._row_coords is not None and self._col_coords is not None

    @property
    def row_coords(self) -> Optional[np.ndarray]:
        """行坐标数组 (只读)"""
        return self._row_coords

    @property
    def col_coords(self) -> Optional[np.ndarray]:
        """列坐标数组 (只读)"""
        return self._col_coords

    @property
    def landscape(self) -> bool:
        """是否横向布局"""
        return self._landscape

    def pixel_to_logic(
        self, px: float, py: float,
    ) -> Optional[Tuple[str, str]]:
        """像素坐标 → 逻辑坐标 (行号, 列名) — frame_pixel_to_logic 的公共别名"""
        return self.frame_pixel_to_logic(px, py)

    def build_synthetic_grid(self, image_shape: Tuple[int, int]):
        """公共接口: 根据图像尺寸生成合成面包板网格"""
        self._build_synthetic_grid(image_shape)

    def ensure_calibrated(self, image: np.ndarray) -> bool:
        """确保校准器已校准: 先尝试孔洞智能校准, 再尝试视觉校准, 最后合成网格"""
        if self.is_grid_ready:
            return True

        # 1) 基于孔洞检测的智能校准 (自动识别朝向)
        if self._smart_calibrate(image):
            logger.info("[Calibrator] Smart hole-based calibration succeeded (landscape=%s)", self._landscape)
            return True

        # 2) 尝试视觉自动校准
        if self.auto_calibrate(image):
            logger.info("[Calibrator] Visual auto-calibrate succeeded")
            return True

        # 2.5) RANSAC 多点单应性估计 (参考 Document Scanner)
        if self._calibrate_from_holes_ransac(image):
            logger.info("[Calibrator] RANSAC homography calibration succeeded")
            return True

        # 3) 视觉校准失败 → 使用合成网格
        logger.info("[Calibrator] All calibration failed, using synthetic grid")
        self._build_synthetic_grid(image.shape[:2])
        return self.is_grid_ready

    def _build_synthetic_grid(self, image_shape: Tuple[int, int]):
        """根据图像尺寸生成合成面包板网格

        假设面包板占据图像的大部分区域 (标准俯拍):
        - 行方向: 图像从上到下均匀分布 rows 行
        - 列方向: 左侧 a-e, 中间 gap, 右侧 f-j
        """
        h, w = image_shape
        self._img_h = h
        self._img_w = w
        self._synthetic_grid = True
        # 清除可能的部分校准状态
        self._perspective_matrix = None
        self._inv_perspective = None

        # 行: 均匀分布在图像 5%~95% 高度范围
        margin_y = h * 0.05
        self._row_coords = np.linspace(margin_y, h - margin_y, self.rows)

        # 列: a-e 在左半, f-j 在右半, 中间有 gap
        margin_x = w * 0.08
        gap = w * 0.06  # 中间沟槽宽度
        left_start = margin_x
        left_end = w / 2 - gap / 2
        right_start = w / 2 + gap / 2
        right_end = w - margin_x

        left_cols = np.linspace(left_start, left_end, self.cols_per_side)
        right_cols = np.linspace(right_start, right_end, self.cols_per_side)
        self._col_coords = np.concatenate([left_cols, right_cols])

        # 计算空间哈希参数
        self._compute_grid_params()

        logger.info(
            "[Calibrator] Synthetic grid: %d rows × %d cols on %dx%d image",
            self.rows, self.total_cols, w, h,
        )

    def get_nearest_hole_px(
        self, px: float, py: float,
    ) -> Optional[Tuple[float, float]]:
        """返回最近孔洞的像素坐标 (用于可视化)"""
        logic = self.pixel_to_logic(px, py)
        if logic is None:
            return None
        row_name, col_name = logic
        try:
            row_idx = int(row_name) - 1
        except (ValueError, TypeError):
            return None

        if self._row_coords is None or self._col_coords is None:
            return None
        if row_idx >= len(self._row_coords):
            return None

        # 电轨: 直接返回行坐标 + 电轨坐标
        if col_name.startswith("rail_"):
            rail_coord = None
            for rails in [self._top_rails, self._bot_rails]:
                for rp in rails:
                    if rail_coord is None:
                        rail_coord = rp
            if rail_coord is None:
                return None
            row_coord = float(self._row_coords[row_idx])
            if self._landscape:
                return (row_coord, rail_coord)  # X=行, Y=列
            else:
                return (rail_coord, row_coord)  # X=列, Y=行

        col_idx = self._col_names.index(col_name) if col_name in self._col_names else -1
        if col_idx < 0 or col_idx >= len(self._col_coords):
            return None

        row_coord = float(self._row_coords[row_idx])
        col_coord = float(self._col_coords[col_idx])

        if self._landscape:
            return (row_coord, col_coord)  # X=行坐标, Y=列坐标
        else:
            return (col_coord, row_coord)  # X=列坐标, Y=行坐标

    # ---- 智能校准 (孔洞检测 + 自动朝向识别) ----

    def _detect_holes_raw(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """形态学检测面包板孔洞, 返回 (cx, cy) 列表"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        holes: List[Tuple[float, float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 4 < area < 100:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    perimeter = cv2.arcLength(cnt, True)
                    circ = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
                    if circ > 0.25:
                        holes.append((cx, cy))
        return holes

    @staticmethod
    def _find_peaks_1d(values: List[float], span: int, sigma: float = 3.0, min_density: float = 1.5) -> List[int]:
        """1D 高斯平滑后找峰值"""
        from scipy.ndimage import gaussian_filter1d

        if span < 10 or not values:
            return []
        profile = np.zeros(span)
        for v in values:
            iv = int(v)
            if 0 <= iv < span:
                profile[iv] += 1
        smoothed = gaussian_filter1d(profile, sigma=sigma)
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1] and smoothed[i] > min_density:
                peaks.append(i)
        return peaks

    def _smart_calibrate(self, image: np.ndarray) -> bool:
        """基于孔洞检测自动确定面包板朝向和网格"""
        holes = self._detect_holes_raw(image)
        if len(holes) < 50:
            logger.info("[Calibrator] Too few holes (%d), skip smart calibration", len(holes))
            return False

        h, w = image.shape[:2]
        self._img_h = h
        self._img_w = w
        self._synthetic_grid = False
        self._perspective_matrix = None
        self._inv_perspective = None

        xs = [p[0] for p in holes]
        ys = [p[1] for p in holes]

        y_peaks = self._find_peaks_1d(ys, span=h, sigma=3.0, min_density=1.5)
        x_peaks = self._find_peaks_1d(xs, span=w, sigma=3.0, min_density=1.5)
        logger.info("[Calibrator] Y-peaks=%d, X-peaks=%d (from %d holes)", len(y_peaks), len(x_peaks), len(holes))

        # 通过中央 gap 模式判断: 面包板的列方向一定有 a-e gap f-j 结构
        # 在哪个轴找到这种结构, 哪个轴就是列方向
        y_gap_score = self._check_center_gap_pattern(y_peaks)
        x_gap_score = self._check_center_gap_pattern(x_peaks)
        logger.info("[Calibrator] Center-gap score: Y=%.2f, X=%.2f", y_gap_score, x_gap_score)

        if y_gap_score < float("inf") and (x_gap_score == float("inf") or y_gap_score < x_gap_score):
            logger.info("[Calibrator] => LANDSCAPE (cols=Y, rows=X)")
            self._landscape = True
            return self._build_from_peaks(holes, col_peaks=y_peaks, row_axis="x", img_shape=(h, w))
        elif x_gap_score < float("inf"):
            logger.info("[Calibrator] => PORTRAIT (cols=X, rows=Y)")
            self._landscape = False
            return self._build_from_peaks(holes, col_peaks=x_peaks, row_axis="y", img_shape=(h, w))
        else:
            # 都找不到 center gap → 用峰值数量作后备判断
            if len(y_peaks) > 0 and len(y_peaks) < len(x_peaks):
                logger.info("[Calibrator] Fallback => LANDSCAPE (y fewer peaks)")
                self._landscape = True
                return self._build_from_peaks(holes, col_peaks=y_peaks, row_axis="x", img_shape=(h, w))
            elif len(x_peaks) > 0:
                logger.info("[Calibrator] Fallback => PORTRAIT (x fewer peaks)")
                self._landscape = False
                return self._build_from_peaks(holes, col_peaks=x_peaks, row_axis="y", img_shape=(h, w))
            logger.warning("[Calibrator] Cannot determine orientation")
            return False

    @staticmethod
    def _check_center_gap_pattern(peaks: List[int]) -> float:
        """检查给定峰值列表中是否有面包板的中央 gap 结构

        返回最佳 score (越小越好), 如果找不到返回 inf
        """
        if len(peaks) < 10:
            return float("inf")
        best_score = float("inf")
        for start in range(len(peaks) - 9):
            subset = peaks[start : start + 10]
            gaps = [subset[i + 1] - subset[i] for i in range(9)]
            max_gap_idx = int(np.argmax(gaps))
            if max_gap_idx != 4:
                continue  # 中央 gap 应在 e→f 之间
            # 中央 gap 应该明显大于两侧间距
            center_gap = gaps[4]
            left_gaps = gaps[:4]
            right_gaps = gaps[5:]
            avg_side = (np.mean(left_gaps) + np.mean(right_gaps)) / 2
            if center_gap < avg_side * 1.5:
                continue  # 中央 gap 不够明显
            score = float(np.std(left_gaps)) + float(np.std(right_gaps))
            if score < best_score:
                best_score = score
        return best_score

    def _build_from_peaks(
        self,
        holes: List[Tuple[float, float]],
        col_peaks: List[int],
        row_axis: str,
        img_shape: Tuple[int, int],
    ) -> bool:
        """从列峰值 + 行轴方向建立校准网格"""
        col_peaks = sorted(col_peaks)
        if len(col_peaks) < 10:
            logger.warning("[Calibrator] Not enough column peaks (%d)", len(col_peaks))
            return False

        # --- 从 col_peaks 中识别主 grid (10列) 和电轨 ---
        # 主 grid: 连续 10 个峰值, 第 5→6 之间有中央 gap
        best_start = 0
        best_score = float("inf")
        found = False
        for start in range(len(col_peaks) - 9):
            subset = col_peaks[start : start + 10]
            gaps = [subset[i + 1] - subset[i] for i in range(9)]
            max_gap_idx = int(np.argmax(gaps))
            if max_gap_idx != 4:
                continue  # 中央 gap 应在 e→f 之间 (index 4)
            left_gaps = gaps[:4]
            right_gaps = gaps[5:]
            score = float(np.std(left_gaps)) + float(np.std(right_gaps))
            if score < best_score:
                best_score = score
                best_start = start
                found = True

        if not found:
            # 退而求其次: 取间距最均匀的 10 个
            best_start = 0
            logger.warning("[Calibrator] No clean center-gap found, using first 10 peaks")

        main_cols = col_peaks[best_start : best_start + 10]
        rail_peaks = [p for p in col_peaks if p not in main_cols]
        top_rails = sorted([p for p in rail_peaks if p < main_cols[0]])
        bot_rails = sorted([p for p in rail_peaks if p > main_cols[-1]])

        self._top_rails = [float(r) for r in top_rails]
        self._bot_rails = [float(r) for r in bot_rails]
        self._col_coords = np.array(main_cols, dtype=float)

        logger.info("[Calibrator] Main cols=%s, top_rails=%s, bot_rails=%s", main_cols, top_rails, bot_rails)

        # --- 确定行坐标 ---
        if row_axis == "x":
            col_val_func = lambda p: p[1]
            row_val_func = lambda p: p[0]
        else:
            col_val_func = lambda p: p[0]
            row_val_func = lambda p: p[1]

        # 收集主 grid 孔洞的行坐标
        row_values: List[float] = []
        for col_center in main_cols:
            for p in holes:
                if abs(col_val_func(p) - col_center) < 12:
                    row_values.append(row_val_func(p))

        if not row_values:
            logger.warning("[Calibrator] No row values from main grid holes")
            return False

        row_values.sort()
        # 估算行间距
        diffs = np.diff(row_values)
        valid_diffs = diffs[(diffs > 5) & (diffs < 35)]
        if len(valid_diffs) == 0:
            logger.warning("[Calibrator] Cannot determine row pitch")
            return False
        row_pitch = float(np.median(valid_diffs))
        r_min, r_max = min(row_values), max(row_values)
        num_rows = max(1, round((r_max - r_min) / row_pitch) + 1)
        self._row_coords = np.linspace(r_min, r_max, num_rows)
        self.rows = num_rows

        # 计算空间哈希参数
        self._compute_grid_params()

        logger.info("[Calibrator] %d rows, pitch=%.1f, range=[%.0f, %.0f]", num_rows, row_pitch, r_min, r_max)
        return True
