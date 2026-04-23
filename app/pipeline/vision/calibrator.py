"""
面包板校准器 (← src_v2/vision/calibrator.py)

提供像素坐标 ↔ 逻辑坐标 (Row×Col) 映射
服务端简化版: 保留核心校准 + 坐标映射, 去掉 GUI 交互
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

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
        self._expected_rows = rows
        self._expected_rail_rows = 50
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
        self._board_mask: Optional[np.ndarray] = None  # warped 图中的面包板区域 mask
        self._top_rail_matrix: Optional[np.ndarray] = None  # grid[row][rail_idx] 上电源轨孔洞中心
        self._bot_rail_matrix: Optional[np.ndarray] = None  # grid[row][rail_idx] 下电源轨孔洞中心
        self._rail_row_coords: Optional[np.ndarray] = None
        self._observed_main_mask: Optional[np.ndarray] = None
        self._observed_top_mask: Optional[np.ndarray] = None
        self._observed_bot_mask: Optional[np.ndarray] = None
        self._valid_main_mask: Optional[np.ndarray] = None
        self._valid_top_mask: Optional[np.ndarray] = None
        self._valid_bot_mask: Optional[np.ndarray] = None

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
            self._synthetic_grid = False
            candidates = self._detect_board_region_candidates(image)
            if not candidates:
                return False
            best_corners = None
            best_holes: List[Tuple[float, float]] = []
            for corners in candidates:
                try:
                    self.calibrate(corners)
                    warped = self.warp(image)
                    holes = self._detect_holes_raw(warped)
                    if len(holes) > len(best_holes):
                        best_holes = holes
                        best_corners = corners
                except Exception:
                    continue
            if best_corners is None:
                return False
            self.calibrate(best_corners)
            warped = self.warp(image)
            self.detect_holes(warped)
            return self.is_grid_ready
        except Exception as e:
            logger.warning(f"[Calibrator] Auto-calibrate failed: {e}")
            return False

    def _detect_board_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """检测面包板白色区域轮廓, 返回四角坐标"""
        candidates = self._detect_board_region_candidates(image)
        if not candidates:
            return None
        return candidates[0]

    def _detect_board_region_candidates(self, image: np.ndarray) -> List[np.ndarray]:
        """生成多个角点候选，后续通过 warp 后的孔洞质量选择最优。"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 白色区域掩码
        mask = cv2.inRange(hsv, (0, 0, 160), (180, 50, 255))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, kernel_e, iterations=3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[np.ndarray] = []
        img_area = image.shape[0] * image.shape[1]

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(contour)
            if area < img_area * 0.05:
                continue
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                candidates.append(self._order_corners(approx.reshape(4, 2).astype(np.float32)))
            rect = cv2.minAreaRect(contour)
            candidates.append(self._order_corners(cv2.boxPoints(rect).astype(np.float32)))

        hole_pts = self._detect_holes_raw(image)
        if len(hole_pts) >= 80:
            rect = cv2.minAreaRect(np.array(hole_pts, dtype=np.float32))
            box = cv2.boxPoints(rect).astype(np.float32)
            expanded = self._expand_quad(box, scale=1.18, image_shape=image.shape[:2])
            candidates.append(self._order_corners(expanded))

        unique: List[np.ndarray] = []
        seen = set()
        for pts in candidates:
            key = tuple(int(round(v)) for v in pts.flatten())
            if key in seen:
                continue
            seen.add(key)
            unique.append(pts)
        return unique

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        return np.array([
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)],
        ], dtype=np.float32)

    @staticmethod
    def _expand_quad(pts: np.ndarray, scale: float, image_shape: Tuple[int, int]) -> np.ndarray:
        center = pts.mean(axis=0, keepdims=True)
        expanded = center + (pts - center) * float(scale)
        h, w = image_shape
        expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
        return expanded.astype(np.float32)

    def warp(self, image: np.ndarray) -> np.ndarray:
        """透视变换"""
        if self._perspective_matrix is None:
            return image
        return cv2.warpPerspective(image, self._perspective_matrix, (800, 600))

    def detect_holes(self, warped_image: np.ndarray):
        """在校正后的图像中检测孔洞"""
        raw_holes = self._detect_holes_raw(warped_image)
        blob_holes = self._detect_holes_blob(warped_image)
        merged_holes = self._merge_hole_centers(raw_holes + blob_holes)
        self._board_mask = self._estimate_warped_board_mask(warped_image)
        board_holes = self._filter_points_by_mask(merged_holes, self._board_mask, pad=4)
        candidate_holes = board_holes if len(board_holes) >= 400 else merged_holes
        self.hole_centers = self._limit_hole_candidates(candidate_holes, self._max_hole_count())
        self._landscape = True
        self._top_rails = []
        self._bot_rails = []
        logger.info(
            "[Calibrator] Detected %d holes (board-filtered=%d, capped=%d)",
            len(merged_holes),
            len(board_holes),
            len(self.hole_centers),
        )

        if len(self.hole_centers) >= 50:
            if not self._build_warped_four_region_grid(warped_image):
                self._build_grid()

    def _max_hole_count(self) -> int:
        return int(self._expected_rows * self.total_cols + self._expected_rail_rows * 4)

    @staticmethod
    def _limit_hole_candidates(points: List[Tuple[float, float]], max_count: int) -> List[Tuple[float, float]]:
        if len(points) <= max_count or max_count <= 0:
            return points
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) < 8:
            return points[:max_count]

        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        nn = np.sort(dist, axis=1)
        first_nn = nn[:, 0]
        finite_first = first_nn[np.isfinite(first_nn)]
        if len(finite_first) == 0:
            return points[:max_count]
        pitch = float(np.median(finite_first))
        if pitch <= 0:
            return points[:max_count]

        near_min = pitch * 0.45
        near_max = pitch * 1.9
        axis_tol = pitch * 0.35
        long_max = pitch * 3.2

        scores: List[Tuple[float, int]] = []
        for idx in range(len(pts)):
            dx = np.abs(pts[:, 0] - pts[idx, 0])
            dy = np.abs(pts[:, 1] - pts[idx, 1])
            d = dist[idx]

            neighbor_support = int(np.sum((d >= near_min) & (d <= near_max)))
            horizontal_support = int(np.sum((dy <= axis_tol) & (dx >= near_min) & (dx <= long_max)))
            vertical_support = int(np.sum((dx <= axis_tol) & (dy >= near_min) & (dy <= long_max)))
            pitch_penalty = abs(float(first_nn[idx]) - pitch) / max(pitch, 1e-6)
            score = (
                neighbor_support * 1.0
                + horizontal_support * 1.5
                + vertical_support * 1.5
                - pitch_penalty * 0.75
            )
            scores.append((score, idx))

        scores.sort(key=lambda item: item[0], reverse=True)
        keep_indices = sorted(idx for _, idx in scores[:max_count])
        return [points[idx] for idx in keep_indices]

    def _build_grid(self):
        """从散点孔洞构建行列网格"""
        if not self.hole_centers:
            return

        pts = np.array(self.hole_centers)
        self._row_coords = self._cluster_1d(pts[:, 1], self.rows)
        self._col_coords = self._cluster_1d(pts[:, 0], self.total_cols)

        # 计算空间哈希参数 + 2D 网格矩阵
        self._compute_grid_params()

    def _build_warped_four_region_grid(self, warped_image: np.ndarray) -> bool:
        """在 warp 后图像中按四个区域拟合网格:
        上电源轨 / 上主区 / 下主区 / 下电源轨。
        """
        if not self.hole_centers:
            return False

        pts = np.asarray(self.hole_centers, dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]

        coarse_y_levels = self._fit_axis_coords(ys, expected_count=14, span=warped_image.shape[0])
        if coarse_y_levels is None or len(coarse_y_levels) != 14:
            logger.info("[Calibrator] Four-region fit failed: could not fit 14 vertical levels")
            return False

        coarse_y_levels = np.sort(coarse_y_levels.astype(np.float32))
        level_gaps = np.diff(coarse_y_levels)
        base_gap = float(np.median(level_gaps)) if len(level_gaps) else 12.0
        level_tolerance = max(4.0, base_gap * 0.45)
        coarse_edges = self._level_band_edges(coarse_y_levels)

        coarse_top_rails = coarse_y_levels[:2]
        coarse_upper_main = coarse_y_levels[2:7]
        coarse_lower_main = coarse_y_levels[7:12]
        coarse_bot_rails = coarse_y_levels[12:]

        main_low = float(coarse_edges[2]) if len(coarse_edges) > 2 else float(coarse_upper_main[0] - level_tolerance)
        main_high = float(coarse_edges[12]) if len(coarse_edges) > 12 else float(coarse_lower_main[-1] + level_tolerance)
        main_points_mask = np.array([(main_low <= float(y) <= main_high) for y in ys], dtype=bool)
        main_xs = xs[main_points_mask]
        row_coords = self._fit_axis_coords(
            main_xs if len(main_xs) >= self._expected_rows * 4 else xs,
            expected_count=self._expected_rows,
            span=warped_image.shape[1],
        )
        if row_coords is None or len(row_coords) != self._expected_rows:
            logger.info("[Calibrator] Four-region fit failed: could not fit %d row coords", self._expected_rows)
            return False

        top_rails = self._refine_level_centers(ys, coarse_top_rails, tolerance=level_tolerance)
        upper_main = self._refine_level_centers(ys, coarse_upper_main, tolerance=level_tolerance)
        lower_main = self._refine_level_centers(ys, coarse_lower_main, tolerance=level_tolerance)
        bot_rails = self._refine_level_centers(ys, coarse_bot_rails, tolerance=level_tolerance)

        if len(upper_main) != 5 or len(lower_main) != 5 or len(top_rails) != 2 or len(bot_rails) != 2:
            return False

        self._landscape = True
        self._row_coords = np.sort(row_coords.astype(np.float32))
        self._col_coords = np.concatenate([upper_main, lower_main]).astype(np.float32)
        self._top_rails = [float(v) for v in top_rails]
        self._bot_rails = [float(v) for v in bot_rails]
        self._build_indexed_hole_matrices(
            pts=pts,
            row_coords=self._row_coords,
            top_rails=np.asarray(self._top_rails, dtype=np.float32),
            upper_main=upper_main,
            lower_main=lower_main,
            bot_rails=np.asarray(self._bot_rails, dtype=np.float32),
            row_tolerance=max(5.0, float(np.median(np.diff(self._row_coords))) * 0.55) if len(self._row_coords) > 1 else 6.0,
            level_tolerance=level_tolerance,
        )
        self._compute_grid_params()
        logger.info(
            "[Calibrator] Four-region warped grid built: rows=%d, cols=%d, rails=(%d,%d)",
            len(self._row_coords),
            len(self._col_coords),
            len(self._top_rails),
            len(self._bot_rails),
        )
        return True

    @staticmethod
    def _level_band_edges(levels: np.ndarray) -> np.ndarray:
        if len(levels) == 0:
            return np.asarray([], dtype=np.float32)
        if len(levels) == 1:
            return np.asarray([float(levels[0]) - 1.0, float(levels[0]) + 1.0], dtype=np.float32)
        mids = (levels[:-1] + levels[1:]) * 0.5
        first = float(levels[0] - (mids[0] - levels[0]))
        last = float(levels[-1] + (levels[-1] - mids[-1]))
        return np.concatenate([[first], mids.astype(np.float32), [last]]).astype(np.float32)

    @staticmethod
    def _refine_level_centers(values: np.ndarray, coarse_levels: np.ndarray, *, tolerance: float) -> np.ndarray:
        refined: List[float] = []
        for level in coarse_levels:
            nearby = values[np.abs(values - float(level)) <= tolerance]
            if len(nearby) >= 3:
                refined.append(float(np.median(nearby)))
            else:
                refined.append(float(level))
        return np.asarray(sorted(refined), dtype=np.float32)

    def _build_indexed_hole_matrices(
        self,
        *,
        pts: np.ndarray,
        row_coords: np.ndarray,
        top_rails: np.ndarray,
        upper_main: np.ndarray,
        lower_main: np.ndarray,
        bot_rails: np.ndarray,
        row_tolerance: float,
        level_tolerance: float,
    ) -> None:
        main_levels = np.concatenate([upper_main, lower_main]).astype(np.float32)
        all_levels = np.concatenate([top_rails, upper_main, lower_main, bot_rails]).astype(np.float32)
        level_edges = self._level_band_edges(all_levels)
        buckets: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        assigned_indices: set[int] = set()

        for point_idx, (x, y) in enumerate(pts):
            row_idx = int(np.argmin(np.abs(row_coords - x)))
            row_delta = float(abs(row_coords[row_idx] - x))
            if row_delta > row_tolerance:
                continue
            level_idx = int(np.searchsorted(level_edges, float(y), side="right") - 1)
            level_idx = max(0, min(level_idx, len(all_levels) - 1))
            level_delta = float(abs(all_levels[level_idx] - y))
            if level_delta > level_tolerance:
                continue
            if level_idx < 2 or level_idx >= 12:
                continue
            buckets.setdefault((row_idx, level_idx), []).append((float(x), float(y)))
            assigned_indices.add(point_idx)
        main_grid = np.zeros((len(row_coords), len(main_levels), 2), dtype=np.float32)
        top_grid = np.zeros((self._expected_rail_rows, 2, 2), dtype=np.float32)
        bot_grid = np.zeros((self._expected_rail_rows, 2, 2), dtype=np.float32)
        observed_main = np.zeros((len(row_coords), len(main_levels)), dtype=bool)
        observed_top = np.zeros((self._expected_rail_rows, 2), dtype=bool)
        observed_bot = np.zeros((self._expected_rail_rows, 2), dtype=bool)

        for row_idx in range(len(row_coords)):
            for level_idx in range(2, 12):
                samples = buckets.get((row_idx, level_idx), [])
                if samples:
                    point = np.median(np.asarray(samples, dtype=np.float32), axis=0)
                else:
                    point = np.asarray([row_coords[row_idx], all_levels[level_idx]], dtype=np.float32)
                main_ci = level_idx - 2
                main_grid[row_idx, main_ci] = point
                observed_main[row_idx, main_ci] = bool(samples)

        rail_seed_levels = np.concatenate([top_rails, bot_rails]).astype(np.float32)
        rail_points_mask = np.array(
            [np.min(np.abs(rail_seed_levels - y)) <= level_tolerance for y in pts[:, 1]],
            dtype=bool,
        )
        rail_xs = pts[rail_points_mask, 0]
        rail_row_coords = self._initial_rail_row_coords(
            rail_xs=rail_xs.astype(np.float32),
            row_coords=row_coords.astype(np.float32),
        )

        rail_buckets: Dict[Tuple[str, int, int], List[Tuple[float, float]]] = {}
        rail_line_points: Dict[Tuple[str, int], List[Tuple[float, float]]] = {
            ("top", 0): [],
            ("top", 1): [],
            ("bot", 0): [],
            ("bot", 1): [],
        }
        for point_idx, (x, y) in enumerate(pts):
            if point_idx in assigned_indices:
                continue
            rail_row_idx = int(np.argmin(np.abs(rail_row_coords - x)))
            row_delta = float(abs(rail_row_coords[rail_row_idx] - x))
            if row_delta > max(row_tolerance * 1.2, 8.0):
                continue
            level_idx = int(np.searchsorted(level_edges, float(y), side="right") - 1)
            level_idx = max(0, min(level_idx, len(all_levels) - 1))
            level_delta = float(abs(all_levels[level_idx] - y))
            if level_delta > level_tolerance:
                continue
            if level_idx < 2:
                rail_line_points[("top", level_idx)].append((float(x), float(y)))
                continue
            if level_idx >= 12:
                bot_level_idx = level_idx - 12
                rail_line_points[("bot", bot_level_idx)].append((float(x), float(y)))

        for (side, rail_idx), samples in rail_line_points.items():
            if not samples:
                continue
            sample_arr = np.asarray(samples, dtype=np.float32)
            xs_line = sample_arr[:, 0]
            target_count = min(self._expected_rail_rows, len(xs_line))
            if len(xs_line) > target_count and target_count >= 2:
                x_centers = self._cluster_1d(xs_line.astype(np.float32), target_count)
            else:
                x_centers = np.sort(xs_line.astype(np.float32))
            for center_x in x_centers:
                nearby = sample_arr[np.abs(sample_arr[:, 0] - float(center_x)) <= max(row_tolerance * 0.75, 6.0)]
                if len(nearby) == 0:
                    nearby = sample_arr[np.argmin(np.abs(sample_arr[:, 0] - float(center_x))) : np.argmin(np.abs(sample_arr[:, 0] - float(center_x))) + 1]
                point = np.median(nearby, axis=0)
                rail_row_idx = int(np.argmin(np.abs(rail_row_coords - point[0])))
                row_delta = float(abs(rail_row_coords[rail_row_idx] - point[0]))
                if row_delta > max(row_tolerance * 1.35, 10.0):
                    continue
                rail_buckets.setdefault((side, rail_row_idx, rail_idx), []).append((float(point[0]), float(point[1])))

        for row_idx in range(len(rail_row_coords)):
            for rail_idx in range(2):
                top_samples = rail_buckets.get(("top", row_idx, rail_idx), [])
                if top_samples:
                    top_grid[row_idx, rail_idx] = np.median(np.asarray(top_samples, dtype=np.float32), axis=0)
                    observed_top[row_idx, rail_idx] = True
                else:
                    top_grid[row_idx, rail_idx] = [rail_row_coords[row_idx], top_rails[rail_idx]]

                bot_samples = rail_buckets.get(("bot", row_idx, rail_idx), [])
                if bot_samples:
                    bot_grid[row_idx, rail_idx] = np.median(np.asarray(bot_samples, dtype=np.float32), axis=0)
                    observed_bot[row_idx, rail_idx] = True
                else:
                    bot_grid[row_idx, rail_idx] = [rail_row_coords[row_idx], bot_rails[rail_idx]]

        unused_indices = sorted(set(range(len(pts))) - assigned_indices)
        if unused_indices:
            self._rescue_missing_cells(
                pts=pts,
                unused_indices=unused_indices,
                main_grid=main_grid,
                top_grid=top_grid,
                bot_grid=bot_grid,
                observed_main=observed_main,
                observed_top=observed_top,
                observed_bot=observed_bot,
                row_tolerance=max(row_tolerance * 1.15, 6.0),
                level_tolerance=max(level_tolerance * 1.35, 6.0),
            )

        refined_rows = row_coords.astype(np.float32).copy()
        for row_idx in range(len(row_coords)):
            row_samples = main_grid[row_idx, observed_main[row_idx], 0].tolist()
            if row_samples:
                refined_rows[row_idx] = float(np.median(np.asarray(row_samples, dtype=np.float32)))

        refined_cols = main_levels.astype(np.float32).copy()
        for col_idx in range(len(main_levels)):
            col_samples = main_grid[observed_main[:, col_idx], col_idx, 1]
            if len(col_samples) > 0:
                refined_cols[col_idx] = float(np.median(col_samples))

        refined_top = top_rails.astype(np.float32).copy()
        for rail_idx in range(2):
            rail_samples = top_grid[observed_top[:, rail_idx], rail_idx, 1]
            if len(rail_samples) > 0:
                refined_top[rail_idx] = float(np.median(rail_samples))

        refined_bot = bot_rails.astype(np.float32).copy()
        for rail_idx in range(2):
            rail_samples = bot_grid[observed_bot[:, rail_idx], rail_idx, 1]
            if len(rail_samples) > 0:
                refined_bot[rail_idx] = float(np.median(rail_samples))

        support_counts = observed_main.sum(axis=1).astype(np.int32)
        regularized_rows = self._regularize_axis_by_pitch(refined_rows, support_counts=support_counts, min_support=2)
        refined_rows = regularized_rows.astype(np.float32)
        refined_upper = self._regularize_group_axis(refined_cols[:5]).astype(np.float32)
        refined_lower = self._regularize_group_axis(refined_cols[5:]).astype(np.float32)
        refined_cols = np.concatenate([refined_upper, refined_lower]).astype(np.float32)
        refined_top = self._regularize_group_axis(refined_top).astype(np.float32)
        refined_bot = self._regularize_group_axis(refined_bot).astype(np.float32)
        rail_support_counts = (observed_top.sum(axis=1) + observed_bot.sum(axis=1)).astype(np.int32)
        refined_rail_rows = self._regularize_grouped_rail_axis(
            initial_coords=rail_row_coords.astype(np.float32),
            support_counts=rail_support_counts,
            top_grid=top_grid,
            bot_grid=bot_grid,
            observed_top=observed_top,
            observed_bot=observed_bot,
        ).astype(np.float32)

        for row_idx in range(len(refined_rows)):
            for col_idx in range(len(refined_cols)):
                if not observed_main[row_idx, col_idx]:
                    main_grid[row_idx, col_idx] = [refined_rows[row_idx], refined_cols[col_idx]]
        for row_idx in range(len(refined_rail_rows)):
            for rail_idx in range(2):
                if not observed_top[row_idx, rail_idx]:
                    top_grid[row_idx, rail_idx] = [refined_rail_rows[row_idx], refined_top[rail_idx]]
                if not observed_bot[row_idx, rail_idx]:
                    bot_grid[row_idx, rail_idx] = [refined_rail_rows[row_idx], refined_bot[rail_idx]]

        self._row_coords = refined_rows
        self._rail_row_coords = refined_rail_rows
        self._col_coords = refined_cols
        self._top_rails = [float(v) for v in refined_top]
        self._bot_rails = [float(v) for v in refined_bot]
        self._grid_matrix = main_grid
        self._top_rail_matrix = top_grid
        self._bot_rail_matrix = bot_grid
        self._observed_main_mask = observed_main
        self._observed_top_mask = observed_top
        self._observed_bot_mask = observed_bot
        self._compute_valid_hole_masks()

    def _compute_valid_hole_masks(self) -> None:
        if self._grid_matrix is None or self._top_rail_matrix is None or self._bot_rail_matrix is None:
            return
        if self._observed_main_mask is None or self._observed_top_mask is None or self._observed_bot_mask is None:
            return

        hole_arr = np.asarray(self.hole_centers, dtype=np.float32) if self.hole_centers else None
        row_pitch = float(np.median(np.diff(self._row_coords))) if self._row_coords is not None and len(self._row_coords) > 1 else 10.0
        col_pitch = float(np.median(np.diff(self._col_coords))) if self._col_coords is not None and len(self._col_coords) > 1 else 10.0
        main_threshold = max(min(row_pitch, col_pitch) * 0.85, 6.0)
        rail_threshold = max(row_pitch * 0.9, 7.0)

        self._valid_main_mask = self._observed_main_mask.copy()
        self._valid_top_mask = self._observed_top_mask.copy()
        self._valid_bot_mask = self._observed_bot_mask.copy()

        def nearest_dist(point: np.ndarray) -> float:
            if hole_arr is None or len(hole_arr) == 0:
                return float("inf")
            d = np.sqrt(np.sum((hole_arr - point.reshape(1, 2)) ** 2, axis=1))
            return float(np.min(d))

        for row_idx in range(self._grid_matrix.shape[0]):
            for col_idx in range(self._grid_matrix.shape[1]):
                if self._valid_main_mask[row_idx, col_idx]:
                    continue
                point = self._grid_matrix[row_idx, col_idx]
                if nearest_dist(point) <= main_threshold:
                    self._valid_main_mask[row_idx, col_idx] = True

        for row_idx in range(self._top_rail_matrix.shape[0]):
            for rail_idx in range(self._top_rail_matrix.shape[1]):
                if not self._valid_top_mask[row_idx, rail_idx]:
                    point = self._top_rail_matrix[row_idx, rail_idx]
                    if nearest_dist(point) <= rail_threshold:
                        self._valid_top_mask[row_idx, rail_idx] = True
                if not self._valid_bot_mask[row_idx, rail_idx]:
                    point = self._bot_rail_matrix[row_idx, rail_idx]
                    if nearest_dist(point) <= rail_threshold:
                        self._valid_bot_mask[row_idx, rail_idx] = True

    @staticmethod
    def _rail_position_units(group_count: int = 10, holes_per_group: int = 5) -> np.ndarray:
        units: List[float] = [0.0]
        pos = 0.0
        for group_idx in range(group_count):
            for hole_idx in range(holes_per_group - 1):
                pos += 1.0
                units.append(pos)
            if group_idx < group_count - 1:
                pos += 2.0
        return np.asarray(units, dtype=np.float32)

    def _build_grouped_rail_axis(self, start: float, end: float) -> np.ndarray:
        units = self._rail_position_units()
        if len(units) != self._expected_rail_rows:
            return np.linspace(start, end, self._expected_rail_rows, dtype=np.float32)
        total_units = float(units[-1]) if len(units) > 1 else 1.0
        if total_units <= 0:
            return np.linspace(start, end, self._expected_rail_rows, dtype=np.float32)
        scale = float(end - start) / total_units
        return (float(start) + units * scale).astype(np.float32)

    def _initial_rail_row_coords(self, *, rail_xs: np.ndarray, row_coords: np.ndarray) -> np.ndarray:
        low = float(row_coords[0])
        high = float(row_coords[-1])
        if len(rail_xs) >= 8:
            rail_low = float(np.percentile(rail_xs, 2))
            rail_high = float(np.percentile(rail_xs, 98))
            low = min(low, rail_low)
            high = max(high, rail_high)
        elif len(rail_xs) >= 2:
            rail_low = float(np.min(rail_xs))
            rail_high = float(np.max(rail_xs))
            low = min(low, rail_low)
            high = max(high, rail_high)
        if high <= low:
            low = float(row_coords[0])
            high = float(row_coords[-1])
        return self._build_grouped_rail_axis(low, high)

    def _regularize_grouped_rail_axis(
        self,
        *,
        initial_coords: np.ndarray,
        support_counts: np.ndarray,
        top_grid: np.ndarray,
        bot_grid: np.ndarray,
        observed_top: np.ndarray,
        observed_bot: np.ndarray,
    ) -> np.ndarray:
        units = self._rail_position_units()
        if len(initial_coords) != len(units):
            return initial_coords.astype(np.float32)
        x_obs = np.full(len(initial_coords), np.nan, dtype=np.float32)
        weights = np.maximum(support_counts.astype(np.float32), 0.0)
        for row_idx in range(len(initial_coords)):
            samples: List[float] = []
            if np.any(observed_top[row_idx]):
                samples.extend(top_grid[row_idx, observed_top[row_idx], 0].astype(np.float32).tolist())
            if np.any(observed_bot[row_idx]):
                samples.extend(bot_grid[row_idx, observed_bot[row_idx], 0].astype(np.float32).tolist())
            if samples:
                x_obs[row_idx] = float(np.median(np.asarray(samples, dtype=np.float32)))
        valid = ~np.isnan(x_obs)
        if int(np.sum(valid)) >= 2:
            fit_w = np.maximum(weights[valid], 1.0)
            slope, intercept = np.polyfit(units[valid], x_obs[valid], 1, w=fit_w)
            return (intercept + slope * units).astype(np.float32)
        return initial_coords.astype(np.float32)

    @staticmethod
    def _regularize_axis_by_pitch(
        coords: np.ndarray,
        *,
        support_counts: np.ndarray,
        min_support: int = 2,
    ) -> np.ndarray:
        if len(coords) < 2:
            return coords
        indices = np.arange(len(coords), dtype=np.float32)
        anchor_mask = support_counts >= min_support
        if np.sum(anchor_mask) >= 2:
            x = indices[anchor_mask]
            y = coords[anchor_mask].astype(np.float32)
            weights = np.maximum(support_counts[anchor_mask].astype(np.float32), 1.0)
            slope, intercept = np.polyfit(x, y, 1, w=weights)
            regularized = intercept + slope * indices
            return regularized.astype(np.float32)
        pitch = float(np.median(np.diff(coords)))
        return (float(coords[0]) + pitch * indices).astype(np.float32)

    @staticmethod
    def _regularize_group_axis(coords: np.ndarray) -> np.ndarray:
        if len(coords) < 2:
            return coords
        indices = np.arange(len(coords), dtype=np.float32)
        slope, intercept = np.polyfit(indices, coords.astype(np.float32), 1)
        regularized = intercept + slope * indices
        return regularized.astype(np.float32)

    @staticmethod
    def _rescue_missing_cells(
        *,
        pts: np.ndarray,
        unused_indices: List[int],
        main_grid: np.ndarray,
        top_grid: np.ndarray,
        bot_grid: np.ndarray,
        observed_main: np.ndarray,
        observed_top: np.ndarray,
        observed_bot: np.ndarray,
        row_tolerance: float,
        level_tolerance: float,
    ) -> None:
        remaining = set(int(idx) for idx in unused_indices)

        def assign_if_close(expected: np.ndarray) -> Optional[np.ndarray]:
            ex, ey = float(expected[0]), float(expected[1])
            best_idx: Optional[int] = None
            best_dist: Optional[float] = None
            for idx in list(remaining):
                x, y = pts[idx]
                if abs(float(x) - ex) > row_tolerance or abs(float(y) - ey) > level_tolerance:
                    continue
                dist = float((float(x) - ex) ** 2 + (float(y) - ey) ** 2)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is None:
                return None
            remaining.remove(best_idx)
            return pts[best_idx].astype(np.float32)

        for row_idx in range(main_grid.shape[0]):
            for col_idx in range(main_grid.shape[1]):
                if observed_main[row_idx, col_idx]:
                    continue
                found = assign_if_close(main_grid[row_idx, col_idx])
                if found is not None:
                    main_grid[row_idx, col_idx] = found
                    observed_main[row_idx, col_idx] = True

        for row_idx in range(top_grid.shape[0]):
            for rail_idx in range(top_grid.shape[1]):
                if not observed_top[row_idx, rail_idx]:
                    found = assign_if_close(top_grid[row_idx, rail_idx])
                    if found is not None:
                        top_grid[row_idx, rail_idx] = found
                        observed_top[row_idx, rail_idx] = True

        for row_idx in range(bot_grid.shape[0]):
            for rail_idx in range(bot_grid.shape[1]):
                if not observed_bot[row_idx, rail_idx]:
                    found = assign_if_close(bot_grid[row_idx, rail_idx])
                    if found is not None:
                        bot_grid[row_idx, rail_idx] = found
                        observed_bot[row_idx, rail_idx] = True

    def _fit_axis_coords(self, values: np.ndarray, *, expected_count: int, span: int) -> Optional[np.ndarray]:
        peaks = self._find_peaks_1d(list(map(float, values)), span=span, sigma=2.0, min_density=1.0)
        if len(peaks) == expected_count:
            return np.asarray(peaks, dtype=np.float32)
        if len(peaks) > expected_count:
            peak_arr = np.asarray(peaks, dtype=np.float32)
            return self._cluster_1d(peak_arr, expected_count)
        if len(values) >= expected_count * 4:
            return self._cluster_1d(values.astype(np.float32), expected_count)
        return None

    def _estimate_warped_board_mask(self, warped_image: np.ndarray) -> Optional[np.ndarray]:
        hsv = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 120), (180, 70, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        white_mask = cv2.dilate(
            white_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=1,
        )
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(white_mask)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        return mask

    @staticmethod
    def _filter_points_by_mask(
        points: List[Tuple[float, float]],
        mask: Optional[np.ndarray],
        *,
        pad: int = 2,
    ) -> List[Tuple[float, float]]:
        if mask is None:
            return points
        h, w = mask.shape[:2]
        kept: List[Tuple[float, float]] = []
        for x, y in points:
            xi = int(round(x))
            yi = int(round(y))
            x1 = max(0, xi - pad)
            y1 = max(0, yi - pad)
            x2 = min(w, xi + pad + 1)
            y2 = min(h, yi + pad + 1)
            patch = mask[y1:y2, x1:x2]
            if patch.size > 0 and int(np.max(patch)) > 0:
                kept.append((x, y))
        return kept


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
        if self._grid_matrix is None or self._grid_matrix.shape != (nr, nc, 2):
            self._grid_matrix = np.zeros((nr, nc, 2), dtype=np.float32)
            for r in range(nr):
                for c in range(nc):
                    self._grid_matrix[r, c] = [self._row_coords[r], self._col_coords[c]]

        rail_nr = len(self._rail_row_coords) if self._rail_row_coords is not None else nr
        rail_rows = self._rail_row_coords if self._rail_row_coords is not None else self._row_coords

        if self._top_rail_matrix is None or self._top_rail_matrix.shape != (rail_nr, 2, 2):
            self._top_rail_matrix = np.zeros((rail_nr, 2, 2), dtype=np.float32)
            for r in range(rail_nr):
                for rail_idx in range(2):
                    rail_y = self._top_rails[rail_idx] if rail_idx < len(self._top_rails) else self._col_coords[0]
                    self._top_rail_matrix[r, rail_idx] = [rail_rows[r], rail_y]

        if self._bot_rail_matrix is None or self._bot_rail_matrix.shape != (rail_nr, 2, 2):
            self._bot_rail_matrix = np.zeros((rail_nr, 2, 2), dtype=np.float32)
            for r in range(rail_nr):
                for rail_idx in range(2):
                    rail_y = self._bot_rails[rail_idx] if rail_idx < len(self._bot_rails) else self._col_coords[-1]
                    self._bot_rail_matrix[r, rail_idx] = [rail_rows[r], rail_y]

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

        nearest = self._nearest_indexed_holes(row_val, col_val, k=1)
        if nearest:
            return tuple(nearest[0]["logic_loc"])  # type: ignore[return-value]

        # 检查是否落入电轨区域 (包括超出面包板范围的引脚)
        rail_tolerance = self._rail_tolerance
        grid_min = float(self._col_coords[0]) if len(self._col_coords) > 0 else 0
        grid_max = float(self._col_coords[-1]) if len(self._col_coords) > 0 else 0
        rail_rows = self._rail_row_coords if self._rail_row_coords is not None else self._row_coords

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
                row_idx = int(np.argmin(np.abs(rail_rows - row_val)))
                rail_name = "+" if closest_rail_idx == 0 else "-"
                return (str(row_idx + 1), f"rail_top{rail_name}")

        if self._bot_rails and col_val > grid_max:
            dist_to_grid = col_val - grid_max
            grid_spacing = float(self._col_coords[-1] - self._col_coords[-2]) if len(self._col_coords) > 1 else 20
            if dist_to_grid < grid_spacing:
                pass  # 离 grid 够近
            else:
                closest_rail_idx = int(np.argmin([abs(col_val - r) for r in self._bot_rails]))
                row_idx = int(np.argmin(np.abs(rail_rows - row_val)))
                rail_name = "+" if closest_rail_idx == 0 else "-"
                return (str(row_idx + 1), f"rail_bot{rail_name}")

        for i, rail_pos in enumerate(self._top_rails):
            if abs(col_val - rail_pos) < rail_tolerance:
                row_idx = int(np.argmin(np.abs(rail_rows - row_val)))
                rail_name = "+" if i == 0 else "-"
                return (str(row_idx + 1), f"rail_top{rail_name}")

        for i, rail_pos in enumerate(self._bot_rails):
            if abs(col_val - rail_pos) < rail_tolerance:
                row_idx = int(np.argmin(np.abs(rail_rows - row_val)))
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

        indexed_candidates = self._nearest_indexed_holes(row_val, col_val, k=max(k * 3, 8))
        if indexed_candidates:
            ordered: List[Tuple[str, str]] = []
            seen = set()
            for item in indexed_candidates:
                logic_loc = tuple(item["logic_loc"])
                if logic_loc not in seen:
                    seen.add(logic_loc)
                    ordered.append(logic_loc)  # type: ignore[arg-type]
                if len(ordered) >= k:
                    break
            if ordered:
                return ordered

        # 先检查电轨 (包括超出 grid 范围的引脚)
        rail_tolerance = self._rail_tolerance
        grid_min = float(self._col_coords[0]) if len(self._col_coords) > 0 else 0
        grid_max = float(self._col_coords[-1]) if len(self._col_coords) > 0 else 0
        grid_spacing = float(self._col_coords[1] - self._col_coords[0]) if len(self._col_coords) > 1 else 20
        rail_rows = self._rail_row_coords if self._rail_row_coords is not None else self._row_coords

        # 超出范围但离 grid 近的 → 映射到 grid, 离 grid 远的 → 映射到 rail
        if self._top_rails and col_val < grid_min:
            dist_to_grid = grid_min - col_val
            if dist_to_grid >= grid_spacing:
                closest_idx = int(np.argmin([abs(col_val - r) for r in self._top_rails]))
                row_dists = np.abs(rail_rows - row_val)
                top_rows = np.argsort(row_dists)[:k]
                rail_name = "+" if closest_idx == 0 else "-"
                return [(str(ri + 1), f"rail_top{rail_name}") for ri in top_rows]

        if self._bot_rails and col_val > grid_max:
            dist_to_grid = col_val - grid_max
            if dist_to_grid >= grid_spacing:
                closest_idx = int(np.argmin([abs(col_val - r) for r in self._bot_rails]))
                row_dists = np.abs(rail_rows - row_val)
                top_rows = np.argsort(row_dists)[:k]
                rail_name = "+" if closest_idx == 0 else "-"
                return [(str(ri + 1), f"rail_bot{rail_name}") for ri in top_rows]

        for rails, prefix in [(self._top_rails, "rail_top"), (self._bot_rails, "rail_bot")]:
            for i, rail_pos in enumerate(rails):
                if abs(col_val - rail_pos) < rail_tolerance:
                    row_dists = np.abs(rail_rows - row_val)
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
        """确保校准器已校准: 优先使用板区透视校准, 峰值法作为兜底"""
        if self.is_grid_ready:
            return True

        # 1) 优先尝试视觉自动校准。对真实面包板照片来说，
        #    先做板区透视校正，再在校正图上找孔，更稳定。
        if self.auto_calibrate(image):
            logger.info("[Calibrator] Visual auto-calibrate succeeded")
            return True

        # 2) 基于孔洞峰值的智能校准作为兜底。
        if self._smart_calibrate(image):
            logger.info("[Calibrator] Smart hole-based calibration succeeded (landscape=%s)", self._landscape)
            return True

        # 3) RANSAC 多点单应性估计 (参考 Document Scanner)
        if self._calibrate_from_holes_ransac(image):
            logger.info("[Calibrator] RANSAC homography calibration succeeded")
            return True

        # 4) 视觉校准失败 → 使用合成网格
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

    def iter_indexed_holes(self) -> List[Dict[str, Any]]:
        indexed: List[Dict[str, Any]] = []
        if self._grid_matrix is not None:
            for row_idx in range(self._grid_matrix.shape[0]):
                for col_idx in range(self._grid_matrix.shape[1]):
                    if self._valid_main_mask is not None and not bool(self._valid_main_mask[row_idx, col_idx]):
                        continue
                    x, y = self._grid_matrix[row_idx, col_idx]
                    col_name = self._col_names[col_idx] if col_idx < len(self._col_names) else str(col_idx)
                    observed = bool(self._observed_main_mask[row_idx, col_idx]) if self._observed_main_mask is not None else False
                    indexed.append(
                        {
                            "pixel": (float(x), float(y)),
                            "logic_loc": (str(row_idx + 1), col_name),
                            "hole_id": f"{col_name.upper()}{row_idx + 1}",
                            "group": "main",
                            "observed": observed,
                            "anchor": observed,
                            "source": "anchor" if observed else "inferred_from_anchor",
                        }
                    )
        if self._top_rail_matrix is not None:
            for row_idx in range(self._top_rail_matrix.shape[0]):
                for rail_idx in range(self._top_rail_matrix.shape[1]):
                    if self._valid_top_mask is not None and not bool(self._valid_top_mask[row_idx, rail_idx]):
                        continue
                    x, y = self._top_rail_matrix[row_idx, rail_idx]
                    logic_col = "rail_top+" if rail_idx == 0 else "rail_top-"
                    observed = bool(self._observed_top_mask[row_idx, rail_idx]) if self._observed_top_mask is not None else False
                    indexed.append(
                        {
                            "pixel": (float(x), float(y)),
                            "logic_loc": (str(row_idx + 1), logic_col),
                            "hole_id": f"{'LP' if rail_idx == 0 else 'LN'}{row_idx + 1}",
                            "group": "top_rail",
                            "observed": observed,
                            "anchor": observed,
                            "source": "anchor" if observed else "inferred_from_anchor",
                        }
                    )
        if self._bot_rail_matrix is not None:
            for row_idx in range(self._bot_rail_matrix.shape[0]):
                for rail_idx in range(self._bot_rail_matrix.shape[1]):
                    if self._valid_bot_mask is not None and not bool(self._valid_bot_mask[row_idx, rail_idx]):
                        continue
                    x, y = self._bot_rail_matrix[row_idx, rail_idx]
                    logic_col = "rail_bot+" if rail_idx == 0 else "rail_bot-"
                    observed = bool(self._observed_bot_mask[row_idx, rail_idx]) if self._observed_bot_mask is not None else False
                    indexed.append(
                        {
                            "pixel": (float(x), float(y)),
                            "logic_loc": (str(row_idx + 1), logic_col),
                            "hole_id": f"{'RP' if rail_idx == 0 else 'RN'}{row_idx + 1}",
                            "group": "bot_rail",
                            "observed": observed,
                            "anchor": observed,
                            "source": "anchor" if observed else "inferred_from_anchor",
                        }
                    )
        return indexed

    def _nearest_indexed_holes(self, row_val: float, col_val: float, k: int = 5) -> List[Dict[str, Any]]:
        indexed = self.iter_indexed_holes()
        if not indexed:
            return []
        scored = []
        for item in indexed:
            hx, hy = item["pixel"]
            dist = float((hx - row_val) ** 2 + (hy - col_val) ** 2)
            scored.append((dist, item))
        scored.sort(key=lambda x: x[0])
        return [item for _, item in scored[:k]]

    def get_nearest_hole_px(
        self, px: float, py: float,
    ) -> Optional[Tuple[float, float]]:
        """返回最近孔洞的像素坐标 (用于可视化)"""
        if self._perspective_matrix is not None:
            pt = np.array([[[px, py]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self._perspective_matrix)
            px, py = transformed[0, 0]

        if self._landscape:
            row_val, col_val = px, py
        else:
            row_val, col_val = py, px

        nearest = self._nearest_indexed_holes(row_val, col_val, k=1)
        if nearest:
            return tuple(nearest[0]["pixel"])  # type: ignore[return-value]
        return None

    # ---- 智能校准 (孔洞检测 + 自动朝向识别) ----

    def _detect_holes_raw(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """形态学检测面包板孔洞, 返回 (cx, cy) 列表"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
        blackhat = cv2.morphologyEx(
            clahe,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        _, thresh_otsu = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_adapt = cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            6,
        )
        thresh = cv2.bitwise_or(thresh_otsu, thresh_adapt)
        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        holes: List[Tuple[float, float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 4 < area < 180:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    perimeter = cv2.arcLength(cnt, True)
                    circ = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
                    if circ > 0.15:
                        holes.append((cx, cy))
        return self._merge_hole_centers(holes)

    def _detect_holes_blob(self, image: np.ndarray) -> List[Tuple[float, float]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = 8
        params.maxArea = 220
        params.filterByCircularity = True
        params.minCircularity = 0.12
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        inv = cv2.bitwise_not(gray)
        keypoints = detector.detect(inv)
        return [(kp.pt[0], kp.pt[1]) for kp in keypoints]

    @staticmethod
    def _merge_hole_centers(points: List[Tuple[float, float]], distance_threshold: float = 6.0) -> List[Tuple[float, float]]:
        if not points:
            return []
        ordered = sorted(points, key=lambda p: (p[0], p[1]))
        merged: List[List[Tuple[float, float]]] = []
        for point in ordered:
            if not merged:
                merged.append([point])
                continue
            last_cluster = merged[-1]
            last_x = float(np.mean([p[0] for p in last_cluster]))
            last_y = float(np.mean([p[1] for p in last_cluster]))
            if (point[0] - last_x) ** 2 + (point[1] - last_y) ** 2 <= distance_threshold ** 2:
                last_cluster.append(point)
            else:
                merged.append([point])
        return [
            (float(np.mean([p[0] for p in cluster])), float(np.mean([p[1] for p in cluster])))
            for cluster in merged
        ]

    @staticmethod
    def _find_peaks_1d(values: List[float], span: int, sigma: float = 3.0, min_density: float = 1.5) -> List[int]:
        """1D 高斯平滑后找峰值"""
        if span < 10 or not values:
            return []
        profile = np.zeros(span, dtype=np.float32)
        for v in values:
            iv = int(v)
            if 0 <= iv < span:
                profile[iv] += 1
        sig = float(max(0.5, sigma))
        radius = int(max(3, round(sig * 3)))
        xs = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(xs * xs) / (2.0 * sig * sig))
        kernel_sum = float(kernel.sum())
        if kernel_sum > 0:
            kernel /= kernel_sum
        smoothed = np.convolve(profile, kernel, mode="same")
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
