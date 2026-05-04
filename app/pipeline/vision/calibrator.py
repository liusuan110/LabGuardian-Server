"""
面包板校准器 (← src_v2/vision/calibrator.py)

提供像素坐标 ↔ 逻辑坐标 (Row×Col) 映射
服务端简化版: 保留核心校准 + 坐标映射, 去掉 GUI 交互
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BREAD_DETECT_DIR = Path(__file__).resolve().parents[3] / "bread_detect"
if str(_BREAD_DETECT_DIR) not in sys.path:
    sys.path.append(str(_BREAD_DETECT_DIR))

from breadboard_detect import GridModel, fit_grid
from breadboard_detect_white_region import detect_white_region_quad


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
        self._detected_hole_map = False

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
        """自动校准: 使用白区域检测 + 标准网格拟合建立主链校准。"""
        try:
            self._synthetic_grid = False
            corners, _, _ = detect_white_region_quad(
                image,
                fallback_auto=True,
            )
            self.calibrate(corners)
            warped = self.warp(image)
            self.detect_holes(warped)
            return self.is_grid_ready
        except Exception as e:
            logger.warning(f"[Calibrator] Auto-calibrate failed: {e}")
            return False

    def warp(self, image: np.ndarray) -> np.ndarray:
        """透视变换"""
        if self._perspective_matrix is None:
            return image
        return cv2.warpPerspective(image, self._perspective_matrix, (800, 600))

    def detect_holes(self, warped_image: np.ndarray):
        """在校正后的图像中检测孔洞，并按队友的标准网格模型建孔位矩阵。"""
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
            self._build_from_teammate_model(warped_image)

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

    def _build_from_teammate_model(self, warped_image: np.ndarray) -> None:
        """用队友的 GridModel 直接生成主区/电源轨孔位矩阵。"""
        model = fit_grid(warped_image, self._expected_rows)
        self._apply_grid_model(model)
        logger.info(
            "[Calibrator] GridModel applied: rows=%d, cols=%d, rails=(%d,%d)",
            len(self._row_coords) if self._row_coords is not None else 0,
            len(self._col_coords) if self._col_coords is not None else 0,
            len(self._top_rails),
            len(self._bot_rails),
        )

    def _apply_grid_model(self, model: GridModel) -> None:
        self._landscape = True
        self._row_coords = np.asarray(
            [model.main_x0 + idx * model.main_pitch_x for idx in range(self._expected_rows)],
            dtype=np.float32,
        )
        upper_main = np.asarray(
            [model.upper_y0 + idx * model.upper_pitch_y for idx in range(5)],
            dtype=np.float32,
        )
        lower_main = np.asarray(
            [model.lower_y0 + idx * model.lower_pitch_y for idx in range(5)],
            dtype=np.float32,
        )
        self._col_coords = np.concatenate([upper_main, lower_main]).astype(np.float32)
        self._top_rails = [
            float(model.top_rail_y0),
            float(model.top_rail_y0 + model.top_rail_pitch_y),
        ]
        self._bot_rails = [
            float(model.bottom_rail_y0),
            float(model.bottom_rail_y0 + model.bottom_rail_pitch_y),
        ]
        rail_offsets = self._rail_position_units()
        self._rail_row_coords = np.asarray(
            [model.rail_x0 + offset * model.rail_pitch_x for offset in rail_offsets],
            dtype=np.float32,
        )

        self._grid_matrix = np.zeros((self._expected_rows, self.total_cols, 2), dtype=np.float32)
        for row_idx, row_x in enumerate(self._row_coords):
            for col_idx, col_y in enumerate(self._col_coords):
                self._grid_matrix[row_idx, col_idx] = [row_x, col_y]

        self._top_rail_matrix = np.zeros((self._expected_rail_rows, 2, 2), dtype=np.float32)
        self._bot_rail_matrix = np.zeros((self._expected_rail_rows, 2, 2), dtype=np.float32)
        for row_idx, row_x in enumerate(self._rail_row_coords):
            for rail_idx, rail_y in enumerate(self._top_rails):
                self._top_rail_matrix[row_idx, rail_idx] = [row_x, rail_y]
            for rail_idx, rail_y in enumerate(self._bot_rails):
                self._bot_rail_matrix[row_idx, rail_idx] = [row_x, rail_y]

        self._observed_main_mask = self._observed_mask_for_grid(self._grid_matrix)
        self._observed_top_mask = self._observed_mask_for_grid(self._top_rail_matrix)
        self._observed_bot_mask = self._observed_mask_for_grid(self._bot_rail_matrix)
        self._compute_valid_hole_masks()
        self._compute_grid_params()

    def _observed_mask_for_grid(self, grid: np.ndarray) -> np.ndarray:
        if not self.hole_centers:
            return np.zeros(grid.shape[:-1], dtype=bool)
        hole_arr = np.asarray(self.hole_centers, dtype=np.float32)
        flat_grid = grid.reshape(-1, 2)
        diff = flat_grid[:, None, :] - hole_arr[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        nearest = np.min(dist, axis=1)
        threshold = max(self._grid_observation_threshold(grid), 5.0)
        return (nearest <= threshold).reshape(grid.shape[:-1])

    @staticmethod
    def _grid_observation_threshold(grid: np.ndarray) -> float:
        x_pitch = float("inf")
        y_pitch = float("inf")
        if grid.shape[0] > 1:
            x_pitch = float(np.median(np.abs(np.diff(grid[:, 0, 0]))))
        if len(grid.shape) >= 3 and grid.shape[1] > 1:
            y_pitch = float(np.median(np.abs(np.diff(grid[0, :, 1]))))
        finite = [value for value in (x_pitch, y_pitch) if np.isfinite(value) and value > 0]
        if not finite:
            return 6.0
        return min(finite) * 0.72

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


    def frame_pixel_to_logic(
        self, px: float, py: float,
    ) -> Optional[Tuple[str, str]]:
        """像素坐标 → 逻辑坐标 (行号, 列名)"""
        board_x, board_y = self.frame_pixel_to_board_point(px, py)
        return self.board_point_to_logic(board_x, board_y)

    def frame_pixel_to_board_point(self, px: float, py: float) -> Tuple[float, float]:
        """Map a source image pixel into the calibrated 2D board plane."""
        if self._perspective_matrix is not None:
            pt = np.array([[[px, py]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self._perspective_matrix)
            px, py = transformed[0, 0]
        return (float(px), float(py))

    def board_point_to_logic_candidates(
        self, board_x: float, board_y: float, k: int = 5,
    ) -> List[Tuple[str, str]]:
        """Return nearest logic candidates for a point already on the 2D board plane."""
        board_values = self._board_values(board_x, board_y)
        if board_values is None:
            return []
        row_val, col_val = board_values

        rail_tolerance = self._rail_tolerance
        grid_min = float(self._col_coords[0]) if len(self._col_coords) > 0 else 0
        grid_max = float(self._col_coords[-1]) if len(self._col_coords) > 0 else 0
        grid_spacing = float(self._col_coords[1] - self._col_coords[0]) if len(self._col_coords) > 1 else 20
        rail_rows = self._rail_row_coords if self._rail_row_coords is not None else self._row_coords
        rail_row_pitch = self._median_pitch(rail_rows)
        main_row_pitch = self._median_pitch(self._row_coords)
        main_col_pitch = self._median_pitch(self._col_coords)
        main_pitch = min(main_row_pitch, main_col_pitch)

        if self._top_rails and col_val < grid_min:
            dist_to_grid = grid_min - col_val
            if dist_to_grid >= grid_spacing:
                closest_idx = int(np.argmin([abs(col_val - r) for r in self._top_rails]))
                top_rows = self._candidate_row_indices(
                    rail_rows,
                    row_val,
                    k=k,
                    pitch=rail_row_pitch,
                )
                rail_name = "+" if closest_idx == 0 else "-"
                return [(str(ri + 1), f"rail_top{rail_name}") for ri in top_rows]

        if self._bot_rails and col_val > grid_max:
            dist_to_grid = col_val - grid_max
            if dist_to_grid >= grid_spacing:
                closest_idx = int(np.argmin([abs(col_val - r) for r in self._bot_rails]))
                top_rows = self._candidate_row_indices(
                    rail_rows,
                    row_val,
                    k=k,
                    pitch=rail_row_pitch,
                )
                rail_name = "+" if closest_idx == 0 else "-"
                return [(str(ri + 1), f"rail_bot{rail_name}") for ri in top_rows]

        for rails, prefix in [(self._top_rails, "rail_top"), (self._bot_rails, "rail_bot")]:
            for i, rail_pos in enumerate(rails):
                if abs(col_val - rail_pos) < rail_tolerance:
                    top_rows = self._candidate_row_indices(
                        rail_rows,
                        row_val,
                        k=k,
                        pitch=rail_row_pitch,
                    )
                    rail_name = "+" if i == 0 else "-"
                    return [(str(ri + 1), f"{prefix}{rail_name}") for ri in top_rows]

        center_r, center_c = self._spatial_hash(row_val, col_val)
        scored = []
        radius = 2
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                ri = center_r + dr
                ci = center_c + dc
                if ri < 0 or ri >= len(self._row_coords):
                    continue
                if ci < 0 or ci >= len(self._col_coords):
                    continue
                if self._valid_main_mask is not None and not bool(self._valid_main_mask[ri, ci]):
                    continue
                dist = float(
                    (self._row_coords[ri] - row_val) ** 2
                    + (self._col_coords[ci] - col_val) ** 2
                )
                row_name = str(ri + 1)
                col_name = self._col_names[ci] if ci < len(self._col_names) else str(ci)
                scored.append((dist, row_name, col_name))
        if not scored:
            return []
        scored.sort(key=lambda x: x[0])
        min_dist = float(np.sqrt(scored[0][0]))
        distance_gate = min(
            min_dist + max(main_pitch * 0.35, 3.0),
            max(main_pitch * 0.95, 8.0),
        )
        filtered = [
            (r, c)
            for dist2, r, c in scored
            if float(np.sqrt(dist2)) <= distance_gate
        ]
        if not filtered:
            filtered = [(scored[0][1], scored[0][2])]
        return filtered[:k]

    @staticmethod
    def _median_pitch(coords: np.ndarray) -> float:
        if coords is None or len(coords) < 2:
            return 10.0
        diffs = np.abs(np.diff(coords.astype(np.float32)))
        diffs = diffs[diffs > 1e-6]
        if len(diffs) == 0:
            return 10.0
        return float(np.median(diffs))

    @staticmethod
    def _candidate_row_indices(
        row_coords: np.ndarray,
        row_val: float,
        *,
        k: int,
        pitch: float,
    ) -> List[int]:
        row_dists = np.abs(row_coords - row_val)
        nearest = float(np.min(row_dists))
        distance_gate = min(
            nearest + max(pitch * 0.35, 2.0),
            max(pitch * 0.95, 6.0),
        )
        indices = [int(idx) for idx, dist in enumerate(row_dists) if float(dist) <= distance_gate]
        if not indices:
            indices = [int(np.argmin(row_dists))]
        indices.sort(key=lambda idx: float(row_dists[idx]))
        return indices[:k]

    def board_point_to_logic(
        self, board_x: float, board_y: float,
    ) -> Optional[Tuple[str, str]]:
        """Map a calibrated 2D board-plane point to one logic location."""
        candidates = self.board_point_to_logic_candidates(board_x, board_y, k=1)
        return candidates[0] if candidates else None

    def frame_pixel_to_logic_candidates(
        self, px: float, py: float, k: int = 5,
    ) -> List[Tuple[str, str]]:
        """Return nearest logic candidates for a source image pixel."""
        board_x, board_y = self.frame_pixel_to_board_point(px, py)
        return self.board_point_to_logic_candidates(board_x, board_y, k=k)

    def logic_to_board_point(self, logic_loc: Tuple[str, str]) -> Optional[Tuple[float, float]]:
        """Map a logic location to its calibrated 2D board-plane point."""
        if self._row_coords is None or self._col_coords is None:
            return None
        if len(logic_loc) < 2:
            return None

        row_raw, col_raw = str(logic_loc[0]).strip(), str(logic_loc[1]).strip()
        try:
            row_idx = int(row_raw) - 1
        except ValueError:
            return None
        if row_idx < 0:
            return None

        col_lower = col_raw.lower()
        rail_rows = self._rail_row_coords if self._rail_row_coords is not None else self._row_coords

        if col_lower in {"rail_top+", "rail_top-", "lp", "ln"}:
            rail_idx = 0 if col_lower in {"rail_top+", "lp"} else 1
            if row_idx >= len(rail_rows) or rail_idx >= len(self._top_rails):
                return None
            row_val = float(rail_rows[row_idx])
            col_val = float(self._top_rails[rail_idx])
            return (row_val, col_val) if self._landscape else (col_val, row_val)

        if col_lower in {"rail_bot+", "rail_bot-", "rp", "rn"}:
            rail_idx = 0 if col_lower in {"rail_bot+", "rp"} else 1
            if row_idx >= len(rail_rows) or rail_idx >= len(self._bot_rails):
                return None
            row_val = float(rail_rows[row_idx])
            col_val = float(self._bot_rails[rail_idx])
            return (row_val, col_val) if self._landscape else (col_val, row_val)

        if col_lower not in self._col_names:
            return None
        col_idx = self._col_names.index(col_lower)
        if row_idx >= len(self._row_coords) or col_idx >= len(self._col_coords):
            return None

        row_val = float(self._row_coords[row_idx])
        col_val = float(self._col_coords[col_idx])
        return (row_val, col_val) if self._landscape else (col_val, row_val)

    def _board_values(self, board_x: float, board_y: float) -> Optional[Tuple[float, float]]:
        if self._row_coords is None or self._col_coords is None:
            return None
        px, py = float(board_x), float(board_y)
        if self._landscape:
            return (px, py)
        return (py, px)

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

    def load_detected_holes_json(self, path: str | Path) -> bool:
        """Load a bread_detect holes.json file as the calibrated board grid."""
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return self.load_detected_holes(payload)

    def load_detected_holes(self, payload: Dict[str, Any]) -> bool:
        """Load detected breadboard holes exported by bread_detect.

        bread_detect names terminal holes as A00..J62. The pipeline's legacy
        logic coordinate is (numeric row along the board, column name), so
        A00 becomes ("1", "a"), and BoardSchema resolves that to hole_id A1.
        """
        holes = payload.get("holes") or []
        if not holes:
            return False

        def as_float(item: Dict[str, Any], key: str) -> Optional[float]:
            value = item.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def board_point(item: Dict[str, Any]) -> Optional[Tuple[float, float]]:
            x = as_float(item, "x_warp")
            y = as_float(item, "y_warp")
            if x is None or y is None:
                x = as_float(item, "x_image")
                y = as_float(item, "y_image")
            if x is None or y is None:
                return None
            return (x, y)

        terminal_rows = set("ABCDEFGHIJ")
        main_items = [
            item
            for item in holes
            if str(item.get("row") or "").upper() in terminal_rows
        ]
        if not main_items:
            return False

        main_cols = [
            int(item.get("column"))
            for item in main_items
            if item.get("column") is not None
        ]
        if not main_cols:
            return False
        row_count = max(main_cols) + 1
        col_names = list("abcdefghij")
        main_grid = np.full((row_count, len(col_names), 2), np.nan, dtype=np.float32)
        observed_main = np.zeros((row_count, len(col_names)), dtype=bool)

        for item in main_items:
            point = board_point(item)
            if point is None:
                continue
            row_letter = str(item.get("row") or "").strip().lower()
            if row_letter not in col_names:
                continue
            try:
                row_idx = int(item.get("column"))
            except (TypeError, ValueError):
                continue
            if row_idx < 0 or row_idx >= row_count:
                continue
            col_idx = col_names.index(row_letter)
            main_grid[row_idx, col_idx] = point
            observed_main[row_idx, col_idx] = True

        if not observed_main.any():
            return False

        row_coords = self._median_axis_from_grid(
            main_grid[:, :, 0],
            observed_main,
            axis=1,
        )
        col_coords = self._median_axis_from_grid(
            main_grid[:, :, 1],
            observed_main,
            axis=0,
        )
        if row_coords is None or col_coords is None:
            return False

        for row_idx in range(row_count):
            for col_idx in range(len(col_names)):
                if not observed_main[row_idx, col_idx]:
                    main_grid[row_idx, col_idx] = [row_coords[row_idx], col_coords[col_idx]]

        rail_items = [
            item
            for item in holes
            if str(item.get("row") or "").lower()
            in {"top_pos", "top_neg", "bottom_pos", "bottom_neg"}
        ]
        rail_cols = [
            int(item.get("column"))
            for item in rail_items
            if item.get("column") is not None
        ]
        rail_count = max(rail_cols) + 1 if rail_cols else self._expected_rail_rows
        top_grid = np.full((rail_count, 2, 2), np.nan, dtype=np.float32)
        bot_grid = np.full((rail_count, 2, 2), np.nan, dtype=np.float32)
        observed_top = np.zeros((rail_count, 2), dtype=bool)
        observed_bot = np.zeros((rail_count, 2), dtype=bool)
        rail_name_to_target = {
            "top_pos": (top_grid, observed_top, 0),
            "top_neg": (top_grid, observed_top, 1),
            "bottom_pos": (bot_grid, observed_bot, 0),
            "bottom_neg": (bot_grid, observed_bot, 1),
        }

        for item in rail_items:
            point = board_point(item)
            if point is None:
                continue
            target = rail_name_to_target.get(str(item.get("row") or "").lower())
            if target is None:
                continue
            try:
                row_idx = int(item.get("column"))
            except (TypeError, ValueError):
                continue
            if row_idx < 0 or row_idx >= rail_count:
                continue
            grid, observed, rail_idx = target
            grid[row_idx, rail_idx] = point
            observed[row_idx, rail_idx] = True

        rail_row_coords = None
        top_rails = None
        bot_rails = None
        if rail_items and (observed_top.any() or observed_bot.any()):
            rail_x_values = np.concatenate(
                [top_grid[:, :, 0], bot_grid[:, :, 0]],
                axis=1,
            )
            rail_observed = np.concatenate([observed_top, observed_bot], axis=1)
            rail_row_coords = self._median_axis_from_grid(
                rail_x_values,
                rail_observed,
                axis=1,
            )
            if rail_row_coords is None:
                rail_row_coords = np.linspace(
                    float(row_coords[0]),
                    float(row_coords[-1]),
                    rail_count,
                ).astype(np.float32)

            top_rails = self._rail_levels_from_grid(top_grid, observed_top)
            bot_rails = self._rail_levels_from_grid(bot_grid, observed_bot)
            for row_idx in range(rail_count):
                for rail_idx in range(2):
                    if top_rails is not None and not observed_top[row_idx, rail_idx]:
                        top_grid[row_idx, rail_idx] = [
                            rail_row_coords[row_idx],
                            top_rails[rail_idx],
                        ]
                    if bot_rails is not None and not observed_bot[row_idx, rail_idx]:
                        bot_grid[row_idx, rail_idx] = [
                            rail_row_coords[row_idx],
                            bot_rails[rail_idx],
                        ]

        self.rows = row_count
        self._expected_rows = row_count
        self._expected_rail_rows = rail_count
        self.total_cols = len(col_names)
        self._col_names = col_names
        self._landscape = True
        self._synthetic_grid = False
        self._detected_hole_map = True
        self.is_calibrated = True
        self._row_coords = row_coords.astype(np.float32)
        self._col_coords = col_coords.astype(np.float32)
        self._grid_matrix = main_grid.astype(np.float32)
        self._observed_main_mask = observed_main
        self._valid_main_mask = observed_main.copy()
        self._rail_row_coords = (
            rail_row_coords.astype(np.float32)
            if rail_row_coords is not None
            else None
        )
        self._top_rails = [float(v) for v in top_rails] if top_rails is not None else []
        self._bot_rails = [float(v) for v in bot_rails] if bot_rails is not None else []
        self._top_rail_matrix = (
            top_grid.astype(np.float32)
            if top_rails is not None
            else None
        )
        self._bot_rail_matrix = (
            bot_grid.astype(np.float32)
            if bot_rails is not None
            else None
        )
        self._observed_top_mask = observed_top if top_rails is not None else None
        self._observed_bot_mask = observed_bot if bot_rails is not None else None
        self._valid_top_mask = observed_top.copy() if top_rails is not None else None
        self._valid_bot_mask = observed_bot.copy() if bot_rails is not None else None

        self.hole_centers = []
        for item in holes:
            point = board_point(item)
            if point is not None:
                self.hole_centers.append(tuple(map(float, point)))

        warped_size = payload.get("warped_size") or {}
        quad = payload.get("quad_tl_tr_br_bl")
        if quad and warped_size.get("width") and warped_size.get("height"):
            try:
                src = np.asarray(quad, dtype=np.float32)
                width = float(warped_size["width"])
                height = float(warped_size["height"])
                dst = np.asarray(
                    [
                        [0.0, 0.0],
                        [width, 0.0],
                        [width, height],
                        [0.0, height],
                    ],
                    dtype=np.float32,
                )
                self._perspective_matrix = cv2.getPerspectiveTransform(src, dst)
                self._inv_perspective = cv2.getPerspectiveTransform(dst, src)
            except Exception as exc:
                logger.warning(
                    "[Calibrator] Could not load detected hole homography: %s",
                    exc,
                )
                self._perspective_matrix = None
                self._inv_perspective = None
        else:
            self._perspective_matrix = None
            self._inv_perspective = None

        self._compute_grid_params()
        logger.info(
            "[Calibrator] Loaded detected hole map: main=%d×%d, rails=%d",
            row_count,
            len(col_names),
            rail_count,
        )
        return self.is_grid_ready

    @staticmethod
    def _median_axis_from_grid(
        values: np.ndarray,
        observed: np.ndarray,
        *,
        axis: int,
    ) -> Optional[np.ndarray]:
        count = values.shape[0] if axis == 1 else values.shape[1]
        centers: List[float] = []
        for idx in range(count):
            mask = observed[idx, :] if axis == 1 else observed[:, idx]
            vals = values[idx, :] if axis == 1 else values[:, idx]
            finite = vals[mask & np.isfinite(vals)]
            if len(finite) == 0:
                return None
            centers.append(float(np.median(finite)))
        return np.asarray(centers, dtype=np.float32)

    @staticmethod
    def _rail_levels_from_grid(
        grid: np.ndarray,
        observed: np.ndarray,
    ) -> Optional[np.ndarray]:
        if not observed.any():
            return None
        levels: List[float] = []
        for rail_idx in range(grid.shape[1]):
            vals = grid[:, rail_idx, 1]
            mask = observed[:, rail_idx] & np.isfinite(vals)
            if not mask.any():
                return None
            levels.append(float(np.median(vals[mask])))
        return np.asarray(levels, dtype=np.float32)

    def ensure_calibrated(self, image: np.ndarray) -> bool:
        """确保校准器已校准: 主路径只走队友方案，失败后直接退 synthetic grid。"""
        if self.is_grid_ready:
            return True

        if self.auto_calibrate(image):
            logger.info("[Calibrator] Visual auto-calibrate succeeded")
            return True

        logger.info("[Calibrator] Teammate calibration failed, using synthetic grid")
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
        self._detected_hole_map = False
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
