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

from breadboard_detect import build_holes, fit_grid, score_image
from breadboard_detect_white_region import (
    _locate_score_peak,
    _refine_quad_from_hole_lattice,
    detect_white_region_quad,
)


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
        """自动校准: 只走队友的白区域 + 孔洞晶格细化 + detected-hole payload 主路径。"""
        try:
            self._synthetic_grid = False
            corners, _, _ = detect_white_region_quad(
                image,
                fallback_auto=True,
            )
            refined_corners, _ = _refine_quad_from_hole_lattice(
                image,
                corners,
                main_columns=self._expected_rows,
            )
            self.calibrate(refined_corners)
            return self._load_teammate_detected_holes(image, refined_corners)
        except Exception as e:
            logger.warning(f"[Calibrator] Auto-calibrate failed: {e}")
            return False

    def _load_teammate_detected_holes(
        self,
        image: np.ndarray,
        corners: np.ndarray,
    ) -> bool:
        """Run teammate breadboard pipeline and load its detected hole payload."""
        if self._perspective_matrix is None or self._inv_perspective is None:
            return False

        warped = self.warp(image)
        model = fit_grid(warped, self._expected_rows)
        warped_score = score_image(warped)
        holes = build_holes(
            model,
            self._inv_perspective,
            warped_score,
            self._expected_rows,
        )
        if not holes:
            return False
        holes = self._refine_detected_hole_payload(holes, warped_score)

        payload = {
            "holes": holes,
            "warped_size": {
                "width": warped.shape[1],
                "height": warped.shape[0],
            },
            "quad_tl_tr_br_bl": [
                [float(x), float(y)]
                for x, y in corners.astype(np.float32)
            ],
        }
        return self.load_detected_holes(payload)

    def _refine_detected_hole_payload(
        self,
        holes: List[Dict[str, Any]],
        warped_score: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Snap teammate grid holes to local score peaks in warped space."""
        if not holes:
            return holes

        visibility_scores = np.asarray(
            [float(item.get("visible_score", 0.0)) for item in holes],
            dtype=np.float32,
        )
        visibility_cut = max(8.0, float(np.percentile(visibility_scores, 30)))
        search_radius = int(np.clip(round(min(warped_score.shape[:2]) * 0.010), 8, 14))
        refined: List[Dict[str, Any]] = []
        for item in holes:
            updated = dict(item)
            visible_score = float(updated.get("visible_score", 0.0))
            if visible_score < visibility_cut:
                refined.append(updated)
                continue

            found = _locate_score_peak(
                warped_score,
                float(updated["x_warp"]),
                float(updated["y_warp"]),
                radius=search_radius,
                min_peak=12.0,
            )
            if found is None:
                refined.append(updated)
                continue

            refined_center, peak_value = found
            shift = float(
                np.linalg.norm(
                    refined_center
                    - np.array(
                        [float(updated["x_warp"]), float(updated["y_warp"])],
                        dtype=np.float32,
                    )
                )
            )
            if shift > search_radius * 0.95:
                refined.append(updated)
                continue

            updated["x_warp"] = round(float(refined_center[0]), 3)
            updated["y_warp"] = round(float(refined_center[1]), 3)
            if self._inv_perspective is not None:
                src = cv2.perspectiveTransform(
                    np.array([[[updated["x_warp"], updated["y_warp"]]]], dtype=np.float32),
                    self._inv_perspective,
                )[0, 0]
                updated["x_image"] = round(float(src[0]), 3)
                updated["y_image"] = round(float(src[1]), 3)
            updated["visible_score"] = round(max(visible_score, float(peak_value)), 3)
            refined.append(updated)
        return refined

    def warp(self, image: np.ndarray) -> np.ndarray:
        """透视变换"""
        if self._perspective_matrix is None:
            return image
        return cv2.warpPerspective(image, self._perspective_matrix, (800, 600))

    # ============================================================
    # 空间哈希 (Spatial Hashing)
    # 队友方案已提供标准孔位 payload, 这里仅保留坐标查找层
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
                point = (
                    self._grid_matrix[ri, ci]
                    if self._grid_matrix is not None
                    else np.array([self._row_coords[ri], self._col_coords[ci]], dtype=np.float32)
                )
                dist = float(
                    (float(point[0]) - row_val) ** 2
                    + (float(point[1]) - col_val) ** 2
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

    def representative_pitch_px(self) -> float:
        """Median grid pitch in board-plane pixels, used to normalize snap distance."""
        if self._row_coords is None or self._col_coords is None:
            return 10.0
        row_pitch = self._median_pitch(self._row_coords)
        col_pitch = self._median_pitch(self._col_coords)
        pitch = min(row_pitch, col_pitch)
        return float(pitch) if pitch > 1e-3 else 10.0

    def board_point_to_logic_candidates_scored(
        self, board_x: float, board_y: float, k: int = 5,
    ) -> List[Tuple[Tuple[str, str], float]]:
        """Like board_point_to_logic_candidates, but each item carries the snap distance.

        Returns a list of `(logic_loc, distance_px)` tuples in the same order as
        the unscored API. Distance is the Euclidean distance from the input
        board-plane point to the candidate hole's stored grid coordinate; if a
        candidate has no resolvable board point, distance is `inf` so it sorts
        last but stays available as a fallback.
        """
        candidates = self.board_point_to_logic_candidates(board_x, board_y, k=k)
        if not candidates:
            return []
        scored: List[Tuple[Tuple[str, str], float]] = []
        for logic_loc in candidates:
            target = self.logic_to_board_point(logic_loc)
            if target is None:
                scored.append((logic_loc, float("inf")))
                continue
            dx = float(target[0]) - float(board_x)
            dy = float(target[1]) - float(board_y)
            scored.append((logic_loc, float((dx * dx + dy * dy) ** 0.5)))
        return scored

    def frame_pixel_to_logic_candidates_scored(
        self, px: float, py: float, k: int = 5,
    ) -> List[Tuple[Tuple[str, str], float]]:
        """Frame-pixel variant of `board_point_to_logic_candidates_scored`.

        Routes through the unscored API so test/runtime monkey-patches still apply,
        then computes the snap distance in the board plane.
        """
        candidates = self.frame_pixel_to_logic_candidates(px, py, k=k)
        if not candidates:
            return []
        board_x, board_y = self.frame_pixel_to_board_point(px, py)
        scored: List[Tuple[Tuple[str, str], float]] = []
        for logic_loc in candidates:
            target = self.logic_to_board_point(logic_loc)
            if target is None:
                scored.append((logic_loc, float("inf")))
                continue
            dx = float(target[0]) - float(board_x)
            dy = float(target[1]) - float(board_y)
            scored.append((logic_loc, float((dx * dx + dy * dy) ** 0.5)))
        return scored

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
            if self._top_rail_matrix is not None and row_idx < self._top_rail_matrix.shape[0] and rail_idx < self._top_rail_matrix.shape[1]:
                point = self._top_rail_matrix[row_idx, rail_idx]
                return (
                    (float(point[0]), float(point[1]))
                    if self._landscape
                    else (float(point[1]), float(point[0]))
                )
            if row_idx >= len(rail_rows) or rail_idx >= len(self._top_rails):
                return None
            row_val = float(rail_rows[row_idx])
            col_val = float(self._top_rails[rail_idx])
            return (row_val, col_val) if self._landscape else (col_val, row_val)

        if col_lower in {"rail_bot+", "rail_bot-", "rp", "rn"}:
            rail_idx = 0 if col_lower in {"rail_bot+", "rp"} else 1
            if self._bot_rail_matrix is not None and row_idx < self._bot_rail_matrix.shape[0] and rail_idx < self._bot_rail_matrix.shape[1]:
                point = self._bot_rail_matrix[row_idx, rail_idx]
                return (
                    (float(point[0]), float(point[1]))
                    if self._landscape
                    else (float(point[1]), float(point[0]))
                )
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
        if self._grid_matrix is not None and row_idx < self._grid_matrix.shape[0] and col_idx < self._grid_matrix.shape[1]:
            point = self._grid_matrix[row_idx, col_idx]
            return (
                (float(point[0]), float(point[1]))
                if self._landscape
                else (float(point[1]), float(point[0]))
            )

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
        """Load detected breadboard holes exported by bread_detect as the active grid."""
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
                        [max(width - 1.0, 0.0), 0.0],
                        [max(width - 1.0, 0.0), max(height - 1.0, 0.0)],
                        [0.0, max(height - 1.0, 0.0)],
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
