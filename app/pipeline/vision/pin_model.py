"""
Component ROI pin detector.

这一层承担两件事:
1. 为后续真实第二模型提供稳定接口
2. 在模型未接入前, 基于 ROI 图像内容做启发式 pin 定位
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.pipeline.vision.pin_schema import default_pin_names

logger = logging.getLogger(__name__)


@dataclass
class PinPrediction:
    pin_id: int
    pin_name: str
    keypoint: tuple[float, float] | None
    confidence: float
    visibility: int
    source: str
    metadata: dict[str, object]


class PinRoiDetector:
    """ROI pin detector.

    当前优先使用真实模型接口; 当第二模型尚未接入时, 回退到
    基于 ROI 图像内容的启发式 pin 定位, 不再使用固定比例占位点。
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self.model = None
        if model_path:
            self.load(model_path)

    @property
    def interface_version(self) -> str:
        return "pin_detector_v1"

    @property
    def backend_type(self) -> str:
        return "yolo_pose"

    @property
    def backend_mode(self) -> str:
        return "model" if self.model is not None else "heuristic_fallback"

    def load(self, model_path: str | None = None) -> bool:
        """预留真实第二模型加载入口.

        当前支持在提供兼容推理接口的模型时挂接; 若加载失败,
        自动回退到启发式实现。
        """
        path = model_path or self.model_path
        if not path:
            return False
        try:
            from ultralytics import YOLO

            self.model = YOLO(path)
            self.model_path = path
            logger.info("[PinDetector] Loaded ROI pin model: %s", path)
            return True
        except Exception as exc:
            logger.warning("[PinDetector] Failed to load ROI pin model %s: %s", path, exc)
            self.model = None
            return False

    def predict_component_pins(
        self,
        *,
        component_id: str,
        component_type: str,
        package_type: str,
        pin_schema_id: str,
        roi_image: np.ndarray | None,
        roi_offset: tuple[int, int],
        view_id: str = "top",
        confidence: float = 1.0,
    ) -> list[PinPrediction]:
        pin_count = _infer_pin_count(component_type, package_type)
        pin_names = default_pin_names(component_type, pin_count)

        # 优先走真实第二模型; 模型未接入或推理失败时再退回图像启发式,
        # 这样不会把占位逻辑继续混进主链接口。
        model_keypoints = self._predict_with_model(
            roi_image=roi_image,
            component_type=component_type,
            package_type=package_type,
            pin_count=pin_count,
        )
        if model_keypoints is not None:
            ox, oy = roi_offset
            return [
                PinPrediction(
                    pin_id=idx + 1,
                    pin_name=pin_names[idx],
                    keypoint=(
                        (
                            float(model_keypoints[idx][0] + ox),
                            float(model_keypoints[idx][1] + oy),
                        )
                        if idx < len(model_keypoints) and model_keypoints[idx] is not None
                        else None
                    ),
                    confidence=confidence,
                    visibility=2 if idx < len(model_keypoints) and model_keypoints[idx] is not None else 0,
                    source="model",
                    metadata={
                        "backend_type": self.backend_type,
                        "backend_mode": "model",
                        "interface_version": self.interface_version,
                        "view_id": view_id,
                    },
                )
                for idx in range(pin_count)
            ]

        keypoints, heuristic_score = self._heuristic_keypoints(
            component_type=component_type,
            package_type=package_type,
            pin_schema_id=pin_schema_id,
            roi_image=roi_image,
            roi_offset=roi_offset,
            pin_count=pin_count,
        )
        return [
            PinPrediction(
                pin_id=idx + 1,
                pin_name=pin_names[idx],
                keypoint=keypoints[idx] if idx < len(keypoints) else None,
                confidence=min(1.0, max(0.1, confidence * heuristic_score)),
                visibility=2 if idx < len(keypoints) and keypoints[idx] is not None else 0,
                source="heuristic_fallback",
                metadata={
                    "backend_type": self.backend_type,
                    "backend_mode": "heuristic_fallback",
                    "interface_version": self.interface_version,
                    "heuristic_score": round(float(heuristic_score), 4),
                    "view_id": view_id,
                },
            )
            for idx in range(pin_count)
        ]

    def _predict_with_model(
        self,
        *,
        roi_image: np.ndarray | None,
        component_type: str,
        package_type: str,
        pin_count: int,
    ) -> list[tuple[float, float] | None] | None:
        """真实第二模型接口.

        约定:
        - 输入为单组件 ROI
        - 输出应为与 pin schema 对齐的有序 keypoints
        当前若无模型或解析失败, 返回 None 走启发式路径。
        """
        if self.model is None or roi_image is None or roi_image.size == 0:
            return None
        try:
            results = self.model(roi_image, verbose=False, device=self.device)
            if not results:
                return None
            first = results[0]
            if not hasattr(first, "keypoints") or first.keypoints is None:
                return None
            xy = first.keypoints.xy
            if xy is None or len(xy) == 0:
                return None
            points = xy[0].cpu().numpy()
            ordered: list[tuple[float, float] | None] = []
            for idx in range(min(pin_count, len(points))):
                ordered.append((float(points[idx][0]), float(points[idx][1])))
            while len(ordered) < pin_count:
                ordered.append(None)
            return ordered
        except Exception as exc:
            logger.warning(
                "[PinDetector] Model inference failed for %s/%s: %s",
                component_type,
                package_type,
                exc,
            )
            return None

    def _heuristic_keypoints(
        self,
        *,
        component_type: str,
        package_type: str,
        pin_schema_id: str,
        roi_image: np.ndarray | None,
        roi_offset: tuple[int, int],
        pin_count: int,
    ) -> tuple[list[tuple[float, float] | None], float]:
        """基于 ROI 内容的启发式 pin 定位.

        核心思路:
        - 先用亮板暗件的先验提取前景 mask
        - 再沿主轴做投影, 找有效 pin 位置
        - 如果 mask 质量不足, 才退回到前景包围盒几何
        """
        if roi_image is None or roi_image.size == 0:
            return [None] * pin_count, 0.1

        mask = _build_foreground_mask(roi_image)
        bbox = _foreground_bbox(mask)
        if bbox is None:
            h, w = roi_image.shape[:2]
            bbox = (0, 0, w - 1, h - 1)
            score = 0.25
        else:
            score = _mask_quality(mask)

        orientation = _major_orientation(mask, bbox)
        local_points = _estimate_pin_points_from_mask(
            mask=mask,
            bbox=bbox,
            orientation=orientation,
            pin_count=pin_count,
            component_type=component_type,
            package_type=package_type,
        )
        if not any(point is not None for point in local_points):
            local_points = _fallback_points_from_bbox(
                bbox=bbox,
                orientation=orientation,
                pin_count=pin_count,
                component_type=component_type,
                package_type=package_type,
            )
            score = min(score, 0.35)

        ox, oy = roi_offset
        global_points = []
        for point in local_points:
            if point is None:
                global_points.append(None)
            else:
                global_points.append((float(point[0] + ox), float(point[1] + oy)))
        return global_points, score


def _infer_pin_count(component_type: str, package_type: str) -> int:
    ctype = component_type.lower()
    if ctype == "ic" and package_type == "dip8":
        return 2
    if ctype == "potentiometer":
        return 3
    return 2


def _build_foreground_mask(roi_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, dark_mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    mask = cv2.bitwise_or(dark_mask, edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def _foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 10 or len(ys) < 10:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _mask_quality(mask: np.ndarray) -> float:
    nonzero = float(np.count_nonzero(mask))
    total = float(mask.shape[0] * mask.shape[1]) or 1.0
    density = nonzero / total
    return float(min(0.9, max(0.35, density * 4.0)))


def _major_orientation(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1 + 1)
    height = max(1, y2 - y1 + 1)
    return "horizontal" if width >= height else "vertical"


def _estimate_pin_points_from_mask(
    *,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    orientation: str,
    pin_count: int,
    component_type: str,
    package_type: str,
) -> list[tuple[float, float] | None]:
    x1, y1, x2, y2 = bbox
    if orientation == "horizontal":
        band_y1 = max(0, y1 - max(1, (y2 - y1) // 5))
        band_y2 = min(mask.shape[0], y2 + max(2, (y2 - y1) // 5) + 1)
        band = mask[band_y1:band_y2, x1:x2 + 1]
        proj = band.sum(axis=0)
        active = _active_indices(proj)
        if len(active) < 2:
            return _fallback_points_from_bbox(
                bbox=bbox,
                orientation=orientation,
                pin_count=pin_count,
                component_type=component_type,
                package_type=package_type,
            )
        xs = _select_positions(active, pin_count)
        if component_type.lower() == "ic" and package_type == "dip8":
            side_x = _choose_side_x(mask, bbox, vertical=False)
            ys = _select_positions(_active_indices(mask[y1:y2 + 1, x1:x2 + 1].sum(axis=1)), 2, offset=y1)
            return [(side_x, ys[0]), (side_x, ys[-1])]
        return [
            (float(x), float(_centroid_y(mask, x, y1, y2)))
            for x in xs
        ]

    band_x1 = max(0, x1 - max(1, (x2 - x1) // 5))
    band_x2 = min(mask.shape[1], x2 + max(2, (x2 - x1) // 5) + 1)
    band = mask[y1:y2 + 1, band_x1:band_x2]
    proj = band.sum(axis=1)
    active = _active_indices(proj)
    if len(active) < 2:
        return _fallback_points_from_bbox(
            bbox=bbox,
            orientation=orientation,
            pin_count=pin_count,
            component_type=component_type,
            package_type=package_type,
        )
    ys = _select_positions(active, pin_count, offset=y1)
    if component_type.lower() == "ic" and package_type == "dip8":
        side_x = _choose_side_x(mask, bbox, vertical=True)
        return [(side_x, ys[0]), (side_x, ys[-1])]
    return [
        (float(_centroid_x(mask, y, x1, x2)), float(y))
        for y in ys
    ]


def _active_indices(projection: np.ndarray) -> np.ndarray:
    if projection.size == 0:
        return np.array([], dtype=int)
    threshold = max(float(projection.max()) * 0.25, 1.0)
    return np.where(projection >= threshold)[0]


def _select_positions(indices: np.ndarray, count: int, offset: int = 0) -> list[float]:
    if len(indices) == 0:
        return []
    if count == 2:
        return [float(indices[0] + offset), float(indices[-1] + offset)]
    if count == 3:
        q = np.quantile(indices, [0.1, 0.5, 0.9])
        return [float(v + offset) for v in q]
    q = np.linspace(0.0, 1.0, count)
    vals = np.quantile(indices, q)
    return [float(v + offset) for v in vals]


def _centroid_y(mask: np.ndarray, x: float, y1: int, y2: int) -> float:
    xi = int(round(x))
    x_left = max(0, xi - 2)
    x_right = min(mask.shape[1], xi + 3)
    region = mask[y1:y2 + 1, x_left:x_right]
    ys, _ = np.where(region > 0)
    if len(ys) == 0:
        return float((y1 + y2) / 2.0)
    return float(y1 + ys.mean())


def _centroid_x(mask: np.ndarray, y: float, x1: int, x2: int) -> float:
    yi = int(round(y))
    y_top = max(0, yi - 2)
    y_bottom = min(mask.shape[0], yi + 3)
    region = mask[y_top:y_bottom, x1:x2 + 1]
    _, xs = np.where(region > 0)
    if len(xs) == 0:
        return float((x1 + x2) / 2.0)
    return float(x1 + xs.mean())


def _choose_side_x(mask: np.ndarray, bbox: tuple[int, int, int, int], vertical: bool) -> float:
    x1, y1, x2, y2 = bbox
    width = max(2, x2 - x1 + 1)
    height = max(2, y2 - y1 + 1)
    if vertical:
        band_w = max(2, width // 4)
        left_score = mask[y1:y2 + 1, x1:x1 + band_w].sum()
        right_score = mask[y1:y2 + 1, x2 - band_w + 1:x2 + 1].sum()
        if left_score >= right_score:
            return float(x1 + band_w / 2.0)
        return float(x2 - band_w / 2.0)

    band_h = max(2, height // 4)
    top_score = mask[y1:y1 + band_h, x1:x2 + 1].sum()
    bottom_score = mask[y2 - band_h + 1:y2 + 1, x1:x2 + 1].sum()
    if top_score >= bottom_score:
        return float((x1 + x2) / 2.0)
    return float((x1 + x2) / 2.0)


def _fallback_points_from_bbox(
    *,
    bbox: tuple[int, int, int, int],
    orientation: str,
    pin_count: int,
    component_type: str,
    package_type: str,
) -> list[tuple[float, float] | None]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if component_type.lower() == "ic" and package_type == "dip8":
        if orientation == "horizontal":
            return [(x1 + width * 0.2, cy), (x2 - width * 0.2, cy)]
        return [(cx, y1 + height * 0.2), (cx, y2 - height * 0.2)]

    if pin_count == 3:
        if orientation == "horizontal":
            return [
                (x1 + width * 0.15, cy),
                (cx, cy),
                (x2 - width * 0.15, cy),
            ]
        return [
            (cx, y1 + height * 0.15),
            (cx, cy),
            (cx, y2 - height * 0.15),
        ]

    if orientation == "horizontal":
        return [(x1, cy), (x2, cy)]
    return [(cx, y1), (cx, y2)]
