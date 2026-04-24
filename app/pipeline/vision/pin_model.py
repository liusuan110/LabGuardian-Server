"""
Component ROI pin detector.

这一层承担两件事:
1. 为后续真实第二模型提供稳定接口
2. 在模型未接入前, 基于 ROI 图像内容做启发式 pin 定位
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.pipeline.vision.label_mapping import default_pin_count
from app.pipeline.vision.model_inspector import inspect_yolo_weight
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


@dataclass
class ModelPinParseResult:
    ordered_keypoints: list[tuple[float, float] | None]
    raw_keypoint_count: int
    raw_visible_keypoint_count: int
    used_keypoint_count: int
    extra_keypoints_ignored: int
    ignored_keypoints_reason: str


@dataclass
class WireTraceResult:
    ordered_keypoints: list[tuple[float, float] | None]
    score: float
    color_mode: str
    target_hue: float | None
    target_sat: float | None
    component_area: int
    span_px: float
    center_overlap_px: int


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
        self.model_contract: dict[str, object] = {
            "path": "",
            "exists": False,
            "task": "unknown",
            "model_class": "unknown",
            "names": [],
            "kpt_shape": None,
            "loaded": False,
        }
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
        contract = inspect_yolo_weight(path)
        contract["loaded"] = False
        self.model_contract = contract
        task = str(contract.get("task") or "unknown")
        if task in {"detect", "obb"}:
            logger.error("[PinDetector] Refusing non-pose weight for pin detector: %s (task=%s)", path, task)
            self.model = None
            return False
        try:
            from ultralytics import YOLO

            self.model = YOLO(path)
            self.model_path = path
            self.model_contract["loaded"] = True
            logger.info("[PinDetector] Loaded ROI pin model: %s", path)
            return True
        except Exception as exc:
            logger.warning("[PinDetector] Failed to load ROI pin model %s: %s", path, exc)
            self.model = None
            self.model_contract["loaded"] = False
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

        wire_trace = self._trace_wire_keypoints(
            roi_image=roi_image,
            component_type=component_type,
            package_type=package_type,
            pin_count=pin_count,
        )
        if wire_trace is not None:
            ox, oy = roi_offset
            return [
                PinPrediction(
                    pin_id=idx + 1,
                    pin_name=pin_names[idx],
                    keypoint=(
                        (
                            float(wire_trace.ordered_keypoints[idx][0] + ox),
                            float(wire_trace.ordered_keypoints[idx][1] + oy),
                        )
                        if idx < len(wire_trace.ordered_keypoints) and wire_trace.ordered_keypoints[idx] is not None
                        else None
                    ),
                    confidence=min(1.0, max(0.1, confidence * wire_trace.score)),
                    visibility=2 if idx < len(wire_trace.ordered_keypoints) and wire_trace.ordered_keypoints[idx] is not None else 0,
                    source="wire_color_trace",
                    metadata={
                        "backend_type": self.backend_type,
                        "backend_mode": "wire_color_trace",
                        "interface_version": self.interface_version,
                        "view_id": view_id,
                        "trace_score": round(float(wire_trace.score), 4),
                        "trace_color_mode": wire_trace.color_mode,
                        "trace_target_hue": None if wire_trace.target_hue is None else round(float(wire_trace.target_hue), 2),
                        "trace_target_sat": None if wire_trace.target_sat is None else round(float(wire_trace.target_sat), 2),
                        "trace_component_area": int(wire_trace.component_area),
                        "trace_span_px": round(float(wire_trace.span_px), 2),
                        "trace_center_overlap_px": int(wire_trace.center_overlap_px),
                        "raw_keypoint_count": 0,
                        "raw_visible_keypoint_count": 0,
                        "used_keypoint_count": sum(1 for point in wire_trace.ordered_keypoints if point is not None),
                        "extra_keypoints_ignored": 0,
                        "ignored_keypoints_reason": "",
                    },
                )
                for idx in range(pin_count)
            ]

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
                            float(model_keypoints.ordered_keypoints[idx][0] + ox),
                            float(model_keypoints.ordered_keypoints[idx][1] + oy),
                        )
                        if idx < len(model_keypoints.ordered_keypoints) and model_keypoints.ordered_keypoints[idx] is not None
                        else None
                    ),
                    confidence=confidence,
                    visibility=2 if idx < len(model_keypoints.ordered_keypoints) and model_keypoints.ordered_keypoints[idx] is not None else 0,
                    source="model",
                    metadata={
                        "backend_type": self.backend_type,
                        "backend_mode": "model",
                        "interface_version": self.interface_version,
                        "view_id": view_id,
                        "raw_keypoint_count": model_keypoints.raw_keypoint_count,
                        "raw_visible_keypoint_count": model_keypoints.raw_visible_keypoint_count,
                        "used_keypoint_count": model_keypoints.used_keypoint_count,
                        "extra_keypoints_ignored": model_keypoints.extra_keypoints_ignored,
                        "ignored_keypoints_reason": model_keypoints.ignored_keypoints_reason,
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
                    "raw_keypoint_count": 0,
                    "raw_visible_keypoint_count": 0,
                    "used_keypoint_count": sum(1 for point in keypoints if point is not None),
                    "extra_keypoints_ignored": 0,
                    "ignored_keypoints_reason": "",
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
    ) -> ModelPinParseResult | None:
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
            conf_obj = getattr(first.keypoints, "conf", None)
            confs = None
            if conf_obj is not None and len(conf_obj) > 0:
                confs = conf_obj[0].cpu().numpy()
            return _parse_model_keypoints(points=points, confs=confs, pin_count=pin_count)
        except Exception as exc:
            logger.warning(
                "[PinDetector] Model inference failed for %s/%s: %s",
                component_type,
                package_type,
                exc,
            )
            return None

    def _trace_wire_keypoints(
        self,
        *,
        roi_image: np.ndarray | None,
        component_type: str,
        package_type: str,
        pin_count: int,
    ) -> WireTraceResult | None:
        if not _is_wire_component(component_type, package_type):
            return None
        if roi_image is None or roi_image.size == 0 or pin_count != 2:
            return None
        return _trace_wire_endpoints(roi_image)

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
    return default_pin_count(component_type, package_type)


def _is_wire_component(component_type: str, package_type: str) -> bool:
    component_key = str(component_type or "").lower()
    package_key = str(package_type or "").lower()
    return component_key in {"wire", "jumper_wire"} or package_key == "jumper_wire_2pin"


def _is_valid_model_keypoint(x: float, y: float, score: float | None) -> bool:
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    if score is not None and (not np.isfinite(score) or score <= 0.0):
        return False
    # 当前 pose 训练为全局 kpt_shape=[3,3]。
    # 对两脚器件，第 3 个点是 padding 槽位，通常会落成 (0, 0, 0)。
    if abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6:
        return False
    return True


def _parse_model_keypoints(
    *,
    points: np.ndarray,
    confs: np.ndarray | None,
    pin_count: int,
) -> ModelPinParseResult:
    ordered: list[tuple[float, float] | None] = []
    raw_keypoint_count = int(len(points))
    raw_visible_keypoint_count = 0
    used_keypoint_count = 0

    for idx in range(raw_keypoint_count):
        score = float(confs[idx]) if confs is not None and idx < len(confs) else None
        x = float(points[idx][0])
        y = float(points[idx][1])
        if _is_valid_model_keypoint(x, y, score):
            raw_visible_keypoint_count += 1

    for idx in range(pin_count):
        if idx >= raw_keypoint_count:
            ordered.append(None)
            continue
        score = float(confs[idx]) if confs is not None and idx < len(confs) else None
        x = float(points[idx][0])
        y = float(points[idx][1])
        if _is_valid_model_keypoint(x, y, score):
            ordered.append((x, y))
            used_keypoint_count += 1
        else:
            ordered.append(None)

    extra_keypoints_ignored = max(0, raw_keypoint_count - pin_count)
    if extra_keypoints_ignored and pin_count == 2:
        ignored_reason = "schema_padding_for_2pin"
    elif extra_keypoints_ignored:
        ignored_reason = "schema_excess_keypoints"
    else:
        ignored_reason = ""

    return ModelPinParseResult(
        ordered_keypoints=ordered,
        raw_keypoint_count=raw_keypoint_count,
        raw_visible_keypoint_count=raw_visible_keypoint_count,
        used_keypoint_count=used_keypoint_count,
        extra_keypoints_ignored=extra_keypoints_ignored,
        ignored_keypoints_reason=ignored_reason,
    )


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


def _trace_wire_endpoints(roi_image: np.ndarray) -> WireTraceResult | None:
    h, w = roi_image.shape[:2]
    if h < 12 or w < 12:
        return None

    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    color_mask, color_meta = _build_wire_color_mask(hsv)
    if color_mask is None:
        return None

    component_mask, center_overlap = _select_wire_component(color_mask)
    if component_mask is None:
        return None

    points = np.column_stack(np.where(component_mask > 0))
    if len(points) < 12:
        return None

    pts_xy = np.column_stack((points[:, 1].astype(np.float64), points[:, 0].astype(np.float64)))
    center = pts_xy.mean(axis=0)
    centered = pts_xy - center
    cov = np.cov(centered, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
        return None
    major_axis = eigvecs[:, int(np.argmax(eigvals))]
    axis_norm = float(np.linalg.norm(major_axis))
    if axis_norm < 1e-6:
        return None
    major_axis = major_axis / axis_norm

    projections = centered[:, 0] * float(major_axis[0]) + centered[:, 1] * float(major_axis[1])
    start = pts_xy[int(np.argmin(projections))]
    end = pts_xy[int(np.argmax(projections))]

    foreground_mask = _build_foreground_mask(roi_image)
    start = _extend_endpoint_along_axis(
        foreground_mask,
        start,
        -major_axis,
        lateral_tol=max(5.0, min(h, w) * 0.06),
        search_radius=max(18.0, max(h, w) * 0.25),
    )
    end = _extend_endpoint_along_axis(
        foreground_mask,
        end,
        major_axis,
        lateral_tol=max(5.0, min(h, w) * 0.06),
        search_radius=max(18.0, max(h, w) * 0.25),
    )

    span_px = float(np.linalg.norm(end - start))
    if span_px < max(18.0, max(h, w) * 0.18):
        return None

    score = min(
        0.95,
        0.45
        + min(0.25, center_overlap / max(1.0, float(color_meta["center_area"])) * 0.35)
        + min(0.25, span_px / max(1.0, float(max(h, w))) * 0.35)
        + min(0.1, int(np.count_nonzero(component_mask)) / max(1.0, float(h * w)) * 0.8),
    )

    return WireTraceResult(
        ordered_keypoints=[
            (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1])),
        ],
        score=score,
        color_mode=str(color_meta["mode"]),
        target_hue=color_meta["target_hue"],
        target_sat=color_meta["target_sat"],
        component_area=int(np.count_nonzero(component_mask)),
        span_px=span_px,
        center_overlap_px=int(center_overlap),
    )


def _build_wire_color_mask(hsv_image: np.ndarray) -> tuple[np.ndarray | None, dict[str, object]]:
    h, w = hsv_image.shape[:2]
    hue = hsv_image[:, :, 0]
    sat = hsv_image[:, :, 1]
    val = hsv_image[:, :, 2]

    cx1 = max(0, int(round(w * 0.25)))
    cx2 = min(w, int(round(w * 0.75)))
    cy1 = max(0, int(round(h * 0.25)))
    cy2 = min(h, int(round(h * 0.75)))
    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_mask[cy1:cy2, cx1:cx2] = 255

    saturated = (sat >= 45) & (val >= 35)
    seeded = saturated & (center_mask > 0)
    seeded_hues = hue[seeded]
    seeded_sats = sat[seeded]

    meta: dict[str, object] = {
        "mode": "color_hsv",
        "target_hue": None,
        "target_sat": None,
        "center_area": int(np.count_nonzero(center_mask)),
    }

    if seeded_hues.size >= 24:
        hist = np.bincount(seeded_hues.astype(np.int32), weights=seeded_sats.astype(np.float32), minlength=180)
        target_hue = int(np.argmax(hist))
        hue_dist = np.minimum(np.abs(hue.astype(np.int16) - target_hue), 180 - np.abs(hue.astype(np.int16) - target_hue))
        mask = ((hue_dist <= 12) & (sat >= 35) & (val >= 25)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        meta["target_hue"] = float(target_hue)
        meta["target_sat"] = float(np.median(seeded_sats))
        return mask, meta

    dark_seed = (val <= 150) & (center_mask > 0)
    dark_mask = ((val <= 170) & (sat <= 80)).astype(np.uint8) * 255
    if int(np.count_nonzero(dark_seed)) >= 24:
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        meta["mode"] = "dark_fallback"
        meta["target_sat"] = float(np.median(sat[dark_seed])) if int(np.count_nonzero(dark_seed)) else None
        return dark_mask, meta

    return None, meta


def _select_wire_component(mask: np.ndarray) -> tuple[np.ndarray | None, int]:
    h, w = mask.shape[:2]
    cx1 = max(0, int(round(w * 0.25)))
    cx2 = min(w, int(round(w * 0.75)))
    cy1 = max(0, int(round(h * 0.25)))
    cy2 = min(h, int(round(h * 0.75)))
    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_mask[cy1:cy2, cx1:cx2] = 1

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_label = -1
    best_score = -1.0
    best_overlap = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        overlap = int(np.count_nonzero((labels == label) & (center_mask > 0)))
        if overlap <= 0:
            continue
        span = max(ww, hh)
        score = overlap * 8.0 + span * 1.5 + area * 0.05
        if score > best_score:
            best_score = score
            best_label = label
            best_overlap = overlap

    if best_label < 0:
        return None, 0

    component_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    return component_mask, best_overlap


def _extend_endpoint_along_axis(
    foreground_mask: np.ndarray,
    endpoint_xy: np.ndarray,
    direction_xy: np.ndarray,
    *,
    lateral_tol: float,
    search_radius: float,
) -> np.ndarray:
    points = np.column_stack(np.where(foreground_mask > 0))
    if len(points) == 0:
        return endpoint_xy

    pts_xy = np.column_stack((points[:, 1].astype(np.float32), points[:, 0].astype(np.float32)))
    axis = direction_xy / max(float(np.linalg.norm(direction_xy)), 1e-6)
    perp = np.array([-axis[1], axis[0]], dtype=np.float32)
    delta = pts_xy - endpoint_xy.reshape(1, 2)
    proj = delta[:, 0] * float(axis[0]) + delta[:, 1] * float(axis[1])
    perp_dist = np.abs(delta[:, 0] * float(perp[0]) + delta[:, 1] * float(perp[1]))
    candidates = pts_xy[(proj >= 0.0) & (proj <= search_radius) & (perp_dist <= lateral_tol)]
    if len(candidates) == 0:
        return endpoint_xy
    candidate_delta = candidates - endpoint_xy.reshape(1, 2)
    candidate_proj = candidate_delta[:, 0] * float(axis[0]) + candidate_delta[:, 1] * float(axis[1])
    return candidates[int(np.argmax(candidate_proj))]


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
