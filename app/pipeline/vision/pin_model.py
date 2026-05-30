"""Full-image YOLO-Pose pin model loader and keypoint parsing helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.pipeline.vision.model_inspector import inspect_yolo_weight

logger = logging.getLogger(__name__)


@dataclass
class ModelPinParseResult:
    ordered_keypoints: list[tuple[float, float] | None]
    raw_keypoint_count: int
    raw_visible_keypoint_count: int
    used_keypoint_count: int
    extra_keypoints_ignored: int
    ignored_keypoints_reason: str


class PinRoiDetector:
    """Compatibility wrapper for the full-image YOLO-Pose pin model."""

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
        return "model" if self.model is not None else "unavailable"

    def load(self, model_path: str | None = None) -> bool:
        """Load a YOLO-Pose model for full-image pin detection."""
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

            if task == "unknown":
                self.model = YOLO(path, task="pose")
            else:
                self.model = YOLO(path)
            self.model_path = path
            self.model_contract["loaded"] = True
            logger.info("[PinDetector] Loaded full-image pin model: %s", path)
            return True
        except Exception as exc:
            logger.warning("[PinDetector] Failed to load full-image pin model %s: %s", path, exc)
            self.model = None
            self.model_contract["loaded"] = False
            return False


def _is_valid_model_keypoint(
    x: float, y: float, score: float | None, *, vis_threshold: float = 0.5
) -> bool:
    """判定单个 keypoint 是否是"真实存在的端点"。

    YOLOv8-pose 标注约定:
    - vis=2 / 高 score: 端点真实存在且可见
    - vis=1 / 中等 score: 标注存在但被遮挡
    - vis=0 / 低 score (~0.0-0.1): 模型学到的"该槽位不存在"

    对两脚器件 (jumper_wire / resistor / capacitor / led / diode), kpt_shape=[3,3]
    的第 3 个槽位标注为 vis=0; 模型在 NPU 上的预测 score 通常 0.00-0.15,
    并可能给出 (x, y) 幻觉位置（不是 (0,0)）。

    历史问题: 旧版只过滤 score <= 0.0, 0.06 这种低 score 幻觉端点会被当成
    真端点喂进 S2 mapping, 跳线尤其受影响（两脚但模型有第 3 槽预测）。
    """
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    if score is not None:
        if not np.isfinite(score) or score < vis_threshold:
            return False
    # 老兜底: (0, 0, 0) padding。新阈值已经覆盖，但保留防御性。
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
