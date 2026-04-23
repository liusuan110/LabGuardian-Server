"""
YOLO 组件检测器封装.

当前视觉主路径使用 YOLO-Detect（HBB）。
OBB 解析能力仅作为兼容分支保留, 方便后续接历史模型或离线对比,
但不再作为当前项目的默认检测方案。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.pipeline.vision.model_inspector import inspect_yolo_weight

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """单个检测结果"""

    class_name: str = ""
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2

    # OBB 兼容字段（当前主路径默认不会产出）
    is_obb: bool = False
    obb_corners: Optional[np.ndarray] = None  # (4, 2)

    # YOLO-Pose 关键点
    keypoints: Optional[np.ndarray] = None
    keypoints_conf: Optional[float] = None

    # Wire 颜色
    wire_color: str = ""


class ComponentDetector:
    """组件检测器.

    主路径:
    - 加载 detect 模型
    - 输出标准 bbox

    兼容路径:
    - 若历史权重被识别为 OBBModel，仍可解析旋转框
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        obb_model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model = None
        self.model_path = model_path
        self.obb_model_path = obb_model_path
        self.device = device
        self._is_obb = False
        self.model_contract: dict[str, object] = {
            "path": "",
            "exists": False,
            "task": "unknown",
            "model_class": "unknown",
            "names": [],
            "kpt_shape": None,
            "loaded": False,
        }

        # 有路径就自动加载
        if model_path:
            self.load(model_path)

    @property
    def backend_type(self) -> str:
        return "yolo_obb_component" if self._is_obb else "yolo_detect_component"

    def load(self, model_path: Optional[str] = None) -> bool:
        """加载组件检测模型.

        当前会优先按 detect 主路径工作；若权重合同显示是 OBBModel，
        则自动切到兼容解析分支。
        """
        path = model_path or self.model_path
        if not path:
            logger.warning("[Detector] No model path specified")
            return False
        if not Path(path).exists():
            logger.error("[Detector] Model path does not exist: %s", path)
            self.model = None
            self.model_contract = {
                "path": str(path),
                "exists": False,
                "task": "unknown",
                "model_class": "unknown",
                "names": [],
                "kpt_shape": None,
                "loaded": False,
            }
            return False

        contract = inspect_yolo_weight(path)
        contract["loaded"] = False
        self.model_contract = contract
        task = str(contract.get("task") or "unknown")
        if task == "pose":
            logger.error("[Detector] Refusing pose weight for component detector: %s", path)
            self.model = None
            self._is_obb = False
            return False

        try:
            from ultralytics import YOLO

            self.model = YOLO(path)
            self.model_path = path
            self._is_obb = task == "obb" or "obb" in Path(path).name.lower()
            self.model_contract["loaded"] = True
            logger.info(
                "[Detector] Loaded: %s (task=%s backend=%s)",
                path,
                task,
                self.backend_type,
            )
            return True
        except Exception as e:
            logger.error(f"[Detector] Load failed: {e}")
            self.model_contract["loaded"] = False
            return False

    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 960,
    ) -> List[Detection]:
        """执行检测, 返回 Detection 列表"""
        if self.model is None:
            return []

        results = self.model(
            image, conf=conf, iou=iou, imgsz=imgsz,
            device=self.device, verbose=False,
        )

        detections = []
        for r in results:
            if self._is_obb and hasattr(r, "obb") and r.obb is not None:
                detections.extend(self._parse_obb(r))
            elif hasattr(r, "boxes") and r.boxes is not None:
                detections.extend(self._parse_hbb(r))

        return detections

    def _parse_hbb(self, result) -> List[Detection]:
        dets = []
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = result.names.get(cls_id, str(cls_id))
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            dets.append(
                Detection(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
        return dets

    def _parse_obb(self, result) -> List[Detection]:
        dets = []
        obb = result.obb
        for i in range(len(obb)):
            cls_id = int(obb.cls[i])
            cls_name = result.names.get(cls_id, str(cls_id))
            conf = float(obb.conf[i])
            corners = obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)

            x_min = int(corners[:, 0].min())
            y_min = int(corners[:, 1].min())
            x_max = int(corners[:, 0].max())
            y_max = int(corners[:, 1].max())

            det = Detection(
                class_name=cls_name,
                confidence=conf,
                bbox=(x_min, y_min, x_max, y_max),
                is_obb=True,
                obb_corners=corners,
            )
            dets.append(det)
        return dets

    @staticmethod
    def offset_detections(detections: List[Detection], dx: int, dy: int):
        """将 ROI 坐标系的检测结果偏移到全图坐标系"""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            det.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            if det.obb_corners is not None:
                det.obb_corners[:, 0] += dx
                det.obb_corners[:, 1] += dy

    def annotate_frame(
        self, image: np.ndarray, detections: List[Detection],
    ) -> np.ndarray:
        """在图片上绘制检测标注"""
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
        return annotated
