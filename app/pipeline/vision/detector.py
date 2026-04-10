"""
YOLO 检测器封装 (← src_v2/vision/detector.py)

封装 Ultralytics YOLO, 支持 HBB 和 OBB 模型
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """单个检测结果"""

    class_name: str = ""
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2

    # OBB 支持
    is_obb: bool = False
    obb_corners: Optional[np.ndarray] = None  # (4, 2)

    # YOLO-Pose 关键点
    keypoints: Optional[np.ndarray] = None
    keypoints_conf: Optional[float] = None

    # Wire 颜色
    wire_color: str = ""


class ComponentDetector:
    """YOLO 检测器"""

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

        # 有路径就自动加载
        if model_path:
            self.load(model_path)

    def load(self, model_path: Optional[str] = None) -> bool:
        """加载 YOLO 模型"""
        path = model_path or self.model_path
        if not path:
            logger.warning("[Detector] No model path specified")
            return False

        try:
            from ultralytics import YOLO

            self.model = YOLO(path)
            self._is_obb = "obb" in str(path).lower()
            logger.info(f"[Detector] Loaded: {path} (OBB={self._is_obb})")
            return True
        except Exception as e:
            logger.error(f"[Detector] Load failed: {e}")
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
