"""
YOLO 检测器封装 (← src_v2/vision/detector.py)

封装 Ultralytics YOLO, 支持 HBB 和 OBB 模型
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

    # 引脚像素坐标 (OBB 短边中点 或 YOLO-Pose 关键点)
    pin1_pixel: Optional[Tuple[float, float]] = None
    pin2_pixel: Optional[Tuple[float, float]] = None

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

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model = None
        self.model_path = model_path
        self.device = device
        self._is_obb = False

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

            det = Detection(
                class_name=cls_name,
                confidence=conf,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            )
            # 默认引脚: bbox 短边中点
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            if w > h:
                det.pin1_pixel = (float(x1), float(cy))
                det.pin2_pixel = (float(x2), float(cy))
            else:
                det.pin1_pixel = (float(cx), float(y1))
                det.pin2_pixel = (float(cx), float(y2))

            dets.append(det)
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

            # 短边中点作为引脚
            d01 = np.linalg.norm(corners[0] - corners[1])
            d12 = np.linalg.norm(corners[1] - corners[2])
            if d01 < d12:
                pin1 = (corners[0] + corners[1]) / 2
                pin2 = (corners[2] + corners[3]) / 2
            else:
                pin1 = (corners[1] + corners[2]) / 2
                pin2 = (corners[3] + corners[0]) / 2

            det = Detection(
                class_name=cls_name,
                confidence=conf,
                bbox=(x_min, y_min, x_max, y_max),
                pin1_pixel=(float(pin1[0]), float(pin1[1])),
                pin2_pixel=(float(pin2[0]), float(pin2[1])),
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
            if det.pin1_pixel:
                det.pin1_pixel = (det.pin1_pixel[0] + dx, det.pin1_pixel[1] + dy)
            if det.pin2_pixel:
                det.pin2_pixel = (det.pin2_pixel[0] + dx, det.pin2_pixel[1] + dy)
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
            if det.pin1_pixel:
                cv2.circle(annotated, (int(det.pin1_pixel[0]), int(det.pin1_pixel[1])), 4, (0, 0, 255), -1)
            if det.pin2_pixel:
                cv2.circle(annotated, (int(det.pin2_pixel[0]), int(det.pin2_pixel[1])), 4, (255, 0, 0), -1)
        return annotated
