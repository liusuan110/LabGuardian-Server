"""Mock 对象 — 在视觉模型未训练时提供可控的假数据."""

from __future__ import annotations

from typing import Any

from app.pipeline.vision.detector import Detection


class MockComponentDetector:
    """Mock YOLO 检测器 — 返回预设 bbox，不依赖真实模型."""

    def __init__(self, detections: list[dict[str, Any]]):
        """
        Args:
            detections: 预设检测结果列表, 每个 dict 包含:
                - class_name: str 元件类型
                - bbox: tuple (x1, y1, x2, y2)
                - confidence: float 置信度
                - is_obb: bool 是否有旋转框
                - obb_corners: np.ndarray (4, 2) 旋转框顶点
                - wire_color: str 导线颜色
        """
        self._dets = detections
        self.detected_count = 0

    def detect(
        self,
        image,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 1280,
    ) -> list[Detection]:
        self.detected_count += 1
        return [self._to_detection(d) for d in self._dets]

    def _to_detection(self, d: dict) -> Detection:
        import numpy as np
        obb = d.get("obb_corners")
        if obb is not None:
            obb = np.array(obb, dtype=np.float32)
        return Detection(
            class_name=d["class_name"],
            confidence=float(d.get("confidence", 0.9)),
            bbox=tuple(d["bbox"]),  # (x1, y1, x2, y2)
            is_obb=d.get("is_obb", False),
            obb_corners=obb,
            wire_color=d.get("wire_color", ""),
        )

    def offset_detections(self, dets: list[Detection], dx: int, dy: int) -> None:
        """将检测结果偏移到全图坐标系."""
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            det.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            if det.obb_corners is not None:
                det.obb_corners = det.obb_corners.copy()
                det.obb_corners[:, 0] += dx
                det.obb_corners[:, 1] += dy


class MockPinDetector:
    """Mock Pin 检测器 — 返回预设 keypoints，不依赖真实模型."""
    backend_mode = "mock_model"
    backend_type = "mock_pose"
    interface_version = "pin_detector_v1"

    def __init__(self, predictions: list[dict[str, Any]]):
        """
        Args:
            predictions: 预设 pin 预测列表, 每个 dict 包含:
                - pin_id: int
                - pin_name: str
                - keypoint: tuple (x, y) 或 None
                - confidence: float
                - visibility: int
        """
        self._preds = predictions

    def predict_component_pins(
        self,
        *,
        component_id: str,
        component_type: str,
        package_type: str,
        pin_schema_id: str,
        roi_image,
        roi_offset: tuple[int, int],
        view_id: str = "top",
        confidence: float = 1.0,
    ) -> list:
        from app.pipeline.vision.pin_model import PinPrediction
        return [
            PinPrediction(
                pin_id=p["pin_id"],
                pin_name=p["pin_name"],
                keypoint=p.get("keypoint"),
                confidence=float(p.get("confidence", 0.9)),
                visibility=int(p.get("visibility", 2)),
                source="mock_model",
                metadata={
                    "mock": True,
                    "view_id": view_id,
                },
            )
            for p in self._preds
        ]
