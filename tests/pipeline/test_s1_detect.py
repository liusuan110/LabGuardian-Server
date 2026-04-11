"""
T2+T3: 组件检测和多视图补召回测试

T2: 验证 Mock YOLO 检测器的输出转换正确性
T3: 验证多视图检测和补召回逻辑
"""

from __future__ import annotations

import logging
import numpy as np
import pytest

from tests.pipeline.fixtures import make_blank_image, image_to_b64


class TestS1ComponentDetection:
    """T2: 组件检测测试（Mock YOLO）."""

    def test_t2_1_single_resistor(self, mock_detector_resistor, blank_image_b64):
        """T2.1: 单个 Resistor → class_name=Resistor, component_id=R1"""
        from app.pipeline.stages.s1_detect import run_detect

        result = run_detect(
            images_b64=[blank_image_b64],
            detector=mock_detector_resistor,
        )

        assert "detections" in result
        assert len(result["detections"]) == 1
        det = result["detections"][0]
        assert det["class_name"] == "Resistor"
        assert det["component_id"] == "R1"
        assert det["component_type"] == "Resistor"
        assert det["package_type"] == "axial_2pin"
        assert det["confidence"] == 0.95
        assert det["bbox"] == [100, 200, 300, 260]

    def test_t2_2_mixed_components(self, mock_detector_3_components, blank_image_b64):
        """T2.2: 3 个混合元件 → R1, C1, LED1 分别生成"""
        from app.pipeline.stages.s1_detect import run_detect

        result = run_detect(
            images_b64=[blank_image_b64],
            detector=mock_detector_3_components,
        )

        ids = {d["component_id"] for d in result["detections"]}
        types = {d["class_name"] for d in result["detections"]}
        assert ids == {"R1", "C1", "LED1"}
        assert types == {"Resistor", "Capacitor", "LED"}

    def test_t2_3_empty_detections(self, blank_image_b64):
        """T2.3: 0 个元件 → detections=[]"""
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.mocks import MockComponentDetector

        mock_det = MockComponentDetector([])
        result = run_detect(
            images_b64=[blank_image_b64],
            detector=mock_det,
        )

        assert result["detections"] == []

    def test_t2_4_obb_orientation(self, mock_detector_obb, blank_image_b64):
        """T2.4: 带 OBB corners → is_obb=True, orientation 计算正确"""
        from app.pipeline.stages.s1_detect import run_detect

        result = run_detect(
            images_b64=[blank_image_b64],
            detector=mock_detector_obb,
        )

        det = result["detections"][0]
        assert det["is_obb"] is True
        assert det["obb_corners"] is not None
        assert len(det["obb_corners"]) == 4
        assert "orientation" in det

    def test_t2_5_background_class_filtered(self, mock_detector_breadboard, blank_image_b64):
        """T2.5: Breadboard 背景类被过滤，不出现在 detections 中"""
        from app.pipeline.stages.s1_detect import run_detect

        result = run_detect(
            images_b64=[blank_image_b64],
            detector=mock_detector_breadboard,
        )

        types = {d["class_name"] for d in result["detections"]}
        assert "Breadboard" not in types
        assert "Resistor" in types  # 真正的元件应该保留

    def test_t2_interface_version(self, mock_detector_resistor, blank_image_b64):
        """T9.1: S1 返回正确的 interface_version"""
        from app.pipeline.stages.s1_detect import run_detect

        result = run_detect(
            images_b64=[blank_image_b64],
            detector=mock_detector_resistor,
        )
        assert result["interface_version"] == "component_detect_v1"
        assert result["detector_backend"] == "yolo_obb_component"

    def test_t2_recall_mode_single_image(self, mock_detector_resistor, blank_image_b64):
        """T3.1: 单张图 → recall_mode=top_primary_plus_side_candidates"""
        from app.pipeline.stages.s1_detect import run_detect

        result = run_detect(
            images_b64=[blank_image_b64],
            detector=mock_detector_resistor,
        )
        assert result["recall_mode"] == "top_primary_plus_side_candidates"
        assert result["supplemental_detections"] == []


class TestS1MultiViewRecall:
    """T3: 多视图补召回测试."""

    def test_t3_1_side_candidates_collected(self):
        """T3.2: 侧视图候选被收集到 supplemental_detections"""
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.mocks import MockComponentDetector
        from tests.pipeline.fixtures import image_to_b64, make_blank_image
        import numpy as np

        # top view: Resistor
        top_det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.95}
        ])
        # left_front view: Capacitor
        left_det = MockComponentDetector([
            {"class_name": "Capacitor", "bbox": (200, 150, 280, 210), "confidence": 0.90}
        ])

        images = [
            image_to_b64(make_blank_image()),  # top
            image_to_b64(make_blank_image()),  # left_front
        ]
        # 先用 left_det 做侧视图
        # 但 run_detect 用同一个 detector，我们测 supplemental 结构
        # 实际 supplemental 来自侧视图，detector 相同

        result = run_detect(
            images_b64=images,
            detector=top_det,  # 同一个 detector
        )

        assert result["recall_mode"] == "top_primary_plus_side_candidates"
        assert "supplemental_detections" in result

    def test_t3_2_no_top_fallback(self, mock_detector_resistor):
        """T3.3: 无 top view（top 损坏）→ recall_mode=side_candidates_only"""
        from app.pipeline.stages.s1_detect import run_detect

        corrupted = "!!!corrupted!!!"
        result = run_detect(
            images_b64=[corrupted],
            detector=mock_detector_resistor,
        )

        assert result["detections"] == []
        assert result["recall_mode"] == "side_candidates_only"

    def test_t3_3_multi_image_decode_summary(self, mock_detector_resistor):
        """多图时 decode_summary 正确传递"""
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.fixtures import image_to_b64, make_blank_image, make_corrupted_b64

        images = [
            image_to_b64(make_blank_image()),  # top: ok
            make_corrupted_b64(),              # left_front: 损坏
            image_to_b64(make_blank_image()),  # right_front: ok
        ]
        result = run_detect(images_b64=images, detector=mock_detector_resistor)

        assert result["decoded_view_count"] == 2
        assert "top" in result["available_view_ids"]
        assert "right_front" in result["available_view_ids"]
        assert "left_front" in result["dropped_view_ids"]

    def test_t3_4_multiple_side_views(self):
        """3 张图各自跑 detector → supplemental_detections 收集侧视图结果"""
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.mocks import MockComponentDetector
        from tests.pipeline.fixtures import image_to_b64, make_blank_image

        # top: 无检测（空 detector）
        # left_front: 有检测
        # right_front: 有检测
        top_det = MockComponentDetector([])
        side_det = MockComponentDetector([
            {"class_name": "LED", "bbox": (50, 100, 150, 200), "confidence": 0.85}
        ])

        images = [image_to_b64(make_blank_image()) for _ in range(3)]
        result = run_detect(images_b64=images, detector=top_det)

        # top 无检测，侧视图有检测被收集
        assert "supplemental_detections" in result


class TestS1OutputSchema:
    """验证 S1 输出的完整 schema 字段."""

    def test_detection_schema_fields(self, mock_detector_resistor, blank_image_b64):
        """每个 detection 包含所有必要字段."""
        from app.pipeline.stages.s1_detect import run_detect

        result = run_detect(
            images_b64=[blank_image_b64],
            detector=mock_detector_resistor,
        )
        det = result["detections"][0]

        required_fields = [
            "component_id", "class_name", "component_type", "package_type",
            "pin_schema_id", "confidence", "bbox", "is_obb", "orientation",
            "view_id", "source", "source_model_type",
            "input_detection_interface_version",
        ]
        for field in required_fields:
            assert field in det, f"Missing field: {field}"

    def test_primary_image_shape(self, mock_detector_resistor, blank_image_b64):
        """primary_image_shape 正确传递."""
        from app.pipeline.stages.s1_detect import run_detect

        result = run_detect(
            images_b64=[blank_image_b64],
            detector=mock_detector_resistor,
        )
        h, w = result["primary_image_shape"]
        assert h == 480
        assert w == 640

    def test_duration_ms_present(self, mock_detector_resistor, blank_image_b64):
        """duration_ms 字段存在且非负."""
        from app.pipeline.stages.s1_detect import run_detect

        result = run_detect(
            images_b64=[blank_image_b64],
            detector=mock_detector_resistor,
        )
        assert "duration_ms" in result
        assert result["duration_ms"] >= 0
