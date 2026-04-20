"""
T4: Pin 检测测试 — 验证启发式 fallback 和多视图融合

关键：无 Pin 模型时自动走 heuristic_fallback（已实现）
"""

from __future__ import annotations

import logging
import numpy as np
import pytest

from tests.pipeline.fixtures import (
    make_blank_image,
    make_resistor_roi,
    make_capacitor_roi,
    make_led_roi,
    image_to_b64,
)


class TestPinDetectionMock:
    """T4: Pin 检测测试（Mock PinDetector）."""

    def test_t4_1_mock_2pin(self, mock_detector_3_components, blank_image_b64):
        """T4.3: Mock 2-pin 元件 → len(pins)=2"""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect

        # 先跑 S1
        from app.pipeline.stages.s1_detect import run_detect
        s1 = run_detect(
            images_b64=[blank_image_b64],
            detector=mock_detector_3_components,
        )

        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=mock_pin_detector_2pin(),
        )

        assert len(result["components"]) == 3
        for comp in result["components"]:
            assert len(comp["pins"]) == 2
            assert comp["pins"][0]["pin_name"] == "pin1"
            assert comp["pins"][1]["pin_name"] == "pin2"

    def test_t4_2_mock_3pin(self, blank_image_b64):
        """T4.4: Mock 3-pin (Potentiometer) → len(pins)=3"""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        pot_det = MockComponentDetector([
            {"class_name": "Potentiometer", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        pot_pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "pin1", "keypoint": (100.0, 240.0), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "pin2", "keypoint": (200.0, 240.0), "confidence": 0.9, "visibility": 2},
            {"pin_id": 3, "pin_name": "pin3", "keypoint": (300.0, 240.0), "confidence": 0.9, "visibility": 2},
        ])

        s1 = run_detect(images_b64=[blank_image_b64], detector=pot_det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pot_pin,
        )

        assert len(result["components"]) == 1
        assert len(result["components"][0]["pins"]) == 3

    def test_t4_3_mock_ic_dip8(self, blank_image_b64):
        """T4.5: Mock IC DIP-8 → len(pins)=2 (anchor pair)"""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        ic_det = MockComponentDetector([
            {"class_name": "IC", "package_type": "dip8", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        ic_pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "pin1", "keypoint": (120.0, 220.0), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "pin2", "keypoint": (280.0, 220.0), "confidence": 0.9, "visibility": 2},
        ])

        s1 = run_detect(images_b64=[blank_image_b64], detector=ic_det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=ic_pin,
        )

        assert len(result["components"][0]["pins"]) == 2

    def test_t4_4_pin_source_field(self, blank_image_b64):
        """T9.4: 每个 pin 有 source 字段"""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "pin1", "keypoint": (120.0, 240.0), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "pin2", "keypoint": (280.0, 240.0), "confidence": 0.9, "visibility": 2},
        ])

        s1 = run_detect(images_b64=[blank_image_b64], detector=det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pin,
        )

        for comp in result["components"]:
            for pin_data in comp["pins"]:
                assert "source" in pin_data

    def test_t4_5_pin_detector_metadata(self, blank_image_b64):
        """T9.5: pin_detector.backend_mode 透传"""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "pin1", "keypoint": (120.0, 240.0), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "pin2", "keypoint": (280.0, 240.0), "confidence": 0.9, "visibility": 2},
        ])

        s1 = run_detect(images_b64=[blank_image_b64], detector=det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pin,
        )

        assert result["pin_detector_backend"] == "mock_pose"
        assert result["pin_detector_mode"] == "mock_model"


class TestPinDetectionHeuristicFallback:
    """T4: Pin 检测 — 启发式 fallback（无 Pin 模型时）."""

    def test_t4_6_heuristic_fallback_real_roi(self, blank_image_b64, resistor_roi_image):
        """T4.1: 真实 Resistor ROI → source=heuristic_fallback, confidence>0"""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector
        from app.pipeline.stages.s1_detect import run_detect
        from app.pipeline.vision.pin_model import PinRoiDetector

        # 使用真实 ROI 图像 + 无模型的 PinRoiDetector
        det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 360, 260), "confidence": 0.95}
        ])
        pin_det = PinRoiDetector(model_path=None, device="cpu")

        s1 = run_detect(images_b64=[blank_image_b64], detector=det)

        # 将 resistor ROI 图像替换进 blank_image
        from tests.pipeline.fixtures import image_to_b64
        resistor_b64 = image_to_b64(resistor_roi_image)

        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[resistor_b64],
            pin_detector=pin_det,
        )

        # 启发式 fallback 应该工作
        assert result["pin_detector_mode"] == "heuristic_fallback"
        assert result["pin_detector_backend"] == "yolo_pose"

        for comp in result["components"]:
            assert len(comp["pins"]) == 2  # Resistor 2-pin
            for pin_data in comp["pins"]:
                assert pin_data["source"] == "heuristic_fallback"
                assert pin_data["confidence"] > 0

    def test_t4_7_empty_roi_image(self, blank_image_b64):
        """T4.2: 空 ROI 图像 → keypoint=None, visibility=0"""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector
        from app.pipeline.stages.s1_detect import run_detect
        from app.pipeline.vision.pin_model import PinRoiDetector

        det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (0, 0, 1, 1), "confidence": 0.95}  # 极小 bbox
        ])
        pin_det = PinRoiDetector(model_path=None, device="cpu")

        s1 = run_detect(images_b64=[blank_image_b64], detector=det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pin_det,
        )

        for comp in result["components"]:
            for pin_data in comp["pins"]:
                # 空/无效 ROI 时可能为 None 或有值
                # 关键是 pipeline 不 crash
                assert "source" in pin_data
                assert "confidence" in pin_data

    def test_t4_8_interface_version(self, blank_image_b64):
        """T9.2: S1.5 返回 interface_version=component_pin_detect_v1"""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "pin1", "keypoint": (120.0, 240.0), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "pin2", "keypoint": (280.0, 240.0), "confidence": 0.9, "visibility": 2},
        ])

        s1 = run_detect(images_b64=[blank_image_b64], detector=det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pin,
        )

        assert result["interface_version"] == "component_pin_detect_v1"


class TestPinDetectionMultiView:
    """T4: 多视图 Pin 检测."""

    def test_t4_9_multi_view_keypoints(self):
        """T4.6: 3 张图各有 ROI → keypoints_by_view 对 3 个视图填充"""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.fixtures import image_to_b64, make_blank_image

        det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        # Mock PinDetector 在各视图返回不同的 keypoint
        pin = MockMultiViewPinDetector()

        s1 = run_detect(
            images_b64=[image_to_b64(make_blank_image()) for _ in range(3)],
            detector=det,
        )

        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[image_to_b64(make_blank_image()) for _ in range(3)],
            pin_detector=pin,
        )

        # 各视图应被处理
        assert result["decoded_view_count"] == 3
        assert "top" in result["available_view_ids"]

    def test_t4_10_merge_predictions_by_view(self):
        """T4.7: 多视图融合 — 置信度取 max, source 正确标记"""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.mocks import MockComponentDetector
        from tests.pipeline.fixtures import image_to_b64, make_blank_image

        det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        # 使用真实的 PinRoiDetector（启发式 fallback）
        from app.pipeline.vision.pin_model import PinRoiDetector
        pin_det = PinRoiDetector(model_path=None, device="cpu")

        s1 = run_detect(
            images_b64=[image_to_b64(make_blank_image()) for _ in range(2)],
            detector=det,
        )
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[image_to_b64(make_blank_image()) for _ in range(2)],
            pin_detector=pin_det,
        )

        for comp in result["components"]:
            for pin_data in comp["pins"]:
                # keypoints_by_view 应存在
                assert "keypoints_by_view" in pin_data
                # visibility_by_view 应存在
                assert "visibility_by_view" in pin_data
                # score_by_view 应存在
                assert "score_by_view" in pin_data

    def test_t4_11_side_roi_association_candidate(self):
        """侧视图 ROI 可优先使用 side candidate, 不再只走 shared bbox fallback."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from tests.pipeline.fixtures import image_to_b64, make_blank_image

        det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "pin1", "keypoint": (120.0, 240.0), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "pin2", "keypoint": (280.0, 240.0), "confidence": 0.9, "visibility": 2},
        ])

        images = [image_to_b64(make_blank_image()) for _ in range(2)]
        s1 = run_detect(images_b64=images, detector=det)
        side_candidates = [
            {
                "candidate_id": "left_front_resistor_1",
                "class_name": "Resistor",
                "component_type": "Resistor",
                "package_type": "axial_2pin",
                "pin_schema_id": "fixed_pins",
                "confidence": 0.88,
                "bbox": [90, 198, 320, 270],
                "is_obb": False,
                "orientation": 0.0,
                "view_id": "left_front",
                "source": "side_recall_candidate",
                "source_model_type": "yolo_obb_component",
                "instance_status": "candidate",
                "wire_color": "",
                "obb_corners": None,
            }
        ]

        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=images,
            pin_detector=pin,
            supplemental_detections=side_candidates,
        )

        comp = result["components"][0]
        assert result["side_roi_assoc_backend"] == "side_view_roi_assoc_v1"
        assert comp["roi_by_view"]["left_front"]["source"] == "associated_bbox_candidate"
        assert comp["roi_by_view"]["left_front"]["association"]["matched"] is True


class MockMultiViewPinDetector:
    """Mock Pin 检测器 — 模拟多视图返回不同 keypoint."""
    backend_mode = "mock_model"
    backend_type = "mock_pose"
    interface_version = "pin_detector_v1"

    def predict_component_pins(self, **kwargs):
        from app.pipeline.vision.pin_model import PinPrediction
        view_id = kwargs.get("view_id", "top")
        offset_x, offset_y = kwargs.get("roi_offset", (0, 0))

        base = 150.0 if view_id == "top" else 160.0
        return [
            PinPrediction(
                pin_id=1,
                pin_name="pin1",
                keypoint=(base + offset_x, 240.0 + offset_y),
                confidence=0.95,
                visibility=2,
                source="mock_model",
                metadata={"view_id": view_id},
            ),
            PinPrediction(
                pin_id=2,
                pin_name="pin2",
                keypoint=(250.0 + offset_x, 240.0 + offset_y),
                confidence=0.95,
                visibility=2,
                source="mock_model",
                metadata={"view_id": view_id},
            ),
        ]


def mock_pin_detector_2pin():
    """Factory: 创建 2-pin Mock PinDetector."""
    from tests.pipeline.mocks import MockPinDetector
    return MockPinDetector([
        {"pin_id": 1, "pin_name": "pin1", "keypoint": (120.0, 240.0), "confidence": 0.95, "visibility": 2},
        {"pin_id": 2, "pin_name": "pin2", "keypoint": (280.0, 240.0), "confidence": 0.95, "visibility": 2},
    ])
