"""
T9: 接口版本与元数据测试 — 验证所有 stage 的接口完整性
"""

from __future__ import annotations

import pytest


class TestInterfaceVersions:
    """T9: 接口版本验证."""

    def test_t9_1_s1_interface_version(self, blank_image_b64):
        """T9.1: S1 返回 interface_version=component_detect_v1"""
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.mocks import MockComponentDetector

        mock_det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.95}
        ])
        result = run_detect(
            images_b64=[blank_image_b64],
            detector=mock_det,
        )

        assert result["interface_version"] == "component_detect_v1"
        assert "detector_backend" in result

    def test_t9_2_s1_5_interface_version(self, blank_image_b64):
        """T9.2: S1.5 返回 interface_version=component_pin_detect_v1"""
        from app.pipeline.stages.s1_detect import run_detect
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.pin_model import PinRoiDetector
        from tests.pipeline.mocks import MockComponentDetector

        det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.95}
        ])
        s1 = run_detect(images_b64=[blank_image_b64], detector=det)

        pin_det = PinRoiDetector(model_path=None, device="cpu")
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pin_det,
        )

        assert result["interface_version"] == "component_pin_detect_v1"
        assert result["pin_detector_mode"] == "heuristic_fallback"
        assert result["pin_detector_backend"] == "yolo_pose"

    def test_t9_3_s2_interface_version(self):
        """T9.3: S2 返回 interface_version=hole_mapping_v1"""
        from app.pipeline.stages.s2_mapping import run_mapping
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.fixtures import image_to_b64, make_blank_image

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {
                        "pin_id": 1, "pin_name": "pin1",
                        "keypoints_by_view": {"top": [160.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95, "source": "heuristic_fallback",
                    },
                    {
                        "pin_id": 2, "pin_name": "pin2",
                        "keypoints_by_view": {"top": [340.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95, "source": "heuristic_fallback",
                    },
                ],
            }
        ]

        result = run_mapping(
            components=components,
            calibrator=calibrator,
            image_shape=(480, 640),
            images_b64=[image_to_b64(make_blank_image())],
        )

        assert result["interface_version"] == "hole_mapping_v1"
        assert "board_schema_id" in result

    def test_t9_4_s3_interface_fields(self):
        """S3 返回完整的接口字段"""
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {
                        "pin_id": 1, "pin_name": "pin1",
                        "hole_id": "A1", "electrical_node_id": "ROW_1_L",
                        "confidence": 0.95, "observations": [],
                        "source": "heuristic_fallback",
                    },
                    {
                        "pin_id": 2, "pin_name": "pin2",
                        "hole_id": "A3", "electrical_node_id": "ROW_3_L",
                        "confidence": 0.95, "observations": [],
                        "source": "heuristic_fallback",
                    },
                ],
            }
        ]

        result = run_topology(components=components)

        assert "circuit_description" in result
        assert "netlist_v2" in result
        assert "topology_graph" in result
        assert "component_count" in result
        assert "duration_ms" in result

    def test_t9_5_s4_interface_fields(self):
        """S4 返回完整的接口字段"""
        from app.pipeline.stages.s3_topology import run_topology
        from app.pipeline.stages.s4_validate import run_validate

        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {
                        "pin_id": 1, "pin_name": "pin1",
                        "hole_id": "A1", "electrical_node_id": "ROW_1_L",
                        "confidence": 0.95, "observations": [],
                        "source": "heuristic_fallback",
                    },
                    {
                        "pin_id": 2, "pin_name": "pin2",
                        "hole_id": "A3", "electrical_node_id": "ROW_3_L",
                        "confidence": 0.95, "observations": [],
                        "source": "heuristic_fallback",
                    },
                ],
            }
        ]

        s3 = run_topology(components=components)
        result = run_validate(
            topology_graph=s3["topology_graph"],
            reference_circuit=None,
            components=components,
        )

        required = [
            "risk_level", "diagnosis", "diagnostics",
            "risk_reasons", "details", "duration_ms",
            "is_correct", "similarity",
        ]
        for field in required:
            assert field in result, f"Missing S4 field: {field}"

    def test_t9_6_pin_source_field(self, blank_image_b64):
        """T9.4: 每个 pin 有 source 字段"""
        from app.pipeline.stages.s1_detect import run_detect
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.pin_model import PinRoiDetector
        from tests.pipeline.mocks import MockComponentDetector

        det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.95}
        ])
        s1 = run_detect(images_b64=[blank_image_b64], detector=det)
        pin_det = PinRoiDetector(model_path=None, device="cpu")
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pin_det,
        )

        for comp in result["components"]:
            for pin_data in comp["pins"]:
                assert "source" in pin_data
                assert pin_data["source"] in ("heuristic_fallback", "model", "unavailable")

    def test_t9_7_pin_detector_backend_mode(self, blank_image_b64):
        """T9.5: pin_detector.backend_mode 透传到输出"""
        from app.pipeline.stages.s1_detect import run_detect
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.pin_model import PinRoiDetector
        from tests.pipeline.mocks import MockComponentDetector

        det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.95}
        ])
        pin_det = PinRoiDetector(model_path=None, device="cpu")

        s1 = run_detect(images_b64=[blank_image_b64], detector=det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pin_det,
        )

        assert "pin_detector_mode" in result
        assert result["pin_detector_mode"] == "heuristic_fallback"
        assert "pin_detector_backend" in result
        assert result["pin_detector_backend"] == "yolo_pose"
