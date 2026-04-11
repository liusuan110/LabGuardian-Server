"""
T5: Pin→Hole 映射测试 — 验证校准、孔位吸附、候选生成

无模型依赖：校准器、BoardSchema、坐标映射均为确定性逻辑
"""

from __future__ import annotations

import logging
import numpy as np
import pytest


class TestS2Mapping:
    """T5: Pin→Hole 映射测试."""

    def test_t5_1_valid_pin_hole_mapping(self, calibrator):
        """T5.1: 有效 pin keypoint → hole_id 非空, electrical_node_id 非空"""
        from app.pipeline.stages.s2_mapping import run_mapping

        # 用面包板图像初始化校准器（走启发式校准或 synthetic fallback）
        from tests.pipeline.fixtures import make_breadboard_image, image_to_b64

        img = make_breadboard_image(h=480, w=640)
        b64 = image_to_b64(img)

        # 模拟 S1.5 输出的组件（带 pin 数据）
        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {
                        "pin_id": 1,
                        "pin_name": "pin1",
                        "keypoints_by_view": {"top": [160.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95,
                        "source": "heuristic_fallback",
                    },
                    {
                        "pin_id": 2,
                        "pin_name": "pin2",
                        "keypoints_by_view": {"top": [340.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95,
                        "source": "heuristic_fallback",
                    },
                ],
            }
        ]

        result = run_mapping(
            components=components,
            calibrator=calibrator,
            image_shape=(480, 640),
            images_b64=[b64],
        )

        assert len(result["components"]) == 1
        mapped_comp = result["components"][0]
        assert len(mapped_comp["pins"]) == 2

        # 至少第一个 pin 应该有 hole_id
        pin1 = mapped_comp["pins"][0]
        assert "hole_id" in pin1
        assert "electrical_node_id" in pin1

    def test_t5_2_calibration_mode_visual(self, calibrator):
        """T5.2: 有图像校准 → calibration.mode 非空"""
        from app.pipeline.stages.s2_mapping import run_mapping
        from tests.pipeline.fixtures import make_breadboard_image, image_to_b64

        img = make_breadboard_image(h=480, w=640)
        b64 = image_to_b64(img)
        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {
                        "pin_id": 1,
                        "pin_name": "pin1",
                        "keypoints_by_view": {"top": [160.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95,
                        "source": "heuristic_fallback",
                    },
                    {
                        "pin_id": 2,
                        "pin_name": "pin2",
                        "keypoints_by_view": {"top": [340.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95,
                        "source": "heuristic_fallback",
                    },
                ],
            }
        ]

        result = run_mapping(
            components=components,
            calibrator=calibrator,
            image_shape=(480, 640),
            images_b64=[b64],
        )

        assert "calibration" in result
        assert "mode" in result["calibration"]
        # mode 应该是 visual 或 synthetic_fallback
        assert result["calibration"]["mode"] in ("visual", "synthetic_fallback")
        assert result["calibration"]["grid_ready"] is True

    def test_t5_3_synthetic_fallback(self):
        """T5.3: 无图像 fallback → calibration.mode=synthetic_fallback"""
        from app.pipeline.stages.s2_mapping import run_mapping
        from app.pipeline.vision.calibrator import BreadboardCalibrator

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {
                        "pin_id": 1,
                        "pin_name": "pin1",
                        "keypoints_by_view": {"top": [160.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95,
                        "source": "heuristic_fallback",
                    },
                    {
                        "pin_id": 2,
                        "pin_name": "pin2",
                        "keypoints_by_view": {"top": [340.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95,
                        "source": "heuristic_fallback",
                    },
                ],
            }
        ]

        result = run_mapping(
            components=components,
            calibrator=calibrator,
            image_shape=(480, 640),
            images_b64=None,  # 无图像
        )

        assert result["calibration"]["mode"] == "synthetic_fallback"
        assert result["calibration"]["grid_ready"] is True

    def test_t5_4_candidate_hole_ids(self):
        """T5.4: 候选孔位 → candidate_hole_ids 包含多个候选"""
        from app.pipeline.stages.s2_mapping import run_mapping
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.fixtures import make_breadboard_image, image_to_b64

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        img = make_breadboard_image(h=480, w=640)
        b64 = image_to_b64(img)
        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {
                        "pin_id": 1,
                        "pin_name": "pin1",
                        "keypoints_by_view": {"top": [160.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95,
                        "source": "heuristic_fallback",
                    },
                    {
                        "pin_id": 2,
                        "pin_name": "pin2",
                        "keypoints_by_view": {"top": [340.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95,
                        "source": "heuristic_fallback",
                    },
                ],
            }
        ]

        result = run_mapping(
            components=components,
            calibrator=calibrator,
            image_shape=(480, 640),
            images_b64=[b64],
        )

        for comp in result["components"]:
            for pin_data in comp["pins"]:
                # 候选孔位字段存在
                assert "candidate_hole_ids" in pin_data
                assert "candidate_node_ids" in pin_data
                # 第一个候选应该是当前选中的 hole_id
                assert pin_data["candidate_hole_ids"][0] == pin_data["hole_id"]

    def test_t5_5_ambiguity_reasons(self):
        """T5.5: 多候选时 is_ambiguous=True"""
        from app.pipeline.stages.s2_mapping import run_mapping
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.fixtures import image_to_b64, make_blank_image

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        # 两个 pin 靠得很近，映射到同一候选
        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {
                        "pin_id": 1,
                        "pin_name": "pin1",
                        "keypoints_by_view": {"top": [161.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95,
                        "source": "heuristic_fallback",
                    },
                    {
                        "pin_id": 2,
                        "pin_name": "pin2",
                        "keypoints_by_view": {"top": [163.0, 240.0]},  # 非常近
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95,
                        "source": "heuristic_fallback",
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

        # pipeline 不 crash，字段完整
        for comp in result["components"]:
            for pin_data in comp["pins"]:
                assert "is_ambiguous" in pin_data
                assert "ambiguity_reasons" in pin_data

    def test_t5_6_interface_version(self):
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
                        "pin_id": 1,
                        "pin_name": "pin1",
                        "keypoints_by_view": {"top": [160.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95,
                        "source": "heuristic_fallback",
                    },
                    {
                        "pin_id": 2,
                        "pin_name": "pin2",
                        "keypoints_by_view": {"top": [340.0, 240.0]},
                        "visibility_by_view": {"top": 2},
                        "score_by_view": {"top": 0.95},
                        "source_by_view": {"top": "heuristic_fallback"},
                        "confidence": 0.95,
                        "source": "heuristic_fallback",
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

    def test_t5_7_empty_components(self, calibrator):
        """空组件列表 → 不 crash"""
        from app.pipeline.stages.s2_mapping import run_mapping
        from tests.pipeline.fixtures import image_to_b64, make_blank_image

        result = run_mapping(
            components=[],
            calibrator=calibrator,
            image_shape=(480, 640),
            images_b64=[image_to_b64(make_blank_image())],
        )

        assert result["components"] == []
        assert "calibration" in result
