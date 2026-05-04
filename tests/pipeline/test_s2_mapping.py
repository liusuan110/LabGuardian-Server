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

    def test_t5_8_multi_view_weighted_vote(self):
        """多视图候选冲突时，应按加权投票选出最终 hole。"""
        from app.pipeline.stages.s2_mapping import run_mapping
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.fixtures import image_to_b64, make_blank_image

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)

        def fake_candidates(x: float, y: float, k: int = 5):
            if x < 200:
                return [("12", "b")]
            return [("13", "b")]

        calibrator.frame_pixel_to_logic_candidates = fake_candidates  # type: ignore[method-assign]

        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {
                        "pin_id": 1,
                        "pin_name": "pin1",
                        "keypoints_by_view": {
                            "top": [120.0, 240.0],
                            "left_front": [320.0, 240.0],
                        },
                        "visibility_by_view": {
                            "top": 2,
                            "left_front": 2,
                        },
                        "score_by_view": {
                            "top": 0.20,
                            "left_front": 0.95,
                        },
                        "source_by_view": {
                            "top": "model",
                            "left_front": "model",
                        },
                        "confidence": 0.95,
                        "source": "model",
                        "metadata": {
                            "per_view": {
                                "top": {"roi_source": "detected_bbox"},
                                "left_front": {"roi_source": "associated_bbox_candidate"},
                            }
                        },
                    }
                ],
            }
        ]

        result = run_mapping(
            components=components,
            calibrator=calibrator,
            image_shape=(480, 640),
            images_b64=[image_to_b64(make_blank_image()) for _ in range(2)],
        )

        pin_data = result["components"][0]["pins"][0]
        assert pin_data["hole_id"] == "B13"
        assert pin_data["candidate_hole_ids"][0] == "B13"
        assert pin_data["metadata"]["selected_by"] == "multi_view_weighted_vote"
        assert pin_data["metadata"]["vote_scores"]["B13"] > pin_data["metadata"]["vote_scores"]["B12"]

    def test_t5_9_side_view_3d_projection_maps_to_board_2d(self):
        from app.pipeline.stages.s2_mapping import run_mapping
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.fixtures import image_to_b64, make_blank_image

        probe = BreadboardCalibrator(rows=63, cols_per_side=5)
        probe.build_synthetic_grid((480, 640))
        target_point = probe.logic_to_board_point(("13", "b"))
        assert target_point is not None

        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {
                        "pin_id": 1,
                        "pin_name": "pin1",
                        "keypoints_by_view": {
                            "top": [600.0, 440.0],
                            "left_front": [8.0, 8.0],
                        },
                        "visibility_by_view": {
                            "top": 2,
                            "left_front": 2,
                        },
                        "score_by_view": {
                            "top": 0.05,
                            "left_front": 0.99,
                        },
                        "source_by_view": {
                            "top": "model",
                            "left_front": "model",
                        },
                        "confidence": 0.99,
                        "source": "model",
                        "metadata": {
                            "per_view": {
                                "top": {"roi_source": "detected_bbox"},
                                "left_front": {
                                    "roi_source": "associated_bbox_candidate",
                                    "point_3d": [target_point[0], target_point[1], 1.0],
                                    "projection": {
                                        "camera_matrix": [
                                            [1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0],
                                        ],
                                        "rvec": [0.0, 0.0, 0.0],
                                        "tvec": [0.0, 0.0, 0.0],
                                    },
                                },
                            }
                        },
                    }
                ],
            }
        ]

        result = run_mapping(
            components=components,
            calibrator=BreadboardCalibrator(rows=63, cols_per_side=5),
            image_shape=(480, 640),
            images_b64=[image_to_b64(make_blank_image()) for _ in range(2)],
        )

        pin_data = result["components"][0]["pins"][0]
        side_obs = next(obs for obs in pin_data["observations"] if obs["view_id"] == "left_front")

        assert pin_data["hole_id"] == "B13"
        assert pin_data["board_2d_point"] == pytest.approx([target_point[0], target_point[1]])
        assert side_obs["board_2d_point"] == pytest.approx([target_point[0], target_point[1]])
        assert side_obs["projection"]["used_3d"] is True
        assert side_obs["projection"]["method"] == "project_points_3d_to_top_2d"

    def test_t5_10_board_point_candidates_prioritize_rail(self):
        from app.pipeline.vision.calibrator import BreadboardCalibrator

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))

        assert calibrator._row_coords is not None
        assert calibrator._col_coords is not None

        calibrator._landscape = True
        calibrator._rail_row_coords = calibrator._row_coords.copy()
        calibrator._top_rails = [float(calibrator._col_coords[0] - 30.0), float(calibrator._col_coords[0] - 15.0)]
        calibrator._bot_rails = [float(calibrator._col_coords[-1] + 15.0), float(calibrator._col_coords[-1] + 30.0)]
        calibrator._rail_tolerance = 12.0

        row_val = float(calibrator._row_coords[12])
        col_val = float(calibrator._top_rails[0])

        candidates = calibrator.board_point_to_logic_candidates(row_val, col_val, k=3)

        assert candidates
        assert candidates[0] == ("13", "rail_top+")

    def test_t5_11_board_mask_alone_does_not_validate_inferred_hole(self):
        from app.pipeline.vision.calibrator import BreadboardCalibrator

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator._grid_matrix = np.array([[[50.0, 50.0]]], dtype=np.float32)
        calibrator._top_rail_matrix = np.array([[[10.0, 10.0], [20.0, 10.0]]], dtype=np.float32)
        calibrator._bot_rail_matrix = np.array([[[10.0, 90.0], [20.0, 90.0]]], dtype=np.float32)
        calibrator._observed_main_mask = np.array([[False]], dtype=bool)
        calibrator._observed_top_mask = np.array([[False, False]], dtype=bool)
        calibrator._observed_bot_mask = np.array([[False, False]], dtype=bool)
        calibrator._row_coords = np.array([50.0, 60.0], dtype=np.float32)
        calibrator._col_coords = np.array([50.0, 60.0], dtype=np.float32)
        calibrator.hole_centers = []
        calibrator._board_mask = np.ones((120, 120), dtype=np.uint8) * 255

        calibrator._compute_valid_hole_masks()

        assert calibrator._valid_main_mask is not None
        assert calibrator._valid_top_mask is not None
        assert calibrator._valid_bot_mask is not None
        assert bool(calibrator._valid_main_mask[0, 0]) is False
        assert bool(calibrator._valid_top_mask[0, 0]) is False
        assert bool(calibrator._valid_top_mask[0, 1]) is False
        assert bool(calibrator._valid_bot_mask[0, 0]) is False
        assert bool(calibrator._valid_bot_mask[0, 1]) is False

    def test_t5_12_detected_holes_json_maps_original_pixels(self):
        from pathlib import Path

        from app.pipeline.stages.s2_mapping import run_mapping
        from app.pipeline.vision.calibrator import BreadboardCalibrator

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        assert calibrator.load_detected_holes_json(
            Path("bread_detect/outputs/bread_4_white_holes.json")
        )

        candidates = calibrator.frame_pixel_to_logic_candidates(126.94, 549.51, k=3)
        assert candidates[0] == ("1", "a")

        result = run_mapping(
            components=[
                {
                    "component_id": "R1",
                    "component_type": "Resistor",
                    "pins": [
                        {
                            "pin_id": 1,
                            "pin_name": "pin1",
                            "keypoints_by_view": {"top": [126.94, 549.51]},
                            "visibility_by_view": {"top": 2},
                            "score_by_view": {"top": 0.95},
                            "source_by_view": {"top": "model"},
                            "confidence": 0.95,
                            "source": "model",
                        }
                    ],
                }
            ],
            calibrator=calibrator,
            image_shape=(0, 0),
            images_b64=None,
        )

        pin = result["components"][0]["pins"][0]
        assert result["calibration"]["mode"] == "detected_hole_map"
        assert pin["hole_id"] == "A1"
        assert pin["candidate_hole_ids"][0] == "A1"

    def test_t5_13_exact_board_point_returns_single_main_candidate(self):
        from app.pipeline.vision.calibrator import BreadboardCalibrator

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))

        assert calibrator._row_coords is not None
        assert calibrator._col_coords is not None

        nr = len(calibrator._row_coords)
        nc = len(calibrator._col_coords)
        calibrator._landscape = True
        calibrator._grid_matrix = np.zeros((nr, nc, 2), dtype=np.float32)
        for ri in range(nr):
            for ci in range(nc):
                calibrator._grid_matrix[ri, ci] = [calibrator._row_coords[ri], calibrator._col_coords[ci]]
        calibrator._valid_main_mask = np.ones((nr, nc), dtype=bool)

        row_val = float(calibrator._row_coords[12])
        col_val = float(calibrator._col_coords[1])

        candidates = calibrator.board_point_to_logic_candidates(row_val, col_val, k=5)

        assert candidates == [("13", "b")]
