"""
T4: Pin 检测测试 — 验证启发式 fallback 和多视图融合

关键：无 Pin 模型时自动走 heuristic_fallback（已实现）
"""

from __future__ import annotations

import logging
import cv2
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

    def test_t4_2_mock_3pin_horizontal_snap(self, blank_image_b64):
        """POT with 3 keypoints near a horizontal triplet snaps to 3 adjacent digit columns on one letter row."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        # Horizontal insert: three pins share letter "e", spread across digits 10..12.
        e_x = float(calibrator.col_coords[4])  # letter 'e' X
        ys = [float(calibrator.row_coords[i]) for i in (9, 10, 11)]  # digits 10, 11, 12

        pot_det = MockComponentDetector([
            {"class_name": "Potentiometer", "bbox": (e_x - 6.0, ys[0] - 4.0, e_x + 6.0, ys[2] + 4.0), "confidence": 0.9}
        ])
        pot_pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "terminal_b", "keypoint": (e_x, ys[2]), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "terminal_a", "keypoint": (e_x, ys[0]), "confidence": 0.9, "visibility": 2},
            {"pin_id": 3, "pin_name": "wiper", "keypoint": (e_x, ys[1]), "confidence": 0.9, "visibility": 2},
        ])

        s1 = run_detect(images_b64=[blank_image_b64], detector=pot_det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pot_pin,
            calibrator=calibrator,
        )

        comp = result["components"][0]
        assert comp["symmetry_group"] == [["terminal_a", "terminal_b"]]
        pins = comp["pins"]
        assert [pin["pin_name"] for pin in pins] == ["terminal_a", "wiper", "terminal_b"]
        for pin in pins:
            assert pin["metadata"]["potentiometer_role_source"] == "board_plane_3collinear_snap"
            assert pin["metadata"]["pot_orientation"] == "horizontal"
            assert pin["metadata"]["row_lock"] == "e"
        # logic_slots: same letter "e", 3 adjacent digits.
        slots = pins[0]["metadata"]["pot_logic_slots"]
        assert [s[1] for s in slots] == ["e", "e", "e"]
        digits = [int(s[0]) for s in slots]
        assert digits == [digits[0] + i for i in range(3)]
        # wiper sits on the middle hole's board point.
        assert pins[1]["metadata"]["board_2d_point"] == pytest.approx([e_x, ys[1]], abs=1e-3)
        # keypoints_by_view is the candidate hole's frame pixel (no perspective in synthetic mode).
        assert pins[0]["keypoints_by_view"]["top"] == pytest.approx([e_x, ys[0]], abs=1e-3)
        assert pins[1]["keypoints_by_view"]["top"] == pytest.approx([e_x, ys[1]], abs=1e-3)
        assert pins[2]["keypoints_by_view"]["top"] == pytest.approx([e_x, ys[2]], abs=1e-3)
        # Snap cost should be ~0 since keypoints sit exactly on holes.
        assert pins[0]["metadata"]["pot_snap_cost_sq"] < 1e-6

    def test_t4_2a_potentiometer_missing_terminal_snaps_to_correct_triplet(self, blank_image_b64):
        """One missing terminal: bbox fallback + remaining 2 keypoints still resolve the right 3-collinear triplet."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        e_x = float(calibrator.col_coords[4])
        ys = [float(calibrator.row_coords[i]) for i in (20, 21, 22)]  # digits 21,22,23

        pot_det = MockComponentDetector([
            {"class_name": "Potentiometer", "bbox": (e_x - 6.0, ys[0] - 4.0, e_x + 6.0, ys[2] + 4.0), "confidence": 0.9}
        ])
        pot_pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "terminal_a", "keypoint": (e_x, ys[0]), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "wiper", "keypoint": (e_x, ys[1]), "confidence": 0.9, "visibility": 2},
            {"pin_id": 3, "pin_name": "terminal_b", "keypoint": None, "confidence": 0.0, "visibility": 0},
        ])

        s1 = run_detect(images_b64=[blank_image_b64], detector=pot_det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pot_pin,
            calibrator=calibrator,
        )

        pins = result["components"][0]["pins"]
        assert [pin["pin_name"] for pin in pins] == ["terminal_a", "wiper", "terminal_b"]
        # Even with one missing keypoint the snap recovers the correct logical triplet.
        slots = pins[0]["metadata"]["pot_logic_slots"]
        digits = [int(s[0]) for s in slots]
        assert [s[1] for s in slots] == ["e", "e", "e"]
        assert digits == [21, 22, 23]
        # terminal_a / wiper keypoints land on holes 21 and 22 respectively.
        assert pins[0]["keypoints_by_view"]["top"] == pytest.approx([e_x, ys[0]], abs=1e-3)
        assert pins[1]["keypoints_by_view"]["top"] == pytest.approx([e_x, ys[1]], abs=1e-3)
        # terminal_b (previously missing) gets snapped to hole 23 by geometry.
        assert pins[2]["keypoints_by_view"]["top"] == pytest.approx([e_x, ys[2]], abs=1e-3)
        assert pins[2]["visibility_by_view"]["top"] == 2
        assert pins[2]["metadata"]["potentiometer_input_source"] == "potentiometer_bbox_fallback"

    def test_pot_vertical_snap(self, blank_image_b64):
        """竖插 POT: 三脚同一数字列, 跨同一半内 3 个相邻字母行."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        digit_y = float(calibrator.row_coords[14])  # digit 15
        # Vertical insert in the f-j half: letters f, g, h.
        xs = [float(calibrator.col_coords[i]) for i in (5, 6, 7)]

        pot_det = MockComponentDetector([
            {"class_name": "Potentiometer", "bbox": (xs[0] - 4.0, digit_y - 6.0, xs[2] + 4.0, digit_y + 6.0), "confidence": 0.9}
        ])
        pot_pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "terminal_a", "keypoint": (xs[0], digit_y), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "wiper", "keypoint": (xs[1], digit_y), "confidence": 0.9, "visibility": 2},
            {"pin_id": 3, "pin_name": "terminal_b", "keypoint": (xs[2], digit_y), "confidence": 0.9, "visibility": 2},
        ])
        s1 = run_detect(images_b64=[blank_image_b64], detector=pot_det)
        result = run_pin_detect(
            detections=s1["detections"], images_b64=[blank_image_b64], pin_detector=pot_pin, calibrator=calibrator,
        )

        pins = result["components"][0]["pins"]
        assert [p["pin_name"] for p in pins] == ["terminal_a", "wiper", "terminal_b"]
        for pin in pins:
            assert pin["metadata"]["pot_orientation"] == "vertical"
            assert pin["metadata"]["column_lock"] == "15"
        slots = pins[0]["metadata"]["pot_logic_slots"]
        letters = [s[1] for s in slots]
        assert letters == ["f", "g", "h"]
        assert all(s[0] == "15" for s in slots)
        # All three resulting keypoints share Y = digit 15 row.
        for pin in pins:
            assert pin["keypoints_by_view"]["top"][1] == pytest.approx(digit_y, abs=1e-3)

    def test_pot_jitter_stability(self, blank_image_b64):
        """Sub-pitch jitter on detected keypoints must not change the chosen 3-collinear triplet."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        e_x = float(calibrator.col_coords[4])
        ys = [float(calibrator.row_coords[i]) for i in (30, 31, 32)]
        pitch = float(calibrator.row_coords[1] - calibrator.row_coords[0])

        clean_slots: list[list[str]] | None = None
        for jitter in (-0.4, 0.0, 0.4):
            pot_det = MockComponentDetector([
                {"class_name": "Potentiometer", "bbox": (e_x - 6.0, ys[0] - 4.0, e_x + 6.0, ys[2] + 4.0), "confidence": 0.9}
            ])
            pot_pin = MockPinDetector([
                {"pin_id": 1, "pin_name": "terminal_a", "keypoint": (e_x + jitter * pitch, ys[0] + jitter * pitch * 0.5), "confidence": 0.9, "visibility": 2},
                {"pin_id": 2, "pin_name": "wiper", "keypoint": (e_x - jitter * pitch, ys[1] + jitter * pitch * 0.3), "confidence": 0.9, "visibility": 2},
                {"pin_id": 3, "pin_name": "terminal_b", "keypoint": (e_x, ys[2] - jitter * pitch * 0.4), "confidence": 0.9, "visibility": 2},
            ])
            s1 = run_detect(images_b64=[blank_image_b64], detector=pot_det)
            result = run_pin_detect(
                detections=s1["detections"], images_b64=[blank_image_b64], pin_detector=pot_pin, calibrator=calibrator,
            )
            pins = result["components"][0]["pins"]
            slots = [list(s) for s in pins[0]["metadata"]["pot_logic_slots"]]
            if clean_slots is None:
                clean_slots = slots
            else:
                assert slots == clean_slots, f"jitter {jitter} changed triplet"

    def test_ic_refused_without_calibrator(self, blank_image_b64):
        """No calibrator → IC pin output is refused (empty pins, downstream skips)."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        ic_det = MockComponentDetector([
            {"class_name": "IC", "package_type": "dip8", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        s1 = run_detect(images_b64=[blank_image_b64], detector=ic_det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=MockPinDetector([]),
        )

        comp = result["components"][0]
        assert comp["pins"] == []
        assert comp["ic_geometry"]["calibrator_used"] is False

    def test_pot_snap_constrained_to_body_footprint(self, blank_image_b64):
        """Pose keypoints that land on *visible* holes adjacent to the body must NOT
        win the snap — physical 3296 pins are hidden under the body, so the chosen
        triplet must lie inside the bbox footprint even if outside-body holes are
        closer to the (misleading) model keypoints.
        """
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        d_x = float(calibrator.col_coords[3])  # letter 'd'
        e_x = float(calibrator.col_coords[4])  # letter 'e' (inside body)
        f_x = float(calibrator.col_coords[5])  # letter 'f' (visible holes beside body)
        ys_inside = [float(calibrator.row_coords[i]) for i in (19, 20, 21)]  # digits 20..22

        # Body covers letters d..e, digits 20..22 (POT body footprint).
        bbox = (d_x - 4.0, ys_inside[0] - 4.0, e_x + 4.0, ys_inside[2] + 4.0)
        pot_det = MockComponentDetector([
            {"class_name": "Potentiometer", "bbox": bbox, "confidence": 0.9}
        ])
        # Misleading model keypoints: on visible holes at letter 'f', OUTSIDE the body bbox.
        pot_pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "terminal_a", "keypoint": (f_x, ys_inside[0]), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "wiper", "keypoint": (f_x, ys_inside[1]), "confidence": 0.9, "visibility": 2},
            {"pin_id": 3, "pin_name": "terminal_b", "keypoint": (f_x, ys_inside[2]), "confidence": 0.9, "visibility": 2},
        ])
        s1 = run_detect(images_b64=[blank_image_b64], detector=pot_det)
        result = run_pin_detect(
            detections=s1["detections"], images_b64=[blank_image_b64], pin_detector=pot_pin, calibrator=calibrator,
        )

        pins = result["components"][0]["pins"]
        for pin in pins:
            assert pin["metadata"]["pot_body_constrained"] is True
            assert pin["metadata"]["pot_orientation"] == "horizontal"
            # Snap must land on letter 'e' (inside body), NOT letter 'f' (visible holes outside body).
            assert pin["metadata"]["row_lock"] == "e"
        # Verify all 3 snapped board points sit on letter 'e' (X = e_x), not 'f'.
        for pin in pins:
            assert pin["metadata"]["board_2d_point"][0] == pytest.approx(e_x, abs=1e-3)

    def test_pot_all_keypoints_missing_uses_board_fallback(self, blank_image_b64):
        """All 3 pose keypoints missing: bbox + calibrator alone should produce a legal triplet."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        e_x = float(calibrator.col_coords[4])
        # bbox roughly covers digits 5..7 on letter 'e' (horizontal insert), no real keypoints provided.
        ys = [float(calibrator.row_coords[i]) for i in (4, 5, 6)]
        pot_det = MockComponentDetector([
            {"class_name": "Potentiometer", "bbox": (e_x - 6.0, ys[0] - 3.0, e_x + 6.0, ys[2] + 3.0), "confidence": 0.9}
        ])
        pot_pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "terminal_a", "keypoint": None, "confidence": 0.0, "visibility": 0},
            {"pin_id": 2, "pin_name": "wiper", "keypoint": None, "confidence": 0.0, "visibility": 0},
            {"pin_id": 3, "pin_name": "terminal_b", "keypoint": None, "confidence": 0.0, "visibility": 0},
        ])
        s1 = run_detect(images_b64=[blank_image_b64], detector=pot_det)
        result = run_pin_detect(
            detections=s1["detections"], images_b64=[blank_image_b64], pin_detector=pot_pin, calibrator=calibrator,
        )

        pins = result["components"][0]["pins"]
        # All three input keypoints came from the board-plane bbox fallback, so the
        # 3 snapped holes form a legal triplet on letter 'e' across 3 adjacent digits.
        for pin in pins:
            assert pin["metadata"]["pot_orientation"] == "horizontal"
            assert pin["metadata"]["row_lock"] == "e"
        slots = pins[0]["metadata"]["pot_logic_slots"]
        letters = [s[1] for s in slots]
        digits = [int(s[0]) for s in slots]
        assert letters == ["e", "e", "e"]
        assert digits == [digits[0] + i for i in range(3)]
        # Snap cost should be ~0 since fallback already lands on hole positions.
        assert pins[0]["metadata"]["pot_snap_cost_sq"] < 1e-3

    def test_pot_refused_without_calibrator(self, blank_image_b64):
        """No calibrator → POT pin output is refused (no keypoints, degraded reason set)."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        pot_det = MockComponentDetector([
            {"class_name": "Potentiometer", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        pot_pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "terminal_a", "keypoint": (100.0, 240.0), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "wiper", "keypoint": (200.0, 240.0), "confidence": 0.9, "visibility": 2},
            {"pin_id": 3, "pin_name": "terminal_b", "keypoint": (300.0, 240.0), "confidence": 0.9, "visibility": 2},
        ])
        s1 = run_detect(images_b64=[blank_image_b64], detector=pot_det)
        result = run_pin_detect(
            detections=s1["detections"], images_b64=[blank_image_b64], pin_detector=pot_pin,
        )

        pins = result["components"][0]["pins"]
        assert [p["pin_name"] for p in pins] == ["terminal_a", "wiper", "terminal_b"]
        for pin in pins:
            assert pin["source"] == "unavailable"
            assert pin["confidence"] == 0.0
            assert pin["visibility_by_view"]["top"] == 0
            assert pin["keypoints_by_view"]["top"] is None
            assert pin["metadata"]["potentiometer_role_source"] == "refused"
            assert pin["metadata"]["potentiometer_role_degraded_reason"] == "calibrator_unavailable"

    def test_t4_2b_mock_transistor_3pin(self, blank_image_b64):
        """模型新标签 transistor_3pin 会被解释为 3-pin Transistor."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        det = MockComponentDetector([
            {"class_name": "transistor_3pin", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "pin1", "keypoint": (100.0, 240.0), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "pin2", "keypoint": (200.0, 240.0), "confidence": 0.9, "visibility": 2},
            {"pin_id": 3, "pin_name": "pin3", "keypoint": (300.0, 240.0), "confidence": 0.9, "visibility": 2},
        ])

        s1 = run_detect(images_b64=[blank_image_b64], detector=det)
        result = run_pin_detect(detections=s1["detections"], images_b64=[blank_image_b64], pin_detector=pin)

        assert result["components"][0]["component_type"] == "Transistor"
        assert result["components"][0]["package_type"] == "transistor_3pin"
        assert len(result["components"][0]["pins"]) == 3

    def test_t4_2c_electrolytic_pin_names(self, blank_image_b64):
        """细粒度电解电容默认 pin_name 为 positive/negative."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        det = MockComponentDetector([
            {"class_name": "capacitor_electrolytic", "bbox": (100, 200, 240, 320), "confidence": 0.9}
        ])
        pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "positive", "keypoint": (120.0, 240.0), "confidence": 0.9, "visibility": 2},
            {"pin_id": 2, "pin_name": "negative", "keypoint": (200.0, 240.0), "confidence": 0.9, "visibility": 2},
        ])

        s1 = run_detect(images_b64=[blank_image_b64], detector=det)
        result = run_pin_detect(detections=s1["detections"], images_b64=[blank_image_b64], pin_detector=pin)

        pin_names = [p["pin_name"] for p in result["components"][0]["pins"]]
        assert result["components"][0]["component_type"] == "CapacitorElectrolytic"
        assert result["components"][0]["package_type"] == "capacitor_electrolytic_2pin"
        assert pin_names == ["positive", "negative"]

    def test_t4_3_mock_ic_dip8(self, blank_image_b64):
        """IC DIP-8 走 e/f-bridge 几何路径, 不再走 anchor pair."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        e_x = float(calibrator.col_coords[4])
        f_x = float(calibrator.col_coords[5])
        y_top = float(calibrator.row_coords[2]) - 1.0
        y_bot = float(calibrator.row_coords[5]) + 1.0

        ic_det = MockComponentDetector([
            {"class_name": "IC", "package_type": "dip8", "bbox": (e_x - 4.0, y_top, f_x + 4.0, y_bot), "confidence": 0.9}
        ])
        # Pin detector intentionally returns wrong data — IC path must ignore it.
        ic_pin = MockPinDetector([
            {"pin_id": 1, "pin_name": "ignored", "keypoint": (0.0, 0.0), "confidence": 0.0, "visibility": 0},
        ])

        s1 = run_detect(images_b64=[blank_image_b64], detector=ic_det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=ic_pin,
            calibrator=calibrator,
        )

        comp = result["components"][0]
        pins = comp["pins"]
        assert len(pins) == 8
        assert comp["package_type"] == "dip8"
        assert comp["pin_schema_id"] == "ic_dip_ef_bridge"
        # Every pin lives on row e or f and is sourced from the bridge-geometry path.
        for pin in pins:
            assert pin["source"] == "ic_ef_bridge_geometry"
            assert pin["source_by_view"]["top"] == "ic_ef_bridge_geometry"
            assert pin["metadata"]["row_lock"] in {"e", "f"}
            assert pin["metadata"]["package_type"] == "dip8"
            assert pin["metadata"]["notch_direction"] == "left"
            assert pin["metadata"]["numbering_rule"] == "counterclockwise"
            assert "estimated_column" in pin["metadata"]
        # DIP8 with notch=left: pin1..pin4 on e row, pin5..pin8 on f row.
        e_pin_ids = sorted(p["pin_id"] for p in pins if p["metadata"]["row_lock"] == "e")
        f_pin_ids = sorted(p["pin_id"] for p in pins if p["metadata"]["row_lock"] == "f")
        assert e_pin_ids == [1, 2, 3, 4]
        assert f_pin_ids == [5, 6, 7, 8]
        # With board-logic layout: e-row pins all share X = letter 'e' X; spread along Y (digit axis).
        e_row = {p["pin_id"]: p["keypoints_by_view"]["top"] for p in pins if p["metadata"]["row_lock"] == "e"}
        f_row = {p["pin_id"]: p["keypoints_by_view"]["top"] for p in pins if p["metadata"]["row_lock"] == "f"}
        # notch=left: pin1 at lowest digit (smaller Y) → opposite end is pin4.
        assert e_row[1][1] < e_row[4][1]
        # pin5 is f-row pin at digit 6 (same as pin4); pin8 at digit 3 (same as pin1).
        assert abs(e_row[4][1] - f_row[5][1]) < 1e-3
        assert abs(e_row[1][1] - f_row[8][1]) < 1e-3
        # e-row sits on the smaller-X side (letter 'e'); f-row on letter 'f' (larger X).
        assert e_row[1][0] < f_row[8][0]

    def test_t4_3b_mock_ic_dip14(self, blank_image_b64):
        """DIP-14 应该输出 14 个引脚, e/f 行各 7 个."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        e_x = float(calibrator.col_coords[4])
        f_x = float(calibrator.col_coords[5])
        y_top = float(calibrator.row_coords[2]) - 1.0
        y_bot = float(calibrator.row_coords[8]) + 1.0  # 7 consecutive digits

        ic_det = MockComponentDetector([
            {"class_name": "IC", "package_type": "dip14", "bbox": (e_x - 5.0, y_top, f_x + 5.0, y_bot), "confidence": 0.85}
        ])
        ic_pin = MockPinDetector([])

        s1 = run_detect(images_b64=[blank_image_b64], detector=ic_det)
        # S1's component-detect contract always emits dip8 for IC; tests that
        # exercise the DIP14 branch override the package_type after S1.
        for det in s1["detections"]:
            det["package_type"] = "dip14"
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=ic_pin,
            calibrator=calibrator,
        )

        comp = result["components"][0]
        pins = comp["pins"]
        assert len(pins) == 14
        assert comp["package_type"] == "dip14"
        for pin in pins:
            assert pin["source"] == "ic_ef_bridge_geometry"
            assert pin["metadata"]["package_type"] == "dip14"
            assert pin["metadata"]["row_lock"] in {"e", "f"}
        e_pin_ids = sorted(p["pin_id"] for p in pins if p["metadata"]["row_lock"] == "e")
        f_pin_ids = sorted(p["pin_id"] for p in pins if p["metadata"]["row_lock"] == "f")
        assert e_pin_ids == list(range(1, 8))
        assert f_pin_ids == list(range(8, 15))

    def test_t4_3c_mock_ic_dip8_notch_right(self, blank_image_b64):
        """notch=right 时 pin1 应该落在 e 行靠右一端."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        e_x = float(calibrator.col_coords[4])
        f_x = float(calibrator.col_coords[5])
        y_top = float(calibrator.row_coords[2]) - 1.0
        y_bot = float(calibrator.row_coords[5]) + 1.0

        ic_det = MockComponentDetector([
            {
                "class_name": "IC",
                "package_type": "dip8",
                "bbox": (e_x - 4.0, y_top, f_x + 4.0, y_bot),
                "confidence": 0.9,
                "notch_direction": "right",
            }
        ])
        ic_pin = MockPinDetector([])

        s1 = run_detect(images_b64=[blank_image_b64], detector=ic_det)
        # ``notch_direction`` is not part of the standard detection contract; we
        # patch it onto the dict so the IC helper sees it.
        for det in s1["detections"]:
            det["notch_direction"] = "right"
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=ic_pin,
            calibrator=calibrator,
        )

        pins = result["components"][0]["pins"]
        e_row = {p["pin_id"]: p["keypoints_by_view"]["top"] for p in pins if p["metadata"]["row_lock"] == "e"}
        # notch=right → pin1 sits at the *higher* digit (larger Y on synthetic grid).
        assert e_row[1][1] > e_row[4][1]
        for pin in pins:
            assert pin["metadata"]["notch_direction"] == "right"

    def test_t4_3d_dip8_board_logic_locks_to_ef_rows(self, blank_image_b64):
        """calibrator 就绪时, DIP8 8 个槽位必须是 4 个连续数字列 × (e, f) 两行。

        关键: pin board_2d_point 应等于 calibrator.logic_to_board_point((col, 'e'/'f'))。
        不允许沿 a-j 字母行方向展开 (那是旧实现的 bug, 会把 IC 旋转 90 度).
        """
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        # synthetic grid: row_coords 沿 Y 轴 (数字列), col_coords 沿 X 轴 (字母行).
        # 把 bbox 摆成"数字列 3..6 × e/f 行"对应的 board 区域.
        row_coords = calibrator.row_coords
        col_coords = calibrator.col_coords
        e_x = float(col_coords[4])  # 'e' 字母行 X
        f_x = float(col_coords[5])  # 'f' 字母行 X
        y_top = float(row_coords[2]) - 1.0  # 数字列 3 (1-indexed)
        y_bot = float(row_coords[5]) + 1.0  # 数字列 6

        ic_det = MockComponentDetector([
            {
                "class_name": "IC",
                "package_type": "dip8",
                "bbox": (e_x - 4.0, y_top, f_x + 4.0, y_bot),
                "confidence": 0.9,
            }
        ])
        ic_pin = MockPinDetector([])
        s1 = run_detect(images_b64=[blank_image_b64], detector=ic_det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=ic_pin,
            calibrator=calibrator,
        )

        pins = result["components"][0]["pins"]
        assert len(pins) == 8
        for pin in pins:
            assert pin["metadata"]["column_source"] == "board_logic"
            assert pin["metadata"]["digit_column_label"] is not None
            assert "board_2d_point" in pin["metadata"]

        # 必须是 4 个**连续**数字列, 不允许跨字母行.
        e_cols = sorted(int(p["metadata"]["digit_column_label"]) for p in pins if p["metadata"]["row_lock"] == "e")
        f_cols = sorted(int(p["metadata"]["digit_column_label"]) for p in pins if p["metadata"]["row_lock"] == "f")
        assert len(e_cols) == 4
        assert e_cols == [e_cols[0] + i for i in range(4)]  # 4 个连续
        assert e_cols == f_cols  # e 与 f 行覆盖同 4 列
        # bbox 中点对应数字列 3..6 的中点 ≈ 4.5, 期望窗口落在 3..6.
        assert e_cols == [3, 4, 5, 6]

        # board_2d_point 与 logic_to_board_point 完全一致.
        for pin in pins:
            col_label = pin["metadata"]["digit_column_label"]
            row_letter = pin["metadata"]["row_lock"]
            expected = calibrator.logic_to_board_point((col_label, row_letter))
            actual = pin["metadata"]["board_2d_point"]
            assert abs(actual[0] - expected[0]) < 1e-3
            assert abs(actual[1] - expected[1]) < 1e-3

        # notch=left + 4 连续数字列: pin1..pin4 沿 e 行(数字列 3..6), pin5..pin8 沿 f 行(数字列 6..3 逆序).
        pin_by_id = {p["pin_id"]: p for p in pins}
        assert pin_by_id[1]["metadata"]["row_lock"] == "e"
        assert pin_by_id[1]["metadata"]["digit_column_label"] == "3"
        assert pin_by_id[4]["metadata"]["digit_column_label"] == "6"
        assert pin_by_id[5]["metadata"]["row_lock"] == "f"
        assert pin_by_id[5]["metadata"]["digit_column_label"] == "6"
        assert pin_by_id[8]["metadata"]["digit_column_label"] == "3"

    def test_t4_3e_dip14_board_logic_locks_to_ef_rows(self, blank_image_b64):
        """DIP14: e/f 行各 7 个连续数字列, 总 14 个 pin, 每个 pin 的 board_2d_point 由 calibrator 决定."""
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        row_coords = calibrator.row_coords
        col_coords = calibrator.col_coords
        e_x = float(col_coords[4])
        f_x = float(col_coords[5])
        # 数字列 10..16 (1-indexed) — 7 连续列.
        y_top = float(row_coords[9]) - 1.0
        y_bot = float(row_coords[15]) + 1.0

        ic_det = MockComponentDetector([
            {
                "class_name": "IC",
                "package_type": "dip14",
                "bbox": (e_x - 5.0, y_top, f_x + 5.0, y_bot),
                "confidence": 0.85,
            }
        ])
        ic_pin = MockPinDetector([])
        s1 = run_detect(images_b64=[blank_image_b64], detector=ic_det)
        for det in s1["detections"]:
            det["package_type"] = "dip14"
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=ic_pin,
            calibrator=calibrator,
        )

        pins = result["components"][0]["pins"]
        assert len(pins) == 14
        e_cols = sorted(int(p["metadata"]["digit_column_label"]) for p in pins if p["metadata"]["row_lock"] == "e")
        f_cols = sorted(int(p["metadata"]["digit_column_label"]) for p in pins if p["metadata"]["row_lock"] == "f")
        assert len(e_cols) == 7
        assert e_cols == [e_cols[0] + i for i in range(7)]
        assert e_cols == f_cols
        for pin in pins:
            assert pin["metadata"]["column_source"] == "board_logic"
            expected = calibrator.logic_to_board_point(
                (pin["metadata"]["digit_column_label"], pin["metadata"]["row_lock"])
            )
            actual = pin["metadata"]["board_2d_point"]
            assert abs(actual[0] - expected[0]) < 1e-3
            assert abs(actual[1] - expected[1]) < 1e-3

    def test_t4_3f_board_logic_ignores_bbox_aspect(self, blank_image_b64):
        """即使 bbox 是 image-frame 'horizontal' (宽 > 高), 只要 board plane 上覆盖
        的是 4 个连续数字列, board_logic 路径就必须沿数字列方向出 pin, 不沿字母行.

        这条用例直接锁死"不再用 bbox 长短轴决定 IC 朝向"的物理要求.
        """
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector, MockPinDetector
        from app.pipeline.stages.s1_detect import run_detect

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        col_coords = calibrator.col_coords
        row_coords = calibrator.row_coords
        e_x = float(col_coords[4])
        f_x = float(col_coords[5])
        y_top = float(row_coords[2]) - 1.0
        y_bot = float(row_coords[5]) + 1.0

        # bbox 宽 = e/f 字母行间距 (~77px), 高 = 4 数字列间距 (~28px) -> image-frame 是 "横向".
        bbox = (e_x - 4.0, y_top, f_x + 4.0, y_bot)
        assert (bbox[2] - bbox[0]) > (bbox[3] - bbox[1]), "测试前提: bbox 横向"

        ic_det = MockComponentDetector([
            {"class_name": "IC", "package_type": "dip8", "bbox": bbox, "confidence": 0.9}
        ])
        s1 = run_detect(images_b64=[blank_image_b64], detector=ic_det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=MockPinDetector([]),
            calibrator=calibrator,
        )

        pins = result["components"][0]["pins"]
        # e 行的 4 个 pin: board_2d_point 的 X 都应该等于 e_x (字母行 X),
        # Y 在 4 个连续数字列的 Y 坐标上变化.
        e_points = [p["metadata"]["board_2d_point"] for p in pins if p["metadata"]["row_lock"] == "e"]
        f_points = [p["metadata"]["board_2d_point"] for p in pins if p["metadata"]["row_lock"] == "f"]
        assert len(e_points) == 4
        assert len(f_points) == 4
        for pt in e_points:
            assert abs(pt[0] - e_x) < 1e-3, f"e-row pin X 应等于字母行 e 的 X={e_x}, 实际 {pt[0]}"
        for pt in f_points:
            assert abs(pt[0] - f_x) < 1e-3, f"f-row pin X 应等于字母行 f 的 X={f_x}, 实际 {pt[0]}"
        # e 行 4 pin 的 Y 必须沿数字列方向变化 (不是常数), 即 spread > 0.
        e_y_spread = max(p[1] for p in e_points) - min(p[1] for p in e_points)
        assert e_y_spread > 5.0, "e 行 pin 应沿数字列 (Y 轴) 展开, 不是聚成一点"

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
                "source_model_type": "yolo_detect_component",
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


class MockEdgeRetryPinDetector:
    backend_mode = "mock_model"
    backend_type = "mock_pose"
    interface_version = "pin_detector_v1"

    def __init__(self):
        self.calls = 0

    def predict_component_pins(self, **kwargs):
        from app.pipeline.vision.pin_model import PinPrediction

        self.calls += 1
        offset_x, offset_y = kwargs.get("roi_offset", (0, 0))

        if self.calls == 1:
            pin1 = (float(offset_x), float(offset_y + 20))
            pin2 = (float(offset_x + 40), float(offset_y + 20))
        else:
            pin1 = (float(offset_x + 24), float(offset_y + 24))
            pin2 = (float(offset_x + 80), float(offset_y + 24))

        return [
            PinPrediction(
                pin_id=1,
                pin_name="pin1",
                keypoint=pin1,
                confidence=0.95,
                visibility=2,
                source="mock_model",
                metadata={"retry_call": self.calls},
            ),
            PinPrediction(
                pin_id=2,
                pin_name="pin2",
                keypoint=pin2,
                confidence=0.95,
                visibility=2,
                source="mock_model",
                metadata={"retry_call": self.calls},
            ),
        ]


def mock_pin_detector_2pin():
    """Factory: 创建 2-pin Mock PinDetector."""
    from tests.pipeline.mocks import MockPinDetector
    return MockPinDetector([
        {"pin_id": 1, "pin_name": "pin1", "keypoint": (120.0, 240.0), "confidence": 0.95, "visibility": 2},
        {"pin_id": 2, "pin_name": "pin2", "keypoint": (280.0, 240.0), "confidence": 0.95, "visibility": 2},
    ])


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.array(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)


class _FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = [_FakeTensor(xy)]
        self.conf = [_FakeTensor(conf)]


class _FakePoseResult:
    def __init__(self, xy, conf):
        self.keypoints = _FakeKeypoints(xy, conf)


class _FakePoseModel:
    def __init__(self, xy, conf):
        self._xy = xy
        self._conf = conf

    def __call__(self, roi_image, verbose=False, device="cpu"):
        return [_FakePoseResult(self._xy, self._conf)]


class _FakeTensorList:
    def __init__(self, arr):
        self._arr = np.array(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)


class _FakeBoxes:
    def __init__(self, xyxy, cls_ids, confs):
        self.xyxy = _FakeTensorList(xyxy)
        self.cls = _FakeTensorList(cls_ids)
        self.conf = _FakeTensorList(confs)


class _FakeFullImageKeypoints:
    def __init__(self, xy, conf):
        self.xy = _FakeTensorList(xy)
        self.conf = _FakeTensorList(conf)


class _FakeFullImagePoseResult:
    def __init__(self, *, xyxy, cls_ids, box_confs, kpts_xy, kpts_conf):
        self.boxes = _FakeBoxes(xyxy, cls_ids, box_confs)
        self.keypoints = _FakeFullImageKeypoints(kpts_xy, kpts_conf)


class _FakeFullImagePoseModel:
    names = {0: "Resistor"}

    def __call__(self, image, verbose=False, device="cpu"):
        return [
            _FakeFullImagePoseResult(
                xyxy=[[100.0, 200.0, 300.0, 260.0]],
                cls_ids=[0],
                box_confs=[0.93],
                kpts_xy=[[[120.0, 240.0], [280.0, 240.0], [0.0, 0.0]]],
                kpts_conf=[[0.95, 0.92, 0.0]],
            )
        ]


class _FakeIcOnlyFullImagePoseModel:
    names = {0: "IC"}

    def __call__(self, image, verbose=False, device="cpu"):
        return [
            _FakeFullImagePoseResult(
                xyxy=[[760.0, 260.0, 930.0, 390.0]],
                cls_ids=[0],
                box_confs=[0.93],
                kpts_xy=[[[790.0, 300.0], [820.0, 315.0], [850.0, 315.0]]],
                kpts_conf=[[0.95, 0.92, 0.91]],
            )
        ]


class TestPinModelSchemaAlignment:
    def test_two_pin_components_ignore_third_padding_keypoint(self):
        from app.pipeline.vision.pin_model import PinRoiDetector

        detector = PinRoiDetector(model_path=None, device="cpu")
        detector.model = _FakePoseModel(
            xy=[
                [12.0, 10.0],
                [88.0, 10.0],
                [0.0, 0.0],
            ],
            conf=[0.95, 0.91, 0.0],
        )

        preds = detector.predict_component_pins(
            component_id="LED1",
            component_type="LED",
            package_type="led_2pin",
            pin_schema_id="fixed_pins",
            roi_image=np.zeros((32, 100, 3), dtype=np.uint8),
            roi_offset=(5, 7),
            view_id="top",
            confidence=0.9,
        )

        assert len(preds) == 2
        assert [p.pin_name for p in preds] == ["anode", "cathode"]
        assert preds[0].keypoint == (17.0, 17.0)
        assert preds[1].keypoint == (93.0, 17.0)
        assert preds[0].metadata["raw_keypoint_count"] == 3
        assert preds[0].metadata["used_keypoint_count"] == 2
        assert preds[0].metadata["extra_keypoints_ignored"] == 1
        assert preds[0].metadata["ignored_keypoints_reason"] == "schema_padding_for_2pin"

    def test_transistor_keeps_three_schema_keypoints(self):
        from app.pipeline.vision.pin_model import PinRoiDetector

        detector = PinRoiDetector(model_path=None, device="cpu")
        detector.model = _FakePoseModel(
            xy=[
                [10.0, 20.0],
                [20.0, 20.0],
                [30.0, 20.0],
            ],
            conf=[0.9, 0.92, 0.88],
        )

        preds = detector.predict_component_pins(
            component_id="Q1",
            component_type="Transistor",
            package_type="transistor_3pin",
            pin_schema_id="fixed_3pins",
            roi_image=np.zeros((40, 40, 3), dtype=np.uint8),
            roi_offset=(1, 2),
            view_id="top",
            confidence=0.9,
        )

        assert len(preds) == 3
        assert [p.pin_name for p in preds] == ["pin1", "pin2", "pin3"]
        assert preds[2].keypoint == (31.0, 22.0)
        assert preds[0].metadata["raw_keypoint_count"] == 3
        assert preds[0].metadata["used_keypoint_count"] == 3
        assert preds[0].metadata["extra_keypoints_ignored"] == 0
        assert preds[0].metadata["ignored_keypoints_reason"] == ""


class TestAdaptiveRoiRetry:
    def test_retry_when_model_keypoint_is_on_roi_edge(self, blank_image_b64):
        from app.pipeline.stages.s1_detect import run_detect
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from tests.pipeline.mocks import MockComponentDetector

        det = MockComponentDetector([
            {"class_name": "CapacitorCeramic", "bbox": (100, 200, 170, 250), "confidence": 0.95}
        ])
        pin = MockEdgeRetryPinDetector()

        s1 = run_detect(images_b64=[blank_image_b64], detector=det)
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pin,
        )

        comp = result["components"][0]
        assert pin.calls >= 2
        assert comp["roi"]["retry_attempts"] >= 2
        assert float(comp["roi"]["scale_multiplier"]) > 1.0


class TestFullImagePoseMainPath:
    def test_full_image_pose_becomes_default_main_path(self, blank_image_b64):
        from app.pipeline.stages.s1_detect import run_detect
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.pin_model import PinRoiDetector
        from tests.pipeline.mocks import MockComponentDetector

        det = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.95}
        ])
        s1 = run_detect(images_b64=[blank_image_b64], detector=det)

        pin_det = PinRoiDetector(model_path=None, device="cpu")
        pin_det.model = _FakeFullImagePoseModel()

        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pin_det,
        )

        assert result["pin_detector_mode"] == "full_image_model"
        assert result["side_roi_assoc_backend"] == "not_applicable_full_image_pose"
        comp = result["components"][0]
        assert comp["roi"]["source"] == "full_image_pose"
        assert comp["pins"][0]["keypoints_by_view"]["top"] == [120.0, 240.0]
        assert comp["pins"][1]["keypoints_by_view"]["top"] == [280.0, 240.0]

    def test_potentiometer_does_not_steal_ic_pose_instance(self, blank_image_b64):
        from app.pipeline.stages.s1_detect import run_detect
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.pin_model import PinRoiDetector
        from tests.pipeline.mocks import MockComponentDetector

        det = MockComponentDetector([
            {"class_name": "Potentiometer", "bbox": (580, 190, 675, 280), "confidence": 0.9}
        ])
        s1 = run_detect(images_b64=[blank_image_b64], detector=det)

        pin_det = PinRoiDetector(model_path=None, device="cpu")
        pin_det.model = _FakeIcOnlyFullImagePoseModel()

        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pin_det,
        )

        pins = result["components"][0]["pins"]
        assert [pin["pin_name"] for pin in pins] == ["terminal_a", "wiper", "terminal_b"]
        # No calibrator → POT geometry path refuses output rather than stealing
        # the IC's full-image pose keypoints.
        for pin in pins:
            assert pin["keypoints_by_view"]["top"] is None
            assert pin["source_by_view"]["top"] == "unavailable"
            assert pin["metadata"]["potentiometer_role_degraded_reason"] == "calibrator_unavailable"
