"""
S1.5 pin detection tests.

Current contract:
- full-image YOLO-Pose is the only model-driven pin path
- no loaded pose model returns schema-compatible unavailable pin shells
- IC pins use board e/f bridge geometry when a calibrator is available
"""

from __future__ import annotations

import numpy as np
import pytest


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


def _run_s1_with_mock_detector(blank_image_b64, detections):
    from app.pipeline.stages.s1_detect import run_detect
    from tests.pipeline.mocks import MockComponentDetector

    return run_detect(images_b64=[blank_image_b64], detector=MockComponentDetector(detections))


class TestUnavailablePinPath:
    def test_no_pose_model_returns_unavailable_pin_shells(self, blank_image_b64):
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.pin_model import PinRoiDetector

        s1 = _run_s1_with_mock_detector(
            blank_image_b64,
            [{"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.95}],
        )
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=PinRoiDetector(model_path=None, device="cpu"),
        )

        assert result["interface_version"] == "component_pin_detect_v1"
        assert result["pin_detector_backend"] == "yolo_pose"
        assert result["pin_detector_mode"] == "unavailable"
        assert result["side_roi_assoc_backend"] == "removed"

        comp = result["components"][0]
        assert comp["pin_detector"]["backend_mode"] == "unavailable"
        assert comp["full_image_pose_match"]["matched"] is False
        assert comp["roi"]["source"] == "unavailable"
        assert len(comp["pins"]) == 2
        for pin in comp["pins"]:
            assert pin["source"] == "unavailable"
            assert pin["keypoints_by_view"]["top"] is None
            assert pin["visibility_by_view"]["top"] == 0

    def test_interface_metadata_survives_unavailable_path(self, blank_image_b64):
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.pin_model import PinRoiDetector

        s1 = _run_s1_with_mock_detector(
            blank_image_b64,
            [{"class_name": "LED", "bbox": (110, 210, 240, 250), "confidence": 0.9}],
        )
        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=PinRoiDetector(model_path=None, device="cpu"),
        )

        comp = result["components"][0]
        assert comp["input_pin_detect_interface_version"] == "component_pin_detect_v1"
        assert comp["pin_schema_id"] == "fixed_pins"
        assert [pin["pin_name"] for pin in comp["pins"]] == ["anode", "cathode"]


class TestFullImagePoseMainPath:
    def test_full_image_pose_becomes_default_main_path(self, blank_image_b64):
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.pin_model import PinRoiDetector

        s1 = _run_s1_with_mock_detector(
            blank_image_b64,
            [{"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.95}],
        )
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
        assert comp["full_image_pose_match"]["matched"] is True
        assert comp["pins"][0]["keypoints_by_view"]["top"] == [120.0, 240.0]
        assert comp["pins"][1]["keypoints_by_view"]["top"] == [280.0, 240.0]

    def test_potentiometer_does_not_steal_ic_pose_instance(self, blank_image_b64):
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.pin_model import PinRoiDetector

        s1 = _run_s1_with_mock_detector(
            blank_image_b64,
            [{"class_name": "Potentiometer", "bbox": (580, 190, 675, 280), "confidence": 0.9}],
        )
        pin_det = PinRoiDetector(model_path=None, device="cpu")
        pin_det.model = _FakeIcOnlyFullImagePoseModel()

        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=pin_det,
        )

        pins = result["components"][0]["pins"]
        assert [pin["pin_name"] for pin in pins] == ["terminal_a", "wiper", "terminal_b"]
        for pin in pins:
            assert pin["keypoints_by_view"]["top"] is None
            assert pin["source_by_view"]["top"] == "unavailable"
            assert pin["metadata"]["potentiometer_role_degraded_reason"] == "calibrator_unavailable"


class TestBoardGeometryPins:
    def test_ic_dip8_uses_ef_bridge_geometry_with_calibrator(self, blank_image_b64):
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from app.pipeline.vision.pin_model import PinRoiDetector

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        e_x = float(calibrator.col_coords[4])
        f_x = float(calibrator.col_coords[5])
        y_top = float(calibrator.row_coords[2]) - 1.0
        y_bot = float(calibrator.row_coords[5]) + 1.0
        s1 = _run_s1_with_mock_detector(
            blank_image_b64,
            [{"class_name": "IC", "package_type": "dip8", "bbox": (e_x - 4.0, y_top, f_x + 4.0, y_bot), "confidence": 0.9}],
        )

        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=PinRoiDetector(model_path=None, device="cpu"),
            calibrator=calibrator,
        )

        comp = result["components"][0]
        pins = comp["pins"]
        assert len(pins) == 8
        assert comp["pin_schema_id"] == "ic_dip_ef_bridge"
        assert {pin["source"] for pin in pins} == {"ic_ef_bridge_geometry"}
        assert sorted(pin["pin_id"] for pin in pins if pin["metadata"]["row_lock"] == "e") == [1, 2, 3, 4]
        assert sorted(pin["pin_id"] for pin in pins if pin["metadata"]["row_lock"] == "f") == [5, 6, 7, 8]

    def test_potentiometer_bbox_fallback_snaps_to_legal_triplet(self, blank_image_b64):
        from app.pipeline.stages.s1b_pin_detect import run_pin_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from app.pipeline.vision.pin_model import PinRoiDetector

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        e_x = float(calibrator.col_coords[4])
        ys = [float(calibrator.row_coords[i]) for i in (4, 5, 6)]
        s1 = _run_s1_with_mock_detector(
            blank_image_b64,
            [{"class_name": "Potentiometer", "bbox": (e_x - 6.0, ys[0] - 3.0, e_x + 6.0, ys[2] + 3.0), "confidence": 0.9}],
        )

        result = run_pin_detect(
            detections=s1["detections"],
            images_b64=[blank_image_b64],
            pin_detector=PinRoiDetector(model_path=None, device="cpu"),
            calibrator=calibrator,
        )

        pins = result["components"][0]["pins"]
        assert [pin["pin_name"] for pin in pins] == ["terminal_a", "wiper", "terminal_b"]
        assert {pin["source"] for pin in pins} == {"potentiometer_board_logic"}
        slots = pins[0]["metadata"]["pot_logic_slots"]
        assert [slot[1] for slot in slots] == ["e", "e", "e"]
        digits = [int(slot[0]) for slot in slots]
        assert digits == [digits[0] + i for i in range(3)]


class TestPinModelSchemaAlignment:
    def test_two_pin_components_ignore_third_padding_keypoint(self):
        from app.pipeline.vision.pin_model import _parse_model_keypoints

        parsed = _parse_model_keypoints(
            points=np.array([[12.0, 10.0], [88.0, 10.0], [0.0, 0.0]], dtype=np.float32),
            confs=np.array([0.95, 0.91, 0.0], dtype=np.float32),
            pin_count=2,
        )

        assert parsed.ordered_keypoints == [(12.0, 10.0), (88.0, 10.0)]
        assert parsed.raw_keypoint_count == 3
        assert parsed.used_keypoint_count == 2
        assert parsed.extra_keypoints_ignored == 1
        assert parsed.ignored_keypoints_reason == "schema_padding_for_2pin"

    def test_transistor_keeps_three_schema_keypoints(self):
        from app.pipeline.vision.pin_model import _parse_model_keypoints

        parsed = _parse_model_keypoints(
            points=np.array([[10.0, 20.0], [20.0, 20.0], [30.0, 20.0]], dtype=np.float32),
            confs=np.array([0.9, 0.92, 0.88], dtype=np.float32),
            pin_count=3,
        )

        assert parsed.ordered_keypoints == [(10.0, 20.0), (20.0, 20.0), (30.0, 20.0)]
        assert parsed.raw_keypoint_count == 3
        assert parsed.used_keypoint_count == 3
        assert parsed.extra_keypoints_ignored == 0
        assert parsed.ignored_keypoints_reason == ""
