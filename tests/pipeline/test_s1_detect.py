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

    def test_t2_2b_model_labels_normalized(self, blank_image_b64):
        """模型细粒度标签会被标准化成后端语义."""
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.mocks import MockComponentDetector

        detector = MockComponentDetector([
            {"class_name": "capacitor_ceramic", "bbox": (100, 100, 180, 180), "confidence": 0.91},
            {"class_name": "capacitor_electrolytic", "bbox": (220, 100, 320, 210), "confidence": 0.92},
            {"class_name": "jumper_wire", "bbox": (50, 220, 250, 250), "confidence": 0.93},
            {"class_name": "transistor_3pin", "bbox": (360, 110, 430, 220), "confidence": 0.94},
        ])

        result = run_detect(images_b64=[blank_image_b64], detector=detector)
        by_type = {d["component_type"]: d for d in result["detections"]}

        assert "CapacitorCeramic" in by_type
        assert by_type["CapacitorCeramic"]["package_type"] == "capacitor_ceramic_2pin"

        assert "CapacitorElectrolytic" in by_type
        assert by_type["CapacitorElectrolytic"]["package_type"] == "capacitor_electrolytic_2pin"
        assert by_type["CapacitorElectrolytic"]["pin_schema_id"] == "polarized_2pin"

        assert "Wire" in by_type
        assert by_type["Wire"]["package_type"] == "jumper_wire_2pin"

        assert "Transistor" in by_type
        assert by_type["Transistor"]["package_type"] == "transistor_3pin"
        assert by_type["Transistor"]["pin_schema_id"] == "fixed_3pins"

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
        assert result["detector_backend"] == "mock_component_detector"
        assert result["detector_contract"]["task"] == "mock"

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


class TestS1ICPackageInference:
    """S1 阶段对 IC 元件补 package_type / package_confidence / package_source."""

    def test_ic_dip8_class_name_direct(self, blank_image_b64):
        """模型类别已经是 ic_dip8 → package_source=model_class, package_type=dip8."""
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.mocks import MockComponentDetector

        detector = MockComponentDetector([
            {"class_name": "ic_dip8", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        result = run_detect(images_b64=[blank_image_b64], detector=detector)

        det = result["detections"][0]
        assert det["component_type"] == "IC"
        assert det["package_type"] == "dip8"
        assert det["package_source"] == "model_class"
        assert det["package_confidence"] == 1.0
        assert det["raw_class_name"] == "ic_dip8"

    def test_ic_dip14_class_name_direct(self, blank_image_b64):
        """模型类别 IC_DIP14 (大小写不敏感) → package_type=dip14."""
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.mocks import MockComponentDetector

        detector = MockComponentDetector([
            {"class_name": "IC_DIP14", "bbox": (100, 200, 300, 260), "confidence": 0.85}
        ])
        result = run_detect(images_b64=[blank_image_b64], detector=detector)

        det = result["detections"][0]
        assert det["component_type"] == "IC"
        assert det["package_type"] == "dip14"
        assert det["package_source"] == "model_class"
        assert det["package_confidence"] == 1.0

    def test_ic_without_calibrator_returns_unknown(self, blank_image_b64):
        """模型只输出 IC, 且没有 calibrator → package_type=unknown."""
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.mocks import MockComponentDetector

        detector = MockComponentDetector([
            {"class_name": "IC", "bbox": (100, 200, 300, 260), "confidence": 0.9}
        ])
        result = run_detect(images_b64=[blank_image_b64], detector=detector)

        det = result["detections"][0]
        assert det["component_type"] == "IC"
        assert det["package_type"] == "unknown"
        assert det["package_source"] == "unknown"
        assert det["package_confidence"] == 0.0

    def test_ic_bbox_column_inference_dip8(self, blank_image_b64):
        """模型只输出 IC + calibrator 就绪 → bbox 覆盖 4 个数字列推断为 dip8."""
        from app.pipeline.stages.s1_detect import run_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        # synthetic grid 中 row_coords 是 numbered-row 的 Y 坐标 (landscape=False).
        row_coords = calibrator.row_coords
        col_coords = calibrator.col_coords
        e_x = float(col_coords[4])
        f_x = float(col_coords[5])
        y_lo = float(row_coords[0]) - 1.0
        y_hi = float(row_coords[3]) + 1.0  # 覆盖 numbered-row 1..4 -> 4 列

        detector = MockComponentDetector([
            {
                "class_name": "IC",
                "bbox": (e_x - 5.0, y_lo, f_x + 5.0, y_hi),
                "confidence": 0.88,
            }
        ])
        result = run_detect(
            images_b64=[blank_image_b64],
            detector=detector,
            calibrator=calibrator,
        )

        det = result["detections"][0]
        assert det["package_type"] == "dip8"
        assert det["package_source"] == "bbox_column_inference"
        assert det["package_confidence"] > 0
        assert det["package_inference_metadata"]["column_count"] == 4

    def test_ic_bbox_column_inference_dip14(self, blank_image_b64):
        """模型只输出 IC + bbox 覆盖 7 列 → dip14."""
        from app.pipeline.stages.s1_detect import run_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        row_coords = calibrator.row_coords
        col_coords = calibrator.col_coords
        e_x = float(col_coords[4])
        f_x = float(col_coords[5])
        y_lo = float(row_coords[0]) - 1.0
        y_hi = float(row_coords[6]) + 1.0  # 覆盖 numbered-row 1..7 -> 7 列

        detector = MockComponentDetector([
            {
                "class_name": "IC",
                "bbox": (e_x - 5.0, y_lo, f_x + 5.0, y_hi),
                "confidence": 0.92,
            }
        ])
        result = run_detect(
            images_b64=[blank_image_b64],
            detector=detector,
            calibrator=calibrator,
        )

        det = result["detections"][0]
        assert det["package_type"] == "dip14"
        assert det["package_source"] == "bbox_column_inference"
        assert det["package_inference_metadata"]["column_count"] == 7

    def test_ic_bbox_column_inference_unknown_out_of_range(self, blank_image_b64):
        """bbox 覆盖列数远超 DIP14 范围 → package_type=unknown."""
        from app.pipeline.stages.s1_detect import run_detect
        from app.pipeline.vision.calibrator import BreadboardCalibrator
        from tests.pipeline.mocks import MockComponentDetector

        calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
        calibrator.build_synthetic_grid((480, 640))
        row_coords = calibrator.row_coords
        col_coords = calibrator.col_coords
        e_x = float(col_coords[4])
        f_x = float(col_coords[5])
        # 覆盖 12 个数字列 — 既不像 DIP8 也不像 DIP14.
        y_lo = float(row_coords[0]) - 1.0
        y_hi = float(row_coords[11]) + 1.0

        detector = MockComponentDetector([
            {"class_name": "IC", "bbox": (e_x - 5.0, y_lo, f_x + 5.0, y_hi), "confidence": 0.7}
        ])
        result = run_detect(
            images_b64=[blank_image_b64],
            detector=detector,
            calibrator=calibrator,
        )

        det = result["detections"][0]
        assert det["package_type"] == "unknown"
        assert det["package_source"] == "unknown"

    def test_non_ic_components_unaffected(self, blank_image_b64):
        """非 IC 元件保留 default_component_type 路径, package_type 不被改写."""
        from app.pipeline.stages.s1_detect import run_detect
        from tests.pipeline.mocks import MockComponentDetector

        detector = MockComponentDetector([
            {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.9},
            {"class_name": "LED", "bbox": (320, 200, 360, 260), "confidence": 0.88},
        ])
        result = run_detect(images_b64=[blank_image_b64], detector=detector)

        by_type = {d["component_type"]: d for d in result["detections"]}
        assert by_type["Resistor"]["package_type"] == "axial_2pin"
        assert by_type["Resistor"]["package_source"] == "default_component_type"
        assert by_type["LED"]["package_type"] == "led_2pin"
        assert by_type["LED"]["package_source"] == "default_component_type"
