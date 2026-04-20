"""Pytest 全局配置和 fixtures."""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

# 确保 app 在 path 中
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def blank_image_b64():
    """单张空白图像 base64."""
    from tests.pipeline.fixtures import make_blank_image, image_to_b64
    return image_to_b64(make_blank_image())


@pytest.fixture
def three_blank_images_b64():
    """3 张空白图像 base64."""
    from tests.pipeline.fixtures import make_blank_image, image_to_b64
    return [image_to_b64(make_blank_image()) for _ in range(3)]


@pytest.fixture
def resistor_roi_b64():
    """模拟 Resistor ROI base64."""
    from tests.pipeline.fixtures import make_resistor_roi, image_to_b64
    return image_to_b64(make_resistor_roi())


@pytest.fixture
def breadboard_image_b64():
    """模拟面包板图像 base64."""
    from tests.pipeline.fixtures import make_breadboard_image, image_to_b64
    return image_to_b64(make_breadboard_image())


@pytest.fixture
def resistor_roi_image():
    """模拟 Resistor ROI 图像（NumPy 数组）."""
    from tests.pipeline.fixtures import make_resistor_roi
    return make_resistor_roi()


@pytest.fixture
def breadboard_image():
    """模拟面包板图像（NumPy 数组）."""
    from tests.pipeline.fixtures import make_breadboard_image
    return make_breadboard_image()


@pytest.fixture
def mock_detector_resistor():
    """Mock YOLO 检测器 — 返回 1 个 Resistor."""
    from tests.pipeline.mocks import MockComponentDetector
    return MockComponentDetector([
        {
            "class_name": "Resistor",
            "bbox": (100, 200, 300, 260),
            "confidence": 0.95,
            "is_obb": False,
        }
    ])


@pytest.fixture
def mock_detector_3_components():
    """Mock YOLO 检测器 — 返回 3 个混合元件."""
    from tests.pipeline.mocks import MockComponentDetector
    return MockComponentDetector([
        {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.95},
        {"class_name": "Capacitor", "bbox": (400, 100, 500, 180), "confidence": 0.92},
        {"class_name": "LED", "bbox": (200, 300, 350, 380), "confidence": 0.88},
    ])


@pytest.fixture
def mock_detector_obb():
    """Mock YOLO 检测器 — 返回带 OBB 的检测."""
    from tests.pipeline.mocks import MockComponentDetector
    import numpy as np
    return MockComponentDetector([
        {
            "class_name": "Resistor",
            "bbox": (100, 200, 300, 260),
            "confidence": 0.95,
            "is_obb": True,
            "obb_corners": np.array([
                [100, 200], [300, 200], [300, 260], [100, 260]
            ], dtype=np.float32),
        }
    ])


@pytest.fixture
def mock_detector_breadboard():
    """Mock YOLO 检测器 — 返回 Breadboard 背景类（应被过滤）."""
    from tests.pipeline.mocks import MockComponentDetector
    return MockComponentDetector([
        {"class_name": "Breadboard", "bbox": (0, 0, 640, 480), "confidence": 0.99},
        {"class_name": "Resistor", "bbox": (100, 200, 300, 260), "confidence": 0.95},
    ])


@pytest.fixture
def mock_pin_detector_2pin():
    """Mock Pin 检测器 — 2-pin 元件."""
    from tests.pipeline.mocks import MockPinDetector
    return MockPinDetector([
        {"pin_id": 1, "pin_name": "pin1", "keypoint": (120.0, 240.0), "confidence": 0.95, "visibility": 2},
        {"pin_id": 2, "pin_name": "pin2", "keypoint": (280.0, 240.0), "confidence": 0.95, "visibility": 2},
    ])


@pytest.fixture
def mock_pin_detector_3pin():
    """Mock Pin 检测器 — 3-pin 元件."""
    from tests.pipeline.mocks import MockPinDetector
    return MockPinDetector([
        {"pin_id": 1, "pin_name": "pin1", "keypoint": (100.0, 240.0), "confidence": 0.95, "visibility": 2},
        {"pin_id": 2, "pin_name": "pin2", "keypoint": (200.0, 240.0), "confidence": 0.95, "visibility": 2},
        {"pin_id": 3, "pin_name": "pin3", "keypoint": (300.0, 240.0), "confidence": 0.95, "visibility": 2},
    ])


@pytest.fixture
def real_pin_detector():
    """真实的 PinRoiDetector（无模型 → 走启发式 fallback）."""
    from app.pipeline.vision.pin_model import PinRoiDetector
    return PinRoiDetector(model_path=None, device="cpu")


@pytest.fixture
def calibrator():
    """每个请求新建的 BreadboardCalibrator."""
    from app.pipeline.vision.calibrator import BreadboardCalibrator
    return BreadboardCalibrator(rows=63, cols_per_side=5)


@pytest.fixture
def real_detector():
    """真实的 ComponentDetector（使用配置的模型路径）."""
    from app.pipeline.vision.detector import ComponentDetector
    from app.core.config import settings
    return ComponentDetector(
        model_path=settings.YOLO_MODEL_PATH,
        device=settings.YOLO_DEVICE,
    )
