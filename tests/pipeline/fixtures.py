"""测试数据生成工具 — 无需真实模型即可生成模拟图像."""

from __future__ import annotations

import base64

import cv2
import numpy as np


def make_blank_image(h: int = 480, w: int = 640, color: tuple = (128, 128, 128)) -> np.ndarray:
    """纯色空白图像."""
    return np.full((h, w, 3), color, dtype=np.uint8)


def make_resistor_roi(h: int = 60, w: int = 200) -> np.ndarray:
    """模拟 Resistor ROI 图像（暗线引脚 + 彩色主体）."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    # 左引脚线
    cv2.line(img, (0, h // 2), (w // 4, h // 2), (50, 50, 50), 4)
    # 右引脚线
    cv2.line(img, (w * 3 // 4, h // 2), (w, h // 2), (50, 50, 50), 4)
    # 棕色主体
    cv2.rectangle(img, (w // 4, 5), (w * 3 // 4, h - 5), (139, 69, 19), -1)
    # 色环（模拟）
    cv2.rectangle(img, (w // 2 - 5, 5), (w // 2 + 5, h - 5), (255, 200, 0), 2)
    return img


def make_capacitor_roi(h: int = 40, w: int = 100) -> np.ndarray:
    """模拟 Capacitor ROI 图像."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    # 引脚线
    cv2.line(img, (0, h // 2), (w // 4, h // 2), (50, 50, 50), 3)
    cv2.line(img, (w * 3 // 4, h // 2), (w, h // 2), (50, 50, 50), 3)
    # 蓝色主体
    cv2.circle(img, (w // 2, h // 2), min(h // 2 - 5, w // 2 - 5), (200, 100, 50), -1)
    return img


def make_led_roi(h: int = 50, w: int = 80) -> np.ndarray:
    """模拟 LED ROI 图像."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.line(img, (0, h // 2), (w // 3, h // 2), (50, 50, 50), 3)
    cv2.line(img, (w * 2 // 3, h // 2), (w, h // 2), (50, 50, 50), 3)
    # 圆顶（透明效果）
    cv2.ellipse(img, (w // 2, h // 3), (w // 3, h // 3), 0, 0, 180, (50, 255, 50), 2)
    cv2.circle(img, (w // 2, h // 3), 5, (255, 255, 0), -1)
    return img


def make_breadboard_image(h: int = 480, w: int = 640) -> np.ndarray:
    """模拟面包板图像（白底 + 灰孔洞点阵）."""
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    # 主 grid 孔洞（左右两侧）
    for r in range(20, h - 20, 18):
        # 左列 a-e
        for c_offset, col in enumerate(['a', 'b', 'c', 'd', 'e']):
            cx = 30 + c_offset * 18
            cv2.circle(img, (cx, r), 3, (80, 80, 80), -1)
        # 右列 f-j（中间有 gap）
        for c_offset, col in enumerate(['f', 'g', 'h', 'i', 'j']):
            cx = 370 + c_offset * 18
            cv2.circle(img, (cx, r), 3, (80, 80, 80), -1)
    # 顶部电轨
    for c in range(20, w - 20, 18):
        cv2.circle(img, (c, 8), 3, (80, 80, 80), -1)
        cv2.circle(img, (c, 8 + 18), 3, (80, 80, 80), -1)
    return img


def make_breadboard_with_component(h: int = 480, w: int = 640) -> np.ndarray:
    """模拟面包板 + 元件的图像."""
    img = make_breadboard_image(h, w)
    # 在中间行插入一个 Resistor（用矩形模拟）
    row_y = 240
    cv2.rectangle(img, (60, row_y - 15), (160, row_y + 15), (139, 69, 19), -1)
    # 引脚线连接到孔洞
    cv2.line(img, (60, row_y), (54, row_y), (50, 50, 50), 3)
    cv2.line(img, (160, row_y), (180, row_y), (50, 50, 50), 3)
    return img


def image_to_b64(img: np.ndarray, fmt: str = '.jpg') -> str:
    """NumPy 图像 → base64 字符串."""
    _, buf = cv2.imencode(fmt, img)
    return base64.b64encode(buf).decode('utf-8')


def b64_to_image(b64: str) -> np.ndarray | None:
    """base64 字符串 → NumPy 图像."""
    import base64
    try:
        data = base64.b64decode(b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def make_corrupted_b64() -> str:
    """生成一个故意损坏的 base64 字符串."""
    import base64
    # 有效的 base64 头但内容损坏
    return base64.b64encode(b"THIS IS NOT A VALID JPEG").decode('utf-8')


def make_valid_but_unreadable_b64() -> str:
    """生成有效 base64 但无法解码为图像的数据."""
    import base64
    return base64.b64encode(b"\x00\x01\x02\xFF\xFE\xFD").decode('utf-8')
