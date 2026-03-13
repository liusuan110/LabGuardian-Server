"""
AOI 图像工具 — base64 编解码 + 热力图可视化

供 PCBDefectDetector 和 WinCLIPDetector 共享。
"""

from __future__ import annotations

import base64

import cv2
import numpy as np


def decode_b64_image(b64_str: str) -> np.ndarray:
    """解码 base64 图片为 BGR ndarray。"""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img


def make_heatmap_b64(
    original_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.4,
) -> str:
    """将原图与异常热力图叠加，返回 base64 JPEG。"""
    h, w = original_bgr.shape[:2]
    heatmap = cv2.resize(anomaly_map, (w, h))
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_color, alpha, 0)
    _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def normalize_anomaly_map(anomaly_map: np.ndarray) -> np.ndarray:
    """将异常热图归一化到 [0, 1]"""
    a_min, a_max = anomaly_map.min(), anomaly_map.max()
    if a_max > a_min:
        return (anomaly_map - a_min) / (a_max - a_min)
    return np.zeros_like(anomaly_map)
