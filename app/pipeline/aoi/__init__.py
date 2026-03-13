"""
AOI (Automated Optical Inspection) — PCB 缺陷检测模块

双模式检测:
  1. PatchCore (无监督): 需正常样本训练, 适合已知产品线批量检测
  2. WinCLIP  (零样本): 无需任何训练图片, 仅靠文本提示即可检测 PCB 异常

面包板原型验证 → PCB 生产焊接，完整生命周期教学工具。
"""

from app.pipeline.aoi.detector import PCBDefectDetector
from app.pipeline.aoi.image_utils import decode_b64_image, make_heatmap_b64, normalize_anomaly_map
from app.pipeline.aoi.winclip_detector import WinCLIPDetector

__all__ = [
    "PCBDefectDetector",
    "WinCLIPDetector",
    "decode_b64_image",
    "make_heatmap_b64",
    "normalize_anomaly_map",
]
