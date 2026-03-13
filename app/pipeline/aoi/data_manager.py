"""
AOI Data Manager — 管理 PCB 黄金样本数据集

功能:
  - 上传/列出/删除黄金样本图片
  - 管理数据集目录结构
  - 支持 base64 图片上传
"""

from __future__ import annotations

import base64
import logging
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AOIDataManager:
    """管理 PCB 检测所需的图片数据集。

    目录结构:
        datasets_root/
        ├── good/           # 正常 (黄金样本)
        │   ├── img_001.jpg
        │   └── img_002.jpg
        └── defect/         # 缺陷样本 (可选, 仅用于验证)
            ├── img_003.jpg
            └── img_004.jpg
    """

    def __init__(self, datasets_root: str | Path = "datasets/pcb_aoi") -> None:
        self.root = Path(datasets_root)
        self.good_dir = self.root / "good"
        self.defect_dir = self.root / "defect"

        # 确保目录存在
        self.good_dir.mkdir(parents=True, exist_ok=True)
        self.defect_dir.mkdir(parents=True, exist_ok=True)

    def upload_golden_sample(self, image_b64: str, filename: str | None = None) -> str:
        """上传一张黄金样本图片。

        Returns:
            保存的文件名。
        """
        return self._save_image(image_b64, self.good_dir, filename)

    def upload_defect_sample(self, image_b64: str, filename: str | None = None) -> str:
        """上传一张缺陷样本图片 (可选, 用于验证)。"""
        return self._save_image(image_b64, self.defect_dir, filename)

    def list_samples(self) -> dict[str, list[str]]:
        """列出当前数据集中的图片。"""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return {
            "good": sorted(
                f.name for f in self.good_dir.iterdir()
                if f.suffix.lower() in exts
            ),
            "defect": sorted(
                f.name for f in self.defect_dir.iterdir()
                if f.suffix.lower() in exts
            ),
        }

    def clear_dataset(self) -> None:
        """清空数据集。"""
        for d in (self.good_dir, self.defect_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        logger.info("AOI dataset cleared: %s", self.root)

    @property
    def golden_count(self) -> int:
        return len(self.list_samples()["good"])

    @property
    def defect_count(self) -> int:
        return len(self.list_samples()["defect"])

    # ------------------------------------------------------------------ #
    #  内部方法                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _save_image(b64_str: str, target_dir: Path, filename: str | None) -> str:
        """解码 base64 并保存到目标目录。"""
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode base64 image")

        if filename is None:
            filename = f"{uuid.uuid4().hex[:12]}.jpg"
        # 确保文件名安全
        safe_name = Path(filename).name
        save_path = target_dir / safe_name
        cv2.imwrite(str(save_path), img)
        logger.info("AOI saved image: %s (%dx%d)", save_path, img.shape[1], img.shape[0])
        return safe_name
