"""
WinCLIP 零样本 PCB 缺陷检测器

基于 Intel Anomalib 内置的 WinCLIP 算法:
  - 零样本 (Zero-shot): 无需任何训练图片，仅靠文本提示词即可检测 PCB 异常
  - 少样本 (Few-shot):  可选提供 1~N 张正常参考图片进一步提升精度
  - 直接输出像素级热力图 + 图像级异常评分

核心优势:
  1. 开箱即用, 无需训练
  2. 英特尔 Anomalib 官方集成, 可通过 OpenVINO 加速
  3. 输入尺寸固定 240×240, CLIP ViT-B/16+ backbone
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from app.pipeline.aoi.image_utils import decode_b64_image, make_heatmap_b64, normalize_anomaly_map

logger = logging.getLogger(__name__)

WINCLIP_IMAGE_SIZE = 240
DEFAULT_CLASS_NAME = "printed circuit board"

PCB_DEFECT_PROMPTS = [
    "short circuit",
    "open circuit",
    "missing solder",
    "solder bridge",
    "cold solder joint",
    "component misalignment",
    "copper residue",
    "mouse bite defect",
    "pin hole",
    "spur defect",
]


class WinCLIPDetector:
    """WinCLIP 零样本/少样本 PCB 缺陷检测器。

    零样本用法 (无需训练!):
        detector = WinCLIPDetector()
        result = detector.predict(image_b64=b64_str, score_threshold=0.5)

    少样本用法 (提供参考图增强):
        detector = WinCLIPDetector(few_shot_dir="datasets/pcb/good")
        result = detector.predict(image_b64=b64_str)
    """

    def __init__(
        self,
        class_name: str = DEFAULT_CLASS_NAME,
        k_shot: int = 0,
        few_shot_dir: str | Path | None = None,
        score_threshold: float = 0.5,
        scales: tuple[int, ...] = (2, 3),
    ) -> None:
        self.class_name = class_name
        self.k_shot = k_shot
        self.few_shot_dir = Path(few_shot_dir) if few_shot_dir else None
        self.default_score_threshold = score_threshold
        self.scales = scales

        self._model = None
        self._is_ready = False

    # ------------------------------------------------------------------ #
    #  公开 API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def score_threshold(self) -> float:
        return self.default_score_threshold

    def initialize(self) -> None:
        """初始化模型并收集文本嵌入。"""
        if self._is_ready:
            return

        from anomalib.models.image import WinClip

        logger.info(
            "WinCLIP: initializing (class_name=%s, k_shot=%d)",
            self.class_name, self.k_shot,
        )
        t0 = time.time()

        self._lightning_model = WinClip(
            class_name=self.class_name,
            k_shot=self.k_shot,
            scales=self.scales,
            few_shot_source=str(self.few_shot_dir) if self.few_shot_dir else None,
        )
        self._lightning_model.eval()

        self._model = self._lightning_model.model

        ref_images = None
        if self.k_shot > 0 and self.few_shot_dir and self.few_shot_dir.is_dir():
            ref_images = self._load_reference_images()

        self._model.setup(self.class_name, ref_images)
        self._is_ready = True

        logger.info(
            "WinCLIP: ready (%.1fs), text_embeddings=%s",
            time.time() - t0,
            self._model.text_embeddings.shape,
        )

    def predict(
        self,
        image_b64: str | None = None,
        image_path: str | Path | None = None,
        image_bgr: np.ndarray | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """零样本/少样本 PCB 缺陷检测。

        Args:
            image_b64: base64 编码的 JPEG/PNG 图片
            image_path: 图片文件路径
            image_bgr: BGR ndarray 图片
            score_threshold: 异常分数阈值 (本次请求生效, 不修改全局状态)
        """
        t0 = time.time()
        threshold = score_threshold if score_threshold is not None else self.default_score_threshold
        self._ensure_ready()

        if image_b64 is not None:
            img_bgr = decode_b64_image(image_b64)
        elif image_path is not None:
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
        elif image_bgr is not None:
            img_bgr = image_bgr
        else:
            raise ValueError("Must provide image_b64, image_path, or image_bgr")

        # WinCLIP 直接调用底层 torch 模型, 需要手动做 CLIP 归一化
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (WINCLIP_IMAGE_SIZE, WINCLIP_IMAGE_SIZE))

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        batch = img_tensor.unsqueeze(0)

        with torch.no_grad():
            result = self._model(batch)

        anomaly_score = float(result.pred_score.item())
        anomaly_map = result.anomaly_map.squeeze().cpu().numpy()
        anomaly_map_norm = normalize_anomaly_map(anomaly_map)

        is_defective = anomaly_score > threshold
        heatmap_b64 = make_heatmap_b64(img_bgr, anomaly_map_norm)

        duration_ms = (time.time() - t0) * 1000
        logger.info(
            "WinCLIP predict: score=%.4f, defective=%s (%.0fms)",
            anomaly_score, is_defective, duration_ms,
        )

        return {
            "anomaly_score": round(anomaly_score, 4),
            "is_defective": is_defective,
            "anomaly_map_b64": heatmap_b64,
            "raw_anomaly_map": anomaly_map_norm,
            "duration_ms": round(duration_ms, 1),
        }

    def predict_with_fusion(
        self,
        test_image_path: str | Path,
        template_image_path: str | Path | None = None,
        diff_weight: float = 0.3,
        winclip_weight: float = 0.7,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """WinCLIP + 模板差分融合检测。"""
        t0 = time.time()
        self._ensure_ready()

        test_bgr = cv2.imread(str(test_image_path))
        if test_bgr is None:
            raise FileNotFoundError(f"Cannot read: {test_image_path}")

        winclip_result = self.predict(image_bgr=test_bgr, score_threshold=score_threshold)
        winclip_map = winclip_result["raw_anomaly_map"]

        if template_image_path is None:
            winclip_result["duration_ms"] = round((time.time() - t0) * 1000, 1)
            return winclip_result

        template_bgr = cv2.imread(str(template_image_path))
        if template_bgr is None:
            raise FileNotFoundError(f"Cannot read template: {template_image_path}")

        h, w = test_bgr.shape[:2]
        template_resized = cv2.resize(template_bgr, (w, h))
        diff = cv2.absdiff(
            cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY),
        )
        diff_blur = cv2.GaussianBlur(diff.astype(np.float32), (5, 5), 0)
        diff_norm = normalize_anomaly_map(diff_blur)

        winclip_resized = cv2.resize(winclip_map, (w, h))
        fusion_map = diff_weight * diff_norm + winclip_weight * winclip_resized
        fusion_map = np.clip(fusion_map, 0, 1)

        fusion_b64 = make_heatmap_b64(test_bgr, fusion_map)

        duration_ms = (time.time() - t0) * 1000
        return {
            **winclip_result,
            "fusion_map_b64": fusion_b64,
            "duration_ms": round(duration_ms, 1),
        }

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # ------------------------------------------------------------------ #
    #  内部方法                                                            #
    # ------------------------------------------------------------------ #

    def _ensure_ready(self) -> None:
        if not self._is_ready:
            self.initialize()

    def _load_reference_images(self) -> torch.Tensor | None:
        if self.few_shot_dir is None or not self.few_shot_dir.is_dir():
            return None

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = [p for p in self.few_shot_dir.iterdir() if p.suffix.lower() in exts]
        if not paths:
            return None

        paths = sorted(paths)[: self.k_shot] if self.k_shot > 0 else sorted(paths)
        images = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (WINCLIP_IMAGE_SIZE, WINCLIP_IMAGE_SIZE))
            t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            t = (t - mean) / std
            images.append(t)

        if not images:
            return None
        return torch.stack(images)
