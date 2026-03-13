"""
PCB Defect Detector — 基于 Anomalib PatchCore 的无监督缺陷检测

核心流程:
  train()   → 从 golden sample 目录提取正常特征 → 保存 checkpoint
  predict() → 通过 Anomalib Engine 进行标准推理 (避免手动预处理的双重归一化)

返回:
  - anomaly_score: 图像级异常分数 [0, 1]
  - anomaly_map:   像素级热力图 ndarray (H, W), 值域 [0, 1]
  - is_defective:  bool 判定
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.pipeline.aoi.image_utils import decode_b64_image, make_heatmap_b64, normalize_anomaly_map

logger = logging.getLogger(__name__)


class PCBDefectDetector:
    """封装 Anomalib PatchCore 用于 PCB AOI 检测。

    用法:
        detector = PCBDefectDetector(model_dir="models/aoi")
        detector.train(golden_dir="datasets/pcb/good")
        results = detector.predict(image_b64, score_threshold=0.5)
    """

    def __init__(
        self,
        model_dir: str | Path = "models/aoi",
        backbone: str = "wide_resnet50_2",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        image_size: tuple[int, int] = (256, 256),
        score_threshold: float = 0.5,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.backbone = backbone
        self.layers = layers
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.image_size = image_size
        self.default_score_threshold = score_threshold

        self._model = None
        self._engine = None
        self._ckpt_path: Path | None = None

        self._discover_checkpoint()

    # ------------------------------------------------------------------ #
    #  公开 API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def score_threshold(self) -> float:
        return self.default_score_threshold

    def train(
        self,
        golden_dir: str | Path,
        abnormal_dir: str | Path | None = None,
        max_epochs: int = 1,
    ) -> dict[str, Any]:
        """使用正常 PCB 图片训练 PatchCore 模型。"""
        from anomalib.data import Folder
        from anomalib.data.utils import TestSplitMode, ValSplitMode
        from anomalib.engine import Engine
        from anomalib.models import Patchcore

        t0 = time.time()
        golden_dir = Path(golden_dir)
        if not golden_dir.is_dir():
            raise FileNotFoundError(f"Golden sample directory not found: {golden_dir}")

        logger.info("AOI train: golden_dir=%s, abnormal_dir=%s", golden_dir, abnormal_dir)

        model = Patchcore(
            backbone=self.backbone,
            layers=list(self.layers),
            coreset_sampling_ratio=self.coreset_sampling_ratio,
            num_neighbors=self.num_neighbors,
        )

        datamodule = Folder(
            name="pcb_aoi",
            root=golden_dir.parent,
            normal_dir=golden_dir.name,
            abnormal_dir=str(Path(abnormal_dir).name) if abnormal_dir else None,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=0,
            test_split_mode=(
                TestSplitMode.FROM_DIR if abnormal_dir else TestSplitMode.SYNTHETIC
            ),
            val_split_mode=ValSplitMode.SYNTHETIC,
        )

        results_dir = self.model_dir / "results"
        engine = Engine(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            default_root_dir=str(results_dir),
            enable_checkpointing=True,
        )
        engine.fit(model=model, datamodule=datamodule)

        ckpt = self._find_latest_checkpoint(results_dir)
        if ckpt:
            self._ckpt_path = ckpt
            self._model = model
            self._engine = engine
            logger.info("AOI train complete: checkpoint=%s", ckpt)

        duration_ms = (time.time() - t0) * 1000
        return {
            "status": "ok",
            "checkpoint": str(ckpt) if ckpt else None,
            "duration_ms": duration_ms,
        }

    def predict(
        self,
        image_b64: str | None = None,
        image_path: str | Path | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """对单张 PCB 图像进行缺陷检测。

        通过 Anomalib Engine.predict() 进行标准推理管线,
        由 Anomalib 内部处理 resize + normalize, 避免双重归一化。

        Args:
            image_b64: base64 编码的 JPEG 图片
            image_path: 图片文件路径 (与 image_b64 二选一)
            score_threshold: 异常分数阈值 (本次请求生效, 不修改全局状态)
        """
        t0 = time.time()
        threshold = score_threshold if score_threshold is not None else self.default_score_threshold

        if image_b64 is None and image_path is None:
            raise ValueError("Must provide either image_b64 or image_path")

        self._ensure_model_loaded()

        # 解码得到原图 (用于最终热力图叠加)
        if image_b64 is not None:
            img_np = decode_b64_image(image_b64)
        else:
            img_np = cv2.imread(str(image_path))
            if img_np is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")

        # 写入临时文件, 让 Anomalib Engine.predict() 走标准 DataModule 管线
        # 这样由 Anomalib 内部处理 resize / normalize, 避免双重归一化
        tmp_dir = None
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="aoi_pred_"))
            tmp_path = tmp_dir / "test_image.png"
            cv2.imwrite(str(tmp_path), img_np)

            predictions = self._engine.predict(
                model=self._model,
                data_path=str(tmp_dir),
            )
        finally:
            if tmp_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # 从 Engine.predict() 结果中提取异常分数和热图
        if predictions and len(predictions) > 0:
            pred = predictions[0]
            anomaly_score = float(pred.pred_score.item()) if hasattr(pred.pred_score, 'item') else float(pred.pred_score)
            anomaly_map_raw = pred.anomaly_map.squeeze().cpu().numpy()
        else:
            anomaly_score = 0.0
            anomaly_map_raw = np.zeros(self.image_size, dtype=np.float32)

        anomaly_map = normalize_anomaly_map(anomaly_map_raw)
        is_defective = anomaly_score > threshold
        heatmap_b64 = make_heatmap_b64(img_np, anomaly_map)

        duration_ms = (time.time() - t0) * 1000
        logger.info("AOI predict: score=%.3f, defective=%s (%.0fms)",
                     anomaly_score, is_defective, duration_ms)

        return {
            "anomaly_score": round(anomaly_score, 4),
            "is_defective": is_defective,
            "anomaly_map_b64": heatmap_b64,
            "duration_ms": round(duration_ms, 1),
        }

    def predict_batch(
        self,
        image_paths: list[str | Path],
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """批量推理多张 PCB 图片。"""
        return [self.predict(image_path=p, score_threshold=score_threshold) for p in image_paths]

    @property
    def is_trained(self) -> bool:
        return self._ckpt_path is not None and self._ckpt_path.exists()

    @property
    def checkpoint_path(self) -> str | None:
        return str(self._ckpt_path) if self._ckpt_path else None

    # ------------------------------------------------------------------ #
    #  内部方法                                                            #
    # ------------------------------------------------------------------ #

    def _discover_checkpoint(self) -> None:
        results_dir = self.model_dir / "results"
        if results_dir.exists():
            ckpt = self._find_latest_checkpoint(results_dir)
            if ckpt:
                self._ckpt_path = ckpt
                logger.info("AOI: discovered checkpoint %s", ckpt)

    def _ensure_model_loaded(self) -> None:
        """确保模型和 Engine 已加载。"""
        if self._model is not None and self._engine is not None:
            return

        if self._ckpt_path is None or not self._ckpt_path.exists():
            raise RuntimeError(
                "No trained AOI model found. "
                "Call POST /api/v1/pcb/train first to train on golden samples."
            )

        from anomalib.engine import Engine
        from anomalib.models import Patchcore

        logger.info("AOI: loading model from %s", self._ckpt_path)
        self._model = Patchcore.load_from_checkpoint(str(self._ckpt_path))
        self._model.eval()

        results_dir = self.model_dir / "results"
        self._engine = Engine(
            default_root_dir=str(results_dir),
            accelerator="auto",
            devices=1,
        )
        logger.info("AOI: model + engine loaded successfully")

    @staticmethod
    def _find_latest_checkpoint(results_dir: Path) -> Path | None:
        ckpts = list(results_dir.rglob("*.ckpt"))
        if not ckpts:
            return None
        return max(ckpts, key=lambda p: p.stat().st_mtime)
