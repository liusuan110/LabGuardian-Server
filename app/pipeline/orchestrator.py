"""
Pipeline Orchestrator

串联 S1→S2→S3→S4 四阶段，管理共享资源（detector / calibrator），
支持进度回调，供 Celery task 或同步调用使用。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.core.config import settings
from app.pipeline.stages.s1_detect import run_detect
from app.pipeline.stages.s2_mapping import run_mapping
from app.pipeline.stages.s3_topology import run_topology
from app.pipeline.stages.s4_validate import run_validate
from app.pipeline.vision.calibrator import BreadboardCalibrator
from app.pipeline.vision.detector import ComponentDetector

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]  # (stage_name, progress 0-1)


@dataclass
class PipelineContext:
    """流水线上下文 —— 携带跨阶段共享对象"""

    detector: ComponentDetector = field(default=None)  # type: ignore[assignment]
    calibrator: BreadboardCalibrator = field(default=None)  # type: ignore[assignment]
    reference_path: Optional[str] = None
    conf: float = 0.25
    imgsz: int = 1280
    roi_rect: Optional[tuple] = None

    def ensure_resources(self) -> None:
        if self.detector is None:
            self.detector = ComponentDetector(
                model_path=settings.YOLO_MODEL_PATH,
                obb_model_path=settings.YOLO_OBB_MODEL_PATH,
            )
        if self.calibrator is None:
            self.calibrator = BreadboardCalibrator(
                board_rows=settings.BOARD_ROWS,
                board_cols=settings.BOARD_COLS,
            )


# 模块级单例 —— 避免 Celery worker 每次任务重建模型
_shared_ctx: PipelineContext | None = None


def get_shared_context() -> PipelineContext:
    global _shared_ctx
    if _shared_ctx is None:
        _shared_ctx = PipelineContext(
            conf=settings.YOLO_CONF,
            imgsz=settings.YOLO_IMGSZ,
        )
        _shared_ctx.ensure_resources()
    return _shared_ctx


def run_pipeline(
    images_b64: List[str],
    reference_path: str | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Dict[str, Any]:
    """执行完整的 4 阶段流水线

    Returns:
        {
            "stages": {
                "detect": {...},
                "mapping": {...},
                "topology": {...},
                "validate": {...},
            },
            "total_duration_ms": float,
        }
    """
    t0 = time.time()
    ctx = get_shared_context()
    stages: Dict[str, Any] = {}

    def _notify(stage: str, progress: float) -> None:
        if progress_cb:
            progress_cb(stage, progress)

    # ── S1: 检测 ──
    _notify("detect", 0.0)
    s1 = run_detect(
        images_b64,
        detector=ctx.detector,
        conf=ctx.conf,
        imgsz=ctx.imgsz,
        roi_rect=ctx.roi_rect,
    )
    stages["detect"] = s1
    logger.info("S1 detect: %d detections (%.0fms)", len(s1["detections"]), s1["duration_ms"])
    _notify("detect", 1.0)

    # ── S2: 映射 ──
    _notify("mapping", 0.0)
    s2 = run_mapping(
        s1["detections"],
        calibrator=ctx.calibrator,
        image_shape=s1["primary_image_shape"],
    )
    stages["mapping"] = s2
    logger.info("S2 mapping: %d components (%.0fms)", len(s2["components"]), s2["duration_ms"])
    _notify("mapping", 1.0)

    # ── S3: 拓扑 ──
    _notify("topology", 0.0)
    s3 = run_topology(s2["components"])
    stages["topology"] = s3
    logger.info("S3 topology: %d nodes (%.0fms)", s3["component_count"], s3["duration_ms"])
    _notify("topology", 1.0)

    # ── S4: 检错 ──
    _notify("validate", 0.0)
    s4 = run_validate(
        s3["topology_graph"],
        reference_path=reference_path or ctx.reference_path,
    )
    stages["validate"] = s4
    logger.info("S4 validate: risk=%s (%.0fms)", s4["risk_level"], s4["duration_ms"])
    _notify("validate", 1.0)

    total_ms = (time.time() - t0) * 1000
    return {
        "stages": stages,
        "total_duration_ms": total_ms,
    }
