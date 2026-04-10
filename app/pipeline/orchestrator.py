"""
Pipeline Orchestrator

串联 S1→S2→S3→S4 四阶段，管理共享资源（detector / calibrator），
支持进度回调，供 Celery task 或同步调用使用。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.core.config import settings
from app.pipeline.stages.s1_detect import run_detect
from app.pipeline.stages.s1b_pin_detect import run_pin_detect
from app.pipeline.stages.s2_mapping import run_mapping
from app.pipeline.stages.s3_topology import run_topology
from app.pipeline.stages.s4_validate import run_validate
from app.pipeline.vision.calibrator import BreadboardCalibrator
from app.pipeline.vision.detector import ComponentDetector
from app.pipeline.vision.pin_model import PinRoiDetector

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]  # (stage_name, progress 0-1)


@dataclass
class PipelineContext:
    """流水线上下文 —— 携带跨阶段共享对象"""

    detector: ComponentDetector = field(default=None)  # type: ignore[assignment]
    pin_detector: PinRoiDetector = field(default_factory=PinRoiDetector)
    calibrator: BreadboardCalibrator = field(default=None)  # type: ignore[assignment]
    reference_circuit: Optional[Dict[str, Any] | str] = None
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 1280
    roi_rect: Optional[tuple] = None

    def ensure_resources(self) -> None:
        if self.detector is None:
            self.detector = ComponentDetector(
                model_path=settings.YOLO_MODEL_PATH,
                obb_model_path=settings.YOLO_OBB_MODEL_PATH,
                device=settings.YOLO_DEVICE,
            )
        if self.calibrator is None:
            self.calibrator = BreadboardCalibrator(
                rows=settings.BREADBOARD_ROWS,
                cols_per_side=settings.BREADBOARD_COLS_PER_SIDE,
            )


# 线程安全单例 —— 避免 Celery worker 每次任务重建模型
_shared_ctx: PipelineContext | None = None
_ctx_lock = threading.Lock()


def get_shared_context() -> PipelineContext:
    global _shared_ctx
    if _shared_ctx is None:
        with _ctx_lock:
            if _shared_ctx is None:
                _shared_ctx = PipelineContext(
                    conf=settings.YOLO_CONF_THRESHOLD,
                    iou=settings.YOLO_IOU_THRESHOLD,
                    imgsz=settings.YOLO_IMGSZ,
                    reference_circuit=settings.REFERENCE_CIRCUIT_PATH,
                )
                _shared_ctx.ensure_resources()
    return _shared_ctx


def run_pipeline(
    images_b64: List[str],
    reference_circuit: Dict[str, Any] | str | None = None,
    rail_assignments: Dict[str, str] | None = None,
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Dict[str, Any]:
    """执行完整的 4 阶段流水线

    Args:
        images_b64: 1-3 张 base64 图片
        reference_circuit: 参考电路 JSON 路径或内联 reference payload
        rail_assignments: 电源轨道指定, 如 {"top_plus": "VCC", "top_minus": "GND", ...}
        conf: YOLO 置信度阈值, 默认使用 settings
        iou: YOLO NMS IoU 阈值, 默认使用 settings
        imgsz: YOLO 推理尺寸, 默认使用 settings
        progress_cb: 进度回调

    Returns:
        {
            "stages": {
                "detect": {...},
                "pin_detect": {...},
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
    eff_conf = ctx.conf if conf is None else conf
    eff_iou = ctx.iou if iou is None else iou
    eff_imgsz = ctx.imgsz if imgsz is None else imgsz

    def _notify(stage: str, progress: float) -> None:
        if progress_cb:
            progress_cb(stage, progress)

    # ── S1: 检测 ──
    _notify("detect", 0.0)
    s1 = run_detect(
        images_b64,
        detector=ctx.detector,
        conf=eff_conf,
        iou=eff_iou,
        imgsz=eff_imgsz,
        roi_rect=ctx.roi_rect,
    )
    stages["detect"] = s1
    logger.info("S1 detect: %d components (%.0fms)",
                len(s1["detections"]), s1["duration_ms"])
    _notify("detect", 1.0)

    # ── S1.5: 组件 ROI pin 检测 ──
    _notify("pin_detect", 0.0)
    s15 = run_pin_detect(
        detections=s1["detections"],
        images_b64=images_b64,
        pin_detector=ctx.pin_detector,
    )
    stages["pin_detect"] = s15
    logger.info("S1.5 pin detect: %d components (%.0fms)", len(s15["components"]), s15["duration_ms"])
    _notify("pin_detect", 1.0)

    # ── S2: pin -> hole 映射 ──
    _notify("mapping", 0.0)
    s2 = run_mapping(
        s15["components"],
        calibrator=ctx.calibrator,
        image_shape=s1["primary_image_shape"],
        images_b64=images_b64,
    )
    stages["mapping"] = s2
    logger.info("S2 mapping: %d components (%.0fms)", len(s2["components"]), s2["duration_ms"])
    _notify("mapping", 1.0)

    # ── S3: 拓扑 (传入 rail_assignments) ──
    _notify("topology", 0.0)
    # 默认电源轨道: top+=VCC, bot-=GND (学生端可覆盖)
    effective_rails = {
        "top_plus": "VCC",
        "top_minus": "GND",
        "bot_plus": "VCC",
        "bot_minus": "GND",
    }
    if rail_assignments:
        effective_rails.update(rail_assignments)
    s3 = run_topology(s2["components"], rail_assignments=effective_rails)
    stages["topology"] = s3
    logger.info("S3 topology: %d nodes (%.0fms)", s3["component_count"], s3["duration_ms"])
    _notify("topology", 1.0)

    # ── S4: 检错 ──
    _notify("validate", 0.0)
    s4 = run_validate(
        s3["topology_graph"],
        reference_circuit=reference_circuit if reference_circuit is not None else ctx.reference_circuit,
        components=s2["components"],
    )
    stages["validate"] = s4
    logger.info("S4 validate: risk=%s (%.0fms)", s4["risk_level"], s4["duration_ms"])
    _notify("validate", 1.0)

    total_ms = (time.time() - t0) * 1000
    return {
        "stages": stages,
        "total_duration_ms": total_ms,
    }
