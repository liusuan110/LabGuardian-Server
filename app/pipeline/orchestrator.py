"""
Pipeline Orchestrator

串联 S1→S1.5→S2→S3→S4→S5 阶段，管理共享资源（detector / pin_detector），
支持进度回调，供 Celery task 或同步调用使用。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.domain.net_normalization import normalize_current_netlist
from app.pipeline.net_roles import apply_net_role_assignments
from app.pipeline.reference_subtypes import apply_reference_ic_subtypes
from app.pipeline.stages.s1_detect import run_detect
from app.pipeline.stages.s1b_pin_detect import run_pin_detect
from app.pipeline.stages.s2_mapping import run_mapping
from app.pipeline.stages.s3_topology import run_topology
from app.pipeline.stages.s4_validate import run_validate
from app.pipeline.stages.s5_semantic_analysis import run_semantic_analysis
from app.pipeline.vision.calibrator import BreadboardCalibrator
from app.pipeline.vision.detector import ComponentDetector
from app.pipeline.vision.pin_model import PinRoiDetector

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]  # (stage_name, progress 0-1)


@dataclass
class PipelineContext:
    """流水线上下文 —— 仅携带可安全跨请求复用的共享对象"""

    detector: ComponentDetector = field(default=None)  # type: ignore[assignment]
    pin_detector: PinRoiDetector = field(default=None)  # type: ignore[assignment]
    reference_circuit: dict[str, Any] | str | None = None
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 960
    roi_rect: tuple | None = None

    def ensure_resources(self) -> None:
        if self.detector is None:
            self.detector = ComponentDetector(
                model_path=settings.YOLO_MODEL_PATH,
                obb_model_path=settings.YOLO_OBB_MODEL_PATH,
                device=settings.YOLO_DEVICE,
            )
        if self.pin_detector is None:
            self.pin_detector = PinRoiDetector(
                model_path=settings.PIN_MODEL_PATH,
                device=settings.PIN_MODEL_DEVICE,
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
    images_b64: list[str],
    reference_circuit: dict[str, Any] | str | None = None,
    rail_assignments: dict[str, str] | None = None,
    port_annotations: list[Any] | None = None,
    net_role_assignments: list[Any] | None = None,
    net_alias_assignments: list[Any] | None = None,
    net_merge_assignments: list[Any] | None = None,
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """执行完整的 5 阶段流水线

    Args:
        images_b64: 1-3 张 base64 图片
        reference_circuit: 参考电路 JSON 路径或内联 reference payload
        rail_assignments: 电源轨道指定, 如 {"top_plus": "VCC", "top_minus": "GND", ...}
        port_annotations: 用户最小输入/输出端口标注
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
                "semantic_analysis": {...},
            },
            "total_duration_ms": float,
        }
    """
    t0 = time.time()
    ctx = get_shared_context()

    # 校准器携带网格与 fallback 状态, 不能跨请求共享.
    calibrator = BreadboardCalibrator(
        rows=settings.BREADBOARD_ROWS,
        cols_per_side=settings.BREADBOARD_COLS_PER_SIDE,
    )
    stages: dict[str, Any] = {}
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
        calibrator=calibrator,
    )
    stages["detect"] = s1
    logger.info("S1 detect: %d components (%.0fms)",
                len(s1["detections"]), s1["duration_ms"])
    _notify("detect", 1.0)

    # ── S1.5: 整图 pin 检测 ──
    _notify("pin_detect", 0.0)
    s15 = run_pin_detect(
        detections=s1["detections"],
        images_b64=images_b64,
        pin_detector=ctx.pin_detector,
        supplemental_detections=s1.get("supplemental_detections"),
        calibrator=calibrator,
    )
    stages["pin_detect"] = s15
    logger.info(
        "S1.5 pin detect: %d components (%.0fms)",
        len(s15["components"]),
        s15["duration_ms"],
    )
    _notify("pin_detect", 1.0)

    # ── S2: pin -> hole 映射 ──
    _notify("mapping", 0.0)
    s2 = run_mapping(
        s15["components"],
        calibrator=calibrator,
        image_shape=s1["primary_image_shape"],
        images_b64=images_b64,
    )
    stages["mapping"] = s2
    logger.info("S2 mapping: %d components (%.0fms)", len(s2["components"]), s2["duration_ms"])
    _notify("mapping", 1.0)

    # ── S3: 拓扑 (传入 rail_assignments) ──
    _notify("topology", 0.0)
    effective_reference = (
        reference_circuit if reference_circuit is not None else ctx.reference_circuit
    )
    subtype_records = apply_reference_ic_subtypes(
        s2["components"],
        effective_reference if isinstance(effective_reference, dict) else None,
    )
    # 默认电源轨道: op-amp 友好三电位配置；学生端可覆盖。
    effective_rails = {
        "top_plus": "VCC",
        "top_minus": "VCC",
        "bot_plus": "GND",
        "bot_minus": "VEE",
    }
    if rail_assignments:
        effective_rails.update(rail_assignments)
    s3 = run_topology(s2["components"], rail_assignments=effective_rails)
    manual_role_warnings, manual_roles_applied = apply_net_role_assignments(
        s3.get("netlist_v2") or {},
        net_role_assignments,
        port_annotations=port_annotations,
    )
    net_normalization = normalize_current_netlist(
        s3.get("netlist_v2") or {},
        reference_circuit=effective_reference if isinstance(effective_reference, dict) else None,
        net_alias_assignments=net_alias_assignments,
        net_merge_assignments=net_merge_assignments,
    )
    stages["topology"] = s3
    logger.info("S3 topology: %d nodes (%.0fms)", s3["component_count"], s3["duration_ms"])
    _notify("topology", 1.0)

    # ── S4: 检错 ──
    _notify("validate", 0.0)
    s4 = run_validate(
        s3["topology_graph"],
        reference_circuit=effective_reference,
        components=s2["components"],
        current_netlist_v2=s3.get("netlist_v2"),
    )
    stages["validate"] = s4
    logger.info("S4 validate: risk=%s (%.0fms)", s4["risk_level"], s4["duration_ms"])
    _notify("validate", 1.0)

    # ── S5: 语义分析 ──
    _notify("semantic_analysis", 0.0)
    s5 = run_semantic_analysis(
        s3.get("netlist_v2"),
        topology_graph=s3.get("topology_graph"),
        reference_circuit=effective_reference,
    )
    stages["semantic_analysis"] = s5
    logger.info(
        "S5 semantic: type=%s errors=%d (%.0fms)",
        (s5.get("circuit_type_guess") or {}).get("template_id"),
        len(s5.get("wiring_errors") or []),
        s5["duration_ms"],
    )
    _notify("semantic_analysis", 1.0)

    total_ms = (time.time() - t0) * 1000
    runtime_metadata = _build_runtime_metadata(
        conf=eff_conf,
        iou=eff_iou,
        imgsz=eff_imgsz,
    )
    if subtype_records:
        runtime_metadata["reference_ic_subtypes_applied"] = subtype_records
    port_roles_applied = [
        item for item in manual_roles_applied
        if item.get("source") == "port_annotation"
    ]
    if port_annotations:
        runtime_metadata["port_annotations"] = _dump_items(port_annotations)
        runtime_metadata["port_annotations_applied"] = port_roles_applied
        port_warnings = [
            item for item in manual_role_warnings
            if (item.get("assignment") or {}).get("source") == "port_annotation"
        ]
        if port_warnings:
            runtime_metadata["port_annotation_warnings"] = port_warnings
    if net_role_assignments:
        runtime_metadata["manual_net_role_assignments"] = _dump_items(net_role_assignments)
    if net_role_assignments or port_annotations:
        runtime_metadata["manual_roles_applied"] = manual_roles_applied
        if manual_role_warnings:
            runtime_metadata["manual_role_warnings"] = manual_role_warnings
    if net_alias_assignments:
        runtime_metadata["manual_net_alias_assignments"] = _dump_items(net_alias_assignments)
    if net_merge_assignments:
        runtime_metadata["manual_net_merge_assignments"] = _dump_items(net_merge_assignments)
    runtime_metadata["net_normalization"] = net_normalization

    return {
        "stages": stages,
        "total_duration_ms": total_ms,
        "runtime_metadata": runtime_metadata,
    }


def _build_runtime_metadata(*, conf: float, iou: float, imgsz: int) -> dict[str, Any]:
    return {
        "code_version": settings.CODE_VERSION,
        "model_version": settings.MODEL_VERSION,
        "kb_version": settings.KB_VERSION,
        "rule_version": settings.RULE_VERSION,
        "model_root": settings.LABGUARDIAN_MODEL_ROOT,
        "component_model_path": settings.YOLO_MODEL_PATH,
        "pin_model_path": settings.PIN_MODEL_PATH,
        "yolo_device": settings.YOLO_DEVICE,
        "pin_model_device": settings.PIN_MODEL_DEVICE,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "board_rows": settings.BREADBOARD_ROWS,
        "board_cols_per_side": settings.BREADBOARD_COLS_PER_SIDE,
    }


def _dump_items(items: list[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items or []:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(dict(item))
        elif hasattr(item, "__dict__"):
            out.append(dict(item.__dict__))
    return out
