"""
数据模型 — Pipeline 相关
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    """四阶段名称"""

    DETECT = "detect"
    PIN_DETECT = "pin_detect"
    MAPPING = "mapping"
    TOPOLOGY = "topology"
    VALIDATE = "validate"


class JobStatus(str, Enum):
    """任务状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineRequest(BaseModel):
    """Pipeline 任务提交请求"""

    station_id: str
    images_b64: List[str] = Field(
        ..., min_length=1, max_length=3,
        description="1-3 张面包板俯拍图 (base64 JPEG)",
    )
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 1280
    reference_circuit: Optional[Dict[str, Any]] = None
    rail_assignments: Optional[Dict[str, str]] = Field(
        default=None,
        description="面包板电源轨道指定, 如 {\"top_plus\": \"VCC\", \"top_minus\": \"GND\", \"bot_plus\": \"VCC\", \"bot_minus\": \"GND\"}",
    )


class StageResult(BaseModel):
    """单阶段执行结果"""

    stage: PipelineStage
    status: JobStatus = JobStatus.COMPLETED
    duration_ms: float = 0.0
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """完整 Pipeline 结果"""

    job_id: str
    station_id: str
    status: JobStatus = JobStatus.COMPLETED
    stages: List[StageResult] = Field(default_factory=list)
    total_duration_ms: float = 0.0

    # ---- 汇总 ----
    component_count: int = 0
    net_count: int = 0
    progress: float = 0.0
    similarity: float = 0.0
    diagnostics: List[str] = Field(default_factory=list)
    comparison_report: Dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "safe"
    risk_reasons: List[str] = Field(default_factory=list)
    report: str = ""

    @classmethod
    def from_pipeline_run(
        cls,
        *,
        job_id: str,
        station_id: str,
        raw: Dict[str, Any],
    ) -> "PipelineResult":
        """将编排器原始输出标准化为统一的 PipelineResult.

        兼容两种输入:
        1. 编排器原始结果: {"stages": {...}, "total_duration_ms": ...}
        2. 已序列化的 PipelineResult dict
        """
        if isinstance(raw.get("stages"), list) and "status" in raw:
            payload = dict(raw)
            payload.setdefault("job_id", job_id)
            payload.setdefault("station_id", station_id)
            return cls(**payload)

        stages_raw = raw.get("stages", {})
        stages = [
            StageResult(
                stage=PipelineStage(stage_name),
                duration_ms=stage_data.get("duration_ms", 0),
                data={k: v for k, v in stage_data.items() if k != "duration_ms"},
            )
            for stage_name, stage_data in stages_raw.items()
        ]
        s3 = stages_raw.get(PipelineStage.TOPOLOGY.value, {})
        s4 = stages_raw.get(PipelineStage.VALIDATE.value, {})
        return cls(
            job_id=job_id,
            station_id=station_id,
            status=JobStatus.COMPLETED,
            stages=stages,
            total_duration_ms=raw.get("total_duration_ms", 0),
            component_count=s3.get("component_count", 0),
            net_count=len(s3.get("netlist_v2", {}).get("nets", [])),
            progress=s4.get("progress", 0.0),
            similarity=s4.get("similarity", 0.0),
            diagnostics=s4.get("diagnostics", []),
            comparison_report=s4.get("comparison_report", {}),
            risk_level=s4.get("risk_level", "safe"),
            risk_reasons=s4.get("risk_reasons", []),
        )


class JobStatusResponse(BaseModel):
    """任务状态查询响应"""

    job_id: str
    status: JobStatus
    current_stage: Optional[PipelineStage] = None
    result: Optional[PipelineResult] = None
