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


class JobStatusResponse(BaseModel):
    """任务状态查询响应"""

    job_id: str
    status: JobStatus
    current_stage: Optional[PipelineStage] = None
    result: Optional[PipelineResult] = None
