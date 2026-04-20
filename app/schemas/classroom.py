"""
数据模型 — 课堂相关 (← shared/models.py)

所有网络传输的数据结构, 由 OpenAPI 反向生成 Android/Electron 客户端模型。
"""

from __future__ import annotations

import time
from typing import List, Optional

from pydantic import BaseModel, Field


class ComponentInfo(BaseModel):
    """单个元器件信息 (简化版, 用于网络传输)"""

    name: str = ""
    type: str = ""
    polarity: str = "none"
    pin1: List[int | str] = Field(default_factory=list)
    pin2: List[int | str] = Field(default_factory=list)
    pin3: List[int | str] = Field(default_factory=list)
    confidence: float = 0.0


class StationHeartbeat(BaseModel):
    """
    学生工位心跳包 — 每 2 秒由学生端 POST 到教师服务器

    包含: 元器件检测结果 + 电路验证进度 + 诊断问题 + 风险等级 + 系统状态
    """

    # ---- 工位身份 ----
    station_id: str
    student_name: str = ""

    # ---- 时间戳 ----
    timestamp: float = Field(default_factory=time.time)

    # ---- 元器件检测 ----
    component_count: int = 0
    net_count: int = 0
    components: List[ComponentInfo] = Field(default_factory=list)

    # ---- 电路验证 ----
    progress: float = 0.0
    similarity: float = 0.0
    match_level: str = ""
    missing_components: List[str] = Field(default_factory=list)

    # ---- 诊断 ----
    diagnostics: List[str] = Field(default_factory=list)

    # ---- 风险分级 ----
    risk_level: str = "safe"
    risk_reasons: List[str] = Field(default_factory=list)

    # ---- 电路快照 ----
    circuit_snapshot: str = ""

    # ---- 系统状态 ----
    fps: float = 0.0
    detector_ok: str = "ok"
    llm_backend: str = ""
    ocr_backend: str = ""

    # ---- 缩略图 ----
    thumbnail_b64: str = ""


class GuidanceMessage(BaseModel):
    """教师 → 单个学生 的指导消息"""

    station_id: str
    type: str = "hint"
    message: str
    sender: str = "Teacher"
    timestamp: float = Field(default_factory=time.time)


class BroadcastMessage(BaseModel):
    """教师 → 全班广播消息"""

    type: str = "broadcast"
    message: str
    sender: str = "Teacher"
    timestamp: float = Field(default_factory=time.time)


class GuidanceAuditRecord(BaseModel):
    """指导审计记录"""

    audit_id: str
    target_type: str
    target_id: str
    delivery_status: str
    delivery_reason: str = ""
    payload: dict = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class StationDetail(BaseModel):
    """单工位详情 (扩展字段, 用于教师端)"""

    station_id: str
    student_name: str = ""
    progress: float = 0.0
    similarity: float = 0.0
    risk_level: str = "safe"
    risk_reasons: List[str] = Field(default_factory=list)
    diagnostics: List[str] = Field(default_factory=list)
    component_count: int = 0
    online: bool = False
    elapsed_s: float = 0.0
    risk_event_count: int = 0
    peak_progress: float = 0.0


class RankingEntry(BaseModel):
    """排行榜条目"""

    rank: int = 0
    station_id: str
    student_name: str = ""
    progress: float = 0.0
    similarity: float = 0.0
    elapsed_s: float = 0.0
    risk_event_count: int = 0
    component_count: int = 0
    risk_level: str = "safe"
    online: bool = False


class ClassroomStats(BaseModel):
    """班级聚合统计"""

    total_stations: int = 0
    online_count: int = 0
    completed_count: int = 0
    avg_progress: float = 0.0
    total_risk_events: int = 0
    danger_count: int = 0
    error_histogram: dict = Field(default_factory=dict)
    session_duration_s: float = 0.0
