"""
angnt / agent 相关数据模型
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AngntJobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AngntAskRequest(BaseModel):
    station_id: str
    query: str = ""
    mode: str = "diagnose"
    top_k: int = Field(default=5, ge=1, le=10)


class AngntCitation(BaseModel):
    source_type: str
    source_id: str
    title: str
    snippet: str


class AngntEvidence(BaseModel):
    evidence_type: str
    source_id: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AngntAction(BaseModel):
    action_type: str
    label: str
    detail: str


class AngntJobResult(BaseModel):
    job_id: str
    station_id: str
    mode: str
    answer: str
    citations: list[AngntCitation] = Field(default_factory=list)
    evidence: list[AngntEvidence] = Field(default_factory=list)
    actions: list[AngntAction] = Field(default_factory=list)
    used_retrieval: bool = False
    created_at: float = Field(default_factory=time.time)


class AngntJobAcceptedResponse(BaseModel):
    job_id: str
    status: AngntJobState


class AngntJobStatusResponse(BaseModel):
    job_id: str
    status: AngntJobState
    result: AngntJobResult | None = None
    error: str | None = None
