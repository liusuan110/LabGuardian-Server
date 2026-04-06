"""
FastAPI 依赖注入

参考: fastapi/full-stack-fastapi-template 的 deps.py 模式
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, settings
from app.services.classroom_state import ClassroomState
from app.services.agent_service import AgentService
from app.services.guidance_service import GuidanceService
from app.services.pipeline_service import PipelineService
from app.services.rag_service import RagService
from app.services.version_service import VersionService


def get_settings() -> Settings:
    return settings


# 课堂状态单例
_classroom: ClassroomState | None = None
_guidance_service: GuidanceService | None = None
_pipeline_service: PipelineService | None = None
_rag_service: RagService | None = None
_agent_service: AgentService | None = None
_version_service: VersionService | None = None


def get_classroom() -> ClassroomState:
    global _classroom
    if _classroom is None:
        _classroom = ClassroomState(
            online_timeout=settings.STATION_ONLINE_TIMEOUT,
        )
    return _classroom


def get_guidance_service() -> GuidanceService:
    global _guidance_service
    if _guidance_service is None:
        _guidance_service = GuidanceService()
    return _guidance_service


def get_pipeline_service() -> PipelineService:
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    return _pipeline_service


def get_rag_service() -> RagService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService()
    return _rag_service


def get_agent_service() -> AgentService:
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService(rag_service=get_rag_service())
    return _agent_service


def get_version_service() -> VersionService:
    global _version_service
    if _version_service is None:
        _version_service = VersionService()
    return _version_service
