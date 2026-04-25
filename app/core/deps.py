"""
FastAPI 依赖注入

参考: fastapi/full-stack-fastapi-template 的 deps.py 模式
"""

from __future__ import annotations

from app.core.config import Settings, settings
from app.services.agent_service import AgentService
from app.services.classroom_state import ClassroomState
from app.services.error_tag_service import ErrorTagService
from app.services.guidance_service import GuidanceService
from app.services.kb_service import KbService
from app.services.mrag_service import MragService
from app.services.pipeline_service import PipelineService
from app.services.rag_service import RagService
from app.services.teaching_kb_service import TeachingKbService
from app.services.version_service import VersionService
from app.services.vlm_service import VlmService


def get_settings() -> Settings:
    return settings


# 课堂状态单例
_classroom: ClassroomState | None = None
_guidance_service: GuidanceService | None = None
_pipeline_service: PipelineService | None = None
_rag_service: RagService | None = None
_agent_service: AgentService | None = None
_version_service: VersionService | None = None
_kb_service: KbService | None = None
_teaching_kb_service: TeachingKbService | None = None
_error_tag_service: ErrorTagService | None = None
_mrag_service: MragService | None = None
_vlm_service: VlmService | None = None


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


def get_kb_service() -> KbService:
    global _kb_service
    if _kb_service is None:
        _kb_service = KbService()
    return _kb_service


def get_teaching_kb_service() -> TeachingKbService:
    global _teaching_kb_service
    if _teaching_kb_service is None:
        _teaching_kb_service = TeachingKbService()
    return _teaching_kb_service


def get_error_tag_service() -> ErrorTagService:
    global _error_tag_service
    if _error_tag_service is None:
        _error_tag_service = ErrorTagService()
    return _error_tag_service


def get_mrag_service() -> MragService:
    global _mrag_service
    if _mrag_service is None:
        _mrag_service = MragService(teaching_kb_service=get_teaching_kb_service())
    return _mrag_service


def get_vlm_service() -> VlmService:
    global _vlm_service
    if _vlm_service is None:
        _vlm_service = VlmService()
    return _vlm_service


def get_rag_service() -> RagService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService(
            kb_service=get_kb_service(),
            teaching_kb_service=get_teaching_kb_service(),
            error_tag_service=get_error_tag_service(),
            mrag_service=get_mrag_service(),
        )
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
