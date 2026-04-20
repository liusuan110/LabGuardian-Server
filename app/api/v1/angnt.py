"""
angnt API 最小骨架

当前提供:
- POST /api/v1/angnt/ask
- GET  /api/v1/angnt/status/{job_id}
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.deps import get_agent_service, get_classroom
from app.schemas.angnt import AngntAskRequest, AngntJobAcceptedResponse, AngntJobStatusResponse
from app.services.agent_service import AgentService
from app.services.classroom_state import ClassroomState

router = APIRouter(prefix="/angnt", tags=["angnt"])


@router.post("/ask", response_model=AngntJobAcceptedResponse)
async def ask_angnt(
    request: AngntAskRequest,
    classroom: ClassroomState = Depends(get_classroom),
    agent_service: AgentService = Depends(get_agent_service),
):
    """提交最小 agent 任务."""
    accepted = agent_service.submit(request, classroom)
    return AngntJobAcceptedResponse(job_id=accepted.job_id, status=accepted.status)


@router.get("/status/{job_id}", response_model=AngntJobStatusResponse)
async def get_angnt_status(
    job_id: str,
    agent_service: AgentService = Depends(get_agent_service),
):
    """查询 agent 任务状态."""
    return agent_service.get_status(job_id)
