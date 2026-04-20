"""
Pipeline 任务提交 API

客户端提交图片 → 返回 job_id → 轮询/WebSocket 获取结果
参考: GregaVrbancic/fastapi-celery 的 task 提交模式
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.deps import get_classroom, get_guidance_service, get_pipeline_service
from app.schemas.pipeline import (
    JobStatusResponse,
    PipelineRequest,
    PipelineResult,
)
from app.services.classroom_state import ClassroomState
from app.services.guidance_service import GuidanceService
from app.services.pipeline_service import PipelineService

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/run", response_model=PipelineResult)
async def run_pipeline_sync(
    request: PipelineRequest,
    classroom: ClassroomState = Depends(get_classroom),
    guidance_service: GuidanceService = Depends(get_guidance_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """
    同步执行 Pipeline (演示用) — 直接返回完整结果，无需 Celery/Redis
    """
    try:
        return pipeline_service.run_sync(
            request=request,
            classroom=classroom,
            guidance_service=guidance_service,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/submit", response_model=JobStatusResponse)
async def submit_pipeline(
    request: PipelineRequest,
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """
    提交四阶段 Pipeline 任务 (异步，需要 Celery+Redis)
    演示阶段请使用 POST /pipeline/run (同步)
    """
    try:
        return pipeline_service.submit_async(request)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Celery unavailable: {exc}")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_pipeline_status(
    job_id: str,
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """查询 Pipeline 任务状态"""
    return pipeline_service.get_job_status(job_id)
