"""
Pipeline 任务提交 API

客户端提交图片 → 返回 job_id → 轮询/WebSocket 获取结果
参考: GregaVrbancic/fastapi-celery 的 task 提交模式
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.pipeline import (
    JobStatus,
    JobStatusResponse,
    PipelineRequest,
    PipelineResult,
)
from app.worker.tasks import run_pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/submit", response_model=JobStatusResponse)
async def submit_pipeline(request: PipelineRequest):
    """
    提交四阶段 Pipeline 任务 (异步)

    返回 job_id, 客户端通过 GET /pipeline/status/{job_id} 轮询结果
    """
    task = run_pipeline.delay(request.model_dump())
    return JobStatusResponse(
        job_id=task.id,
        status=JobStatus.PENDING,
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_pipeline_status(job_id: str):
    """查询 Pipeline 任务状态"""
    from celery.result import AsyncResult

    from app.core.celery_app import celery_app

    result = AsyncResult(job_id, app=celery_app)

    if result.state == "PENDING":
        return JobStatusResponse(job_id=job_id, status=JobStatus.PENDING)

    if result.state == "STARTED" or result.state == "PROGRESS":
        meta = result.info or {}
        current_stage = meta.get("current_stage")
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.RUNNING,
            current_stage=current_stage,
        )

    if result.state == "SUCCESS":
        pipeline_result = result.result
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            result=PipelineResult(**pipeline_result) if isinstance(pipeline_result, dict) else None,
        )

    if result.state == "FAILURE":
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.FAILED,
        )

    return JobStatusResponse(job_id=job_id, status=JobStatus.PENDING)
