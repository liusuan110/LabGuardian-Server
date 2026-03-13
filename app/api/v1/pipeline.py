"""
Pipeline 任务提交 API

客户端提交图片 → 返回 job_id → 轮询/WebSocket 获取结果
参考: GregaVrbancic/fastapi-celery 的 task 提交模式
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from app.core.deps import get_classroom
from app.api.v1.classroom import _frame_cache, _frame_lock
from app.schemas.pipeline import (
    JobStatus,
    JobStatusResponse,
    PipelineRequest,
    PipelineResult,
    StageResult,
    PipelineStage,
)
from app.pipeline.orchestrator import run_pipeline as _run_pipeline_sync

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/run", response_model=PipelineResult)
async def run_pipeline_sync(request: PipelineRequest):
    """
    同步执行 Pipeline (演示用) — 直接返回完整结果，无需 Celery/Redis
    """
    job_id = str(uuid.uuid4())
    try:
        raw = _run_pipeline_sync(
            images_b64=request.images_b64,
            conf=request.conf,
            iou=request.iou,
            imgsz=request.imgsz,
            rail_assignments=request.rail_assignments,
        )
        stages_raw = raw.get("stages", {})
        stages = [
            StageResult(
                stage=PipelineStage(k),
                duration_ms=v.get("duration_ms", 0),
                data={kk: vv for kk, vv in v.items() if kk != "duration_ms"},
            )
            for k, v in stages_raw.items()
        ]
        s4 = stages_raw.get("validate", {})
        s3 = stages_raw.get("topology", {})
        result = PipelineResult(
            job_id=job_id,
            station_id=request.station_id,
            status=JobStatus.COMPLETED,
            stages=stages,
            total_duration_ms=raw.get("total_duration_ms", 0),
            component_count=s3.get("component_count", 0),
            net_count=len(s3.get("netlist", {}).get("nets", [])),
            progress=s4.get("progress", 0.0),
            similarity=s4.get("similarity", 0.0),
            diagnostics=s4.get("diagnostics", []),
            risk_level=s4.get("risk_level", "safe"),
            risk_reasons=s4.get("risk_reasons", []),
        )

        # 自动同步管线结果到教师端 ClassroomState
        classroom = get_classroom()
        thumb = request.images_b64[0] if request.images_b64 else ""
        classroom.update_station({
            "station_id": request.station_id,
            "thumbnail_b64": thumb,
            "component_count": result.component_count,
            "net_count": result.net_count,
            "progress": result.progress,
            "similarity": result.similarity,
            "diagnostics": result.diagnostics,
            "risk_level": result.risk_level,
            "risk_reasons": result.risk_reasons,
            "circuit_snapshot": s3.get("circuit_description", ""),
            "missing_components": s4.get("missing", []),
            "match_level": s4.get("match_level", ""),
            "detector_ok": "ok",
        })

        # 同步学生上传的图片到 frame cache (供 /thumbnail 端点使用)
        if thumb:
            with _frame_lock:
                _frame_cache[request.station_id] = thumb

        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/submit", response_model=JobStatusResponse)
async def submit_pipeline(request: PipelineRequest):
    """
    提交四阶段 Pipeline 任务 (异步，需要 Celery+Redis)
    演示阶段请使用 POST /pipeline/run (同步)
    """
    try:
        from app.worker.tasks import run_pipeline_task
        task = run_pipeline_task.delay(
            images_b64=request.images_b64,
            reference_path=request.reference_circuit,
            rail_assignments=request.rail_assignments,
            conf=request.conf,
            iou=request.iou,
            imgsz=request.imgsz,
        )
        return JobStatusResponse(job_id=task.id, status=JobStatus.PENDING)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Celery unavailable: {exc}")


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
        current_stage = meta.get("current_stage") or meta.get("stage")
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
