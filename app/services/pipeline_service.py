"""
Pipeline 应用服务

负责:
- 同步/异步 pipeline 调用编排
- 统一结果组装
- 同步课堂态与缩略图缓存
"""

from __future__ import annotations

import uuid
from typing import Any

from celery.result import AsyncResult

from app.core.celery_app import celery_app
from app.pipeline.orchestrator import run_pipeline
from app.schemas.pipeline import (
    JobStatus,
    JobStatusResponse,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    StageResult,
)
from app.services.classroom_state import ClassroomState
from app.services.guidance_service import GuidanceService


class PipelineService:
    """封装 pipeline 运行与结果同步逻辑."""

    def build_pipeline_result(
        self,
        *,
        job_id: str,
        request: PipelineRequest,
        raw: dict[str, Any],
    ) -> PipelineResult:
        stages_raw = raw.get("stages", {})
        stages = [
            StageResult(
                stage=PipelineStage(stage_name),
                duration_ms=stage_data.get("duration_ms", 0),
                data={k: v for k, v in stage_data.items() if k != "duration_ms"},
            )
            for stage_name, stage_data in stages_raw.items()
        ]
        s3 = stages_raw.get("topology", {})
        s4 = stages_raw.get("validate", {})
        return PipelineResult(
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

    def sync_result_to_classroom(
        self,
        *,
        classroom: ClassroomState,
        guidance_service: GuidanceService,
        request: PipelineRequest,
        result: PipelineResult,
    ) -> None:
        stages_by_name = {stage.stage.value: stage.data for stage in result.stages}
        s3 = stages_by_name.get(PipelineStage.TOPOLOGY.value, {})
        s4 = stages_by_name.get(PipelineStage.VALIDATE.value, {})

        thumbnail_b64 = request.images_b64[0] if request.images_b64 else ""
        classroom.update_station(
            {
                "station_id": request.station_id,
                "thumbnail_b64": thumbnail_b64,
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
            }
        )
        guidance_service.cache_thumbnail(request.station_id, thumbnail_b64)

    def run_sync(
        self,
        request: PipelineRequest,
        classroom: ClassroomState,
        guidance_service: GuidanceService,
    ) -> PipelineResult:
        job_id = str(uuid.uuid4())
        raw = run_pipeline(
            images_b64=request.images_b64,
            conf=request.conf,
            iou=request.iou,
            imgsz=request.imgsz,
            rail_assignments=request.rail_assignments,
        )
        result = self.build_pipeline_result(job_id=job_id, request=request, raw=raw)
        self.sync_result_to_classroom(
            classroom=classroom,
            guidance_service=guidance_service,
            request=request,
            result=result,
        )
        return result

    def submit_async(self, request: PipelineRequest) -> JobStatusResponse:
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

    def get_job_status(self, job_id: str) -> JobStatusResponse:
        result = AsyncResult(job_id, app=celery_app)

        if result.state == "PENDING":
            return JobStatusResponse(job_id=job_id, status=JobStatus.PENDING)

        if result.state in {"STARTED", "PROGRESS"}:
            meta = result.info or {}
            current_stage = meta.get("current_stage") or meta.get("stage")
            return JobStatusResponse(
                job_id=job_id,
                status=JobStatus.RUNNING,
                current_stage=current_stage,
            )

        if result.state == "SUCCESS":
            payload = result.result
            return JobStatusResponse(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                result=PipelineResult(**payload) if isinstance(payload, dict) else None,
            )

        if result.state == "FAILURE":
            return JobStatusResponse(job_id=job_id, status=JobStatus.FAILED)

        return JobStatusResponse(job_id=job_id, status=JobStatus.PENDING)
