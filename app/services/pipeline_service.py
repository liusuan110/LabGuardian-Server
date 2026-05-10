"""
Pipeline 应用服务

负责:
- 同步/异步 pipeline 调用编排
- 统一结果组装
- 同步课堂态与缩略图缓存
"""

from __future__ import annotations

import copy
import time
import uuid
from typing import Any

from celery.result import AsyncResult

from app.core.celery_app import celery_app
from app.domain.board_schema import BoardSchema
from app.pipeline.orchestrator import run_pipeline
from app.pipeline.stages.s3_topology import run_topology
from app.pipeline.stages.s4_validate import run_validate
from app.pipeline.stages.s5_semantic_analysis import run_semantic_analysis
from app.schemas.pipeline import (
    CircuitAnalysisResult,
    CorrectedRecomputeRequest,
    JobStatus,
    JobStatusResponse,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
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
        return PipelineResult.from_pipeline_run(
            job_id=job_id,
            station_id=request.station_id,
            raw=raw,
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
        s5 = stages_by_name.get(PipelineStage.SEMANTIC_ANALYSIS.value, {})

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
                "comparison_report": result.comparison_report,
                "risk_level": result.risk_level,
                "risk_reasons": result.risk_reasons,
                "circuit_snapshot": s3.get("circuit_description", ""),
                "netlist_v2": s3.get("netlist_v2", {}),
                "semantic_analysis": s5,
                "runtime_metadata": result.runtime_metadata,
                "missing_components": s4.get("missing", []),
                "match_level": s4.get("match_level", ""),
                "detector_ok": "ok",
            }
        )
        guidance_service.cache_thumbnail(request.station_id, thumbnail_b64)

    def sync_corrected_result_to_classroom(
        self,
        *,
        classroom: ClassroomState,
        request: CorrectedRecomputeRequest,
        result: PipelineResult,
    ) -> None:
        """同步手动修正后的可信拓扑到课堂状态，保留原缩略图等心跳信息。"""
        stages_by_name = {stage.stage.value: stage.data for stage in result.stages}
        s3 = stages_by_name.get(PipelineStage.TOPOLOGY.value, {})
        s4 = stages_by_name.get(PipelineStage.VALIDATE.value, {})
        s5 = stages_by_name.get(PipelineStage.SEMANTIC_ANALYSIS.value, {})
        previous = classroom.get_all_stations().get(request.station_id, {})

        classroom.update_station(
            {
                **previous,
                "station_id": request.station_id,
                "component_count": result.component_count,
                "net_count": result.net_count,
                "progress": result.progress,
                "similarity": result.similarity,
                "diagnostics": result.diagnostics,
                "comparison_report": result.comparison_report,
                "risk_level": result.risk_level,
                "risk_reasons": result.risk_reasons,
                "circuit_snapshot": s3.get("circuit_description", ""),
                "netlist_v2": s3.get("netlist_v2", {}),
                "semantic_analysis": s5,
                "runtime_metadata": result.runtime_metadata,
                "missing_components": s4.get("missing", []),
                "match_level": s4.get("match_level", ""),
                "detector_ok": "manual_corrected",
            }
        )

    def run_sync(
        self,
        request: PipelineRequest,
        classroom: ClassroomState,
        guidance_service: GuidanceService,
    ) -> PipelineResult:
        job_id = str(uuid.uuid4())
        raw = run_pipeline(
            images_b64=request.images_b64,
            reference_circuit=request.reference_circuit,
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

    def recompute_corrected(
        self,
        request: CorrectedRecomputeRequest,
    ) -> PipelineResult:
        """应用前端手动孔位修正后，重跑 S3/S4 并返回正式 PipelineResult."""
        if not request.components:
            raise ValueError("Corrected recompute requires mapping components.")
        if not request.corrections:
            raise ValueError("Corrected recompute requires at least one correction.")

        t0 = time.time()
        board_schema = BoardSchema.default_breadboard()
        components = copy.deepcopy(request.components)
        correction_by_key = {
            (item.component_id, item.pin_name): item
            for item in request.corrections
        }
        applied_keys: set[tuple[str, str]] = set()

        for comp in components:
            component_id = str(comp.get("component_id") or "")
            for pin in comp.get("pins") or []:
                pin_name = str(pin.get("pin_name") or f"pin{pin.get('pin_id', '')}")
                correction = correction_by_key.get((component_id, pin_name))
                if correction is not None:
                    normalized_hole = board_schema.normalize_hole_id(correction.to_hole_id)
                    pin["hole_id"] = normalized_hole
                    metadata = dict(pin.get("metadata") or {})
                    metadata["manual_correction"] = {
                        "from_hole_id": correction.from_hole_id,
                        "to_hole_id": normalized_hole,
                        "source": correction.source,
                    }
                    pin["metadata"] = metadata
                    applied_keys.add((component_id, pin_name))

                if pin.get("hole_id"):
                    hole_id = board_schema.normalize_hole_id(str(pin["hole_id"]))
                    pin["hole_id"] = hole_id
                    pin["electrical_node_id"] = board_schema.resolve_hole_to_node(hole_id)
                    logic_loc = board_schema.hole_id_to_logic_loc(hole_id)
                    if logic_loc is not None:
                        pin["logic_loc"] = [logic_loc[0], logic_loc[1]]

        missing = sorted(
            f"{component_id}.{pin_name}"
            for component_id, pin_name in correction_by_key
            if (component_id, pin_name) not in applied_keys
        )
        if missing:
            raise ValueError(f"Corrections did not match any pins: {', '.join(missing)}")

        s3 = run_topology(components, rail_assignments=request.rail_assignments)
        s4 = run_validate(
            s3["topology_graph"],
            reference_circuit=request.reference_circuit,
            components=components,
        )
        s5 = run_semantic_analysis(
            s3.get("netlist_v2"),
            topology_graph=s3.get("topology_graph"),
            reference_circuit=request.reference_circuit,
        )

        mapping_stage = {
            "components": components,
            "manual_corrections_applied": True,
            "manual_corrections": [item.model_dump() for item in request.corrections],
            "duration_ms": 0.0,
        }
        total_ms = (time.time() - t0) * 1000
        raw = {
            "stages": {
                "mapping": mapping_stage,
                "topology": s3,
                "validate": s4,
                "semantic_analysis": s5,
            },
            "total_duration_ms": total_ms,
            "runtime_metadata": {
                "manual_corrections_applied": True,
                "manual_corrections": [item.model_dump() for item in request.corrections],
                "source_job_id": request.job_id,
            },
        }
        job_id = f"{request.job_id}-corrected" if request.job_id else str(uuid.uuid4())
        return self.build_pipeline_result(job_id=job_id, request=request, raw=raw)

    def submit_async(self, request: PipelineRequest) -> JobStatusResponse:
        from app.worker.tasks import run_pipeline_task

        task = run_pipeline_task.delay(
            station_id=request.station_id,
            images_b64=request.images_b64,
            reference_circuit=request.reference_circuit,
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
            parsed = None
            if isinstance(payload, dict):
                try:
                    parsed = PipelineResult.from_pipeline_run(
                        job_id=job_id,
                        station_id=payload.get("station_id", ""),
                        raw=payload,
                    )
                except Exception:
                    parsed = None
            return JobStatusResponse(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                result=parsed,
            )

        if result.state == "FAILURE":
            return JobStatusResponse(job_id=job_id, status=JobStatus.FAILED)

        return JobStatusResponse(job_id=job_id, status=JobStatus.PENDING)

    def analyze_circuit(
        self,
        request: PipelineRequest,
        classroom: ClassroomState,
        guidance_service: GuidanceService,
    ) -> CircuitAnalysisResult:
        """
        执行电路分析，返回元件引脚二维定位和网表信息。
        
        此方法执行完整的 pipeline 分析，提取并返回：
        - 元件及其引脚的二维定位信息
        - 电气连接关系（网表）
        - 电路拓扑图（用于前端可视化）
        """
        job_id = str(uuid.uuid4())
        
        # 执行完整 pipeline
        raw = run_pipeline(
            images_b64=request.images_b64,
            reference_circuit=request.reference_circuit,
            conf=request.conf,
            iou=request.iou,
            imgsz=request.imgsz,
            rail_assignments=request.rail_assignments,
        )
        
        # 构建 PipelineResult
        pipeline_result = self.build_pipeline_result(job_id=job_id, request=request, raw=raw)
        
        # 同步结果到课堂状态
        self.sync_result_to_classroom(
            classroom=classroom,
            guidance_service=guidance_service,
            request=request,
            result=pipeline_result,
        )
        
        # 转换为 CircuitAnalysisResult
        return CircuitAnalysisResult.from_pipeline_result(
            job_id=job_id,
            station_id=request.station_id,
            pipeline_result=pipeline_result,
        )
