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
from app.domain.logical_reference import normalize_net_role
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
from app.services.reference_service import ReferenceService


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

    def _resolve_reference(
        self,
        *,
        reference_id: str | None = None,
        reference_circuit: dict[str, Any] | str | None = None,
    ) -> tuple[dict[str, Any] | str | None, dict[str, Any]]:
        """根据请求优先级解析参考电路并返回元数据。

        优先级：reference_id > reference_circuit > settings fallback > none
        """
        ref_meta: dict[str, Any] = {"source": "none"}

        if reference_id:
            service = ReferenceService()
            ref_payload = service.load_reference(reference_id)
            ref_meta = {
                "source": "reference_id",
                "reference_id": reference_id,
                "format": ref_payload.get("format"),
                "name": ref_payload.get("name"),
            }
            return ref_payload, ref_meta

        if reference_circuit:
            ref_meta = {
                "source": "inline_payload",
                "format": reference_circuit.get("format")
                if isinstance(reference_circuit, dict)
                else None,
            }
            return reference_circuit, ref_meta

        return None, ref_meta

    def run_sync(
        self,
        request: PipelineRequest,
        classroom: ClassroomState,
        guidance_service: GuidanceService,
    ) -> PipelineResult:
        job_id = str(uuid.uuid4())
        reference_circuit, ref_meta = self._resolve_reference(
            reference_id=request.reference_id,
            reference_circuit=request.reference_circuit,
        )
        raw = run_pipeline(
            images_b64=request.images_b64,
            reference_circuit=reference_circuit,
            conf=request.conf,
            iou=request.iou,
            imgsz=request.imgsz,
            rail_assignments=request.rail_assignments,
        )
        # 将 reference 来源信息写入 runtime_metadata
        runtime_metadata = raw.get("runtime_metadata", {})
        runtime_metadata["reference"] = ref_meta
        raw["runtime_metadata"] = runtime_metadata

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

        has_pin_corrections = bool(request.corrections)
        has_role_assignments = bool(request.net_role_assignments)

        if not has_pin_corrections and not has_role_assignments:
            raise ValueError(
                "Corrected recompute requires at least one pin correction or net role assignment."
            )

        # 统一解析参考电路（支持 reference_id 和 inline payload）
        reference_circuit, ref_meta = self._resolve_reference(
            reference_id=request.reference_id,
            reference_circuit=request.reference_circuit,
        )

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

        # 处理手动网络角色指定
        netlist_v2 = s3.get("netlist_v2") or {}
        manual_role_warnings: list[dict[str, Any]] = []
        manual_roles_applied: list[dict[str, Any]] = []

        if request.net_role_assignments:
            # 建立索引：优先使用 netlist_v2 中的拓扑信息
            netlist_nets: dict[str, dict] = {}
            netlist_nets_by_hole: dict[str, dict] = {}
            netlist_nets_by_node: dict[str, dict] = {}
            netlist_nets_by_comp_pin: dict[tuple[str, str], dict] = {}

            for net in netlist_v2.get("nets", []):
                net_id = net.get("electrical_net_id")
                if not net_id:
                    continue
                netlist_nets[str(net_id)] = net
                for hid in net.get("member_hole_ids", []):
                    netlist_nets_by_hole[str(hid)] = net
                for nid in net.get("member_node_ids", []):
                    netlist_nets_by_node[str(nid)] = net

            for comp in netlist_v2.get("components", []):
                comp_id = str(comp.get("component_id") or "")
                for pin in comp.get("pins", []):
                    pin_name = str(pin.get("pin_name") or "")
                    pin_net_id = pin.get("electrical_net_id")
                    if comp_id and pin_name and pin_net_id:
                        netlist_nets_by_comp_pin[(comp_id, pin_name)] = netlist_nets.get(
                            str(pin_net_id), {"electrical_net_id": str(pin_net_id)}
                        )

            for assignment in request.net_role_assignments:
                normalized_role = normalize_net_role(assignment.role)
                if normalized_role == "signal":
                    manual_role_warnings.append({
                        "warning_code": "ROLE_INVALID",
                        "message": f"非法或未知的网络角色: {assignment.role}",
                        "assignment": assignment.model_dump(),
                    })
                    continue

                target_net_id: str | None = None
                resolved_by: str = ""

                if assignment.electrical_net_id:
                    target_net_id = assignment.electrical_net_id
                    resolved_by = "electrical_net_id"
                    # 校验该 net 确实存在于当前 netlist
                    if target_net_id not in netlist_nets:
                        manual_role_warnings.append({
                            "warning_code": "ROLE_TARGET_NOT_FOUND",
                            "message": f"指定的电气网络 {target_net_id} 在当前 netlist 中不存在",
                            "assignment": assignment.model_dump(),
                        })
                        continue
                elif assignment.component_id and assignment.pin_name:
                    net_obj = netlist_nets_by_comp_pin.get((assignment.component_id, assignment.pin_name))
                    if net_obj:
                        target_net_id = net_obj.get("electrical_net_id")
                        resolved_by = "component_pin"
                elif assignment.hole_id:
                    net_obj = netlist_nets_by_hole.get(str(assignment.hole_id))
                    if net_obj:
                        target_net_id = net_obj.get("electrical_net_id")
                        resolved_by = "hole_id"
                    else:
                        # 尝试通过 electrical_node_id 查找
                        if assignment.electrical_node_id:
                            net_obj = netlist_nets_by_node.get(str(assignment.electrical_node_id))
                            if net_obj:
                                target_net_id = net_obj.get("electrical_net_id")
                                resolved_by = "electrical_node_id"

                if not target_net_id:
                    manual_role_warnings.append({
                        "warning_code": "ROLE_TARGET_NOT_CONNECTED",
                        "message": (
                            f"无法为角色指定 {assignment.role} 找到对应电气网络，"
                            f"该孔位/节点可能未连接到任何元件或导线（空孔）"
                        ),
                        "assignment": assignment.model_dump(),
                    })
                    continue

                # 写入 netlist_v2 中的 net
                net_obj = netlist_nets.get(target_net_id)
                if net_obj is not None:
                    net_obj["role"] = normalized_role
                    net_obj["manual_role"] = normalized_role
                    net_obj["role_label"] = assignment.role
                    net_obj["role_source"] = assignment.source
                    if normalized_role in ("power", "ground"):
                        net_obj["power_role"] = "VCC" if normalized_role == "power" else "GND"
                else:
                    manual_role_warnings.append({
                        "warning_code": "ROLE_TARGET_NOT_FOUND",
                        "message": f"角色指定目标网络 {target_net_id} 在 netlist_v2 中不存在",
                        "assignment": assignment.model_dump(),
                    })
                    continue

                # 写入相关 pin metadata
                for comp in components:
                    comp_id = str(comp.get("component_id") or "")
                    for pin in comp.get("pins", []):
                        pin_net_id = pin.get("electrical_net_id")
                        if pin_net_id == target_net_id:
                            metadata = dict(pin.get("metadata") or {})
                            metadata["manual_net_role"] = normalized_role
                            metadata["manual_role_label"] = assignment.role
                            pin["metadata"] = metadata

                applied_record: dict[str, Any] = {
                    "role": normalized_role,
                    "role_label": assignment.role,
                    "electrical_net_id": target_net_id,
                    "source": assignment.source,
                    "resolved_by": resolved_by,
                }
                if assignment.hole_id:
                    applied_record["hole_id"] = assignment.hole_id
                if assignment.x_image is not None:
                    applied_record["x_image"] = assignment.x_image
                if assignment.y_image is not None:
                    applied_record["y_image"] = assignment.y_image
                manual_roles_applied.append(applied_record)

        s4 = run_validate(
            s3["topology_graph"],
            reference_circuit=reference_circuit,
            components=components,
            current_netlist_v2=netlist_v2,
        )
        s5 = run_semantic_analysis(
            s3.get("netlist_v2"),
            topology_graph=s3.get("topology_graph"),
            reference_circuit=reference_circuit,
        )

        mapping_stage = {
            "components": components,
            "manual_corrections_applied": True,
            "manual_corrections": [item.model_dump() for item in request.corrections],
            "duration_ms": 0.0,
        }
        total_ms = (time.time() - t0) * 1000

        # 组装 runtime_metadata：保留手动修正信息 + 增加 reference 来源信息
        runtime_metadata: dict[str, Any] = {
            "manual_corrections_applied": True,
            "manual_corrections": [item.model_dump() for item in request.corrections],
            "manual_net_role_assignments": [item.model_dump() for item in request.net_role_assignments],
            "manual_roles_applied": manual_roles_applied,
            "source_job_id": request.job_id,
            "reference": ref_meta,
        }
        if manual_role_warnings:
            runtime_metadata["manual_role_warnings"] = manual_role_warnings

        raw = {
            "stages": {
                "mapping": mapping_stage,
                "topology": s3,
                "validate": s4,
                "semantic_analysis": s5,
            },
            "total_duration_ms": total_ms,
            "runtime_metadata": runtime_metadata,
        }
        job_id = f"{request.job_id}-corrected" if request.job_id else str(uuid.uuid4())
        return self.build_pipeline_result(job_id=job_id, request=request, raw=raw)

    def submit_async(self, request: PipelineRequest) -> JobStatusResponse:
        from app.worker.tasks import run_pipeline_task

        reference_circuit, _ = self._resolve_reference(
            reference_id=request.reference_id,
            reference_circuit=request.reference_circuit,
        )
        task = run_pipeline_task.delay(
            station_id=request.station_id,
            images_b64=request.images_b64,
            reference_circuit=reference_circuit,
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
        reference_circuit, ref_meta = self._resolve_reference(
            reference_id=request.reference_id,
            reference_circuit=request.reference_circuit,
        )
        raw = run_pipeline(
            images_b64=request.images_b64,
            reference_circuit=reference_circuit,
            conf=request.conf,
            iou=request.iou,
            imgsz=request.imgsz,
            rail_assignments=request.rail_assignments,
        )
        runtime_metadata = raw.get("runtime_metadata", {})
        runtime_metadata["reference"] = ref_meta
        raw["runtime_metadata"] = runtime_metadata
        
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
