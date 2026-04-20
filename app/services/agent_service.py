"""
Agent 服务最小骨架

当前使用规则化回答生成与内存任务表，
后续可接入 Celery、真实 LLM 与工具路由。
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from app.schemas.angnt import (
    AngntAction,
    AngntAskRequest,
    AngntJobResult,
    AngntJobState,
    AngntJobStatusResponse,
)
from app.services.classroom_state import ClassroomState
from app.services.rag_service import RagService


class AgentService:
    """负责 angnt 作业受理、状态管理与最小回答生成."""

    def __init__(self, rag_service: RagService) -> None:
        self._rag_service = rag_service
        self._lock = threading.Lock()
        self._jobs: dict[str, AngntJobStatusResponse] = {}

    def submit(self, request: AngntAskRequest, classroom: ClassroomState) -> AngntJobStatusResponse:
        job_id = str(uuid.uuid4())
        created_at = time.time()

        with self._lock:
            self._jobs[job_id] = AngntJobStatusResponse(
                job_id=job_id,
                status=AngntJobState.RUNNING,
                result=None,
                error=None,
            )

        try:
            result = self._run_job(job_id=job_id, request=request, classroom=classroom, created_at=created_at)
            response = AngntJobStatusResponse(job_id=job_id, status=AngntJobState.COMPLETED, result=result)
        except Exception as exc:
            response = AngntJobStatusResponse(
                job_id=job_id,
                status=AngntJobState.FAILED,
                result=None,
                error=str(exc),
            )

        with self._lock:
            self._jobs[job_id] = response

        return AngntJobStatusResponse(job_id=job_id, status=response.status, result=None, error=response.error)

    def get_status(self, job_id: str) -> AngntJobStatusResponse:
        with self._lock:
            return self._jobs.get(
                job_id,
                AngntJobStatusResponse(
                    job_id=job_id,
                    status=AngntJobState.FAILED,
                    result=None,
                    error="job not found",
                ),
            )

    def _run_job(
        self,
        *,
        job_id: str,
        request: AngntAskRequest,
        classroom: ClassroomState,
        created_at: float,
    ) -> AngntJobResult:
        base_context = self._rag_service.build_context(
            classroom=classroom,
            station_id=request.station_id,
            query=request.query,
            top_k=request.top_k,
        )
        station = base_context["station"]
        risk_level = station.get("risk_level", "unknown") if station else "unknown"
        diagnostics = station.get("diagnostics", []) if station else []
        risk_reasons = station.get("risk_reasons", []) if station else []

        if request.mode == "rag":
            answer, kb_citations, kb_evidence, used = self._rag_service.answer_with_kb(
                query=request.query,
                top_k=request.top_k,
            )
            citations = list(kb_citations) + list(base_context.get("citations", []))
            evidence = list(kb_evidence) + list(base_context.get("evidence", []))
            used_retrieval = bool(used) or bool(base_context.get("used_retrieval"))
            actions = self._build_actions(risk_level=risk_level, diagnostics=diagnostics)
        else:
            answer = self._build_answer(
                station_id=request.station_id,
                mode=request.mode,
                query=request.query,
                risk_level=risk_level,
                diagnostics=diagnostics,
                risk_reasons=risk_reasons,
            )
            citations = base_context["citations"]
            evidence = base_context["evidence"]
            used_retrieval = base_context["used_retrieval"]
            actions = self._build_actions(risk_level=risk_level, diagnostics=diagnostics)

        return AngntJobResult(
            job_id=job_id,
            station_id=request.station_id,
            mode=request.mode,
            answer=answer,
            citations=citations,
            evidence=evidence,
            actions=actions,
            used_retrieval=used_retrieval,
            created_at=created_at,
        )

    def _build_answer(
        self,
        *,
        station_id: str,
        mode: str,
        query: str,
        risk_level: str,
        diagnostics: list[Any],
        risk_reasons: list[Any],
    ) -> str:
        diagnostics_text = "；".join(str(item) for item in diagnostics[:3]) or "暂无诊断条目"
        reason_text = "；".join(str(item) for item in risk_reasons[:2]) or "暂无高风险原因"
        return (
            f"工位 {station_id} 当前处于 {risk_level} 风险等级。"
            f"模式={mode}，问题={query or '未提供额外问题'}。"
            f"诊断摘要：{diagnostics_text}。"
            f"风险摘要：{reason_text}。"
        )

    def _build_actions(self, *, risk_level: str, diagnostics: list[Any]) -> list[AngntAction]:
        actions: list[AngntAction] = []
        if risk_level == "danger":
            actions.append(
                AngntAction(
                    action_type="safety_check",
                    label="立即断电复查",
                    detail="优先检查短路、极性和电源轨连接情况。",
                )
            )
        elif risk_level == "warning":
            actions.append(
                AngntAction(
                    action_type="guided_fix",
                    label="按诊断逐项排查",
                    detail="先检查最前面的风险原因，再核对参考电路。",
                )
            )
        else:
            actions.append(
                AngntAction(
                    action_type="review",
                    label="继续验证",
                    detail="当前风险较低，建议继续检查剩余元件和连接完整性。",
                )
            )

        if diagnostics:
            actions.append(
                AngntAction(
                    action_type="teacher_hint",
                    label="推送指导",
                    detail=str(diagnostics[0])[:160],
                )
            )
        return actions
