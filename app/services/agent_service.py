"""
Agent 服务最小骨架

当前使用规则化回答生成与内存任务表，
后续可接入 Celery、真实 LLM 与工具路由。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.agent.answering import (
    _build_follow_up_suggestions,
    build_diagnostic_citations,
    build_diagnostic_evidence,
)
from app.agent.evidence import build_runtime_evidence_from_classroom
from app.agent.graph import run_diagnostic_graph
from app.agent.tools import ToolResult
from app.schemas.angnt import (
    AngntAction,
    AngntAskRequest,
    AngntChatDebug,
    AngntJobResult,
    AngntJobState,
    AngntJobStatusResponse,
)
from app.services.classroom_state import ClassroomState
from app.services.error_tag_service import ErrorTagService
from app.services.rag_service import RagService

AGENT_MEMORY_LIMIT = 5
RISK_ORDER = {"unknown": 0, "safe": 0, "warning": 1, "danger": 2}


@dataclass
class AgentMemoryRecord:
    query: str
    risk_level: str
    error_codes: list[str]
    error_family: str
    suggested_actions: list[str] = field(default_factory=list)
    highlight_targets: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = 0.0


class AgentService:
    """负责 angnt 作业受理、状态管理与最小回答生成."""

    def __init__(self, rag_service: RagService) -> None:
        self._rag_service = rag_service
        self._lock = threading.Lock()
        self._jobs: dict[str, AngntJobStatusResponse] = {}
        self._station_memory: dict[str, list[AgentMemoryRecord]] = {}

    def submit(self, request: AngntAskRequest, classroom: ClassroomState) -> AngntJobStatusResponse:
        job_id = request.job_id or str(uuid.uuid4())
        created_at = time.time()

        with self._lock:
            self._jobs[job_id] = AngntJobStatusResponse(
                job_id=job_id,
                status=AngntJobState.RUNNING,
                result=None,
                error=None,
            )

        try:
            result = self._run_job(
                job_id=job_id,
                request=request,
                classroom=classroom,
                created_at=created_at,
            )
            response = AngntJobStatusResponse(
                job_id=job_id,
                status=AngntJobState.COMPLETED,
                result=result,
            )
        except Exception as exc:
            response = AngntJobStatusResponse(
                job_id=job_id,
                status=AngntJobState.FAILED,
                result=None,
                error=str(exc),
            )

        with self._lock:
            self._jobs[job_id] = response

        return AngntJobStatusResponse(
            job_id=job_id,
            status=response.status,
            result=None,
            error=response.error,
        )

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
        if request.mode == "diagnostic_agent":
            return self._run_diagnostic_agent_job(
                job_id=job_id,
                request=request,
                classroom=classroom,
                created_at=created_at,
            )

        base_context = self._rag_service.build_context(
            classroom=classroom,
            station_id=request.station_id,
            query=request.user_message or request.query,
            top_k=request.top_k,
        )
        station = base_context["station"]
        risk_level = station.get("risk_level", "unknown") if station else "unknown"
        diagnostics = station.get("diagnostics", []) if station else []
        risk_reasons = station.get("risk_reasons", []) if station else []

        if request.mode == "rag":
            answer, kb_citations, kb_evidence, used = self._rag_service.answer_with_kb(
                query=request.user_message or request.query,
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
                query=request.user_message or request.query,
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

    def _run_diagnostic_agent_job(
        self,
        *,
        job_id: str,
        request: AngntAskRequest,
        classroom: ClassroomState,
        created_at: float,
    ) -> AngntJobResult:
        stations = classroom.get_all_stations()
        station_data = stations.get(request.station_id, {})

        # 如果 classroom 中没有该 station，尝试用 diagnosis_context 兜底
        if not station_data and request.diagnosis_context:
            station_data = dict(request.diagnosis_context)
            station_data["station_id"] = request.station_id

        # 若仍无数据，构造空 station（用于无状态测试或首次查询），不报错
        if not station_data:
            station_data = {"station_id": request.station_id}

        evidence_contract = build_runtime_evidence_from_classroom(
            station_id=request.station_id,
            stations={request.station_id: station_data},
            error_tag_service=ErrorTagService(),
        )

        # 将 chat_history 注入 history_facts
        if request.chat_history:
            chat_facts = [
                f"chat_history:{item.role}:{item.content[:200]}"
                for item in request.chat_history
            ]
            evidence_contract.history_facts = [
                *evidence_contract.history_facts,
                *chat_facts,
            ]

        history_records = self._get_station_memory(request.station_id)
        history_facts, history_summary = self._build_context_timeline(
            current_evidence=evidence_contract,
            history_records=history_records,
        )
        evidence_contract.history_facts = history_facts
        evidence_contract.history_summary = history_summary

        graph_state = run_diagnostic_graph(
            evidence=evidence_contract,
            query=request.query,
            user_message=request.user_message or request.query,
            chat_history=[item.model_dump() for item in request.chat_history] if request.chat_history else None,
            top_k=request.top_k,
        )
        context_pack = graph_state.context_pack
        if context_pack is None:
            raise RuntimeError("diagnostic graph did not produce context_pack")
        tool_results = [
            ToolResult.model_validate(item)
            for item in graph_state.tool_results
        ]
        verification = graph_state.verification_report
        verification_passed = bool(verification and verification.passed)
        verification_issues = verification.issues if verification else ["verification missing"]
        actions = self._build_actions(
            risk_level=evidence_contract.risk_level,
            diagnostics=evidence_contract.diagnostics,
        )

        used_context_refs = [
            ref.ref_id for ref in evidence_contract.evidence_refs
        ] + [tool.tool_name for tool in tool_results]

        result = AngntJobResult(
            job_id=job_id,
            station_id=request.station_id,
            mode=request.mode,
            answer=graph_state.final_answer,
            follow_up_suggestions=_build_follow_up_suggestions(
                evidence=evidence_contract,
                context_pack=context_pack,
            ),
            citations=build_diagnostic_citations(
                evidence=evidence_contract,
                context_pack=context_pack,
                tool_results=tool_results,
            ),
            evidence=build_diagnostic_evidence(
                evidence=evidence_contract,
                context_pack=context_pack,
                tool_results=tool_results,
                verification_passed=verification_passed,
                verification_issues=verification_issues,
                graph_metrics=[
                    metric.model_dump() for metric in graph_state.graph_metrics
                ],
            ),
            actions=actions,
            used_retrieval=bool(tool_results),
            created_at=created_at,
            debug=AngntChatDebug(
                job_id=job_id,
                used_context_refs=used_context_refs[:10],
            ),
        )
        self._append_station_memory(
            station_id=request.station_id,
            record=AgentMemoryRecord(
                query=request.user_message or request.query,
                risk_level=evidence_contract.risk_level,
                error_codes=list(evidence_contract.error_codes),
                error_family=context_pack.error_family,
                suggested_actions=self._suggested_actions(
                    evidence=evidence_contract,
                    actions=actions,
                ),
                highlight_targets=_highlight_targets_from_report(
                    evidence_contract.validator_report_v2,
                ),
                created_at=created_at,
            ),
        )
        return result

    def _get_station_memory(self, station_id: str) -> list[AgentMemoryRecord]:
        with self._lock:
            return list(self._station_memory.get(station_id, []))

    def _append_station_memory(self, *, station_id: str, record: AgentMemoryRecord) -> None:
        with self._lock:
            records = [*self._station_memory.get(station_id, []), record]
            self._station_memory[station_id] = records[-AGENT_MEMORY_LIMIT:]

    def _build_context_timeline(
        self,
        *,
        current_evidence,
        history_records: list[AgentMemoryRecord],
    ) -> tuple[list[str], str]:
        if not history_records:
            return [], ""

        previous = history_records[-1]
        facts = [
            f"previous_error_codes={','.join(previous.error_codes) or 'none'}",
            f"previous_error_family={previous.error_family}",
            f"previous_risk_level={previous.risk_level}",
        ]
        current_codes = list(current_evidence.error_codes)
        previous_codes = list(previous.error_codes)
        summary_parts: list[str] = []

        if current_codes and current_codes == previous_codes:
            code_text = "、".join(current_codes)
            facts.append(f"repeated_error_codes={','.join(current_codes)}")
            summary_parts.append(f"这个问题仍然存在，上一轮也检测到 {code_text}")
        elif current_codes and previous_codes and current_codes != previous_codes:
            facts.append(
                "error_codes_changed="
                + ",".join(previous_codes)
                + "->"
                + ",".join(current_codes)
            )
            summary_parts.append(
                f"当前主要问题已从 {'、'.join(previous_codes)} 变为 {'、'.join(current_codes)}"
            )
        elif current_codes and not previous_codes:
            facts.append("error_codes_new=" + ",".join(current_codes))
            summary_parts.append(f"当前新出现 {'、'.join(current_codes)}")

        previous_risk = RISK_ORDER.get(previous.risk_level, 0)
        current_risk = RISK_ORDER.get(current_evidence.risk_level, 0)
        if current_risk < previous_risk:
            facts.append(f"risk_level_decreased={previous.risk_level}->{current_evidence.risk_level}")
            summary_parts.append(
                f"风险等级从 {previous.risk_level} 下降到 "
                f"{current_evidence.risk_level}，比上一轮有所改善，但仍需检查当前问题"
            )
        elif current_risk > previous_risk:
            facts.append(f"risk_level_increased={previous.risk_level}->{current_evidence.risk_level}")
            summary_parts.append(
                f"风险等级从 {previous.risk_level} 上升到 "
                f"{current_evidence.risk_level}，需要优先复查"
            )
        else:
            facts.append(f"risk_level_unchanged={current_evidence.risk_level}")

        if previous.suggested_actions and current_codes:
            facts.append("previous_suggested_actions=" + " | ".join(previous.suggested_actions[:3]))
            if current_codes == previous_codes:
                summary_parts.append("学生可能已经尝试修改，但上一轮建议对应的问题仍未消除")

        facts.append(f"history_record_count={len(history_records)}")
        summary = "；".join(summary_parts)
        return facts[:8], summary

    def _suggested_actions(
        self,
        *,
        evidence,
        actions: list[AngntAction],
    ) -> list[str]:
        suggestions: list[str] = []
        for finding in evidence.findings:
            if finding.suggested_action:
                suggestions.append(finding.suggested_action)
        suggestions.extend(action.detail for action in actions if action.detail)
        return _dedupe(suggestions)[:5]

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


def _highlight_targets_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    protocol = report.get("highlight_protocol", {})
    targets = protocol.get("targets", [])
    if isinstance(targets, list):
        return [target for target in targets if isinstance(target, dict)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
