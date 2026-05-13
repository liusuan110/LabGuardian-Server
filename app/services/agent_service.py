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

import httpx

from app.agent.answering import (
    _build_follow_up_suggestions,
    build_concept_answer,
    build_concept_citations,
    build_concept_evidence,
    build_diagnostic_citations,
    build_diagnostic_evidence,
    build_lab_guidance_answer,
    ensure_circuit_opening,
)
from app.agent.concepts import lookup_concept
from app.agent.contracts import AgentIntent, ConceptPack
from app.agent.evidence import build_runtime_evidence_from_classroom
from app.agent.graph import run_diagnostic_graph
from app.agent.intent import classify_intent
from app.agent.llm import get_llm_provider
from app.agent.tools import (
    TeachingConceptLookupInput,
    ToolResult,
    teaching_concept_lookup_tool,
)
from app.agent.verification import verify_draft_answer
from app.schemas.angnt import (
    AngntAction,
    AngntAskRequest,
    AngntChatDebug,
    AngntEvidence,
    AngntJobResult,
    AngntJobState,
    AngntJobStatusResponse,
)
from app.core.config import settings
from app.services.classroom_state import ClassroomState
from app.services.error_tag_service import ErrorTagService
from app.services.rag_service import RagService

AGENT_MEMORY_LIMIT = 5
RISK_ORDER = {"unknown": 0, "safe": 0, "warning": 1, "danger": 2}
_CIRCUIT_TOPIC_KEYWORDS = (
    "电路",
    "电压",
    "电流",
    "电阻",
    "电容",
    "电感",
    "二极管",
    "led",
    "三极管",
    "mos",
    "运放",
    "芯片",
    "ic",
    "面包板",
    "跳线",
    "引脚",
    "孔位",
    "电源",
    "短路",
    "断路",
    "接地",
    "gnd",
    "vcc",
    "传感器",
    "单片机",
    "arduino",
    "stm32",
    "万用表",
    "示波器",
    "测量",
    "焊接",
    "欧姆",
    "分压",
    "滤波",
    "rc",
    "rl",
    "rlc",
    "频率",
    "信号",
    "模拟",
    "数字",
    "原理图",
    "breadboard",
    "resistor",
    "capacitor",
    "inductor",
    "diode",
    "transistor",
    "voltage",
    "current",
)


def _is_circuit_related_question(question: str) -> bool:
    msg = (question or "").strip().lower()
    if not msg:
        return False
    for keyword in _CIRCUIT_TOPIC_KEYWORDS:
        key = keyword.lower()
        if key.isascii() and key.isalnum() and len(key) <= 3:
            if _contains_ascii_token(msg, key):
                return True
            continue
        if key in msg:
            return True
    return False


def _contains_ascii_token(msg: str, token: str) -> bool:
    start = 0
    while True:
        index = msg.find(token, start)
        if index < 0:
            return False
        before = msg[index - 1] if index > 0 else ""
        after_index = index + len(token)
        after = msg[after_index] if after_index < len(msg) else ""
        before_ok = not before or not (before.isascii() and before.isalnum())
        after_ok = not after or not (after.isascii() and after.isalnum())
        if before_ok and after_ok:
            return True
        start = index + 1


def _looks_like_current_context_follow_up(question: str) -> bool:
    msg = (question or "").strip().lower()
    follow_up_phrases = (
        "简单点",
        "再说",
        "换种说法",
        "详细点",
        "讲详细",
        "听不懂",
        "没懂",
        "什么意思",
        "这是什么意思",
        "有什么问题",
        "有啥问题",
        "啥问题",
        "问题在哪",
        "问题出在哪",
        "问题是什么",
        "具体问题",
        "具体的问题",
        "这个电路",
        "这张电路",
        "这张图",
        "电路图",
        "上传的电路",
        "上传电路",
        "参考差异",
        "参考电路",
        "和参考",
        "对比参考",
        "跟参考",
        "相比参考",
        "哪里不对",
        "哪里不对劲",
        "哪错了",
        "这个问题",
        "怎么改",
        "怎么修",
        "怎么处理",
    )
    return any(phrase in msg for phrase in follow_up_phrases)


def _build_concept_not_found_prompt(*, question: str, evidence: Any) -> str:
    risk = str(getattr(evidence, "risk_level", "unknown") or "unknown")
    error_codes = list(getattr(evidence, "error_codes", []) or [])
    diagnostics = list(getattr(evidence, "diagnostics", []) or [])
    has_current_context = bool(error_codes or diagnostics or getattr(evidence, "findings", None))
    is_follow_up = has_current_context and _looks_like_current_context_follow_up(question)
    if is_follow_up:
        topic_mode = "current_context_follow_up"
        topic_rule = (
            "用户可能是在追问当前诊断。请结合当前诊断摘要自然回答，不要机械复述完整报告；"
            "如果问题太模糊，先用一句话说明你理解的是哪一类问题，再引导他指定元件或错误码。"
        )
        context_lines = [
            f"当前电路风险等级：{risk}",
            f"当前电路错误码：{','.join(error_codes[:3]) if error_codes else '无'}",
            f"当前诊断摘要：{'；'.join(str(item) for item in diagnostics[:3]) or '无'}",
        ]
    elif _is_circuit_related_question(question):
        topic_mode = "circuit_related"
        topic_rule = (
            "这个问题仍属于电路/电子实验相关，但本地知识库没有命中。请用你的通用电路知识直接回答，"
            "不要主动复述当前工位诊断，不要声称来自本地知识库，不要编造当前电路的具体孔位、器件位置或测量值。"
        )
        context_lines = [
            f"当前电路风险等级：{risk}（仅作为安全背景；除非用户追问当前诊断，否则不要展开错误码或工位报告。）",
        ]
    else:
        topic_mode = "off_topic"
        topic_rule = (
            "这个问题明显偏离电路实验。可以非常简短回应用户的核心意图，但不要长篇展开非电路内容；"
            "重点把用户引导回电路、元件、接线、测量现象或当前诊断结果；不要主动复述上一轮诊断报告，"
            "不要编造正在测某个具体结果。"
        )
        context_lines = []
    return "\n".join(
        [
            "你是 LabGuardian 的电路实验助教。当前本地教学知识库没有命中条目。",
            f"问题类型：{topic_mode}",
            topic_rule,
            f"用户问题：{question}",
            *_topic_grounding_lines(question),
            *context_lines,
            "输出要求：用自然中文回答，3-5 句即可，不要使用固定编号模板。",
            "内容必须覆盖：直接回应用户；说明和电路实验的关系或把话题拉回电路；给出一个自然追问；包含适合低压教学实验的安全提醒。",
            "电路知识必须严谨；不确定时用保守的定性解释，不要编造机理、参数、测量值或因果关系。",
            "相关问题要给出有用解释；无关问题要简短收束，不要长篇展开；除非用户提到高压，否则不要使用“高压”措辞。",
        ]
    )


def _fallback_follow_up_text(*, question: str, evidence: Any) -> str:
    has_current_context = bool(
        getattr(evidence, "error_codes", None)
        or getattr(evidence, "diagnostics", None)
        or getattr(evidence, "findings", None)
    )
    if has_current_context and _looks_like_current_context_follow_up(question):
        return "你可以告诉我想先看哪个元件、错误码或孔位，我再把那一处拆开讲。"
    if _is_circuit_related_question(question):
        return "你想继续看它的公式、实验现象，还是它和当前电路连接的关系？"
    return "你可以继续问我电路现象、元件作用、接线检查或测量方法。"


def _topic_grounding_lines(question: str) -> list[str]:
    msg = (question or "").strip().lower()
    lines: list[str] = []
    if "电感" in msg or "inductor" in msg:
        lines.append(
            "已知事实：电感阻碍电流变化来自自感与楞次定律；感应电动势的方向总是反抗电流变化趋势，不是正反馈增强变化。"
        )
    if "电容" in msg or "capacitor" in msg:
        lines.append("已知事实：电容直接储存电荷/电场能量，电容电压不能突变。")
    if "二极管" in msg or "led" in msg:
        lines.append("已知事实：LED/二极管具有方向性；LED 需要限流，不能直接并到电源两端。")
    return lines[:2]


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

        if request.mode in ("concept_tutor", "lab_guidance"):
            return self._run_concept_or_guidance_job(
                job_id=job_id,
                request=request,
                classroom=classroom,
                created_at=created_at,
                forced_intent=request.mode,  # type: ignore[arg-type]
            )

        if request.mode == "agent_auto":
            evidence_contract = build_runtime_evidence_from_classroom(
                station_id=request.station_id,
                stations={request.station_id: self._station_payload(classroom, request)},
                error_tag_service=ErrorTagService(),
            )
            intent = classify_intent(
                request.user_message or request.query,
                evidence=evidence_contract,
            )
            if intent == "diagnostic":
                return self._run_diagnostic_agent_job(
                    job_id=job_id,
                    request=request,
                    classroom=classroom,
                    created_at=created_at,
                    intent=intent,
                )
            if intent == "mixed":
                # Mixed intent should prefer concept-guidance path so
                # concept_not_found can trigger local LLM fallback answer.
                return self._run_concept_or_guidance_job(
                    job_id=job_id,
                    request=request,
                    classroom=classroom,
                    created_at=created_at,
                    forced_intent="concept_tutor",
                    pre_evidence=evidence_contract,
                )
            return self._run_concept_or_guidance_job(
                job_id=job_id,
                request=request,
                classroom=classroom,
                created_at=created_at,
                forced_intent=intent,
                pre_evidence=evidence_contract,
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
            actual_llm_provider=self._actual_llm_usage()[0],
            actual_llm_model=self._actual_llm_usage()[1],
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
        intent: AgentIntent = "diagnostic",
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
        question = request.user_message or request.query
        rewritten_answer, actual_provider, actual_model = self._llm_rewrite_diagnostic_answer(
            question=question,
            draft_answer=graph_state.final_answer,
            evidence=evidence_contract,
            context_pack=context_pack,
            tool_results=tool_results,
        )

        result = AngntJobResult(
            job_id=job_id,
            station_id=request.station_id,
            mode=request.mode,
            answer=rewritten_answer,
            actual_llm_provider=actual_provider,
            actual_llm_model=actual_model,
            follow_up_suggestions=_build_follow_up_suggestions(
                evidence=evidence_contract,
                context_pack=context_pack,
            ),
            citations=build_diagnostic_citations(
                evidence=evidence_contract,
                context_pack=context_pack,
                tool_results=tool_results,
            ),
            evidence=self._augment_diagnostic_evidence_with_intent(
                evidence_items=build_diagnostic_evidence(
                    evidence=evidence_contract,
                    context_pack=context_pack,
                    tool_results=tool_results,
                    verification_passed=verification_passed,
                    verification_issues=verification_issues,
                    graph_metrics=[
                        metric.model_dump() for metric in graph_state.graph_metrics
                    ],
                    react_trace=[step.model_dump() for step in graph_state.react_trace],
                    react_iterations=graph_state.react_iterations,
                    react_terminate_reason=graph_state.react_terminate_reason,
                ),
                intent=intent,
                question=request.user_message or request.query,
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

    def _station_payload(
        self,
        classroom: ClassroomState,
        request: AngntAskRequest,
    ) -> dict[str, Any]:
        stations = classroom.get_all_stations()
        station_data = stations.get(request.station_id, {})
        if not station_data and request.diagnosis_context:
            station_data = dict(request.diagnosis_context)
            station_data["station_id"] = request.station_id
        if not station_data:
            station_data = {"station_id": request.station_id}
        return station_data

    def _augment_diagnostic_evidence_with_intent(
        self,
        *,
        evidence_items: list[AngntEvidence],
        intent: AgentIntent,
        question: str,
    ) -> list[AngntEvidence]:
        """Attach intent + (optional) concept_pack to diagnostic evidence.

        Only used for `mode="agent_auto"` mixed routing — diagnostic_agent
        callers see the original list (intent defaults to "diagnostic" and we
        skip the concept attachment).
        """
        if intent == "diagnostic":
            return evidence_items
        augmented = list(evidence_items)
        augmented.append(
            AngntEvidence(
                evidence_type="intent",
                source_id="agent_auto:intent",
                summary=f"intent={intent}",
                payload={"intent": intent},
            )
        )
        if intent == "mixed":
            concept = lookup_concept(question)
            if concept is not None:
                augmented.append(
                    AngntEvidence(
                        evidence_type="concept_pack",
                        source_id=concept.concept_id,
                        summary=concept.title,
                        payload=concept.model_dump(),
                    )
                )
        return augmented

    def _run_concept_or_guidance_job(
        self,
        *,
        job_id: str,
        request: AngntAskRequest,
        classroom: ClassroomState,
        created_at: float,
        forced_intent: AgentIntent,
        pre_evidence: Any = None,
    ) -> AngntJobResult:
        """Deterministic non-LangGraph path for concept_tutor / lab_guidance."""
        station_data = self._station_payload(classroom, request)
        evidence_contract = pre_evidence or build_runtime_evidence_from_classroom(
            station_id=request.station_id,
            stations={request.station_id: station_data},
            error_tag_service=ErrorTagService(),
        )

        question = request.user_message or request.query
        tool_result = teaching_concept_lookup_tool(
            TeachingConceptLookupInput(query=question)
        )
        tool_results: list[ToolResult] = [tool_result]
        concept: ConceptPack | None = None
        if tool_result.status == "ok":
            from app.agent.contracts import ConceptPack as _ConceptPack

            concept_payload = tool_result.payload.get("concept") or {}
            concept = _ConceptPack.model_validate(concept_payload)

        actual_provider, actual_model = self._actual_llm_usage()
        if forced_intent == "lab_guidance":
            draft = build_lab_guidance_answer(
                question=question,
                concept=concept,
                evidence=evidence_contract,
            )
        else:
            if concept is None:
                draft, actual_provider, actual_model = self._llm_concept_not_found_answer(
                    question=question,
                    evidence=evidence_contract,
                )
                if actual_provider == "ollama":
                    tool_results.append(
                        ToolResult(
                            tool_name="ollama_concept_fallback",
                            status="ok",
                            summary="concept_not_found -> ollama 直答兜底",
                            payload={"provider": "ollama", "model": actual_model},
                        )
                    )
            else:
                draft = build_concept_answer(
                    question=question,
                    concept=concept,
                    evidence=evidence_contract,
                )
                actual_provider, actual_model = self._actual_llm_usage()

        # concept/lab paths do not produce a ContextPack; build a minimal stub
        # for the verifier signature compatibility.
        from app.agent.contracts import ContextPack

        stub_pack = ContextPack(
            pack_id=f"{request.station_id}:concept_stub",
            error_family="unknown",
            risk_level=evidence_contract.risk_level,
        )
        verification = verify_draft_answer(
            evidence=evidence_contract,
            context_pack=stub_pack,
            draft_answer=draft,
            intent=forced_intent,
            concept=concept,
        )
        verification_passed = verification.passed
        verification_issues = verification.issues

        actions = self._build_actions(
            risk_level=evidence_contract.risk_level,
            diagnostics=evidence_contract.diagnostics,
        )

        return AngntJobResult(
            job_id=job_id,
            station_id=request.station_id,
            mode=request.mode,
            answer=draft,
            actual_llm_provider=actual_provider,
            actual_llm_model=actual_model,
            follow_up_suggestions=[],
            citations=build_concept_citations(
                station_id=request.station_id,
                concept=concept,
                tool_results=tool_results,
            ),
            evidence=build_concept_evidence(
                station_id=request.station_id,
                intent=forced_intent,
                concept=concept,
                tool_results=tool_results,
                verification_passed=verification_passed,
                verification_issues=verification_issues,
                evidence=evidence_contract,
            ),
            actions=actions,
            used_retrieval=bool(concept is not None),
            created_at=created_at,
            debug=AngntChatDebug(
                job_id=job_id,
                used_context_refs=[concept.concept_id] if concept else [],
            ),
        )

    def _actual_llm_usage(self) -> tuple[str, str]:
        provider = get_llm_provider()
        provider_name = str(getattr(provider, "name", "template") or "template")
        if provider_name == "ollama":
            model_name = str(getattr(provider, "_model", "") or "")
        elif provider_name == "template":
            model_name = "template"
        else:
            model_name = str(getattr(provider, "_model", "") or getattr(provider, "_model_dir", "") or provider_name)
        return provider_name, model_name

    def _llm_concept_not_found_answer(
        self,
        *,
        question: str,
        evidence,
    ) -> tuple[str, str, str]:
        provider_name, model_name = self._actual_llm_usage()
        if provider_name != "ollama":
            return build_concept_answer(question=question, concept=None, evidence=evidence), "template", "template"

        endpoint = f"{getattr(settings, 'AGENT_LLM_OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')}/api/chat"
        timeout_s = float(getattr(settings, "AGENT_LLM_OLLAMA_TIMEOUT_S", 120.0) or 120.0)
        prompt = _build_concept_not_found_prompt(question=question, evidence=evidence)
        payload = {
            "model": model_name or getattr(settings, "AGENT_LLM_OLLAMA_MODEL", "qwen3:4b"),
            "stream": False,
            "keep_alive": "30m",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是严谨的电路实验助教。优先回答电路、电子元件、实验测量和安全相关问题；"
                        "遇到无关问题时，简短回应后把用户引导回电路实验。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0.2,
                "num_predict": 260,
            },
        }
        try:
            body: dict[str, Any] = {}
            for timeout in (min(max(timeout_s, 20.0), 60.0), min(max(timeout_s * 1.5, 45.0), 120.0)):
                try:
                    with httpx.Client(timeout=timeout, trust_env=False) as client:
                        response = client.post(endpoint, json=payload)
                        response.raise_for_status()
                        body = response.json()
                    break
                except Exception:
                    body = {}
                    continue
            text = str(((body or {}).get("message") or {}).get("content") or "").strip()
            if not text:
                raise RuntimeError("empty llm response")
            has_follow_up_line = (
                "引导追问：" in text
                or "追问" in text
                or "疑问" in text
                or "提问" in text
                or "随时问" in text
                or "随时提问" in text
                or "可以继续问" in text
                or "你可以" in text
                or "你觉得" in text
                or "能否" in text
                or "有什么" in text
                or "是否" in text
                or "3)" in text
                or "3）" in text
                or "三、" in text
            )
            has_safety_line = (
                "安全提醒：" in text
                or "4)" in text
                or "4）" in text
                or "四、" in text
                or any(token in text for token in ("断电", "触电", "安全"))
            )
            if not has_follow_up_line:
                text += "\n" + _fallback_follow_up_text(question=question, evidence=evidence)
            if not has_safety_line:
                text += "\n安全提醒：上电或改线前请先断电，并优先复查电源轨与短路风险。"
            text += f"\n知识来源：llm_fallback({payload['model']})"
            return text, "ollama", str(payload["model"])
        except Exception:
            # Keep deterministic fallback when local model is unavailable.
            return build_concept_answer(question=question, concept=None, evidence=evidence), "template", "template"

    def _llm_rewrite_diagnostic_answer(
        self,
        *,
        question: str,
        draft_answer: str,
        evidence,
        context_pack,
        tool_results: list[ToolResult],
    ) -> tuple[str, str, str]:
        """Rewrite rigid diagnostic template answer with LLM while preserving evidence facts."""
        provider_name, model_name = self._actual_llm_usage()
        if provider_name != "ollama":
            return draft_answer, provider_name, model_name

        endpoint = f"{getattr(settings, 'AGENT_LLM_OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')}/api/chat"
        timeout_s = float(getattr(settings, "AGENT_LLM_OLLAMA_TIMEOUT_S", 120.0) or 120.0)
        findings = list(getattr(evidence, "findings", []) or [])
        evidence_refs = list(getattr(evidence, "evidence_refs", []) or [])
        error_codes = list(getattr(evidence, "error_codes", []) or [])
        findings_text = "；".join(
            f"{item.error_code}:{item.component_id or '-'}:{item.pin_name or '-'}"
            for item in findings[:4]
        ) or "无"
        refs_text = "；".join(
            f"{ref.ref_id}:{ref.component_id or '-'}:{ref.pin_name or '-'}:{ref.hole_id or '-'}"
            for ref in evidence_refs[:4]
        ) or "无"
        tool_text = "；".join(
            f"{item.tool_name}:{item.summary}"
            for item in tool_results[:4]
        ) or "无"
        prompt = "\n".join(
            [
                "你是电路实验故障诊断助教，请对现有答案做自然中文改写。",
                "关键要求：只能重写表达，不能新增任何事实、孔位、器件、测量值。",
                "第一句必须直接围绕当前电路的错误码、参考电路和涉及元件展开，不要先寒暄或泛泛说明。",
                "如果原始答案包含历史对比、上一轮、仍然存在、有所改善或错误码变化，必须保留这些判断。",
                "必须保留并原样包含：至少 1 个错误码；至少 1 个 evidence_ref 的 component_id/pin_name/hole_id（如果给出）。",
                f"用户问题：{question}",
                f"原始答案：{draft_answer}",
                f"风险等级：{getattr(evidence, 'risk_level', 'unknown')}",
                f"错误码：{','.join(error_codes[:4]) if error_codes else '无'}",
                f"finding 摘要：{findings_text}",
                f"evidence_ref：{refs_text}",
                f"工具观察：{tool_text}",
                f"context_pack.error_family={getattr(context_pack, 'error_family', 'unknown')}",
                "输出要求：4-7句中文；保留排查步骤与安全提醒；最后一句给出下一步检查建议。",
            ]
        )
        payload = {
            "model": model_name or getattr(settings, "AGENT_LLM_OLLAMA_MODEL", "qwen3:4b"),
            "stream": False,
            "keep_alive": "30m",
            "messages": [
                {"role": "system", "content": "你是严谨的电路故障诊断助教，严禁编造证据。"},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0.2,
                "num_predict": 220,
            },
        }
        try:
            body: dict[str, Any] = {}
            for timeout in (min(max(timeout_s, 20.0), 60.0), min(max(timeout_s * 1.5, 45.0), 120.0)):
                try:
                    with httpx.Client(timeout=timeout, trust_env=False) as client:
                        response = client.post(endpoint, json=payload)
                        response.raise_for_status()
                        body = response.json()
                    break
                except Exception:
                    body = {}
                    continue
            text = str(((body or {}).get("message") or {}).get("content") or "").strip()
            if not text:
                raise RuntimeError("empty llm response")
            text = ensure_circuit_opening(text, evidence)
            history_summary = str(getattr(context_pack, "history_summary", "") or "").strip()
            if history_summary and history_summary not in text:
                text = f"{text}\n历史对比：{history_summary}。"
            if error_codes and not any(code in text for code in error_codes):
                text = f"{text}\n补充定位：本轮关键错误码为 {','.join(error_codes[:3])}。"
            if getattr(evidence, "risk_level", "unknown") in {"danger", "warning"} and not any(
                token in text for token in ("断电", "安全", "电源")
            ):
                text = f"{text}\n安全提醒：调整接线前请先断电并复查电源轨是否短接。"
            if getattr(evidence, "risk_level", "unknown") == "danger" and "断电" not in text:
                text = f"{text}\n安全提醒：当前风险较高，请先断电再调整接线。"

            report = verify_draft_answer(
                evidence=evidence,
                context_pack=context_pack,
                draft_answer=text,
                intent="diagnostic",
            )
            if not report.passed:
                return draft_answer, "template", "template"
            return text, provider_name, str(payload["model"])
        except Exception:
            return draft_answer, "template", "template"

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
