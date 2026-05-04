from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Any

try:  # LangGraph is an optional orchestration shell for the deterministic flow.
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - exercised in environments without langgraph
    END = START = StateGraph = None  # type: ignore[assignment]

from app.agent.answering import (
    build_diagnostic_template_answer,
    repair_diagnostic_answer,
)
from app.agent.context_pack import build_context_pack, classify_error_family
from app.agent.contracts import DiagnosticState, RuntimeEvidence
from app.agent.tool_runner import run_diagnostic_tools
from app.agent.tools import ToolResult
from app.agent.verification import verify_draft_answer


def run_diagnostic_graph(
    *,
    evidence: RuntimeEvidence,
    query: str = "",
    user_message: str = "",
    chat_history: list[dict[str, str]] | None = None,
    top_k: int = 5,
) -> DiagnosticState:
    """Run the deterministic PCM diagnostic state machine.

    This is the first LangGraph shell: no LLM is called, and every node is
    deterministic. Later phases can replace `generate_draft` with an LLM node
    while preserving the same state contract.
    """

    initial = DiagnosticState(
        query=query,
        user_message=user_message or query,
        chat_history=chat_history or [],
        top_k=top_k,
        runtime_evidence=evidence,
    )
    if StateGraph is None:
        result = _run_sequential_graph(initial).model_dump()
    else:
        result = _compiled_graph().invoke(initial.model_dump())
    return DiagnosticState.model_validate(result)


@lru_cache(maxsize=1)
def _compiled_graph():
    if StateGraph is None:
        raise RuntimeError("langgraph is not installed")
    graph = StateGraph(DiagnosticState)
    graph.add_node("classify_error", _classify_error_node)
    graph.add_node("build_context_pack", _build_context_pack_node)
    graph.add_node("run_tools", _run_tools_node)
    graph.add_node("generate_draft", _generate_draft_node)
    graph.add_node("verify_answer", _verify_answer_node)
    graph.add_node("repair_answer", _repair_answer_node)
    graph.add_node("finalize_answer", _finalize_answer_node)

    graph.add_edge(START, "classify_error")
    graph.add_edge("classify_error", "build_context_pack")
    graph.add_edge("build_context_pack", "run_tools")
    graph.add_edge("run_tools", "generate_draft")
    graph.add_edge("generate_draft", "verify_answer")
    graph.add_conditional_edges(
        "verify_answer",
        _route_after_verification,
        {
            "finalize_answer": "finalize_answer",
            "repair_answer": "repair_answer",
        },
    )
    graph.add_edge("repair_answer", "finalize_answer")
    graph.add_edge("finalize_answer", END)
    return graph.compile()


def _run_sequential_graph(initial: DiagnosticState) -> DiagnosticState:
    state = initial
    for node in (
        _classify_error_node,
        _build_context_pack_node,
        _run_tools_node,
        _generate_draft_node,
        _verify_answer_node,
    ):
        state = _apply_node_update(state, node(state))

    if state.verification_report and state.verification_report.passed:
        state = _apply_node_update(state, _finalize_answer_node(state))
    else:
        state = _apply_node_update(state, _repair_answer_node(state))
        state = _apply_node_update(state, _finalize_answer_node(state))
    return state


def _apply_node_update(state: DiagnosticState, update: dict[str, Any]) -> DiagnosticState:
    payload = state.model_dump()
    payload.update(update)
    return DiagnosticState.model_validate(payload)


def _classify_error_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    error_family = classify_error_family(state.runtime_evidence)
    return {
        "error_family": error_family,
        "graph_metrics": _append_metric(
            state,
            node_name="classify_error",
            started_at=started_at,
            payload={"error_family": error_family},
        ),
    }


def _build_context_pack_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    pack = build_context_pack(
        state.runtime_evidence,
        query=state.query,
        user_message=state.user_message,
    )
    return {
        "context_pack": pack.model_dump(),
        "error_family": pack.error_family,
        "graph_metrics": _append_metric(
            state,
            node_name="build_context_pack",
            started_at=started_at,
            payload={
                "pack_id": pack.pack_id,
                "pushed_facts_count": len(pack.pushed_facts),
                "allowed_tool_count": len(pack.allowed_tools),
                "context_char_count": pack.metrics.char_count if pack.metrics else 0,
                "estimated_tokens": pack.metrics.estimated_tokens if pack.metrics else 0,
                "history_facts_count": (
                    pack.metrics.history_facts_count if pack.metrics else 0
                ),
                "history_estimated_tokens": (
                    pack.metrics.history_estimated_tokens if pack.metrics else 0
                ),
            },
        ),
    }


def _run_tools_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    context_pack = _require_context_pack(state)
    results = run_diagnostic_tools(
        evidence=state.runtime_evidence,
        context_pack=context_pack,
        query=state.query,
        top_k=state.top_k,
    )
    return {
        "tool_results": [result.model_dump() for result in results],
        "graph_metrics": _append_metric(
            state,
            node_name="run_tools",
            started_at=started_at,
            payload={
                "tool_count": len(results),
                "tool_names": [result.tool_name for result in results],
            },
        ),
    }


def _generate_draft_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    context_pack = _require_context_pack(state)
    tool_results = _tool_results_from_state(state)
    answer = build_diagnostic_template_answer(
        station_id=state.runtime_evidence.station_id,
        query=state.query,
        user_message=state.user_message,
        evidence=state.runtime_evidence,
        context_pack=context_pack,
        tool_results=tool_results,
    )
    return {
        "draft_answer": answer,
        "graph_metrics": _append_metric(
            state,
            node_name="generate_draft",
            started_at=started_at,
            payload={"draft_length": len(answer)},
        ),
    }


def _verify_answer_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    report = verify_draft_answer(
        evidence=state.runtime_evidence,
        context_pack=_require_context_pack(state),
        draft_answer=state.draft_answer,
    )
    return {
        "verification_report": report.model_dump(),
        "graph_metrics": _append_metric(
            state,
            node_name="verify_answer",
            started_at=started_at,
            payload={"passed": report.passed, "issue_count": len(report.issues)},
        ),
    }


def _repair_answer_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    issues = state.verification_report.issues if state.verification_report else []
    repaired = repair_diagnostic_answer(
        draft_answer=state.draft_answer,
        evidence=state.runtime_evidence,
        verification_issues=issues,
    )
    report = verify_draft_answer(
        evidence=state.runtime_evidence,
        context_pack=_require_context_pack(state),
        draft_answer=repaired,
    )
    return {
        "draft_answer": repaired,
        "verification_report": report.model_dump(),
        "graph_metrics": _append_metric(
            state,
            node_name="repair_answer",
            started_at=started_at,
            payload={
                "passed_after_repair": report.passed,
                "issue_count": len(report.issues),
                "draft_length": len(repaired),
            },
        ),
    }


def _finalize_answer_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    return {
        "final_answer": state.draft_answer,
        "graph_metrics": _append_metric(
            state,
            node_name="finalize_answer",
            started_at=started_at,
            payload={"final_answer_length": len(state.draft_answer)},
        ),
    }


def _route_after_verification(state: DiagnosticState) -> str:
    if state.verification_report and state.verification_report.passed:
        return "finalize_answer"
    return "repair_answer"


def _require_context_pack(state: DiagnosticState):
    if state.context_pack is None:
        raise RuntimeError("context_pack is required before this node")
    return state.context_pack


def _tool_results_from_state(state: DiagnosticState) -> list[ToolResult]:
    return [ToolResult.model_validate(item) for item in state.tool_results]


def _append_metric(
    state: DiagnosticState,
    *,
    node_name: str,
    started_at: float,
    payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    metrics = [metric.model_dump() for metric in state.graph_metrics]
    metrics.append(
        {
            "node_name": node_name,
            "duration_ms": round((perf_counter() - started_at) * 1000, 3),
            "status": "ok",
            "payload": payload or {},
        }
    )
    return metrics
