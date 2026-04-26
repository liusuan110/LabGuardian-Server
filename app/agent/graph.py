from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

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
    top_k: int = 5,
) -> DiagnosticState:
    """Run the deterministic PCM diagnostic state machine.

    This is the first LangGraph shell: no LLM is called, and every node is
    deterministic. Later phases can replace `generate_draft` with an LLM node
    while preserving the same state contract.
    """

    initial = DiagnosticState(
        query=query,
        top_k=top_k,
        runtime_evidence=evidence,
    )
    result = _compiled_graph().invoke(initial.model_dump())
    return DiagnosticState.model_validate(result)


@lru_cache(maxsize=1)
def _compiled_graph():
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


def _classify_error_node(state: DiagnosticState) -> dict:
    return {"error_family": classify_error_family(state.runtime_evidence)}


def _build_context_pack_node(state: DiagnosticState) -> dict:
    pack = build_context_pack(state.runtime_evidence, query=state.query)
    return {
        "context_pack": pack.model_dump(),
        "error_family": pack.error_family,
    }


def _run_tools_node(state: DiagnosticState) -> dict:
    context_pack = _require_context_pack(state)
    results = run_diagnostic_tools(
        evidence=state.runtime_evidence,
        context_pack=context_pack,
        query=state.query,
        top_k=state.top_k,
    )
    return {"tool_results": [result.model_dump() for result in results]}


def _generate_draft_node(state: DiagnosticState) -> dict:
    context_pack = _require_context_pack(state)
    tool_results = _tool_results_from_state(state)
    answer = build_diagnostic_template_answer(
        station_id=state.runtime_evidence.station_id,
        query=state.query,
        evidence=state.runtime_evidence,
        context_pack=context_pack,
        tool_results=tool_results,
    )
    return {"draft_answer": answer}


def _verify_answer_node(state: DiagnosticState) -> dict:
    report = verify_draft_answer(
        evidence=state.runtime_evidence,
        context_pack=_require_context_pack(state),
        draft_answer=state.draft_answer,
    )
    return {"verification_report": report.model_dump()}


def _repair_answer_node(state: DiagnosticState) -> dict:
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
    }


def _finalize_answer_node(state: DiagnosticState) -> dict:
    return {"final_answer": state.draft_answer}


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
