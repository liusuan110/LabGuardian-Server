"""Diagnostic LangGraph orchestrator with ReAct + Self-Reflection (Phase 4).

Topology:

  classify_error
    ↓
  build_context_pack
    ↓
  react_plan ──→ react_observe ──→ react_reflect ─┬─(continue)→ react_plan
                                                    └─(finalize)→ verify_answer
                                                                      │
                                          ┌─ passed ──→ finalize_answer
                                          ├─ failed ──→ repair_answer ──→ finalize_answer
                                          (Phase 6 will add a third branch:
                                           needs_micro_inspection → vlm_explain)

The ReAct sub-loop replaces the previous one-shot `generate_draft` node so we
can attribute every tool call and reflection step in `DiagnosticState
.react_trace`. The non-LangGraph sequential fallback walks the same nodes in
order so deployments without `langgraph` installed still work.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

try:  # LangGraph is an optional orchestration shell for the deterministic flow.
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - exercised in environments without langgraph
    END = START = StateGraph = None  # type: ignore[assignment]

from app.agent.contracts import DiagnosticState, RuntimeEvidence
from app.agent.nodes import (
    build_context_pack_node,
    classify_error_node,
    finalize_answer_node,
    react_observe_node,
    react_plan_node,
    react_reflect_node,
    repair_answer_node,
    should_continue_react,
    verify_answer_node,
    vlm_explain_node,
)
from app.core.config import settings


def run_diagnostic_graph(
    *,
    evidence: RuntimeEvidence,
    query: str = "",
    user_message: str = "",
    chat_history: list[dict[str, str]] | None = None,
    top_k: int = 5,
    max_react_iterations: int | None = None,
) -> DiagnosticState:
    """Run the deterministic PCM diagnostic state machine with ReAct loop."""
    cap = (
        max_react_iterations
        if max_react_iterations is not None
        else getattr(settings, "REACT_MAX_ITERATIONS", 4)
    )

    initial = DiagnosticState(
        query=query,
        user_message=user_message or query,
        chat_history=chat_history or [],
        top_k=top_k,
        runtime_evidence=evidence,
        max_react_iterations=max(1, int(cap)),
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
    graph.add_node("classify_error", classify_error_node)
    graph.add_node("build_context_pack", build_context_pack_node)
    graph.add_node("react_plan", react_plan_node)
    graph.add_node("react_observe", react_observe_node)
    graph.add_node("react_reflect", react_reflect_node)
    graph.add_node("verify_answer", verify_answer_node)
    graph.add_node("repair_answer", repair_answer_node)
    graph.add_node("vlm_explain", vlm_explain_node)
    graph.add_node("finalize_answer", finalize_answer_node)

    graph.add_edge(START, "classify_error")
    graph.add_edge("classify_error", "build_context_pack")
    graph.add_edge("build_context_pack", "react_plan")
    graph.add_edge("react_plan", "react_observe")
    graph.add_edge("react_observe", "react_reflect")
    graph.add_conditional_edges(
        "react_reflect",
        should_continue_react,
        {
            "continue": "react_plan",
            "finalize": "verify_answer",
        },
    )
    graph.add_conditional_edges(
        "verify_answer",
        _route_after_verification,
        {
            "finalize_answer": "finalize_answer",
            "repair_answer": "repair_answer",
            "vlm_explain": "vlm_explain",
        },
    )
    graph.add_edge("repair_answer", "finalize_answer")
    graph.add_edge("vlm_explain", "finalize_answer")
    graph.add_edge("finalize_answer", END)
    return graph.compile()


def _run_sequential_graph(initial: DiagnosticState) -> DiagnosticState:
    """Walk the graph nodes in order without LangGraph (CI / minimal deploys)."""
    state = initial
    state = _apply_node_update(state, classify_error_node(state))
    state = _apply_node_update(state, build_context_pack_node(state))

    # ReAct loop with hard cap
    max_iters = max(1, state.max_react_iterations)
    for _ in range(max_iters):
        state = _apply_node_update(state, react_plan_node(state))
        state = _apply_node_update(state, react_observe_node(state))
        state = _apply_node_update(state, react_reflect_node(state))
        if should_continue_react(state) == "finalize":
            break

    state = _apply_node_update(state, verify_answer_node(state))
    route = _route_after_verification(state)
    if route == "vlm_explain":
        state = _apply_node_update(state, vlm_explain_node(state))
    elif route == "repair_answer":
        state = _apply_node_update(state, repair_answer_node(state))
    state = _apply_node_update(state, finalize_answer_node(state))
    return state


def _apply_node_update(state: DiagnosticState, update: dict[str, Any]) -> DiagnosticState:
    payload = state.model_dump()
    payload.update(update)
    return DiagnosticState.model_validate(payload)


def _route_after_verification(state: DiagnosticState) -> str:
    """Three-way route after verification:

    - failed → repair_answer (always wins; we never expose a broken draft)
    - passed + needs_micro_inspection → vlm_explain (Phase 6 white-box gate)
    - passed only → finalize_answer
    """
    report = state.verification_report
    if report is None or not report.passed:
        return "repair_answer"
    if report.needs_micro_inspection:
        return "vlm_explain"
    return "finalize_answer"
