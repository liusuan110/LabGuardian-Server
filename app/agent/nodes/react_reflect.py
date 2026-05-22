"""react_reflect node: critiques the current draft and decides loop control.

Builds (or refreshes) the draft via `build_diagnostic_template_answer`, then
asks the active provider to reflect. Three outcomes:

- reflection.passed = True               → terminate, route to verify_answer
- reflection.passed = False, room left   → loop back to react_plan
- iteration cap reached                  → terminate, route to verify_answer
"""

from __future__ import annotations

from time import perf_counter

from app.agent.answering import (
    build_concept_answer,
    build_diagnostic_template_answer,
    build_lab_guidance_answer,
)
from app.agent.contracts import ConceptPack, DiagnosticState, ReActStep
from app.agent.llm import ReflectRequest, get_llm_provider
from app.agent.nodes._metrics import append_metric, require_context_pack
from app.agent.tools import ToolResult


def react_reflect_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    context_pack = require_context_pack(state)
    iteration = state.react_iterations
    max_iterations = max(1, state.max_react_iterations)

    tool_results = [ToolResult.model_validate(item) for item in state.tool_results]
    # Intent-aware draft selection. We pick the right answer template based
    # on the resolved intent. For concept_tutor / lab_guidance we first
    # check whether the planner already retrieved a ConceptPack via the
    # teaching_concept_lookup_tool — this keeps Phase 4 ReAct + Phase 7
    # concept tutoring on the same code path.
    concept = state.concept or _extract_concept_from_tools(tool_results)
    intent = state.intent
    if intent == "concept_tutor":
        draft = build_concept_answer(
            question=state.user_message or state.query,
            concept=concept,
            evidence=state.runtime_evidence,
        )
    elif intent == "lab_guidance":
        draft = build_lab_guidance_answer(
            question=state.user_message or state.query,
            concept=concept,
            evidence=state.runtime_evidence,
        )
    else:  # diagnostic / mixed → existing diagnostic template
        draft = build_diagnostic_template_answer(
            station_id=state.runtime_evidence.station_id,
            query=state.query,
            user_message=state.user_message,
            evidence=state.runtime_evidence,
            context_pack=context_pack,
            tool_results=tool_results,
        )

    provider = get_llm_provider()
    reflection = provider.reflect(
        ReflectRequest(
            iteration=iteration,
            evidence=state.runtime_evidence,
            context_pack=context_pack,
            draft_answer=draft,
            verification_report=state.verification_report,
            prior_steps=list(state.react_trace),
        )
    )

    next_iteration = iteration + 1
    cap_reached = next_iteration >= max_iterations
    # Termination policy:
    # - Hard cap always wins.
    # - Otherwise terminate when the planner has nothing more to do
    #   (i.e., it returned tool_call=None this iteration).
    # - reflection.passed is recorded but does NOT short-circuit the loop:
    #   we want to exhaust the planner's tool plan so context_pack tools
    #   that follow root-cause traces (e.g., fault_case_lookup,
    #   safety_rule_lookup) still get a chance to enrich the answer.
    last_step = state.react_trace[-1] if state.react_trace else None
    no_more_tools = last_step is not None and (
        (last_step.tool_call if isinstance(last_step, ReActStep) else last_step.get("tool_call")) is None
    )
    terminate = bool(cap_reached or no_more_tools)
    if cap_reached:
        terminate_reason = "max_iterations_reached"
    elif no_more_tools:
        terminate_reason = (
            "verifier_passed_no_more_tools" if reflection.passed
            else "no_more_tools"
        )
    else:
        terminate_reason = ""

    # Stamp the reflection back onto the latest plan/observe step.
    react_trace = [
        (item.model_dump() if isinstance(item, ReActStep) else dict(item))
        for item in state.react_trace
    ]
    if react_trace:
        last = ReActStep.model_validate(react_trace[-1])
        last.reflection = reflection.reason or reflection.next_hint
        last.terminate = terminate
        react_trace[-1] = last.model_dump()

    update: dict = {
        "draft_answer": draft,
        "react_trace": react_trace,
        "react_iterations": next_iteration,
        "react_terminate_reason": terminate_reason,
        "graph_metrics": append_metric(
            state,
            node_name=f"react_reflect_{iteration}",
            started_at=started_at,
            payload={
                "iteration": iteration,
                "intent": intent,
                "passed": reflection.passed,
                "terminate": terminate,
                "terminate_reason": terminate_reason,
                "draft_length": len(draft),
            },
        ),
    }
    # Stash the concept on the state so the verifier (which needs
    # ConceptPack to apply concept_tutor rules) can pick it up downstream
    # without re-scanning tool_results.
    if concept is not None and state.concept is None:
        update["concept"] = concept.model_dump()
    return update


def _extract_concept_from_tools(
    tool_results: list[ToolResult],
) -> ConceptPack | None:
    """Pick the most recent successful teaching_concept_lookup_tool hit.

    The planner can call the tool multiple times (e.g. a paraphrase
    retry); we want the last successful result so reflection sees the
    freshest knowledge. Returns ``None`` when no hit is present so the
    deterministic concept template falls back to its "concept_not_found"
    branch.
    """
    for result in reversed(tool_results):
        if result.tool_name != "teaching_concept_lookup_tool":
            continue
        if result.status != "ok":
            continue
        payload = result.payload or {}
        concept_dict = payload.get("concept") if isinstance(payload, dict) else None
        if not isinstance(concept_dict, dict):
            continue
        try:
            return ConceptPack.model_validate(concept_dict)
        except Exception:
            continue
    return None


def should_continue_react(state: DiagnosticState) -> str:
    """Conditional edge: 'continue' to plan again, or 'finalize' to verify."""
    if state.react_iterations >= max(1, state.max_react_iterations):
        return "finalize"
    if state.react_terminate_reason:
        return "finalize"
    return "continue"
