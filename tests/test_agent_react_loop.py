"""Phase 4 ReAct loop unit tests.

Covers:
- Hard iteration cap forces termination and tags reason correctly
- Planner respects ContextPack.allowed_tools whitelist (anti-hallucination)
- React_trace records every iteration with tool_call, observation, reflection
- Sequential fallback path (no langgraph) walks the same loop
- TemplateLLMProvider deterministic plan/reflect contract
"""

from __future__ import annotations

import pytest

from app.agent.contracts import (
    AllowedTool,
    ContextPack,
    DiagnosticState,
    EvidenceRef,
    ReActStep,
    ReflectionResult,
    RuntimeEvidence,
    ToolCall,
)
from app.agent.evidence import build_runtime_evidence_from_station
from app.agent.graph import run_diagnostic_graph
from app.agent.llm import PlanRequest, ReflectRequest, clear_llm_provider_cache, get_llm_provider
from app.agent.llm.template_provider import TemplateLLMProvider
from app.agent.nodes.react_plan import react_plan_node
from app.agent.nodes.react_reflect import should_continue_react
from app.services.error_tag_service import ErrorTagService


def _danger_evidence() -> RuntimeEvidence:
    return build_runtime_evidence_from_station(
        station_id="ST_REACT",
        station={
            "risk_level": "danger",
            "diagnostics": ["R1 短路"],
            "risk_reasons": ["R1 两端在同一节点"],
            "comparison_report": {
                "items": [
                    {
                        "error_code": "COMPONENT_SHORTED_SAME_NET",
                        "severity": "danger",
                        "component_id": "R1",
                    }
                ]
            },
            "netlist_v2": {
                "components": [{"component_id": "R1", "pins": []}],
                "nets": [{"net_id": "N1", "members": ["R1.pin1", "R1.pin2"]}],
            },
        },
        error_tag_service=ErrorTagService(),
    )


def _make_state_with_pack(allowed_tool_names: list[str]) -> DiagnosticState:
    evidence = _danger_evidence()
    pack = ContextPack(
        pack_id="pcm_test",
        error_family="short_circuit",
        risk_level="danger",
        pushed_facts=["test"],
        allowed_tools=[
            AllowedTool(name=name, reason="test", required=False)
            for name in allowed_tool_names
        ],
        evidence_refs=[],
    )
    return DiagnosticState(
        runtime_evidence=evidence,
        context_pack=pack,
        error_family="short_circuit",
    )


# -- Iteration cap & termination --------------------------------------------------

def test_react_loop_terminates_at_iteration_cap() -> None:
    evidence = _danger_evidence()
    state = run_diagnostic_graph(
        evidence=evidence,
        query="为什么危险",
        top_k=5,
        max_react_iterations=2,
    )
    assert state.react_iterations <= 2
    assert state.react_terminate_reason in {
        "max_iterations_reached",
        "no_more_tools",
        "verifier_passed_no_more_tools",
    }


def test_react_loop_terminates_when_no_more_tools() -> None:
    """When the ContextPack only allows a single tool, ReAct must terminate
    well before the cap (after at most 2 iterations: 1 tool call + 1 no-op).
    """
    from app.agent.nodes._metrics import append_metric  # noqa: F401 — sanity import
    from app.agent.nodes import (
        react_observe_node,
        react_plan_node,
        react_reflect_node,
    )

    state = _make_state_with_pack(["netlist_trace_tool"]).model_copy(
        update={"max_react_iterations": 4}
    )

    def _apply(s, update):
        payload = s.model_dump()
        payload.update(update)
        return DiagnosticState.model_validate(payload)

    for _ in range(state.max_react_iterations):
        state = _apply(state, react_plan_node(state))
        state = _apply(state, react_observe_node(state))
        state = _apply(state, react_reflect_node(state))
        if should_continue_react(state) == "finalize":
            break

    assert state.react_iterations < 4
    assert state.react_terminate_reason in {
        "no_more_tools",
        "verifier_passed_no_more_tools",
    }


def test_should_continue_react_returns_finalize_when_cap_reached() -> None:
    state = _make_state_with_pack(["netlist_trace_tool"]).model_copy(
        update={"react_iterations": 5, "max_react_iterations": 4}
    )
    assert should_continue_react(state) == "finalize"


def test_should_continue_react_returns_continue_in_middle_of_loop() -> None:
    state = _make_state_with_pack(["netlist_trace_tool"]).model_copy(
        update={"react_iterations": 1, "max_react_iterations": 4, "react_terminate_reason": ""}
    )
    assert should_continue_react(state) == "continue"


# -- Tool whitelist (anti-hallucination) ------------------------------------------

def test_planner_respects_allowed_tool_whitelist() -> None:
    state = _make_state_with_pack(["fault_case_lookup_tool"])  # only this is allowed
    update = react_plan_node(state)
    assert "react_trace" in update
    new_step = ReActStep.model_validate(update["react_trace"][-1])
    assert new_step.tool_call is not None
    assert new_step.tool_call.tool_name == "fault_case_lookup_tool"


def test_planner_prioritizes_circuit_lookup_for_inventory_question() -> None:
    state = _make_state_with_pack(["fault_case_lookup_tool", "circuit_lookup_tool"]).model_copy(
        update={
            "query": "那差分电路一共需要几个电阻",
            "user_message": "那差分电路一共需要几个电阻",
        }
    )
    update = react_plan_node(state)
    new_step = ReActStep.model_validate(update["react_trace"][-1])
    assert new_step.tool_call is not None
    assert new_step.tool_call.tool_name == "circuit_lookup_tool"


def test_planner_blocks_unknown_tool_from_provider() -> None:
    """If a buggy provider returns a tool outside the whitelist, the node drops it."""

    class _BadProvider(TemplateLLMProvider):
        def plan(self, request: PlanRequest) -> ToolCall | None:
            return ToolCall(tool_name="evil_tool", arguments={}, rationale="hallucinated")

    state = _make_state_with_pack(["netlist_trace_tool"])
    # monkey-patch the cached provider via the factory cache
    clear_llm_provider_cache()
    import app.agent.nodes.react_plan as react_plan_module

    original = react_plan_module.get_llm_provider
    try:
        react_plan_module.get_llm_provider = lambda: _BadProvider()  # type: ignore[assignment]
        update = react_plan_node(state)
    finally:
        react_plan_module.get_llm_provider = original  # type: ignore[assignment]
        clear_llm_provider_cache()

    new_step = ReActStep.model_validate(update["react_trace"][-1])
    assert new_step.tool_call is None
    assert "no remaining allowed tools" in new_step.thought


# -- React trace shape ------------------------------------------------------------

def test_react_trace_records_each_iteration() -> None:
    evidence = _danger_evidence()
    state = run_diagnostic_graph(evidence=evidence, query="x", top_k=5, max_react_iterations=4)
    assert len(state.react_trace) == state.react_iterations
    for step in state.react_trace:
        assert isinstance(step, ReActStep)
        assert step.duration_ms >= 0.0
        # Either the step called a tool with non-empty observation, or it terminated cleanly.
        if step.tool_call is not None:
            assert step.observation.get("tool_name") == step.tool_call.tool_name


# -- Provider contract ------------------------------------------------------------

def test_template_provider_plan_returns_none_when_no_allowed_tools() -> None:
    state = _make_state_with_pack([])
    provider = TemplateLLMProvider()
    request = PlanRequest(
        iteration=0,
        evidence=state.runtime_evidence,
        context_pack=state.context_pack,
        query="x",
        user_message="x",
        prior_steps=[],
        tool_results_so_far=[],
    )
    assert provider.plan(request) is None


def test_template_provider_reflect_uses_verify_draft_answer_contract() -> None:
    state = _make_state_with_pack(["netlist_trace_tool"])
    provider = TemplateLLMProvider()
    # An empty draft must always fail the verifier.
    bad = provider.reflect(
        ReflectRequest(
            iteration=0,
            evidence=state.runtime_evidence,
            context_pack=state.context_pack,
            draft_answer="",
            verification_report=None,
            prior_steps=[],
        )
    )
    assert bad.passed is False
    assert "回答不能为空" in bad.reason or bad.next_hint


# -- Sequential fallback path -----------------------------------------------------

def test_sequential_fallback_runs_react_loop(monkeypatch) -> None:
    """When langgraph is unavailable, the sequential walker still loops."""
    import app.agent.graph as graph_module

    monkeypatch.setattr(graph_module, "StateGraph", None)
    evidence = _danger_evidence()
    state = run_diagnostic_graph(evidence=evidence, query="x", top_k=5, max_react_iterations=4)
    assert state.final_answer
    assert state.react_iterations >= 1
