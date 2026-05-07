"""react_plan node: planner step of the ReAct loop.

Picks the next tool to call (or terminates) using the active LLM provider.
The provider is restricted to tools listed in `ContextPack.allowed_tools`,
so the planner cannot invent tool names — a hard white-box guarantee.
"""

from __future__ import annotations

from time import perf_counter

from app.agent.contracts import DiagnosticState, ReActStep
from app.agent.llm import PlanRequest, get_llm_provider
from app.agent.nodes._metrics import append_metric, require_context_pack


def react_plan_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    context_pack = require_context_pack(state)
    provider = get_llm_provider()

    iteration = state.react_iterations
    request = PlanRequest(
        iteration=iteration,
        evidence=state.runtime_evidence,
        context_pack=context_pack,
        query=state.query,
        user_message=state.user_message,
        prior_steps=list(state.react_trace),
        tool_results_so_far=list(state.tool_results),
    )
    tool_call = provider.plan(request)

    # Enforce action whitelist defensively: providers should already obey it,
    # but a buggy provider must not be able to invoke an unlisted tool.
    allowed = {tool.name for tool in context_pack.allowed_tools}
    if tool_call is not None and tool_call.tool_name not in allowed:
        tool_call = None

    if tool_call is None:
        thought = "no remaining allowed tools to call"
    else:
        thought = f"plan to call {tool_call.tool_name} ({tool_call.rationale})"

    new_step = ReActStep(
        iteration=iteration,
        thought=thought,
        tool_call=tool_call,
        observation={},
        reflection="",
        terminate=False,
        duration_ms=round((perf_counter() - started_at) * 1000, 3),
    )
    react_trace = [step.model_dump() for step in state.react_trace] + [new_step.model_dump()]

    return {
        "react_trace": react_trace,
        "graph_metrics": append_metric(
            state,
            node_name=f"react_plan_{iteration}",
            started_at=started_at,
            payload={
                "iteration": iteration,
                "tool_name": tool_call.tool_name if tool_call else None,
                "allowed_tool_count": len(allowed),
            },
        ),
    }
