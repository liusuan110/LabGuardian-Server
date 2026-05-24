"""build_context_pack node: assembles the PCM ContextPack."""

from __future__ import annotations

from time import perf_counter

from app.agent.context_pack import build_context_pack
from app.agent.contracts import DiagnosticState
from app.agent.nodes._metrics import append_metric


def build_context_pack_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    pack = build_context_pack(
        state.runtime_evidence,
        query=state.query,
        user_message=state.user_message,
        intent=state.intent,
    )
    # WP-1 v5 (2026-05-24): dynamic ReAct cap. Without this, dangerous
    # short-circuit turns (short_circuit family → 5 family tools, +1 from
    # circuit_lookup keyword gate = 6 tools) overflow the default
    # ``REACT_MAX_ITERATIONS=4`` budget and required ``safety_rule_lookup_tool``
    # may not get called — that produces distillation samples where the
    # teacher answers a dangerous-circuit question without leading with
    # 断电/限流 rules. See ``docs/retrieval-contract.md`` WP-1 v5 entry.
    #
    # Policy:
    #   - When the caller did not override the cap (``react_cap_auto_expand``
    #     is True), expand to ``max(initial cap, len(allowed_tools))`` so
    #     every allowed tool gets a slot.
    #   - When the caller passed an explicit ``max_react_iterations``, that
    #     is a hard ceiling — don't auto-expand. Tests that intentionally
    #     cap ReAct at 2 to verify termination must keep their semantic.
    if state.react_cap_auto_expand:
        desired_cap = max(state.max_react_iterations, len(pack.allowed_tools))
    else:
        desired_cap = state.max_react_iterations
    return {
        "context_pack": pack.model_dump(),
        "error_family": pack.error_family,
        "max_react_iterations": desired_cap,
        "graph_metrics": append_metric(
            state,
            node_name="build_context_pack",
            started_at=started_at,
            payload={
                "pack_id": pack.pack_id,
                "pushed_facts_count": len(pack.pushed_facts),
                "allowed_tool_count": len(pack.allowed_tools),
                # WP-1 v5: surface the dynamic-cap decision for telemetry.
                "react_cap_initial": state.max_react_iterations,
                "react_cap_applied": desired_cap,
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
