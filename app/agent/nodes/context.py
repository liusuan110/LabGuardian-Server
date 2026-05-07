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
    )
    return {
        "context_pack": pack.model_dump(),
        "error_family": pack.error_family,
        "graph_metrics": append_metric(
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
