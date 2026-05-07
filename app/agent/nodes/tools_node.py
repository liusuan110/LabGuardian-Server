"""run_tools node: bulk-runs all allowed tools eagerly (legacy entry).

The ReAct loop calls one tool at a time via `react_observe_node`; this node
remains for backwards compatibility with the deterministic non-ReAct path
and for the sequential fallback when LangGraph is not installed.
"""

from __future__ import annotations

from time import perf_counter

from app.agent.contracts import DiagnosticState
from app.agent.nodes._metrics import append_metric, require_context_pack
from app.agent.tool_runner import run_diagnostic_tools


def run_tools_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    context_pack = require_context_pack(state)
    results = run_diagnostic_tools(
        evidence=state.runtime_evidence,
        context_pack=context_pack,
        query=state.query,
        top_k=state.top_k,
    )
    return {
        "tool_results": [result.model_dump() for result in results],
        "graph_metrics": append_metric(
            state,
            node_name="run_tools",
            started_at=started_at,
            payload={
                "tool_count": len(results),
                "tool_names": [result.tool_name for result in results],
            },
        ),
    }
