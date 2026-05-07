"""Shared per-node metric helper for the diagnostic LangGraph."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from app.agent.contracts import DiagnosticState


def append_metric(
    state: DiagnosticState,
    *,
    node_name: str,
    started_at: float,
    payload: dict[str, Any] | None = None,
    status: str = "ok",
) -> list[dict[str, Any]]:
    """Append one `GraphNodeMetric`-shaped entry to a copy of state metrics."""
    metrics = [metric.model_dump() for metric in state.graph_metrics]
    metrics.append(
        {
            "node_name": node_name,
            "duration_ms": round((perf_counter() - started_at) * 1000, 3),
            "status": status,
            "payload": payload or {},
        }
    )
    return metrics


def require_context_pack(state: DiagnosticState):
    if state.context_pack is None:
        raise RuntimeError("context_pack is required before this node")
    return state.context_pack
