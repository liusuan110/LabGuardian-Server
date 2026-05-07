"""classify_error node: maps error_codes to ErrorFamily."""

from __future__ import annotations

from time import perf_counter

from app.agent.context_pack import classify_error_family
from app.agent.contracts import DiagnosticState
from app.agent.nodes._metrics import append_metric


def classify_error_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    error_family = classify_error_family(state.runtime_evidence)
    return {
        "error_family": error_family,
        "graph_metrics": append_metric(
            state,
            node_name="classify_error",
            started_at=started_at,
            payload={"error_family": error_family},
        ),
    }
