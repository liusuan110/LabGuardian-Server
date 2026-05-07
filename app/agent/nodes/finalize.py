"""finalize_answer node: pins the draft as the final user-visible answer."""

from __future__ import annotations

from time import perf_counter

from app.agent.contracts import DiagnosticState
from app.agent.nodes._metrics import append_metric


def finalize_answer_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    return {
        "final_answer": state.draft_answer,
        "graph_metrics": append_metric(
            state,
            node_name="finalize_answer",
            started_at=started_at,
            payload={"final_answer_length": len(state.draft_answer)},
        ),
    }
