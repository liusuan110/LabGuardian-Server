"""verify_answer node: rule-based critic emitting VerificationReport."""

from __future__ import annotations

from time import perf_counter

from app.agent.contracts import DiagnosticState
from app.agent.nodes._metrics import append_metric, require_context_pack
from app.agent.verification import verify_draft_answer


def verify_answer_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    report = verify_draft_answer(
        evidence=state.runtime_evidence,
        context_pack=require_context_pack(state),
        draft_answer=state.draft_answer,
    )
    return {
        "verification_report": report.model_dump(),
        "graph_metrics": append_metric(
            state,
            node_name="verify_answer",
            started_at=started_at,
            payload={"passed": report.passed, "issue_count": len(report.issues)},
        ),
    }
