"""repair_answer node: rewrites the draft when verification fails."""

from __future__ import annotations

from time import perf_counter

from app.agent.answering import repair_diagnostic_answer
from app.agent.contracts import DiagnosticState
from app.agent.nodes._metrics import append_metric, require_context_pack
from app.agent.verification import verify_draft_answer


def repair_answer_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    issues = state.verification_report.issues if state.verification_report else []
    repaired = repair_diagnostic_answer(
        draft_answer=state.draft_answer,
        evidence=state.runtime_evidence,
        verification_issues=issues,
    )
    report = verify_draft_answer(
        evidence=state.runtime_evidence,
        context_pack=require_context_pack(state),
        draft_answer=repaired,
    )
    return {
        "draft_answer": repaired,
        "verification_report": report.model_dump(),
        "graph_metrics": append_metric(
            state,
            node_name="repair_answer",
            started_at=started_at,
            payload={
                "passed_after_repair": report.passed,
                "issue_count": len(report.issues),
                "draft_length": len(repaired),
            },
        ),
    }
