"""verify_answer node: rule-based critic emitting VerificationReport."""

from __future__ import annotations

from time import perf_counter

from app.agent.contracts import DiagnosticState
from app.agent.nodes._metrics import append_metric, require_context_pack
from app.agent.tools import ToolResult
from app.agent.verification import verify_draft_answer


def verify_answer_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    tool_results = [
        ToolResult.model_validate(item)
        for item in state.tool_results
    ]
    # Intent-aware verification: concept_tutor / lab_guidance rules don't
    # require error_codes, while diagnostic still enforces them. The
    # verifier reads intent + concept from the state set up by
    # react_reflect.
    report = verify_draft_answer(
        evidence=state.runtime_evidence,
        context_pack=require_context_pack(state),
        draft_answer=state.draft_answer,
        intent=state.intent,
        concept=state.concept,
        tool_results=tool_results,
    )
    return {
        "verification_report": report.model_dump(),
        "graph_metrics": append_metric(
            state,
            node_name="verify_answer",
            started_at=started_at,
            payload={
                "intent": state.intent,
                "passed": report.passed,
                "issue_count": len(report.issues),
            },
        ),
    }
