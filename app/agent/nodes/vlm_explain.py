"""vlm_explain node: white-box-gated VLM micro-defect inspection.

This node only runs when `verification_report.needs_micro_inspection=True`,
preserving the project's white-box rule (deterministic vision + topology
must speak first; VLM only when the rule layer is genuinely uncertain).

For each `suspected_defect_type` it calls `analyze_micro_defect` and records
a `VlmFinding`. The draft answer is augmented with a short micro inspection
appendix so the user sees the VLM's per-defect conclusions.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Iterable

from app.agent.contracts import DiagnosticState, VlmFinding
from app.agent.nodes._metrics import append_metric, require_context_pack
from app.services.vlm.defect_types import MicroDefectType

logger = logging.getLogger(__name__)


def _coerce_defect(name: str) -> MicroDefectType | None:
    try:
        return MicroDefectType(name)
    except ValueError:
        return None


def _resolve_defect_types(state: DiagnosticState) -> list[MicroDefectType]:
    report = state.verification_report
    if report is None:
        return []
    types: list[MicroDefectType] = []
    for raw in report.suspected_defect_types or []:
        candidate = _coerce_defect(str(raw))
        if candidate is not None and candidate not in types:
            types.append(candidate)
    return types


def vlm_explain_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    report = state.verification_report
    if report is None or not report.needs_micro_inspection:
        return {
            "graph_metrics": append_metric(
                state,
                node_name="vlm_explain",
                started_at=started_at,
                payload={"skipped": "gate_closed"},
                status="skipped",
            ),
        }

    defect_types = _resolve_defect_types(state)
    if not defect_types:
        return {
            "graph_metrics": append_metric(
                state,
                node_name="vlm_explain",
                started_at=started_at,
                payload={"skipped": "no_defect_types"},
                status="skipped",
            ),
        }

    context_pack = require_context_pack(state)
    findings = _inspect(state, context_pack, defect_types)

    appended_draft = _append_micro_section(state.draft_answer, findings)

    return {
        "draft_answer": appended_draft,
        "vlm_findings": [f.model_dump() for f in findings],
        "graph_metrics": append_metric(
            state,
            node_name="vlm_explain",
            started_at=started_at,
            payload={
                "defect_count": len(defect_types),
                "finding_count": len(findings),
                "defect_types": [d.value for d in defect_types],
            },
        ),
    }


def _inspect(state: DiagnosticState, context_pack, defect_types: Iterable[MicroDefectType]) -> list[VlmFinding]:
    """Invoke the VLM service per defect type, capture findings.

    All exceptions are swallowed: the white-box answer is already valid; the
    micro inspection is a best-effort enrichment.
    """
    findings: list[VlmFinding] = []
    try:
        from app.core.deps import get_mrag_service, get_vlm_service
        from app.services.vlm import analyze_micro_defect
    except Exception as exc:  # pragma: no cover
        logger.warning("VLM dependencies unavailable: %s", exc)
        return findings

    try:
        mrag_service = get_mrag_service()
        vlm_service = get_vlm_service()
    except Exception as exc:  # pragma: no cover
        logger.warning("VLM service init failed: %s", exc)
        return findings

    pack: dict = {}
    try:
        pack = mrag_service.build_pack(
            error_tags=state.runtime_evidence.error_tags,
            query=state.query,
            circuit_snapshot=state.runtime_evidence.circuit_snapshot or "",
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("MRag pack assembly failed: %s", exc)

    for defect_type in defect_types:
        try:
            result = analyze_micro_defect(
                vlm_service=vlm_service,
                defect_type=defect_type,
                mrag_pack=pack,
                user_query=state.user_message or state.query,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("analyze_micro_defect(%s) failed: %s", defect_type, exc)
            continue
        answer = (result or {}).get("answer", {}) if isinstance(result, dict) else {}
        findings.append(
            VlmFinding(
                defect_type=defect_type.value,
                provider=str((result or {}).get("provider", "")),
                status=str((result or {}).get("status", "")),
                conclusion=str(answer.get("conclusion", "")),
                evidence=str(answer.get("evidence", "")),
                fix_steps=list(answer.get("fix_steps", []))[:4],
                raw=result if isinstance(result, dict) else {},
            )
        )
    return findings


_SECTION_HEADER = "\n\n[微观缺陷复检]"


def _append_micro_section(existing_draft: str, findings: list[VlmFinding]) -> str:
    if not findings:
        return existing_draft
    if _SECTION_HEADER.strip() in (existing_draft or ""):
        return existing_draft  # already appended in a prior pass
    lines = [_SECTION_HEADER]
    for f in findings:
        head = f"- [{f.defect_type}] {f.conclusion or '未发现明显异常'}"
        lines.append(head)
    return (existing_draft or "") + "\n".join(lines)
