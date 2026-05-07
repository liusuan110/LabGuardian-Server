from __future__ import annotations

from app.agent.contracts import ContextPack, RuntimeEvidence, VerificationReport
from app.services.vlm.defect_types import suggest_defect_types

# White-box gate: only families where the netlist/topology layer cannot
# definitively explain the failure. Other families (short_circuit, polarity)
# are well-handled by the deterministic stack and must not invoke the VLM.
_GATE_FAMILIES = {
    "missing_component",
    "incomplete_circuit",
    "unknown",
}


def verify_draft_answer(
    *,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    draft_answer: str,
) -> VerificationReport:
    """Rule-based reflection node for the first LangGraph version.

    注意：不再强制要求回答中出现 error_code 或 raw evidence_refs，
    以避免将内部调试信息泄漏到用户可见的自然语言回答中。
    """

    text = draft_answer or ""
    issues: list[str] = []

    if not text.strip():
        issues.append("回答不能为空。")

    # 检查是否泄漏了不应出现的内部字段
    forbidden_keywords = ("error_codes=", "user_query=", "user_message=", "station_id=", "runtime_metadata")
    if any(kw in text for kw in forbidden_keywords):
        issues.append("回答中出现了不应展示的内部调试字段。")

    safety_words = ("断电", "电源", "短路")
    if evidence.risk_level == "danger" and not any(word in text for word in safety_words):
        issues.append("danger 风险回答必须包含断电或电源短路复查提示。")

    passed = not issues
    hint = ""
    if issues:
        hint = "请重写回答，并补充：" + "；".join(issues)

    needs_micro, suspected = _should_request_micro_inspection(evidence, context_pack)

    return VerificationReport(
        passed=passed,
        issues=issues,
        required_rewrite_hint=hint,
        needs_micro_inspection=needs_micro,
        suspected_defect_types=[d.value for d in suspected],
    )


def _should_request_micro_inspection(
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
) -> tuple[bool, list]:
    """White-box gate for VLM micro-defect inspection.

    Returns `(needs, suspected_defect_types)`. Triggers when:
    - The error tags explicitly hint at a micro defect (burn / unstripped /
      cold solder), OR
    - The error family is one the deterministic layer can't fully judge
      (`missing_component`, `incomplete_circuit`, `unknown`) AND we have at
      least one finding to ground the inspection on.
    """
    suspected = suggest_defect_types(evidence.error_tags)
    if suspected:
        return True, suspected

    family = context_pack.error_family
    if family in _GATE_FAMILIES and evidence.findings:
        # Default to all three defect types when the family is ambiguous;
        # the VLM node will inspect them in priority order.
        from app.services.vlm.defect_types import MicroDefectType

        return True, [
            MicroDefectType.BURN_MARK,
            MicroDefectType.UNSTRIPPED_WIRE,
            MicroDefectType.COLD_SOLDER,
        ]

    return False, []
