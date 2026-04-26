from __future__ import annotations

from app.agent.contracts import ContextPack, RuntimeEvidence, VerificationReport


def verify_draft_answer(
    *,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    draft_answer: str,
) -> VerificationReport:
    """Rule-based reflection node for the first LangGraph version."""

    text = draft_answer or ""
    issues: list[str] = []

    if evidence.error_codes and not any(code in text for code in evidence.error_codes):
        issues.append("回答没有引用当前 validator error_code。")

    required_refs = [ref for ref in context_pack.evidence_refs if ref.component_id or ref.ref_id]
    if required_refs:
        ref_hit = False
        for ref in required_refs:
            candidates = [ref.ref_id, ref.component_id, ref.pin_name, ref.hole_id]
            if any(candidate and candidate in text for candidate in candidates):
                ref_hit = True
                break
        if not ref_hit:
            issues.append("回答没有引用推送的 evidence_refs。")

    safety_words = ("断电", "电源", "短路")
    if evidence.risk_level == "danger" and not any(word in text for word in safety_words):
        issues.append("danger 风险回答必须包含断电或电源短路复查提示。")

    passed = not issues
    hint = ""
    if issues:
        hint = "请重写回答，并补充：" + "；".join(issues)
    return VerificationReport(passed=passed, issues=issues, required_rewrite_hint=hint)
