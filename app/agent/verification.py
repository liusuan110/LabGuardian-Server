from __future__ import annotations

from app.agent.contracts import ContextPack, RuntimeEvidence, VerificationReport


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
    return VerificationReport(passed=passed, issues=issues, required_rewrite_hint=hint)
