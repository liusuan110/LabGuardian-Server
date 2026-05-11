"""Rule-based intent classifier for the multi-intent Agent helper modes.

Deterministic keyword scan. No LLM involved. Routing priority:
1. lab_guidance — explicit operational verbs about measurement / checking.
2. diagnostic — explicit "what's wrong with my circuit" phrasing.
3. concept_tutor — knowledge-seeking phrasing (definition / principle / why X).
4. mixed — concept phrasing AND we have validator findings, i.e. the question
   sits between teaching and diagnosing the current circuit.
5. fallback — diagnostic if findings exist, else concept_tutor.
"""

from __future__ import annotations

from app.agent.contracts import AgentIntent, RuntimeEvidence

_LAB_GUIDANCE_PHRASES: tuple[str, ...] = (
    "万用表",
    "示波器",
    "怎么测",
    "如何测",
    "怎么用",
    "如何用",
    "怎么检查",
    "如何检查",
    "怎么验证",
    "如何验证",
    "怎么确认",
    "如何确认",
    "下一步",
    "下一步怎么",
    "测电压",
    "测电流",
    "测电阻",
    "通断挡",
    "二极管挡",
)

_CONCEPT_PHRASES: tuple[str, ...] = (
    "什么是",
    "为什么要",
    "为什么需要",
    "为什么 led",
    "原理",
    "定律",
    "公式",
    "时间常数",
    "导通规则",
    "面包板",
    "欧姆",
    "分压",
    "滤波",
    "去耦",
    "rc 时间",
    "充放电",
    "知识点",
)

_DIAGNOSTIC_PHRASES: tuple[str, ...] = (
    "哪里错",
    "哪里有问题",
    "为什么不亮",
    "为什么短路",
    "为什么不通",
    "什么问题",
    "诊断",
    "为什么这样",
    "为什么报",
    "我这个电路",
    "我的电路",
    "当前电路",
)


def classify_intent(
    user_message: str,
    evidence: RuntimeEvidence | None = None,
) -> AgentIntent:
    """Map user_message → intent label. evidence is consulted only as tiebreaker."""

    msg = (user_message or "").strip().lower()
    if not msg:
        return "diagnostic" if (evidence and evidence.findings) else "concept_tutor"

    lab_hit = any(phrase in msg for phrase in _LAB_GUIDANCE_PHRASES)
    diag_hit = any(phrase in msg for phrase in _DIAGNOSTIC_PHRASES)
    concept_hit = any(phrase in msg for phrase in _CONCEPT_PHRASES)

    if lab_hit:
        return "lab_guidance"
    if diag_hit:
        return "diagnostic"
    if concept_hit:
        if evidence and evidence.findings:
            return "mixed"
        return "concept_tutor"

    if evidence and evidence.findings:
        return "diagnostic"
    return "concept_tutor"
