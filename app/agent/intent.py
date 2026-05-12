"""Rule-based intent classifier for the multi-intent Agent helper modes.

Deterministic keyword scan. No LLM involved. Routing priority:
1. lab_guidance — explicit operational verbs about measurement / checking.
2. mixed — current-circuit diagnostic phrasing plus concept phrasing, or
   concept phrasing with validator findings.
3. diagnostic — explicit "what's wrong with my circuit" phrasing.
4. concept_tutor — knowledge-seeking phrasing (definition / principle / why X).
5. fallback — concept_tutor, unless the user explicitly asks for diagnosis.
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
    "是什么",
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
    "有什么关系",
    "有啥关系",
    "关系是",
    "区别",
    "联系",
    "电磁",
    "电场",
    "磁场",
    "电感",
    "感应",
)

_DIAGNOSTIC_PHRASES: tuple[str, ...] = (
    "哪里错",
    "哪里错了",
    "哪里有问题",
    "哪里不对",
    "哪里不对劲",
    "哪错了",
    "有什么问题",
    "有啥问题",
    "啥问题",
    "问题在哪",
    "问题出在哪",
    "问题是什么",
    "这有什么问题",
    "这个有什么问题",
    "它有什么问题",
    "具体问题",
    "具体的问题",
    "这个电路",
    "这张电路",
    "这张图",
    "电路图",
    "上传的电路",
    "上传电路",
    "参考差异",
    "参考电路",
    "和参考",
    "对比参考",
    "跟参考",
    "相比参考",
    "帮我看看",
    "帮我看下",
    "检查一下",
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

_CURRENT_CONTEXT_FOLLOW_UP_PHRASES: tuple[str, ...] = (
    "简单点",
    "再说",
    "换种说法",
    "详细点",
    "讲详细",
    "听不懂",
    "没懂",
    "什么意思",
    "这是什么意思",
    "有什么问题",
    "有啥问题",
    "啥问题",
    "问题在哪",
    "问题出在哪",
    "问题是什么",
    "具体问题",
    "具体的问题",
    "这个电路",
    "这张电路",
    "这张图",
    "电路图",
    "上传的电路",
    "上传电路",
    "参考差异",
    "参考电路",
    "和参考",
    "对比参考",
    "跟参考",
    "相比参考",
    "哪里不对",
    "哪里不对劲",
    "哪错了",
    "这个问题",
    "怎么改",
    "怎么修",
    "怎么处理",
)

_CIRCUIT_TOPIC_PHRASES: tuple[str, ...] = (
    "电路",
    "电压",
    "电流",
    "电阻",
    "电容",
    "电感",
    "二极管",
    "led",
    "三极管",
    "mos",
    "运放",
    "芯片",
    "面包板",
    "跳线",
    "引脚",
    "孔位",
    "电源",
    "短路",
    "断路",
    "接地",
    "gnd",
    "vcc",
    "万用表",
    "示波器",
    "欧姆",
    "分压",
    "滤波",
    "原理图",
    "breadboard",
    "resistor",
    "capacitor",
    "inductor",
    "diode",
    "transistor",
    "voltage",
    "current",
)


def _looks_like_theory_question(msg: str) -> bool:
    """Heuristic for concept/theory-like questions that should avoid rigid diagnostic templates."""
    theory_hints = (
        "关系",
        "区别",
        "联系",
        "原理",
        "定律",
        "公式",
        "电磁",
        "电场",
        "磁场",
        "感应",
    )
    return any(hint in msg for hint in theory_hints)


def _looks_like_circuit_topic(msg: str) -> bool:
    return any(phrase in msg for phrase in _CIRCUIT_TOPIC_PHRASES)


def _looks_like_current_context_follow_up(msg: str) -> bool:
    return any(phrase in msg for phrase in _CURRENT_CONTEXT_FOLLOW_UP_PHRASES)


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
    if diag_hit and concept_hit:
        return "mixed"
    if diag_hit:
        return "diagnostic"
    if concept_hit:
        if evidence and evidence.findings:
            return "mixed"
        return "concept_tutor"

    if evidence and evidence.findings:
        # Do not force every unrelated or vague follow-up into the diagnostic
        # template just because the station currently has findings. Only
        # explicit diagnostic wording above should enter the diagnostic path.
        if (
            _looks_like_theory_question(msg)
            or _looks_like_circuit_topic(msg)
            or _looks_like_current_context_follow_up(msg)
        ):
            return "mixed"
        return "concept_tutor"
    return "concept_tutor"
