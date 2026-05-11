from __future__ import annotations

import re

from app.agent.contracts import (
    AgentIntent,
    ConceptPack,
    ContextPack,
    RuntimeEvidence,
    VerificationReport,
)
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
    intent: AgentIntent = "diagnostic",
    concept: ConceptPack | None = None,
) -> VerificationReport:
    """Rule-based reflection node.

    `intent` defaults to "diagnostic" so existing callers (the LangGraph
    verify_answer node) are unaffected. concept_tutor / lab_guidance apply
    intent-specific rule sets that do NOT require validator error_codes.
    """

    text = draft_answer or ""
    issues: list[str] = []

    if not text.strip():
        issues.append("回答不能为空。")

    forbidden_keywords = (
        "error_codes=",
        "user_query=",
        "user_message=",
        "station_id=",
        "runtime_metadata",
    )
    if any(kw in text for kw in forbidden_keywords):
        issues.append("回答中出现了不应展示的内部调试字段。")

    if intent == "diagnostic":
        issues.extend(_diagnostic_rules(evidence, text))
    elif intent == "concept_tutor":
        issues.extend(_concept_rules(evidence, text, concept))
    elif intent == "lab_guidance":
        issues.extend(_lab_guidance_rules(text, concept))
    # mixed intent reuses diagnostic rules — the main answer is a diagnostic
    # answer; concept content is appended as evidence, not into the text.
    elif intent == "mixed":
        issues.extend(_diagnostic_rules(evidence, text))

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


_SAFETY_WORDS = ("断电", "电源", "短路")
_RESHOOT_HINTS = ("复拍", "重新拍照", "人工确认", "识别置信度", "孔位识别")
_STEP_MARKERS = ("1.", "1．", "第一步", "步骤", "①")
_FABRICATED_HOLE_PATTERN = re.compile(
    r"(ROW_\d+|HOLE_\d+|NET_\d+|row_\d+|hole_\d+|net_\d+|节点\s*N\d+)",
    re.IGNORECASE,
)


def _diagnostic_rules(evidence: RuntimeEvidence, text: str) -> list[str]:
    issues: list[str] = []
    if evidence.error_codes and not any(code in text for code in evidence.error_codes):
        issues.append("diagnostic 回答必须包含至少一个当前 error_code。")
    if evidence.evidence_refs and not _mentions_any_evidence_ref(evidence, text):
        issues.append(
            "diagnostic 回答必须引用至少一个 evidence_ref、component_id、pin_name 或 hole_id。"
        )
    if evidence.risk_level == "danger" and not any(w in text for w in _SAFETY_WORDS):
        issues.append("danger 风险回答必须包含断电或电源短路复查提示。")
    if _has_visual_uncertainty(evidence) and not any(h in text for h in _RESHOOT_HINTS):
        issues.append("视觉识别置信度较低，回答必须提示复拍或人工确认孔位。")
    return issues


def _mentions_any_evidence_ref(evidence: RuntimeEvidence, text: str) -> bool:
    tokens: list[str] = []
    for ref in evidence.evidence_refs:
        tokens.extend(
            value
            for value in (ref.ref_id, ref.component_id, ref.pin_name, ref.hole_id)
            if value
        )
    return any(token in text for token in tokens)


_CONCEPT_AUDIT_MARKERS = (
    "知识来源",
    "原理",
    "公式",
    "定律",
    "知识点",
    "概念",
)


def _concept_rules(
    evidence: RuntimeEvidence,
    text: str,
    concept: ConceptPack | None,
) -> list[str]:
    issues: list[str] = []
    has_audit_marker = any(marker in text for marker in _CONCEPT_AUDIT_MARKERS) or (
        concept is not None and concept.concept_id in text
    )
    if not has_audit_marker:
        issues.append("概念回答必须包含 concept_id 或原理/公式等可审计标记。")

    has_evidence = bool(evidence.findings or evidence.error_codes)
    if not has_evidence and _FABRICATED_HOLE_PATTERN.search(text):
        issues.append("没有 evidence 时不允许声称当前电路的具体孔位或节点。")

    if concept is not None and concept.concept_id in (
        "led_current_limit",
        "capacitor_filtering",
        "ohms_law",
    ):
        if not any(w in text for w in _SAFETY_WORDS):
            issues.append("涉及 LED / 电源 / 短路相关概念时必须包含安全提醒。")
    return issues


def _lab_guidance_rules(text: str, concept: ConceptPack | None) -> list[str]:
    issues: list[str] = []
    if not any(marker in text for marker in _STEP_MARKERS):
        issues.append("操作指导必须包含编号步骤（例如 1. / 第一步）。")
    if not any(w in text for w in _SAFETY_WORDS):
        issues.append("操作指导必须包含断电或电源安全提示。")
    return issues


def _has_visual_uncertainty(evidence: RuntimeEvidence) -> bool:
    return (
        evidence.ambiguous_pin_count > 0
        or evidence.fallback_pin_count > 0
        or evidence.snap_conflict_count > 0
        or evidence.low_confidence_component_count > 0
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
