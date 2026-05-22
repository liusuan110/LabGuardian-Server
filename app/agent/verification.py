from __future__ import annotations

import re
from typing import Any, Sequence

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
    tool_results: Sequence[Any] | None = None,
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

    issues.extend(_datasheet_rules(text, tool_results))

    passed = not issues
    hint = ""
    if issues:
        hint = "请重写回答，并补充：" + "；".join(issues)

    needs_micro, suspected = _should_request_micro_inspection(
        evidence,
        context_pack,
        tool_results=tool_results,
        draft_answer=text,
    )

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


_DATASHEET_CHUNK_PROVIDERS = {"local_datasheet_v2", "kb_retrieval"}
_DATASHEET_RULE_PROVIDERS = {"local_fallback"}


def _datasheet_rules(
    text: str,
    tool_results: Sequence[Any] | None,
) -> list[str]:
    """Enforce datasheet citation contract when the tool was invoked.

    Triggered only if `datasheet_lookup_tool` actually produced hits or rules
    in this turn. Chunked providers must surface a `chunk_id`; pure rule-based
    fallback must surface a `rule_id`. Both forms can also be satisfied by the
    raw token appearing in the answer text (a model can cite either way).
    """

    if not tool_results:
        return []

    needs_chunk_id = False
    needs_rule_id = False
    expected_chunk_ids: set[str] = set()
    expected_rule_ids: set[str] = set()

    for result in tool_results:
        payload = _result_payload(result)
        if not payload:
            continue
        if _result_tool_name(result) != "datasheet_lookup_tool":
            continue
        provider = str(payload.get("provider") or "")
        if provider in _DATASHEET_CHUNK_PROVIDERS:
            hits = payload.get("hits") or []
            for hit in hits if isinstance(hits, list) else []:
                if isinstance(hit, dict):
                    cid = hit.get("chunk_id") or hit.get("source_id")
                    if cid:
                        expected_chunk_ids.add(str(cid))
            if expected_chunk_ids:
                needs_chunk_id = True
        elif provider in _DATASHEET_RULE_PROVIDERS:
            structured = payload.get("structured_rules") or []
            for rule in structured if isinstance(structured, list) else []:
                if isinstance(rule, dict) and rule.get("rule_id"):
                    expected_rule_ids.add(str(rule["rule_id"]))
            if expected_rule_ids:
                needs_rule_id = True

    issues: list[str] = []
    if needs_chunk_id and not any(cid in text for cid in expected_chunk_ids):
        issues.append("datasheet 检索命中后，回答必须引用至少一个 datasheet chunk_id。")
    if needs_rule_id and not any(rid in text for rid in expected_rule_ids):
        issues.append("datasheet 回退到本地规则时，回答必须引用至少一个 rule_id。")
    return issues


def _result_payload(result: Any) -> dict[str, Any] | None:
    payload = getattr(result, "payload", None)
    if isinstance(payload, dict):
        return payload
    if isinstance(result, dict):
        nested = result.get("payload")
        if isinstance(nested, dict):
            return nested
    return None


def _result_tool_name(result: Any) -> str:
    name = getattr(result, "tool_name", None)
    if isinstance(name, str):
        return name
    if isinstance(result, dict):
        return str(result.get("tool_name") or "")
    return ""


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
    *,
    tool_results: Sequence[Any] | None = None,
    draft_answer: str = "",
) -> tuple[bool, list]:
    """White-box gate for VLM micro-defect inspection.

    Returns `(needs, suspected_defect_types)`. Triggers when:
    - The error tags explicitly hint at a micro defect (burn / unstripped /
      cold solder), OR
    - The error family is one the deterministic layer can't fully judge
      (`missing_component`, `incomplete_circuit`, `unknown`) AND we have at
      least one finding to ground the inspection on.
    """
    if _is_circuit_kb_grounded_answer(tool_results, draft_answer):
        return False, []

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


def _is_circuit_kb_grounded_answer(
    tool_results: Sequence[Any] | None,
    draft_answer: str,
) -> bool:
    if "本地电路知识库" not in (draft_answer or ""):
        return False
    for result in tool_results or []:
        if isinstance(result, dict):
            tool_name = result.get("tool_name")
            status = result.get("status")
            payload = result.get("payload") or {}
        else:
            tool_name = getattr(result, "tool_name", "")
            status = getattr(result, "status", "")
            payload = getattr(result, "payload", {}) or {}
        if (
            tool_name == "circuit_lookup_tool"
            and status == "ok"
            and payload.get("circuits")
        ):
            return True
    return False
