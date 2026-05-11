from __future__ import annotations

from app.agent.contracts import AgentIntent, ConceptPack, ContextPack, RuntimeEvidence
from app.agent.tools import ToolResult
from app.agent.verification import verify_draft_answer
from app.schemas.angnt import AngntCitation, AngntEvidence


def _classify_user_intent(user_message: str) -> str:
    """简单意图分类，用于选择回答风格。"""
    msg = (user_message or "").lower()
    if any(k in msg for k in ("元件", "组件", "有什么", "哪些", "components", "parts", "器件")):
        return "components"
    if any(k in msg for k in ("为什么", "怎么回事", "原因", "解释", "悬空", "为什么判断", "为何")):
        return "explain"
    return "general"


def _describe_components(evidence: RuntimeEvidence, context_pack: ContextPack) -> str:
    """基于 netlist_v2 和 findings 生成元件清单描述。"""
    netlist = evidence.netlist_v2 or {}
    components = netlist.get("components", [])
    if not isinstance(components, list):
        components = []

    # 收集元件信息
    component_summaries: list[str] = []
    for comp in components:
        if not isinstance(comp, dict):
            continue
        cid = comp.get("component_id") or comp.get("id") or "未知元件"
        ctype = comp.get("component_type") or comp.get("type") or ""
        desc = cid
        if ctype:
            desc = f"{cid}（{ctype}）"
        component_summaries.append(desc)

    # 如果没有 netlist 组件，尝试从 findings 中提取
    if not component_summaries:
        seen: set[str] = set()
        for finding in evidence.findings:
            cid = finding.component_id
            if cid and cid not in seen:
                seen.add(cid)
                component_summaries.append(cid)

    if not component_summaries:
        component_summaries.append("未明确识别到元件")

    # 查找潜在问题
    issue_parts: list[str] = []
    for finding in evidence.findings[:3]:
        cid = finding.component_id
        pin = finding.pin_name
        code = finding.error_code
        if code == "FLOATING_PIN" and cid and pin:
            issue_parts.append(
                f"{cid} 的 {pin} 目前被判断为可能悬空，"
                "因为它只映射到了自身或未形成有效参考连接"
            )
        elif code == "COMPONENT_SHORTED_SAME_NET" and cid:
            issue_parts.append(f"{cid} 的两端似乎落在同一导通节点上，存在短接风险")
        elif code == "POLARITY_REVERSED" and cid:
            issue_parts.append(f"{cid} 的极性方向可能需要复核")
        elif cid and pin:
            issue_parts.append(f"{cid} 的 {pin} 存在 {code} 问题")
        elif cid:
            issue_parts.append(f"{cid} 存在 {code} 问题")

    lines: list[str] = []
    if len(component_summaries) == 1 and "未明确" in component_summaries[0]:
        lines.append("当前诊断结果中未明确识别到元件。请确认图像是否清晰、元件是否完整入镜。")
    else:
        lines.append(
            f"当前诊断结果中识别到了 {len(component_summaries)} 个主要对象："
            f"{', '.join(component_summaries)}。"
        )

    if issue_parts:
        lines.append(issue_parts[0] + "。")
        lines.append("建议你检查相关引脚是否插入了正确孔位，并确认是否与目标电路中的电阻/电源/地线形成有效连接。")
    else:
        lines.append("目前暂未检测到明显的结构化连接异常，建议继续验证剩余元件和连接完整性。")

    return "".join(lines)


def _explain_issue(evidence: RuntimeEvidence, context_pack: ContextPack) -> str:
    """基于 findings 生成原因解释。"""
    if not evidence.findings:
        return "当前没有检测到明确的结构化异常。如果仍有疑问，建议对照参考电路逐项核对连接。"

    finding = evidence.findings[0]
    cid = finding.component_id
    pin = finding.pin_name
    code = finding.error_code
    expected = finding.expected
    actual = finding.actual

    parts: list[str] = []

    if code == "FLOATING_PIN" and cid and pin:
        parts.append(
            f"{cid} 的 {pin} 被判断为可能悬空，"
            f"是因为在网表或 validator 检查中，该引脚没有映射到有效的电气节点或其他元件引脚。"
        )
        if expected and actual:
            parts.append(f"期望连接为 {expected}，但实际映射为 {actual}。")
        parts.append(
            "建议你检查该引脚是否确实插入了面包板孔位，"
            "并确认跳线或元件引脚是否与目标电路中的电源、地线或信号节点连通。"
        )
    elif code == "COMPONENT_SHORTED_SAME_NET" and cid:
        parts.append(
            f"{cid} 的两端被检测到落在同一电气节点上，"
            f"这意味着该元件两端电位相同，没有起到应有的作用，相当于被短接。"
        )
        parts.append("建议你重新跨行插接该元件，确保两端位于不同的导通组。")
    elif code == "POLARITY_REVERSED" and cid:
        parts.append(
            f"{cid} 的极性方向与期望不符。"
            f"对于极性器件（如 LED、电解电容、二极管），引脚方向决定了电流是否能正确导通。"
        )
        parts.append("建议核对器件丝印、引脚长度或datasheet，确认正负极后再接入电路。")
    elif code == "NODE_MISMATCH" and cid:
        parts.append(
            f"{cid} 的连接节点与参考电路不一致。"
        )
        if expected and actual:
            parts.append(f"期望节点为 {expected}，但实际为 {actual}。")
        parts.append("建议对照参考电路的网表，确认该元件各引脚所在的电气节点是否正确。")
    else:
        parts.append(f"检测到 {code} 问题")
        if cid:
            parts[-1] += f"，涉及元件 {cid}"
            if pin:
                parts[-1] += f" 的 {pin}"
        parts[-1] += "。"
        if expected and actual:
            parts.append(f"期望值为 {expected}，实际值为 {actual}。")
        parts.append("建议对照 validator 报告和参考电路逐项排查。")

    if evidence.risk_level == "danger":
        parts.append("当前风险等级较高，建议先断电复查，再重新连接。")

    return "".join(parts)


def _build_general_diagnostic_answer(
    station_id: str,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
) -> str:
    """通用诊断摘要，不暴露原始证据。"""
    conclusion = diagnostic_conclusion(evidence=evidence, context_pack=context_pack)

    parts: list[str] = []
    parts.append(f"工位 {station_id} 的诊断结论：{conclusion}。")

    # 历史上下文摘要（追问时有用）
    if context_pack.history_summary:
        parts.append(context_pack.history_summary + "。")

    # 1-2 条关键发现，用自然语言描述
    finding_descs: list[str] = []
    for finding in evidence.findings[:2]:
        cid = finding.component_id
        pin = finding.pin_name
        code = finding.error_code
        if code == "FLOATING_PIN" and cid and pin:
            finding_descs.append(f"{cid} 的 {pin} 可能未形成有效连接")
        elif code == "COMPONENT_SHORTED_SAME_NET" and cid:
            finding_descs.append(f"{cid} 存在短接风险")
        elif code == "POLARITY_REVERSED" and cid:
            finding_descs.append(f"{cid} 的极性方向需要复核")
        elif code == "NODE_MISMATCH" and cid:
            finding_descs.append(f"{cid} 的连接节点与参考不符")
        elif cid:
            finding_descs.append(f"{cid} 存在连接异常")
        else:
            finding_descs.append("检测到连接异常")

    if finding_descs:
        parts.append("关键发现：" + ", ".join(finding_descs) + "。")

    # 下一步建议
    suggestions: list[str] = []
    for finding in evidence.findings[:3]:
        if finding.suggested_action:
            suggestions.append(finding.suggested_action)

    if not suggestions:
        suggestions.append("对照参考电路逐项核对元件和连接")

    if evidence.risk_level == "danger":
        parts.append("安全提示：当前风险等级较高，建议先断电，再优先检查短路、极性和电源轨连接情况。")
    elif evidence.risk_level == "warning":
        parts.append("建议：按诊断发现逐项排查，先检查最前面的风险原因，再核对参考电路。")
    else:
        parts.append("建议：当前风险较低，继续验证剩余元件和连接完整性即可。")

    if suggestions:
        parts.append("可优先尝试：" + ", ".join(suggestions[:3]) + "。")

    return "".join(parts)


def build_diagnostic_template_answer(
    *,
    station_id: str,
    query: str,
    user_message: str,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
) -> str:
    """基于用户意图生成自然语言回答，不暴露 system prompt / raw JSON。"""
    intent = _classify_user_intent(user_message or query)

    if intent == "components":
        return _with_diagnostic_anchors(
            _describe_components(evidence, context_pack),
            evidence,
        )
    if intent == "explain":
        return _with_diagnostic_anchors(_explain_issue(evidence, context_pack), evidence)

    return _with_diagnostic_anchors(
        _build_general_diagnostic_answer(station_id, evidence, context_pack, tool_results),
        evidence,
    )


def _with_diagnostic_anchors(answer: str, evidence: RuntimeEvidence) -> str:
    anchored = answer
    if evidence.error_codes and not any(code in anchored for code in evidence.error_codes):
        anchored += f"\n校验依据：{evidence.error_codes[0]}。"
    if evidence.evidence_refs and not _mentions_any_runtime_ref(evidence, anchored):
        anchored += f"\n证据引用：{_first_ref_text(evidence)}。"
    return anchored


def repair_diagnostic_answer(
    *,
    draft_answer: str,
    evidence: RuntimeEvidence,
    verification_issues: list[str],
) -> str:
    """修复回答，仅补充 verifier 要求的可审计诊断锚点和安全提示。"""
    repaired = _with_diagnostic_anchors(draft_answer, evidence)
    if evidence.risk_level == "danger" and not any(
        word in repaired for word in ("断电", "电源", "短路")
    ):
        repaired += "\n安全提示：请先断电，再复查电源轨和短路风险。"
    if (
        evidence.ambiguous_pin_count
        or evidence.fallback_pin_count
        or evidence.snap_conflict_count
        or evidence.low_confidence_component_count
    ) and not any(
        hint in repaired
        for hint in ("复拍", "重新拍照", "人工确认", "识别置信度", "孔位识别")
    ):
        repaired += "\n提示：当前孔位识别置信度较低，建议复拍或人工确认引脚孔位。"
    return repaired


def build_verified_diagnostic_answer(
    *,
    station_id: str,
    query: str,
    user_message: str,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
) -> tuple[str, bool, list[str]]:
    draft = build_diagnostic_template_answer(
        station_id=station_id,
        query=query,
        user_message=user_message,
        evidence=evidence,
        context_pack=context_pack,
        tool_results=tool_results,
    )
    verification = verify_draft_answer(
        evidence=evidence,
        context_pack=context_pack,
        draft_answer=draft,
    )
    if verification.passed:
        return draft, True, []

    repaired = repair_diagnostic_answer(
        draft_answer=draft,
        evidence=evidence,
        verification_issues=verification.issues,
    )
    repaired_verification = verify_draft_answer(
        evidence=evidence,
        context_pack=context_pack,
        draft_answer=repaired,
    )
    return repaired, repaired_verification.passed, repaired_verification.issues


def build_diagnostic_citations(
    *,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
) -> list[AngntCitation]:
    citations = [
        AngntCitation(
            source_type="runtime_evidence",
            source_id=evidence.station_id,
            title="结构化运行时证据",
            snippet="、".join(evidence.error_codes) or evidence.risk_level,
        ),
        AngntCitation(
            source_type="context_pack",
            source_id=context_pack.pack_id,
            title="PCM 上下文包",
            snippet=context_pack.error_family,
        ),
    ]
    for result in tool_results:
        citations.append(
            AngntCitation(
                source_type="diagnostic_tool",
                source_id=result.tool_name,
                title=result.tool_name,
                snippet=result.summary[:260],
            )
        )
    return citations


def build_diagnostic_evidence(
    *,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
    verification_passed: bool,
    verification_issues: list[str],
    graph_metrics: list[dict] | None = None,
    react_trace: list[dict] | None = None,
    react_iterations: int = 0,
    react_terminate_reason: str = "",
) -> list[AngntEvidence]:
    items = [
        AngntEvidence(
            evidence_type="runtime_evidence",
            source_id=evidence.station_id,
            summary="PCM Agent 输入证据",
            payload=evidence.model_dump(),
        ),
        AngntEvidence(
            evidence_type="context_pack",
            source_id=context_pack.pack_id,
            summary="按错误类型推送的上下文和工具",
            payload=context_pack.model_dump(),
        ),
        AngntEvidence(
            evidence_type="context_timeline",
            source_id=f"{evidence.station_id}:context_timeline",
            summary=context_pack.history_summary or "暂无历史上下文",
            payload={
                "history_facts": context_pack.history_facts,
                "history_summary": context_pack.history_summary,
            },
        ),
        AngntEvidence(
            evidence_type="tool_results",
            source_id=f"{evidence.station_id}:diagnostic_tools",
            summary="白盒诊断工具输出",
            payload={"results": [result.model_dump() for result in tool_results]},
        ),
        AngntEvidence(
            evidence_type="verification_report",
            source_id=f"{evidence.station_id}:verifier",
            summary="Reflection Node 校验结果",
            payload={
                "passed": verification_passed,
                "issues": verification_issues,
            },
        ),
    ]
    if graph_metrics:
        items.append(
            AngntEvidence(
                evidence_type="graph_metrics",
                source_id=f"{evidence.station_id}:langgraph",
                summary="PCM LangGraph 节点级指标",
                payload={"metrics": graph_metrics},
            )
        )
    if react_trace:
        terminate_reason = react_terminate_reason or "completed"
        items.append(
            AngntEvidence(
                evidence_type="react_trace",
                source_id=f"{evidence.station_id}:react",
                summary=f"ReAct {react_iterations} 轮 ({terminate_reason})",
                payload={
                    "steps": react_trace,
                    "iterations": react_iterations,
                    "terminate_reason": terminate_reason,
                },
            )
        )
    highlight_protocol = evidence.validator_report_v2.get("highlight_protocol", {})
    if highlight_protocol.get("targets"):
        items.append(
            AngntEvidence(
                evidence_type="highlight_protocol",
                source_id=f"{evidence.station_id}:highlight_protocol",
                summary="前端高亮协议",
                payload=highlight_protocol,
            )
        )
    return items


def diagnostic_conclusion(
    *,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
) -> str:
    if context_pack.error_family == "short_circuit":
        return "检测到短路或同网风险，需要优先安全复查"
    if context_pack.error_family == "wiring_mismatch":
        return "检测到接线孔位或电气节点不匹配"
    if context_pack.error_family == "polarity_error":
        return "检测到极性方向需要复核"
    if context_pack.error_family == "missing_protection":
        return "检测到缺少保护或限流元件"
    if evidence.risk_level == "safe":
        return "当前没有明确高风险结构化错误"
    return "检测到需要进一步排查的结构化诊断项"


def extract_fix_steps(tool_results: list[ToolResult]) -> list[str]:
    steps: list[str] = []
    for result in tool_results:
        for case in result.payload.get("fault_cases", []):
            steps.extend(str(step) for step in case.get("fix_steps", []) if step)
        steps.extend(str(rule) for rule in result.payload.get("rules", []) if rule)

    deduped: list[str] = []
    for step in steps:
        if step not in deduped:
            deduped.append(step)
    return deduped


def _build_follow_up_suggestions(
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
) -> list[str]:
    """基于诊断内容生成追问建议。"""
    suggestions: list[str] = []

    first_finding = evidence.findings[0] if evidence.findings else None
    if first_finding:
        cid = first_finding.component_id
        pin = first_finding.pin_name
        code = first_finding.error_code
        if code == "FLOATING_PIN" and cid and pin:
            suggestions.append(f"为什么判断 {cid} 的 {pin} 悬空？")
            suggestions.append(f"我应该如何修复 {cid} 的连接？")
        elif code == "COMPONENT_SHORTED_SAME_NET" and cid:
            suggestions.append(f"为什么 {cid} 会被短接？")
            suggestions.append(f"{cid} 应该如何正确插接？")
        elif code == "POLARITY_REVERSED" and cid:
            suggestions.append(f"{cid} 的正确极性方向是什么？")
        elif cid:
            suggestions.append(f"为什么检测到 {cid} 有问题？")

    if evidence.risk_level == "danger":
        suggestions.append("我应该先检查哪些安全事项？")
    elif not suggestions:
        suggestions.append("这个电路图中都有什么元件？")
        suggestions.append("当前诊断结果是否存在风险？")

    return suggestions[:3]


def _first_ref_text(evidence: RuntimeEvidence) -> str:
    first_ref = evidence.evidence_refs[0] if evidence.evidence_refs else None
    if not first_ref:
        return "暂无 evidence_ref"
    return (
        f"{first_ref.ref_id}"
        + (f" / {first_ref.component_id}" if first_ref.component_id else "")
        + (f".{first_ref.pin_name}" if first_ref.pin_name else "")
    )


def _mentions_any_runtime_ref(evidence: RuntimeEvidence, text: str) -> bool:
    tokens: list[str] = []
    for ref in evidence.evidence_refs:
        tokens.extend(
            value
            for value in (ref.ref_id, ref.component_id, ref.pin_name, ref.hole_id)
            if value
        )
    return any(token in text for token in tokens)


# ---------------------------------------------------------------------------
# Concept-tutor / lab-guidance answer paths (no LangGraph; deterministic).
# ---------------------------------------------------------------------------

_CONCEPT_SAFETY_TRIGGERS: tuple[str, ...] = (
    "led_current_limit",
    "capacitor_filtering",
    "ohms_law",
)


def build_concept_answer(
    *,
    question: str,
    concept: ConceptPack | None,
    evidence: RuntimeEvidence | None = None,
) -> str:
    """Generate a 6-section concept_tutor answer from a local ConceptPack.

    The answer never asserts specific holes / nets / connections of the
    current circuit. The "和当前实验的关系" section either references the
    current risk_level / error_codes at a high level, or explicitly states
    that this is generic knowledge unrelated to the current circuit.
    """

    if concept is None:
        return _generic_concept_fallback(question)

    relate = _concept_relation_to_experiment(concept, evidence)
    lines: list[str] = []
    lines.append(f"直接回答：{concept.summary}")
    if concept.key_points:
        lines.append("原理解释：" + "；".join(concept.key_points))
    if concept.formulas:
        lines.append("公式：" + "；".join(concept.formulas))
    lines.append(f"和当前实验的关系：{relate}")
    if concept.common_mistakes:
        lines.append("常见错误：" + "；".join(concept.common_mistakes))
    if concept.lab_guidance:
        lines.append("如何验证：" + "；".join(concept.lab_guidance))
    safety_notes = list(concept.safety_notes)
    if concept.concept_id in _CONCEPT_SAFETY_TRIGGERS and not any(
        word in "；".join(safety_notes) for word in ("断电", "电源", "短路")
    ):
        safety_notes.append("操作前先断电，再复查电源和短路风险。")
    if safety_notes:
        lines.append("安全提醒：" + "；".join(safety_notes))
    lines.append(f"知识来源：{concept.concept_id}")
    return "\n".join(lines)


def _generic_concept_fallback(question: str) -> str:
    """Returned only when no concept matched — never invents domain facts."""
    return (
        "直接回答：本地知识库未匹配到对应概念，建议补充关键词后再次提问。\n"
        "和当前实验的关系：这是通用问题，未与当前电路状态直接关联。\n"
        "如何验证：可以查阅教材或参考权威资料对照学习。\n"
        "安全提醒：上电或调整接线前请先断电，再复查电源与短路风险。\n"
        "知识来源：concept_not_found"
    )


def _concept_relation_to_experiment(
    concept: ConceptPack,
    evidence: RuntimeEvidence | None,
) -> str:
    if evidence is None or not (evidence.findings or evidence.error_codes):
        return "这是通用知识，与当前电路状态无直接对应。"
    family_hint = ""
    error_codes = "、".join(evidence.error_codes[:2]) if evidence.error_codes else ""
    if error_codes:
        family_hint = f"当前诊断报告中出现 {error_codes}，"
    risk_hint = f"风险等级为 {evidence.risk_level}。" if evidence.risk_level else ""
    return (
        f"{family_hint}{risk_hint}"
        "该概念可作为理解上述现象的背景知识，但具体接线请以诊断结果为准。"
    )


def build_lab_guidance_answer(
    *,
    question: str,
    concept: ConceptPack | None,
    evidence: RuntimeEvidence | None = None,
) -> str:
    """Generate a numbered step-by-step lab-guidance answer with safety hint."""
    steps: list[str] = []
    if concept is not None and concept.lab_guidance:
        steps.extend(concept.lab_guidance)
    if not steps:
        steps = [
            "断电状态下检查接线是否与原理图一致。",
            "用万用表通断挡确认怀疑短路的两点是否真的导通。",
            "通电后用电压挡逐节点验证关键节点电压。",
        ]

    safety: list[str] = []
    if concept is not None:
        safety.extend(concept.safety_notes)
    if not any(
        word in "；".join(safety) for word in ("断电", "电源", "短路")
    ):
        safety.append("先断电再操作，复查电源轨与短路风险后再上电。")

    lines: list[str] = ["实验操作步骤："]
    for idx, step in enumerate(steps, start=1):
        lines.append(f"{idx}. {step}")
    lines.append("安全提醒：" + "；".join(safety))
    if concept is not None:
        lines.append(f"知识来源：{concept.concept_id}")
    return "\n".join(lines)


def build_concept_citations(
    *,
    station_id: str,
    concept: ConceptPack | None,
    tool_results: list[ToolResult],
) -> list[AngntCitation]:
    citations: list[AngntCitation] = []
    if concept is not None:
        citations.append(
            AngntCitation(
                source_type="concept_pack",
                source_id=concept.concept_id,
                title=concept.title,
                snippet=concept.summary[:260],
            )
        )
    for result in tool_results:
        citations.append(
            AngntCitation(
                source_type="diagnostic_tool",
                source_id=result.tool_name,
                title=result.tool_name,
                snippet=result.summary[:260],
            )
        )
    if not citations:
        citations.append(
            AngntCitation(
                source_type="concept_pack",
                source_id="concept_not_found",
                title="未匹配到本地概念",
                snippet="建议补充关键词后再次提问",
            )
        )
    return citations


def build_concept_evidence(
    *,
    station_id: str,
    intent: AgentIntent,
    concept: ConceptPack | None,
    tool_results: list[ToolResult],
    verification_passed: bool,
    verification_issues: list[str],
    evidence: RuntimeEvidence | None = None,
) -> list[AngntEvidence]:
    items: list[AngntEvidence] = []
    items.append(
        AngntEvidence(
            evidence_type="intent",
            source_id=f"{station_id}:intent",
            summary=f"intent={intent}",
            payload={"intent": intent},
        )
    )
    if concept is not None:
        items.append(
            AngntEvidence(
                evidence_type="concept_pack",
                source_id=concept.concept_id,
                summary=concept.title,
                payload=concept.model_dump(),
            )
        )
    if evidence is not None:
        items.append(
            AngntEvidence(
                evidence_type="runtime_evidence",
                source_id=evidence.station_id,
                summary="PCM Agent 输入证据（仅供前端展示当前电路状态）",
                payload=evidence.model_dump(),
            )
        )
    items.append(
        AngntEvidence(
            evidence_type="tool_results",
            source_id=f"{station_id}:concept_tools",
            summary="本地概念查找工具输出",
            payload={"results": [result.model_dump() for result in tool_results]},
        )
    )
    items.append(
        AngntEvidence(
            evidence_type="verification_report",
            source_id=f"{station_id}:verifier",
            summary="Reflection Node 校验结果",
            payload={"passed": verification_passed, "issues": verification_issues},
        )
    )
    return items
