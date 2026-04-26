from __future__ import annotations

from app.agent.contracts import ContextPack, RuntimeEvidence
from app.agent.tools import ToolResult
from app.agent.verification import verify_draft_answer
from app.schemas.angnt import AngntCitation, AngntEvidence


def build_diagnostic_template_answer(
    *,
    station_id: str,
    query: str,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
) -> str:
    error_codes = "、".join(evidence.error_codes) or "暂无结构化错误码"
    ref_text = _first_ref_text(evidence)
    tool_summary = "；".join(result.summary for result in tool_results if result.summary)
    conclusion = diagnostic_conclusion(evidence=evidence, context_pack=context_pack)
    fix_steps = extract_fix_steps(tool_results) or ["按 validator 证据逐项复查连接。"]

    lines = [
        f"工位 {station_id} 诊断结论：{conclusion}",
        f"问题：{query or '未提供额外问题'}。",
        f"错误码：{error_codes}。",
        f"证据：{ref_text}。",
    ]
    if tool_summary:
        lines.append(f"工具结果：{tool_summary}。")
    if evidence.risk_level == "danger":
        lines.append("安全提示：请先断电，再复查电源轨和相关元件连接。")
    lines.append("修改步骤：" + "；".join(fix_steps[:4]))
    return "\n".join(lines)


def repair_diagnostic_answer(
    *,
    draft_answer: str,
    evidence: RuntimeEvidence,
    verification_issues: list[str],
) -> str:
    repaired = draft_answer
    if evidence.error_codes and not any(code in repaired for code in evidence.error_codes):
        repaired += "\n错误码：" + "、".join(evidence.error_codes) + "。"
    if evidence.evidence_refs:
        repaired += "\n补充证据：" + _first_ref_text(evidence) + "。"
    if evidence.risk_level == "danger" and not any(
        word in repaired for word in ("断电", "电源", "短路")
    ):
        repaired += "\n安全提示：请先断电，再复查电源轨和短路风险。"
    if verification_issues:
        repaired += "\n自检修正：" + "；".join(verification_issues) + "。"
    return repaired


def build_verified_diagnostic_answer(
    *,
    station_id: str,
    query: str,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
) -> tuple[str, bool, list[str]]:
    draft = build_diagnostic_template_answer(
        station_id=station_id,
        query=query,
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
) -> list[AngntEvidence]:
    return [
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


def _first_ref_text(evidence: RuntimeEvidence) -> str:
    first_ref = evidence.evidence_refs[0] if evidence.evidence_refs else None
    if not first_ref:
        return "暂无 evidence_ref"
    return (
        f"{first_ref.ref_id}"
        + (f" / {first_ref.component_id}" if first_ref.component_id else "")
        + (f".{first_ref.pin_name}" if first_ref.pin_name else "")
    )
