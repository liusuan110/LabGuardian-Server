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
    # WP-1 (2026-05-24): scene_id must come from the resolved topology
    # context. When empty, skip MRag pack assembly entirely — silently
    # defaulting to RC (the old library default) would pull RC fault
    # cases into VLM micro-defect prompts for, say, UA741 questions.
    scene_id = (state.runtime_evidence.current_scene_id or "").strip()
    if scene_id:
        try:
            pack = mrag_service.build_pack(
                scene_id=scene_id,
                error_tags=state.runtime_evidence.error_tags,
                query=state.query,
                circuit_snapshot=state.runtime_evidence.circuit_snapshot or "",
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("MRag pack assembly failed: %s", exc)
    else:
        logger.debug(
            "vlm_explain: empty current_scene_id, skipping MRag pack (WP-1)."
        )

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


def _default_micro_conclusion(defect_type: str) -> str:
    if defect_type == MicroDefectType.BURN_MARK.value:
        return "建议近距离检查是否存在焦黑、变色或过热痕迹（元件本体/导线/供电端附近）。"
    if defect_type == MicroDefectType.UNSTRIPPED_WIRE.value:
        return "建议检查跳线是否剥皮到位、铜芯是否真正插入孔位并与金属夹片接触。"
    if defect_type == MicroDefectType.COLD_SOLDER.value:
        return "建议检查是否存在虚焊/冷焊：焊点暗灰粗糙、裂纹、轻晃就断续导通（如有焊点）。"
    return "建议做一次近距离目检，排除肉眼可见的接触不良或损坏。"


def _default_micro_steps(defect_type: str) -> list[str]:
    if defect_type == MicroDefectType.BURN_MARK.value:
        return [
            "断电后用强光侧照检查：元件壳体、导线绝缘层、面包板/PCB 是否有焦黑/融化/变色。",
            "闻是否有焦糊味；触摸前确认已断电并等待发热器件冷却。",
            "如果发现可疑元件，先从该元件/那根跳线开始做“减法”：暂时移除或更换，再复测是否恢复。",
        ]
    if defect_type == MicroDefectType.UNSTRIPPED_WIRE.value:
        return [
            "检查每根跳线两端：剥皮长度足够、铜芯裸露且插入到孔位金属夹片里，不是绝缘皮顶住孔口。",
            "轻拉测试：轻拉不会松动；必要时更换更硬的跳线或重新压接杜邦线端子。",
            "断电用万用表蜂鸣档测导线两端是否真导通，并确认不该导通的两点没有被误短接。",
        ]
    if defect_type == MicroDefectType.COLD_SOLDER.value:
        return [
            "如果有焊点：观察是否光亮圆润；暗灰粗糙/裂纹/空洞常见于虚焊或冷焊。",
            "轻轻晃动元件引脚，观察是否出现“时通时断”；必要时重新补焊并清理助焊剂残留。",
            "补焊后断电用通断档复测关键连通关系，再上电验证现象是否改善。",
        ]
    return [
        "断电后近距离检查可疑区域（松动、折断、接触不良、压线、挤压导致短接）。",
        "用通断档确认关键连线可靠，再逐项恢复上电验证。",
    ]


def _normalize_micro_finding(finding: VlmFinding) -> VlmFinding:
    generic = "一阶 rc 实验现象需要结合规则结果排查"
    conclusion = (finding.conclusion or "").strip()
    if not conclusion or conclusion.strip().lower() == generic:
        finding.conclusion = _default_micro_conclusion(finding.defect_type)
    if not finding.fix_steps:
        finding.fix_steps = _default_micro_steps(finding.defect_type)
    return finding


def _append_micro_section(existing_draft: str, findings: list[VlmFinding]) -> str:
    if not findings:
        return existing_draft
    if _SECTION_HEADER.strip() in (existing_draft or ""):
        return existing_draft  # already appended in a prior pass
    lines = [_SECTION_HEADER]
    for raw in findings:
        f = _normalize_micro_finding(raw)
        head = f"- [{f.defect_type}] {f.conclusion or '未发现明显异常'}"
        lines.append(head)
        steps = list(f.fix_steps or [])[:3]
        for idx, step in enumerate(steps, start=1):
            lines.append(f"  {idx}) {step}")
    return (existing_draft or "") + "\n".join(lines)
