from __future__ import annotations

import json
import math

from app.agent.contracts import (
    AllowedTool,
    ContextPack,
    ContextPackMetrics,
    ErrorFamily,
    RuntimeEvidence,
)
from app.services.circuit_kb_service import looks_like_circuit_query

ERROR_CODE_TO_FAMILY: dict[str, ErrorFamily] = {
    "COMPONENT_SHORTED_SAME_NET": "short_circuit",
    "NODE_MISMATCH": "wiring_mismatch",
    "HOLE_MISMATCH": "wiring_mismatch",
    "FLOATING_PIN": "wiring_mismatch",
    "POLARITY_REVERSED": "polarity_error",
    "POLARITY_UNKNOWN": "polarity_error",
    "LED_SERIES_RESISTOR_MISSING": "missing_protection",
    "COMPONENT_MISSING": "missing_component",
    "COMPONENT_INSTANCE_MISSING": "missing_component",
    "PIN_MISSING": "incomplete_circuit",
    "MULTIPLE_DISCONNECTED_SUBGRAPHS": "incomplete_circuit",
    "TOPOLOGY_VALID_SUBSET": "incomplete_circuit",
}


def classify_error_family(evidence: RuntimeEvidence) -> ErrorFamily:
    for code in evidence.error_codes:
        family = ERROR_CODE_TO_FAMILY.get(code)
        if family:
            return family
    if "probe_mode_error" in evidence.error_tags:
        return "measurement_error"
    if "scope_ground_or_short_risk" in evidence.error_tags:
        return "short_circuit"
    return "unknown"


def build_context_pack(evidence: RuntimeEvidence, *, query: str = "", user_message: str = "") -> ContextPack:
    family = classify_error_family(evidence)
    allow_tools = _allowed_tools_for_family(family)
    merged_query = (user_message or query or "").strip().lower()
    if merged_query and _looks_like_datasheet_query(merged_query, evidence):
        if not any(tool.name == "datasheet_lookup_tool" for tool in allow_tools):
            allow_tools.append(
                AllowedTool(
                    name="datasheet_lookup_tool",
                    reason="用户在问数据手册/引脚/参数相关问题，允许检索本地 datasheet PDF。",
                    required=False,
                )
            )

    # Conditional gate: circuit knowledge base for schematic-level theory questions.
    # Only added when the query contains circuit-domain keywords; the tool itself
    # has a further relevance check, so false positives from the keyword gate are
    # harmless (they just result in an empty hit).
    if merged_query and looks_like_circuit_query(merged_query):
        if not any(tool.name == "circuit_lookup_tool" for tool in allow_tools):
            allow_tools.append(
                AllowedTool(
                    name="circuit_lookup_tool",
                    reason="用户在问电路原理/拓扑/元件作用相关问题，允许检索本地电路知识库。",
                    required=False,
                )
            )
    pack = ContextPack(
        pack_id=f"pcm_{family}_v1",
        error_family=family,
        risk_level=evidence.risk_level,
        pushed_facts=_build_pushed_facts(evidence=evidence, family=family, query=query, user_message=user_message),
        allowed_tools=allow_tools,
        prompt_rules=_prompt_rules_for_family(family),
        citation_requirements=[
            "回答必须引用 validator_report_v2 或 netlist_v2 中的具体证据。",
            "不得重新猜测元件、孔位、节点或网表事实。",
        ],
        evidence_refs=evidence.evidence_refs,
        history_facts=evidence.history_facts,
        history_summary=evidence.history_summary,
    )
    pack.metrics = estimate_context_pack_metrics(pack)
    return pack


def _looks_like_datasheet_query(msg: str, evidence: RuntimeEvidence) -> bool:
    """Phase 4: defer to the YAML-defined ``datasheet`` route.

    The router combines:
    - auto-fire on known part numbers,
    - embedding-based pos/neg utterance scoring (when DATASHEET_EMBEDDING_BACKEND
      is active and bge has encoded the utterances), and
    - deterministic keyword overlap as a fallback / safety net.

    If the router's ``datasheet`` route fires, we surface the tool. Otherwise,
    as a last-resort path that the old keyword-only check used to cover, we
    let an explicit part_subtype / part_number / chip / ic mention in
    ``netlist_v2.components`` re-enable the tool — this preserves the case
    where a user names a chip currently on the board without using any of
    the datasheet keywords.
    """

    from app.agent.router import get_router  # local import: avoid cycles

    router = get_router()
    if router.has_route("datasheet"):
        decision = router.decide("datasheet", msg)
        if decision.fired:
            return True
        if decision.matched_via == "embedding":
            # Embedding said no with negative-utterance veto; trust it.
            return False
        # No router signal yet — fall through to the netlist-mention path.

    for comp in (evidence.netlist_v2 or {}).get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        subtype = str(comp.get("part_subtype") or "").strip().lower()
        if subtype and len(subtype) >= 3 and subtype in msg:
            return True
        meta = comp.get("metadata", {}) if isinstance(comp.get("metadata"), dict) else {}
        for key in ("part_number", "model", "chip", "ic", "label", "name"):
            value = str(meta.get(key) or "").strip().lower()
            if value and len(value) >= 3 and value in msg:
                return True
    return False


def estimate_context_pack_metrics(pack: ContextPack) -> ContextPackMetrics:
    payload = {
        "pack_id": pack.pack_id,
        "error_family": pack.error_family,
        "risk_level": pack.risk_level,
        "pushed_facts": pack.pushed_facts,
        "allowed_tools": [tool.model_dump() for tool in pack.allowed_tools],
        "prompt_rules": pack.prompt_rules,
        "citation_requirements": pack.citation_requirements,
        "evidence_refs": [ref.model_dump() for ref in pack.evidence_refs],
        "history_facts": pack.history_facts,
        "history_summary": pack.history_summary,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    char_count = len(serialized)
    history_serialized = json.dumps(
        {
            "history_facts": pack.history_facts,
            "history_summary": pack.history_summary,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    history_char_count = len(history_serialized)
    return ContextPackMetrics(
        pushed_facts_count=len(pack.pushed_facts),
        allowed_tool_count=len(pack.allowed_tools),
        evidence_ref_count=len(pack.evidence_refs),
        history_facts_count=len(pack.history_facts),
        history_char_count=history_char_count,
        history_estimated_tokens=(
            max(1, math.ceil(history_char_count / 4)) if pack.history_facts else 0
        ),
        char_count=char_count,
        estimated_tokens=max(1, math.ceil(char_count / 4)),
    )


def _build_pushed_facts(
    *,
    evidence: RuntimeEvidence,
    family: ErrorFamily,
    query: str,
    user_message: str,
) -> list[str]:
    facts = [
        f"station_id={evidence.station_id}",
        f"risk_level={evidence.risk_level}",
        f"error_family={family}",
    ]
    if evidence.error_codes:
        facts.append("error_codes=" + ",".join(evidence.error_codes))
    if evidence.error_tags:
        facts.append("error_tags=" + ",".join(evidence.error_tags))
    if user_message:
        facts.append(f"user_message={user_message}")
    elif query:
        facts.append(f"user_query={query}")
    for finding in evidence.findings[:3]:
        parts = [finding.error_code]
        if finding.component_id:
            parts.append(f"component={finding.component_id}")
        if finding.pin_name:
            parts.append(f"pin={finding.pin_name}")
        if finding.expected is not None:
            parts.append(f"expected={finding.expected}")
        if finding.actual is not None:
            parts.append(f"actual={finding.actual}")
        facts.append("finding:" + ";".join(parts))
    if evidence.circuit_snapshot:
        facts.append("circuit_snapshot=" + evidence.circuit_snapshot[:300])
    facts.extend(f"history:{fact}" for fact in evidence.history_facts[:5])
    return facts


def _allowed_tools_for_family(family: ErrorFamily) -> list[AllowedTool]:
    common = [
        AllowedTool(
            name="fault_case_lookup_tool",
            reason="检索与当前错误类型匹配的本地教学故障知识。",
            required=True,
        )
    ]
    if family == "short_circuit":
        return [
            AllowedTool(
                name="netlist_trace_tool",
                reason="追踪短路元件两端是否落在同一 electrical net。",
                required=True,
            ),
            AllowedTool(
                name="board_schema_lookup_tool",
                reason="确认相关孔位所在的静态导通节点和电源轨分段。",
            ),
            AllowedTool(
                name="safety_rule_lookup_tool",
                reason="推送断电、限流和电源轨复查规则。",
                required=True,
            ),
            AllowedTool(
                name="heatmap_overlay_tool",
                reason="后续用于生成短路风险可视化热力图。",
            ),
            *common,
        ]
    if family == "wiring_mismatch":
        return [
            AllowedTool(
                name="board_schema_lookup_tool",
                reason="解释 expected/actual hole 或 node 的面包板导通关系。",
                required=True,
            ),
            AllowedTool(
                name="netlist_trace_tool",
                reason="追踪错误连接影响到的网表连接。",
            ),
            *common,
        ]
    if family == "polarity_error":
        return [
            AllowedTool(
                name="netlist_trace_tool",
                reason="定位被反接的元件和引脚。",
                required=True,
            ),
            AllowedTool(
                name="datasheet_lookup_tool",
                reason="查找极性元件引脚或封装规则。",
                required=True,
            ),
            *common,
        ]
    if family == "missing_protection":
        return [
            AllowedTool(
                name="datasheet_lookup_tool",
                reason="查找 LED、二极管或保护器件的基础安全规则。",
                required=True,
            ),
            AllowedTool(
                name="safety_rule_lookup_tool",
                reason="推送限流和低压复查规则。",
            ),
            *common,
        ]
    return common


def _prompt_rules_for_family(family: ErrorFamily) -> list[str]:
    rules = [
        "先给结论，再给证据，再给修改步骤。",
        "只基于推送事实和工具结果回答。",
    ]
    if family == "short_circuit":
        rules.append("若 risk_level=danger，必须优先提醒断电复查。")
    if family == "wiring_mismatch":
        rules.append("必须区分 hole_id 错误和 electrical_node_id 错误。")
    if family == "polarity_error":
        rules.append("必须说明需要核对正负极或器件引脚方向。")
    return rules
