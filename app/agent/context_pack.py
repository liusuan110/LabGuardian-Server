from __future__ import annotations

from app.agent.contracts import AllowedTool, ContextPack, ErrorFamily, RuntimeEvidence

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


def build_context_pack(evidence: RuntimeEvidence, *, query: str = "") -> ContextPack:
    family = classify_error_family(evidence)
    return ContextPack(
        pack_id=f"pcm_{family}_v1",
        error_family=family,
        risk_level=evidence.risk_level,
        pushed_facts=_build_pushed_facts(evidence=evidence, family=family, query=query),
        allowed_tools=_allowed_tools_for_family(family),
        prompt_rules=_prompt_rules_for_family(family),
        citation_requirements=[
            "回答必须引用 validator_report_v2 或 netlist_v2 中的具体证据。",
            "不得重新猜测元件、孔位、节点或网表事实。",
        ],
        evidence_refs=evidence.evidence_refs,
    )


def _build_pushed_facts(
    *,
    evidence: RuntimeEvidence,
    family: ErrorFamily,
    query: str,
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
    if query:
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

