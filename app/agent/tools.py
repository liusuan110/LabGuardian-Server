from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agent.contracts import RuntimeEvidence
from app.domain.board_schema import BoardSchema
from app.services.teaching_kb_service import TeachingKbService


class ToolResult(BaseModel):
    tool_name: str
    status: str = "ok"
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class NetlistTraceInput(BaseModel):
    component_id: str = ""
    pin_name: str = ""
    node_id: str = ""


class BoardSchemaLookupInput(BaseModel):
    hole_id: str = ""
    node_id: str = ""


class FaultCaseLookupInput(BaseModel):
    query: str = ""
    error_tags: list[str] = Field(default_factory=list)
    scene_id: str = "exp_first_order_rc"
    top_k: int = Field(default=3, ge=1, le=10)


class SafetyRuleLookupInput(BaseModel):
    risk_level: str = "unknown"
    error_family: str = "unknown"


class DatasheetLookupInput(BaseModel):
    component_type: str = ""
    component_id: str = ""
    query: str = ""
    error_family: str = "unknown"


LOCAL_DATASHEET_FALLBACKS: dict[str, dict[str, Any]] = {
    "led": {
        "component_type": "LED",
        "package": "two_pin_polarized",
        "pin_rules": ["anode 接高电位侧", "cathode 接低电位侧"],
        "safety_rules": ["必须串联限流电阻", "调试时优先使用低压限流电源"],
        "notes": "LED 为极性器件，反接通常不亮，缺少限流可能导致器件损坏。",
    },
    "diode": {
        "component_type": "Diode",
        "package": "two_pin_polarized",
        "pin_rules": ["anode 到 cathode 为正向导通方向", "cathode 通常由色环或标记识别"],
        "safety_rules": ["确认方向后再通电", "避免直接跨接电源轨"],
        "notes": "二极管方向错误会改变支路导通状态。",
    },
    "capacitorelectrolytic": {
        "component_type": "CapacitorElectrolytic",
        "package": "two_pin_polarized",
        "pin_rules": ["positive 接较高电位", "negative 接较低电位或 GND"],
        "safety_rules": ["通电前确认极性", "反接电解电容存在发热或损坏风险"],
        "notes": "电解电容是极性器件，长脚通常为正极，外壳负极侧常有标记。",
    },
    "resistor": {
        "component_type": "Resistor",
        "package": "two_pin_non_polarized",
        "pin_rules": ["两个引脚无极性", "应跨接到两个不同导通节点"],
        "safety_rules": ["避免两脚落在同一导通组造成元件被短接"],
        "notes": "电阻常用于限流、分压和反馈网络。",
    },
    "transistor": {
        "component_type": "Transistor",
        "package": "three_pin_polarized",
        "pin_rules": ["核对 base / collector / emitter 引脚顺序", "不同封装引脚序可能不同"],
        "safety_rules": ["先查封装方向，再接入电路", "避免把电源直接接到错误引脚"],
        "notes": "三极管引脚顺序强依赖具体型号和封装。",
    },
}


def netlist_trace_tool(
    evidence: RuntimeEvidence,
    args: NetlistTraceInput,
) -> ToolResult:
    """Trace component/pin/node facts inside runtime netlist_v2."""

    netlist = evidence.netlist_v2 or {}
    components = netlist.get("components", [])
    matched_components = []
    if isinstance(components, list):
        for component in components:
            if not isinstance(component, dict):
                continue
            if args.component_id and component.get("component_id") != args.component_id:
                continue
            matched_components.append(component)

    nets = netlist.get("nets", [])
    matched_nets = []
    if isinstance(nets, list):
        for net in nets:
            if not isinstance(net, dict):
                continue
            haystack = str(net)
            if args.node_id and args.node_id in haystack:
                matched_nets.append(net)
            elif args.component_id and args.component_id in haystack:
                matched_nets.append(net)

    summary = "未在 netlist_v2 中找到匹配项。"
    if matched_components or matched_nets:
        summary = (
            f"匹配 components={len(matched_components)}, "
            f"nets={len(matched_nets)}。"
        )
    return ToolResult(
        tool_name="netlist_trace_tool",
        summary=summary,
        payload={
            "components": matched_components[:5],
            "nets": matched_nets[:5],
        },
    )


def board_schema_lookup_tool(
    args: BoardSchemaLookupInput,
    *,
    board_schema: BoardSchema | None = None,
) -> ToolResult:
    schema = board_schema or BoardSchema.default_breadboard()
    payload: dict[str, Any] = {"schema_id": schema.schema_id}
    summaries: list[str] = []

    if args.hole_id:
        spec = schema.hole_to_spec(args.hole_id)
        payload["hole"] = {
            "hole_id": spec.hole_id,
            "electrical_node_id": spec.electrical_node_id,
            "group_type": spec.group_type,
            "row": spec.row,
            "col": spec.col,
        }
        summaries.append(f"{spec.hole_id}->{spec.electrical_node_id}")

    if args.node_id:
        matched = [
            {
                "hole_id": spec.hole_id,
                "row": spec.row,
                "col": spec.col,
                "group_type": spec.group_type,
            }
            for spec in schema.holes.values()
            if spec.electrical_node_id == args.node_id
        ]
        payload["node_holes"] = matched[:20]
        summaries.append(f"{args.node_id} contains {len(matched)} holes")

    return ToolResult(
        tool_name="board_schema_lookup_tool",
        summary="；".join(summaries) or "未提供 hole_id 或 node_id。",
        payload=payload,
    )


def fault_case_lookup_tool(
    args: FaultCaseLookupInput,
    *,
    teaching_kb_service: TeachingKbService | None = None,
) -> ToolResult:
    service = teaching_kb_service or TeachingKbService()
    cases = service.search_fault_cases(
        query=args.query,
        scene_id=args.scene_id,
        error_tags=args.error_tags,
        top_k=args.top_k,
    )
    return ToolResult(
        tool_name="fault_case_lookup_tool",
        summary=f"命中 fault_cases={len(cases)}。",
        payload={
            "fault_cases": [
                {
                    "knowledge_id": case.get("knowledge_id", ""),
                    "title": case.get("title", ""),
                    "error_tags": case.get("error_tags", []),
                    "related_error_codes": case.get("related_error_codes", []),
                    "fix_steps": case.get("fix_steps", [])[:4],
                }
                for case in cases
            ]
        },
    )


def datasheet_lookup_tool(args: DatasheetLookupInput) -> ToolResult:
    key = _datasheet_key(args.component_type or args.component_id or args.query)
    fallback = LOCAL_DATASHEET_FALLBACKS.get(key)
    if fallback is None:
        fallback = {
            "component_type": args.component_type or "unknown",
            "package": "local_fallback_unknown",
            "pin_rules": ["本地 fallback 未收录该器件，请以实物丝印和课程参考电路为准。"],
            "safety_rules": ["通电前先确认器件方向、限流条件和电源轨连接。"],
            "notes": "未访问外部 datasheet，仅返回本地保守规则。",
        }

    return ToolResult(
        tool_name="datasheet_lookup_tool",
        summary=(
            f"本地 datasheet fallback: {fallback['component_type']} / "
            f"{fallback['package']}。"
        ),
        payload={
            "provider": "local_fallback",
            "component_id": args.component_id,
            "component_type": fallback["component_type"],
            "package": fallback["package"],
            "pin_rules": fallback["pin_rules"],
            "safety_rules": fallback["safety_rules"],
            "notes": fallback["notes"],
            "matched_key": key if key in LOCAL_DATASHEET_FALLBACKS else "",
            "query": args.query,
            "error_family": args.error_family,
        },
    )


def safety_rule_lookup_tool(args: SafetyRuleLookupInput) -> ToolResult:
    rules: list[str] = []
    if args.risk_level == "danger" or args.error_family == "short_circuit":
        rules.extend(
            [
                "先断开电源，再移动导线或元件。",
                "检查电源轨 VCC/GND 是否被同一元件或导线直接连通。",
                "复查限流元件，避免 LED 或电源输出直接短路。",
            ]
        )
    else:
        rules.append("保持低压限流条件下逐项复查连接。")

    return ToolResult(
        tool_name="safety_rule_lookup_tool",
        summary="；".join(rules),
        payload={"rules": rules},
    )


def _datasheet_key(value: str) -> str:
    normalized = str(value or "").replace("_", "").replace("-", "").replace(" ", "").lower()
    if "electrolytic" in normalized or normalized.startswith("ce"):
        return "capacitorelectrolytic"
    if "led" in normalized:
        return "led"
    if "diode" in normalized:
        return "diode"
    if "resistor" in normalized or normalized.startswith("r"):
        return "resistor"
    if "transistor" in normalized or normalized.startswith("q"):
        return "transistor"
    return normalized
