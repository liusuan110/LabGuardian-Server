from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from app.domain.logical_reference import CRITICAL_ROLE_LABELS, normalize_component_type, normalize_net_role, normalize_pin_role, normalize_role_label
from .matcher import PASSIVE_TWO_PIN_TYPES, STRICT_PIN_ROLE_TYPES, _component_type_key, _component_types_equivalent, _edge_match, _find_any_isomorphism_mapping, _node_match, _role_labels_equivalent


def _result(*, logic_correct: bool, similarity: float, progress: float, message: str, items: list[dict[str, Any]], details: dict[str, Any], ref_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    items = _dedupe_items(items)
    summary: dict[str, Any] = {"total_item_count": len(items), "logic_correct": logic_correct, "similarity": round(float(max(0.0, min(1.0, similarity))), 3), "comparison_mode": "logical_graph", "ignore_component_id": True, "ignore_hole_id": True, "ignore_passive_pin_order": True, "allow_extra_wires": True, "strict_functional_pin_roles": True, "equivalence_rule": "logical_topology_with_port_semantics", "match_type": details.get("match_type"), "report_layers": {"erc": {"source": "semantic_analysis", "included": False}, "reference_compare": {"source": "s4_validate", "included": True}}}
    if ref_payload:
        summary["reference_id"] = ref_payload.get("reference_id")
        summary["reference_name"] = ref_payload.get("name")
    report = {"version": "validator_report_v2", "summary": summary, "items": items, "topology_errors": [i for i in items if i.get("error_family") in {"wiring_mismatch", "open_circuit", "extra_connection", "incomplete_circuit"}], "node_errors": [], "hole_errors": [], "component_errors": [i for i in items if i.get("error_family") in {"missing_component", "extra_component"}], "polarity_errors": []}
    return {"logic_correct": logic_correct, "is_correct": logic_correct, "is_match": logic_correct, "message": message, "similarity": report["summary"]["similarity"], "progress": round(float(max(0.0, min(1.0, progress))), 3), "items": items, "report": report, "details": details}


def _item(error_code: str, error_family: str, severity: str, message: str, *, expected: Any = None, actual: Any = None, suggested_action: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    item = {"error_code": error_code, "error_family": error_family, "severity": severity, "title": _default_title(error_code), "message": message, "expected": expected, "actual": actual, "component_ref": None, "component_actual": None, "evidence_refs": [], "suggested_action": suggested_action}
    if evidence:
        item["evidence"] = evidence
    return item


def _default_title(error_code: str) -> str:
    return {"PIN_ROLE_MISMATCH": "功能引脚错误", "SHORT_CIRCUIT": "短路", "ROLE_LABEL_MISMATCH": "端口标签错误", "UNSUPPORTED_REFERENCE_FORMAT": "不支持的参考格式", "REFERENCE_NOT_SET": "未设置参考电路", "COMPONENT_MISSING": "缺元件", "COMPONENT_EXTRA": "多余元件", "OPEN_CIRCUIT": "断路", "WRONG_CONNECTION": "错接", "EXTRA_CONNECTION": "多余连接", "INCOMPLETE_CIRCUIT": "电路未完成", "ROLE_MISMATCH": "网络角色错误", "INPUT_NODE_MISMATCH": "输入节点错误", "OUTPUT_NODE_MISMATCH": "输出节点错误", "POWER_NODE_MISMATCH": "电源节点错误", "GROUND_NODE_MISMATCH": "地节点错误", "WARN_GNN_DISAGREES_WITH_RULE": "GNN 与规则结果不一致（仅提醒）", "CRITICAL_EXTRA_CONNECTION": "关键网络上多余连接"}.get(error_code, "电路异常")


def _detailed_item(*, error_code: str, error_family: str, severity: str, message: str, expected: Any, actual: Any, component_ref: dict[str, Any] | None, component_actual: dict[str, Any] | None, evidence_refs: list[dict[str, Any]], suggested_action: str, title: str = "") -> dict[str, Any]:
    return {"error_code": error_code, "error_family": error_family, "severity": severity, "title": title or _default_title(error_code), "message": message, "expected": expected, "actual": actual, "component_ref": component_ref, "component_actual": component_actual, "evidence_refs": evidence_refs, "suggested_action": suggested_action}


def _enrich_result(result: dict[str, Any], ref_graph: nx.Graph, cur_graph: nx.Graph, ref_payload: dict[str, Any], cur_netlist_v2: dict[str, Any]) -> dict[str, Any]:
    match_type = result.get("details", {}).get("match_type")
    comp_map, net_map = _build_mappings(ref_graph, cur_graph, ref_payload, cur_netlist_v2)
    _attach_mappings(result, comp_map, net_map)
    detailed_items = _generate_detailed_items(ref_graph, cur_graph, ref_payload, cur_netlist_v2, match_type, comp_map, net_map)
    if not detailed_items:
        return result
    result["items"] = detailed_items
    if any(item.get("severity") == "error" for item in detailed_items):
        result["logic_correct"] = result["is_correct"] = result["is_match"] = False
    report = dict(result.get("report", {}))
    report["items"] = detailed_items
    report["summary"] = dict(report.get("summary", {}))
    report["summary"]["total_item_count"] = len(detailed_items)
    report["summary"]["logic_correct"] = result["logic_correct"]
    report["topology_errors"] = [i for i in detailed_items if i.get("error_family") in {"wiring_mismatch", "open_circuit", "extra_connection", "incomplete_circuit"}]
    report["component_errors"] = [i for i in detailed_items if i.get("error_family") in {"missing_component", "extra_component"}]
    result["report"] = report
    return result


def _attach_mappings(result: dict[str, Any], comp_map: dict[str, str], net_map: dict[str, str]) -> None:
    details = dict(result.get("details", {}))
    details["ref_to_current_component_mapping"] = comp_map
    details["ref_to_current_net_mapping"] = net_map
    result["details"] = details
    report = dict(result.get("report", {}))
    summary = dict(report.get("summary", {}))
    summary["ref_to_current_component_mapping"] = comp_map
    summary["ref_to_current_net_mapping"] = net_map
    report["summary"] = summary
    result["report"] = report


def _generate_detailed_items(ref_graph: nx.Graph, cur_graph: nx.Graph, ref_payload: dict[str, Any], cur_netlist_v2: dict[str, Any], match_type: str | None, comp_map: dict[str, str] | None = None, net_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    if comp_map is None or net_map is None:
        comp_map, net_map = _build_mappings(ref_graph, cur_graph, ref_payload, cur_netlist_v2)
    ref_comp_by_id = {c["ref_id"]: c for c in ref_payload.get("components", [])}
    cur_comp_by_id = {c["component_id"]: c for c in cur_netlist_v2.get("components", []) if c.get("component_type") != "Wire"}
    ref_net_roles = {n["net"]: normalize_net_role(n.get("role") or n.get("role_label") or n.get("label") or n.get("net")) for n in ref_payload.get("nets", [])}
    ref_net_labels = {n["net"]: normalize_role_label(n.get("role_label") or n.get("label") or n.get("net")) for n in ref_payload.get("nets", [])}
    cur_net_by_id = {n.get("electrical_net_id"): n for n in cur_netlist_v2.get("nets", []) if n.get("electrical_net_id")}
    items: list[dict[str, Any]] = []
    for ref_id in sorted(set(ref_comp_by_id) - set(comp_map)):
        ref_comp = ref_comp_by_id[ref_id]
        items.append(_detailed_item(error_code="COMPONENT_MISSING", error_family="missing_component", severity="error", message=f"参考电路需要元件 {ref_id}（{ref_comp.get('type')}），但当前电路中未找到。", expected={"ref_id": ref_id, "type": ref_comp.get("type")}, actual=None, component_ref={"ref_id": ref_id, "type": ref_comp.get("type")}, component_actual=None, evidence_refs=[{"type": "reference_component", "ref_id": ref_id}], suggested_action=f"请添加元件 {ref_id}（{ref_comp.get('type')}）并完成其引脚连接。"))
    mapped_current_nets = {net for net in net_map.values() if net}
    for cur_id in sorted(set(cur_comp_by_id) - set(comp_map.values())):
        cur_comp = cur_comp_by_id[cur_id]
        extra_nets = {
            pin.get("electrical_net_id")
            for pin in cur_comp.get("pins", []) or []
            if pin.get("electrical_net_id")
        }
        severity = "error" if extra_nets & mapped_current_nets else "warning"
        items.append(_detailed_item(error_code="COMPONENT_EXTRA", error_family="extra_component", severity=severity, message=f"当前电路包含多余的元件 {cur_id}（{cur_comp.get('component_type')}）。", expected=None, actual={"component_id": cur_id, "type": cur_comp.get("component_type")}, component_ref=None, component_actual={"component_id": cur_id, "type": cur_comp.get("component_type")}, evidence_refs=[{"type": "component", "component_id": cur_id}], suggested_action=f"请移除多余的元件 {cur_id}（{cur_comp.get('component_type')}）或确认是否需要它。"))
    items.extend(_wrong_connection_items(ref_payload, cur_netlist_v2, comp_map, net_map, ref_comp_by_id, cur_comp_by_id, cur_net_by_id))
    items.extend(_role_mismatch_items(ref_payload, ref_graph, cur_graph, ref_net_roles, ref_net_labels, net_map, cur_net_by_id, cur_netlist_v2))
    if not _is_topology_equivalent_match(match_type):
        items.extend(_open_circuit_items(ref_payload, cur_netlist_v2, comp_map, cur_net_by_id))
    items.extend(_short_circuit_items(ref_net_roles, ref_net_labels, net_map, cur_net_by_id))
    if not _is_topology_equivalent_match(match_type):
        items.extend(_pin_level_short_circuit_items(ref_payload, cur_netlist_v2, comp_map, cur_net_by_id))
    if match_type == "current_subgraph_in_reference":
        items.append(_detailed_item(error_code="INCOMPLETE_CIRCUIT", error_family="incomplete_circuit", severity="error", message="当前电路只匹配到参考电路的一部分，电路尚未完整实现。", expected={"reference_component_count": len(ref_comp_by_id), "reference_edge_count": ref_graph.number_of_edges()}, actual={"current_component_count": len(cur_comp_by_id), "current_edge_count": cur_graph.number_of_edges()}, component_ref=None, component_actual=None, evidence_refs=[], suggested_action="请补齐缺失的元件和连接后重新验证。"))
    if _net_count(cur_graph) < _net_count(ref_graph):
        all_cur_nets = sorted(n.get("electrical_net_id") for n in cur_netlist_v2.get("nets", []) if n.get("electrical_net_id"))
        items.append(_detailed_item(error_code="EXTRA_CONNECTION", error_family="extra_connection", severity="warning", message=f"当前电路的电气网络数（{_net_count(cur_graph)}）少于参考电路（{_net_count(ref_graph)}），可能存在不应有的短接。", expected={"net_count": _net_count(ref_graph), "separate_nets": sorted(ref_net_roles)}, actual={"net_count": _net_count(cur_graph), "merged_nets": all_cur_nets}, component_ref=None, component_actual=None, evidence_refs=[], suggested_action="请检查是否有不应相连的节点被错误地连接到了一起。"))
    if (
        match_type == "graph_edit_distance_or_fallback"
        and items
        and not any(item.get("error_code") == "WRONG_CONNECTION" for item in items)
    ):
        items.append(
            _detailed_item(
                error_code="WRONG_CONNECTION",
                error_family="wiring_mismatch",
                severity="error",
                message="检测到元件连接关系与参考电路不一致，可能存在错接。",
                expected={"edge_signatures": _edge_signatures(ref_graph)},
                actual={"edge_signatures": _edge_signatures(cur_graph)},
                component_ref=None,
                component_actual=None,
                evidence_refs=[],
                suggested_action="请检查相关元件是否连接到正确的电气节点。",
            )
        )
    return _dedupe_detailed_items(items)


def _is_topology_equivalent_match(match_type: str | None) -> bool:
    return match_type in {
        "full_isomorphism",
        "full_isomorphism_with_inferred_roles",
        "equivalent_with_allowed_symmetry",
    }


def _wrong_connection_items(ref_payload: dict[str, Any], cur_netlist_v2: dict[str, Any], comp_map: dict[str, str], net_map: dict[str, str], ref_comp_by_id: dict[str, dict[str, Any]], cur_comp_by_id: dict[str, dict[str, Any]], cur_net_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for ref_id, cur_id in comp_map.items():
        ref_comp = ref_comp_by_id.get(ref_id, {})
        cur_comp = cur_comp_by_id.get(cur_id, {})
        ctype = normalize_component_type(ref_comp.get("type"))
        connected_ref_pins = [p for p in ref_comp.get("pins", []) if p.get("nc") is not True]
        if ctype in PASSIVE_TWO_PIN_TYPES:
            ref_nets = {p["net"] for p in connected_ref_pins}
            cur_nets = {p.get("electrical_net_id") for p in cur_comp.get("pins", [])}
            mapped = {net_map.get(rn) for rn in ref_nets}
            known = {n for n in mapped if n}
            if None not in mapped and mapped != cur_nets:
                items.append(_wire_item(ref_id, cur_id, ref_comp, cur_comp, sorted(ref_nets), cur_nets, cur_net_by_id, f"{ref_id} 的连接网络与参考电路不一致。"))
            elif None in mapped and known and (cur_nets - known):
                items.append(_wire_item(ref_id, cur_id, ref_comp, cur_comp, sorted(ref_nets), cur_nets - known, cur_net_by_id, f"{ref_id} 的部分参考网络无法映射到当前电路，连接关系不完整。"))
            continue
        cur_pins = {p.get("pin_name"): p for p in cur_comp.get("pins", [])}
        cur_pins_by_role = {normalize_pin_role(ctype, p): p for p in cur_comp.get("pins", [])}
        for pin in connected_ref_pins:
            ref_pin_role = normalize_pin_role(ctype, pin)
            mapped_cur_net = net_map.get(pin.get("net"))
            cur_pin = _current_pin_for_reference_role(
                ctype=ctype,
                ref_pin_role=ref_pin_role,
                mapped_cur_net=mapped_cur_net,
                cur_pins_by_role=cur_pins_by_role,
            ) or cur_pins.get(pin.get("pin"))
            if not cur_pin:
                if ctype in STRICT_PIN_ROLE_TYPES:
                    items.append(_detailed_item(error_code="PIN_ROLE_MISMATCH", error_family="wiring_mismatch", severity="error", message=f"{ref_id}.{pin.get('pin')} 需要功能引脚 {ref_pin_role}，但当前元件 {cur_id} 未找到对应功能引脚。", expected={"ref_pin": f"{ref_id}.{pin.get('pin')}", "pin_role": ref_pin_role}, actual={"actual_component_id": cur_id, "available_pin_roles": sorted(cur_pins_by_role)}, component_ref={"ref_id": ref_id, "type": ref_comp.get("type")}, component_actual={"component_id": cur_id, "type": cur_comp.get("component_type")}, evidence_refs=[{"type": "component", "component_id": cur_id}], suggested_action=f"请重新标注 {cur_id} 的功能引脚，确保 {ref_pin_role} 接到正确网络。"))
                continue
            cur_net = cur_pin.get("electrical_net_id")
            if mapped_cur_net and cur_net != mapped_cur_net:
                actual_pin_role = normalize_pin_role(ctype, cur_pin)
                error_code = "PIN_ROLE_MISMATCH" if ctype in STRICT_PIN_ROLE_TYPES and actual_pin_role != ref_pin_role else "WRONG_CONNECTION"
                desc = _current_net_descriptor(cur_net, cur_net_by_id)
                items.append(_detailed_item(error_code=error_code, error_family="wiring_mismatch", severity="error", message=f"{ref_id}.{pin.get('pin')} 应连接到参考网络 {pin.get('net')}，但当前实际连接到 {desc['canonical_name']}。", expected={"ref_pin": f"{ref_id}.{pin.get('pin')}", "pin_role": ref_pin_role, "expected_net": pin.get("net")}, actual={"actual_component_id": cur_id, "actual_pin": cur_pin.get("pin_name"), "pin_role": actual_pin_role, "actual_net": desc, "expected_current_net": _current_net_descriptor(mapped_cur_net, cur_net_by_id), "hole_id": cur_pin.get("hole_id")}, component_ref={"ref_id": ref_id, "type": ref_comp.get("type")}, component_actual={"component_id": cur_id, "type": cur_comp.get("component_type")}, evidence_refs=[{"type": "component", "component_id": cur_id}, {"type": "net", "electrical_net_id": cur_net, "canonical_name": desc["canonical_name"]}], suggested_action=f"请将 {cur_id}.{pin.get('pin')} 从 {desc['canonical_name']} 改接到与 {pin.get('net')} 对应的网络。"))
    return items


def _current_pin_for_reference_role(
    *,
    ctype: str,
    ref_pin_role: str,
    mapped_cur_net: str | None,
    cur_pins_by_role: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if ctype == "Potentiometer" and ref_pin_role in {"terminal_a", "terminal_b"}:
        terminal_roles = ("terminal_a", "terminal_b")
        if mapped_cur_net:
            for role in terminal_roles:
                pin = cur_pins_by_role.get(role)
                if pin and pin.get("electrical_net_id") == mapped_cur_net:
                    return pin
        return cur_pins_by_role.get(ref_pin_role) or next((cur_pins_by_role.get(role) for role in terminal_roles if cur_pins_by_role.get(role)), None)
    return cur_pins_by_role.get(ref_pin_role)


def _wire_item(ref_id: str, cur_id: str, ref_comp: dict[str, Any], cur_comp: dict[str, Any], ref_nets: list[str], cur_nets: set[Any], cur_net_by_id: dict[str, dict[str, Any]], message: str) -> dict[str, Any]:
    return _detailed_item(error_code="WRONG_CONNECTION", error_family="wiring_mismatch", severity="error", message=message, expected={"ref_id": ref_id, "nets": ref_nets}, actual={"actual_component_id": cur_id, "nets": [_current_net_descriptor(n, cur_net_by_id) for n in sorted(n for n in cur_nets if n)]}, component_ref={"ref_id": ref_id, "type": ref_comp.get("type")}, component_actual={"component_id": cur_id, "type": cur_comp.get("component_type")}, evidence_refs=[{"type": "component", "component_id": cur_id}], suggested_action=f"请将 {cur_id} 改接到与 {ref_id} 对应的网络。")


def _role_mismatch_items(ref_payload: dict[str, Any], ref_graph: nx.Graph, cur_graph: nx.Graph, ref_net_roles: dict[str, str], ref_net_labels: dict[str, str], net_map: dict[str, str], cur_net_by_id: dict[str, dict[str, Any]], cur_netlist_v2: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cur_net_roles = {nid: _current_net_role(n) for nid, n in cur_net_by_id.items()}
    cur_net_labels = {nid: _current_net_label(n) for nid, n in cur_net_by_id.items()}
    code_map = {"input": "INPUT_NODE_MISMATCH", "output": "OUTPUT_NODE_MISMATCH", "power": "POWER_NODE_MISMATCH", "ground": "GROUND_NODE_MISMATCH"}
    for ref_net, ref_role in ref_net_roles.items():
        if ref_role == "signal" or not net_map.get(ref_net):
            continue
        mapped = net_map[ref_net]
        cur_role = cur_net_roles.get(mapped, "signal")
        if cur_role != ref_role:
            desc = _current_net_descriptor(mapped, cur_net_by_id)
            items.append(_detailed_item(error_code=code_map.get(ref_role, "ROLE_MISMATCH"), error_family="wiring_mismatch", severity="error", message=f"参考电路中 {ref_net} 为 {ref_role} 节点，但当前映射网络 {desc['canonical_name']} 实际为 {cur_role} 节点。", expected={"role": ref_role, "reference_net": ref_net, "pins": _ref_pins(ref_payload, ref_net)}, actual={"role": cur_role, "current_net": desc, "connected_pins": _cur_pins(cur_netlist_v2, mapped)}, component_ref=None, component_actual=None, evidence_refs=[{"type": "net", "electrical_net_id": mapped, "canonical_name": desc["canonical_name"]}], suggested_action=f"请在二维面包板图上重新点选正确的 {ref_role} 节点。"))
        ref_label = ref_net_labels.get(ref_net, "")
        cur_label = cur_net_labels.get(mapped, "")
        if ref_label in CRITICAL_ROLE_LABELS and cur_label and cur_label != ref_label:
            if not _role_labels_equivalent(ref_graph.nodes.get(f"ref_net:{ref_net}", {}), cur_graph.nodes.get(f"cur_net:{mapped}", {})):
                desc = _current_net_descriptor(mapped, cur_net_by_id)
                items.append(_detailed_item(error_code="ROLE_LABEL_MISMATCH", error_family="wiring_mismatch", severity="error", message=f"参考网络 {ref_net} 应匹配 role_label={ref_label} 的当前网络，但当前映射到了 role_label={cur_label}。", expected={"reference_net": ref_net, "role": ref_role, "role_label": ref_label}, actual={"current_net": desc, "role": cur_role, "role_label": cur_label}, component_ref=None, component_actual=None, evidence_refs=[{"type": "net", "electrical_net_id": mapped, "canonical_name": desc["canonical_name"]}], suggested_action=f"请将当前网络标注为 {ref_label}，或检查端口是否接反。"))
    return items


def _current_net_role(net: dict[str, Any]) -> str:
    manual_role = net.get("manual_role")
    if manual_role:
        return normalize_net_role(manual_role)
    label = _current_net_label(net)
    if label:
        return normalize_net_role(label)
    if net.get("power_role"):
        return normalize_net_role(net.get("power_role"))
    return normalize_net_role(net.get("role"))


def _current_net_label(net: dict[str, Any]) -> str:
    label = normalize_role_label(net.get("role_label"))
    if label:
        return label
    power_role = normalize_role_label(net.get("power_role"))
    if power_role in {"VCC", "VDD", "VEE", "VSS", "GND"}:
        return power_role
    canonical_name = normalize_role_label(net.get("canonical_name"))
    if canonical_name and not canonical_name.startswith("NET_"):
        return canonical_name
    return ""


def _ref_pins(ref_payload: dict[str, Any], ref_net: str) -> list[str]:
    return [f"{c['ref_id']}.{p['pin']}" for c in ref_payload.get("components", []) for p in c.get("pins", []) if p.get("net") == ref_net]


def _cur_pins(cur_netlist_v2: dict[str, Any], cur_net: str) -> list[str]:
    return [f"{c.get('component_id')}.{p.get('pin_name')}" for c in cur_netlist_v2.get("components", []) for p in c.get("pins", []) if p.get("electrical_net_id") == cur_net]


def _open_circuit_items(ref_payload: dict[str, Any], cur_netlist_v2: dict[str, Any], comp_map: dict[str, str], cur_net_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cur_comp_by_id = {c.get("component_id"): c for c in cur_netlist_v2.get("components", [])}
    items: list[dict[str, Any]] = []
    for ref_net_id in {p["net"] for c in ref_payload.get("components", []) for p in c.get("pins", []) if p.get("nc") is not True and p.get("net")}:
        pins = [(c["ref_id"], p["pin"]) for c in ref_payload.get("components", []) for p in c.get("pins", []) if p.get("nc") is not True and p.get("net") == ref_net_id]
        if len(pins) < 2:
            continue
        cur_nets_for_pins = []
        for ref_id, pin_name in pins:
            cur_id = comp_map.get(ref_id)
            for pin in cur_comp_by_id.get(cur_id, {}).get("pins", []):
                if pin.get("pin_name") == pin_name:
                    cur_nets_for_pins.append((cur_id, pin_name, pin.get("electrical_net_id"), pin.get("hole_id")))
        if len({n for _, _, n, _ in cur_nets_for_pins if n}) > 1:
            items.append(_detailed_item(error_code="OPEN_CIRCUIT", error_family="wiring_mismatch", severity="error", message=f"参考电路中网络 {ref_net_id} 连接的引脚在当前电路中被分到了多个不同网络。", expected={"shared_net": ref_net_id, "pins": [f"{rid}.{p}" for rid, p in pins]}, actual={"connected": False, "pins": [{"actual_component_id": cid, "actual_pin": pname, "actual_net": _current_net_descriptor(cnet, cur_net_by_id), "hole_id": hid} for cid, pname, cnet, hid in cur_nets_for_pins]}, component_ref=None, component_actual=None, evidence_refs=[{"type": "net", "electrical_net_id": cnet, "canonical_name": _current_net_descriptor(cnet, cur_net_by_id)["canonical_name"]} for _, _, cnet, _ in cur_nets_for_pins if cnet], suggested_action=f"请将上述引脚连接到同一电气网络，以构成 {ref_net_id} 连接。"))
    return items


def _short_circuit_items(ref_net_roles: dict[str, str], ref_net_labels: dict[str, str], net_map: dict[str, str], cur_net_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    mapped_by_current: dict[str, list[str]] = defaultdict(list)
    for ref_net, cur_net in net_map.items():
        if cur_net:
            mapped_by_current[cur_net].append(ref_net)
    items: list[dict[str, Any]] = []
    for cur_net, ref_nets in mapped_by_current.items():
        for idx, left in enumerate(ref_nets):
            for right in ref_nets[idx + 1:]:
                if _is_harmful_merge(left, right, ref_net_roles, ref_net_labels):
                    items.append(_short_item(left, right, cur_net, ref_nets, ref_net_labels, cur_net_by_id))
    return _dedupe_detailed_items(items)


def _pin_level_short_circuit_items(ref_payload: dict[str, Any], cur_netlist_v2: dict[str, Any], comp_map: dict[str, str], cur_net_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cur_pin_to_net = {(c.get("component_id"), p.get("pin_name")): p.get("electrical_net_id") for c in cur_netlist_v2.get("components", []) for p in c.get("pins", [])}
    ref_roles = {n["net"]: normalize_net_role(n.get("role") or n.get("role_label") or n.get("label") or n["net"]) for n in ref_payload.get("nets", [])}
    ref_labels = {n["net"]: normalize_role_label(n.get("role_label") or n.get("label") or n["net"]) for n in ref_payload.get("nets", [])}
    ref_net_pins: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for comp in ref_payload.get("components", []):
        for pin in comp.get("pins", []):
            if pin.get("nc") is not True and pin.get("net"):
                ref_net_pins[pin.get("net")].append((comp["ref_id"], pin.get("pin")))
    items: list[dict[str, Any]] = []
    sorted_nets = sorted(ref_net_pins)
    for idx, left in enumerate(sorted_nets):
        for right in sorted_nets[idx + 1:]:
            if not _is_harmful_merge(left, right, ref_roles, ref_labels):
                continue
            left_cur = {cur_pin_to_net.get((comp_map.get(rid), pin)) for rid, pin in ref_net_pins[left]} - {None}
            right_cur = {cur_pin_to_net.get((comp_map.get(rid), pin)) for rid, pin in ref_net_pins[right]} - {None}
            for cur_net in left_cur & right_cur:
                items.append(_short_item(left, right, cur_net, [left, right], ref_labels, cur_net_by_id))
    return items


def _short_item(left: str, right: str, cur_net: str, ref_nets: list[str], ref_labels: dict[str, str], cur_net_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left_label = ref_labels.get(left, normalize_role_label(left))
    right_label = ref_labels.get(right, normalize_role_label(right))
    desc = _current_net_descriptor(cur_net, cur_net_by_id)
    return _detailed_item(error_code="SHORT_CIRCUIT", error_family="extra_connection", severity="error", message=f"参考中应分离的关键网络 {left_label} 与 {right_label} 在当前电路中被合并到 {desc['canonical_name']}。", expected={"separate_nets": [left, right], "role_labels": [left_label, right_label]}, actual={"current_net": desc, "merged_reference_nets": ref_nets}, component_ref=None, component_actual=None, evidence_refs=[{"type": "net", "electrical_net_id": cur_net, "canonical_name": desc["canonical_name"]}], suggested_action="请检查是否有多余导线或元件把两个关键网络短接在一起。")


def _is_harmful_merge(left: str, right: str, ref_net_roles: dict[str, str], ref_net_labels: dict[str, str]) -> bool:
    roles = {ref_net_roles.get(left, "signal"), ref_net_roles.get(right, "signal")}
    labels = {ref_net_labels.get(left, normalize_role_label(left)), ref_net_labels.get(right, normalize_role_label(right))}
    return roles == {"power", "ground"} or {"VCC", "VEE"} <= labels or {"input", "output"} <= roles or {"UO1", "UO2"} <= labels or {"UI1", "UI2"} <= labels


def _build_mappings(ref_graph: nx.Graph, cur_graph: nx.Graph, ref_payload: dict[str, Any], cur_netlist_v2: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    iso_mapping = _find_any_isomorphism_mapping(ref_graph, cur_graph)
    if iso_mapping is not None:
        return _extract_comp_mapping(iso_mapping, ref_graph, cur_graph), _extract_net_mapping(iso_mapping, ref_graph, cur_graph)
    comp_map = _fallback_comp_mapping(ref_graph, cur_graph)
    return comp_map, _build_net_mapping(ref_graph, cur_graph, comp_map)


def _extract_net_mapping(mapping: dict[Any, Any], ref_graph: nx.Graph, cur_graph: nx.Graph) -> dict[str, str]:
    return {ref_graph.nodes[r]["source_id"]: cur_graph.nodes[c]["source_id"] for r, c in mapping.items() if ref_graph.nodes.get(r, {}).get("kind") == "net" and cur_graph.nodes.get(c, {}).get("kind") == "net"}


def _extract_comp_mapping(mapping: dict[Any, Any], ref_graph: nx.Graph, cur_graph: nx.Graph) -> dict[str, str]:
    return {ref_graph.nodes[r]["source_id"]: cur_graph.nodes[c]["source_id"] for r, c in mapping.items() if ref_graph.nodes.get(r, {}).get("kind") == "comp" and cur_graph.nodes.get(c, {}).get("kind") == "comp"}


def _fallback_comp_mapping(ref_graph: nx.Graph, cur_graph: nx.Graph) -> dict[str, str]:
    ref_by_type: dict[str, list[str]] = defaultdict(list)
    cur_by_type: dict[str, list[str]] = defaultdict(list)
    for n, data in ref_graph.nodes(data=True):
        if data.get("kind") == "comp":
            ref_by_type[_component_type_key(data.get("ctype", "UNKNOWN"))].append(n)
    for n, data in cur_graph.nodes(data=True):
        if data.get("kind") == "comp":
            cur_by_type[_component_type_key(data.get("ctype", "UNKNOWN"))].append(n)
    comp_map: dict[str, str] = {}
    for ctype, refs in ref_by_type.items():
        ref_sigs = {n: _graph_neighbor_signature(n, ref_graph) for n in refs}
        cur_sigs = {n: _graph_neighbor_signature(n, cur_graph) for n in cur_by_type.get(ctype, [])}
        for ref_node, cur_node in _greedy_match_by_score(refs, cur_by_type.get(ctype, []), ref_sigs, cur_sigs).items():
            comp_map[ref_graph.nodes[ref_node].get("source_id", ref_node)] = cur_graph.nodes[cur_node].get("source_id", cur_node)
    return comp_map


def _graph_neighbor_signature(node: str, graph: nx.Graph) -> tuple:
    neighbors = list(graph.neighbors(node))
    roles = tuple(sorted(graph.nodes[n].get("role", "signal") for n in neighbors if graph.nodes[n].get("kind") == "net"))
    pins = tuple(sorted(str((graph.get_edge_data(node, n) or {}).get("pin", "")) for n in neighbors))
    return (len(neighbors), roles, pins)


def _greedy_match_by_score(ref_nodes: list[str], cur_nodes: list[str], ref_sigs: dict[str, tuple], cur_sigs: dict[str, tuple]) -> dict[str, str]:
    scores = sorted(
        (_neighbor_signature_similarity(ref_sigs[r], cur_sigs[c]), r, c)
        for r in ref_nodes
        for c in cur_nodes
    )
    matched: dict[str, str] = {}
    used: set[str] = set()
    for score, ref_node, cur_node in sorted(scores, key=lambda item: (-item[0], item[1], item[2])):
        if score > 0 and ref_node not in matched and cur_node not in used:
            matched[ref_node] = cur_node
            used.add(cur_node)
    return matched


def _neighbor_signature_similarity(ref_sig: tuple, cur_sig: tuple) -> float:
    return (10.0 if ref_sig[0] == cur_sig[0] else 0.0) + len(set(ref_sig[1]) & set(cur_sig[1])) * 5.0 + len(set(ref_sig[2]) & set(cur_sig[2])) * 3.0


def _build_net_mapping(ref_graph: nx.Graph, cur_graph: nx.Graph, comp_map: dict[str, str]) -> dict[str, str]:
    scores: list[tuple[float, str, str]] = []
    ref_nets = [(n, d) for n, d in ref_graph.nodes(data=True) if d.get("kind") == "net"]
    cur_nets = [(n, d) for n, d in cur_graph.nodes(data=True) if d.get("kind") == "net"]
    ref_comp_order = {
        data.get("source_id"): idx
        for idx, (_node, data) in enumerate(
            (item for item in ref_graph.nodes(data=True) if item[1].get("kind") == "comp")
        )
    }
    for ref_node, ref_data in ref_nets:
        ref_neighbors = {ref_graph.nodes[n]["source_id"] for n in ref_graph.neighbors(ref_node) if ref_graph.nodes[n].get("kind") == "comp"}
        for cur_node, cur_data in cur_nets:
            cur_neighbors = {cur_graph.nodes[n]["source_id"] for n in cur_graph.neighbors(cur_node) if cur_graph.nodes[n].get("kind") == "comp"}
            matched_ref_neighbors = [
                ref_comp
                for ref_comp in ref_neighbors
                if ref_comp in comp_map and comp_map[ref_comp] in cur_neighbors
            ]
            score = float(len(matched_ref_neighbors))
            if matched_ref_neighbors:
                earliest = min(ref_comp_order.get(ref_comp, 9999) for ref_comp in matched_ref_neighbors)
                score += 0.01 / (earliest + 1)
            if str(ref_data.get("role") or "signal") == str(cur_data.get("role") or "signal"):
                score += 2.0
            ref_label = normalize_role_label(ref_data.get("role_label"))
            cur_label = normalize_role_label(cur_data.get("role_label"))
            if ref_label and cur_label and ref_label == cur_label:
                score += 4.0
            elif ref_label in CRITICAL_ROLE_LABELS and cur_label in CRITICAL_ROLE_LABELS:
                score -= 4.0
            scores.append((score, ref_data["source_id"], cur_data["source_id"]))
    net_map: dict[str, str] = {}
    used: set[str] = set()
    for score, ref_net, cur_net in sorted(scores, reverse=True):
        if score > 0 and ref_net not in net_map and cur_net not in used:
            net_map[ref_net] = cur_net
            used.add(cur_net)
    return net_map


def _missing_items(reference_graph: nx.Graph, current_graph: nx.Graph) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    ref_counts = _component_type_counts(reference_graph)
    cur_counts = _component_type_counts(current_graph)
    for ctype in sorted(ref_counts.keys() | cur_counts.keys()):
        missing = ref_counts[ctype] - cur_counts[ctype]
        if missing > 0:
            items.append(_item("COMPONENT_MISSING", "missing_component", "error", f"参考电路需要 {ref_counts[ctype]} 个 {ctype}，当前电路未匹配到 {missing} 个。", expected={"component_type": ctype, "count": ref_counts[ctype]}, actual={"component_type": ctype, "count": cur_counts[ctype]}, suggested_action=f"请检查是否漏接 {ctype}。"))
    if current_graph.number_of_edges() < reference_graph.number_of_edges():
        items.append(_item("OPEN_CIRCUIT", "open_circuit", "error", "参考电路存在当前未完成的逻辑连接，可能有断路。", expected={"edge_count": reference_graph.number_of_edges()}, actual={"edge_count": current_graph.number_of_edges()}, suggested_action="请检查相关元件引脚是否接到同一个电气节点。"))
    return items


def _extra_items(reference_graph: nx.Graph, current_graph: nx.Graph) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    ref_counts = _component_type_counts(reference_graph)
    cur_counts = _component_type_counts(current_graph)
    for ctype in sorted(ref_counts.keys() | cur_counts.keys()):
        extra = cur_counts[ctype] - ref_counts[ctype]
        if extra > 0:
            items.append(_item("COMPONENT_EXTRA", "extra_component", "warning", f"当前电路比参考电路多出 {extra} 个 {ctype}。", expected={"component_type": ctype, "count": ref_counts[ctype]}, actual={"component_type": ctype, "count": cur_counts[ctype]}, suggested_action=f"请检查是否误放了多余的 {ctype}。"))
    if current_graph.number_of_edges() > reference_graph.number_of_edges():
        items.append(_item("EXTRA_CONNECTION", "extra_connection", "warning", "当前电路包含参考电路之外的额外连接。", expected={"edge_count": reference_graph.number_of_edges()}, actual={"edge_count": current_graph.number_of_edges()}, suggested_action="请检查是否有多余导线或多余元件连接。"))
    return items


def _critical_extra_items(reference_graph: nx.Graph, current_graph: nx.Graph) -> list[dict[str, Any]]:
    """**R1 Position B (RULE_SEMANTICS §3)** — return one item per
    role-critical net that has more edges in cur than ref.

    Used by :func:`compare_logical_graphs` to promote
    ``logic_correct=True`` (lenient ``equivalent_with_extra``) to
    ``logic_correct=False`` whenever an extra component / wire actually
    touches a role-critical net (``power``, ``ground``, ``input``,
    ``output``). Extras on signal / internal nets stay as soft warnings.

    The check is **edge-count based** per net role. Pros: deterministic,
    cheap, no monomorphism mapping needed. Cons: cannot catch
    topology-preserving relabels like ``input_output_swapped`` (per-role
    degrees stay the same after swap) — that case is intentionally left
    to a follow-up (see RULE_SEMANTICS §4 Q2).
    """

    from app.domain.compare.matcher import CRITICAL_NET_ROLES

    def _role_degrees(g: nx.Graph) -> dict[str, int]:
        per_role: dict[str, int] = {}
        for node, data in g.nodes(data=True):
            if data.get("kind") != "net":
                continue
            role = normalize_net_role(data.get("role"))
            if role not in CRITICAL_NET_ROLES:
                continue
            per_role[role] = per_role.get(role, 0) + g.degree(node)
        return per_role

    ref_deg = _role_degrees(reference_graph)
    cur_deg = _role_degrees(current_graph)

    items: list[dict[str, Any]] = []
    for role in sorted(CRITICAL_NET_ROLES):
        extra = cur_deg.get(role, 0) - ref_deg.get(role, 0)
        if extra > 0:
            items.append(_item(
                "CRITICAL_EXTRA_CONNECTION",
                "extra_connection",
                "error",
                (
                    f"在关键网络（role={role}）上检测到 {extra} 条多余连接，"
                    "这可能影响电路核心功能。"
                ),
                expected={"role": role, "edge_count_on_role_nets": ref_deg.get(role, 0)},
                actual={"role": role, "edge_count_on_role_nets": cur_deg.get(role, 0)},
                suggested_action=(
                    f"请移除连接到 {role} 网络的多余元件或导线 — "
                    "电源 / 地 / 输入 / 输出 不应出现非参考要求的额外接线。"
                ),
            ))
    return items


def _difference_items(reference_graph: nx.Graph, current_graph: nx.Graph) -> list[dict[str, Any]]:
    items = _missing_items(reference_graph, current_graph) + _extra_items(reference_graph, current_graph)
    ref_net_count = _net_count(reference_graph)
    cur_net_count = _net_count(current_graph)
    if cur_net_count > ref_net_count:
        items.append(_item("OPEN_CIRCUIT", "open_circuit", "error", "当前电路的电气网络比参考电路更多，可能存在应相连但未相连的断点。", expected={"net_count": ref_net_count}, actual={"net_count": cur_net_count}, suggested_action="请检查应共用节点的元件引脚是否断开。"))
    if cur_net_count < ref_net_count:
        items.append(_item("EXTRA_CONNECTION", "extra_connection", "error", "当前电路的电气网络比参考电路更少，可能存在额外短接。", expected={"net_count": ref_net_count}, actual={"net_count": cur_net_count}, suggested_action="请检查是否把不应相连的节点接在一起。"))
    items.append(_item("WRONG_CONNECTION", "wiring_mismatch", "error", "检测到元件连接关系与参考电路不一致，可能存在错接。", expected={"edge_signatures": _edge_signatures(reference_graph)}, actual={"edge_signatures": _edge_signatures(current_graph)}, suggested_action="请检查相关元件是否连接到正确的电气节点。"))
    return _dedupe_items(items)


def _ged_similarity(reference_graph: nx.Graph, current_graph: nx.Graph) -> float:
    graph_size = max(reference_graph.number_of_nodes() + reference_graph.number_of_edges(), current_graph.number_of_nodes() + current_graph.number_of_edges(), 1)
    if reference_graph.number_of_nodes() > 30 or current_graph.number_of_nodes() > 30:
        return _approximate_similarity(reference_graph, current_graph)
    try:
        best = None
        for ged in nx.optimize_graph_edit_distance(reference_graph, current_graph, node_subst_cost=_node_subst_cost, node_del_cost=lambda _a: 1.0, node_ins_cost=lambda _a: 1.0, edge_subst_cost=lambda _a, _b: 0.0, edge_del_cost=lambda _a: 1.0, edge_ins_cost=lambda _a: 1.0, timeout=0.25):
            best = ged
            break
        return _approximate_similarity(reference_graph, current_graph) if best is None else max(0.0, min(1.0, 1.0 - float(best) / graph_size))
    except Exception:
        return _approximate_similarity(reference_graph, current_graph)


def _node_subst_cost(a: dict[str, Any], b: dict[str, Any]) -> float:
    if a.get("kind") != b.get("kind"):
        return 2.0
    if a.get("kind") == "comp":
        return 0.0 if _component_types_equivalent(a.get("ctype"), b.get("ctype")) else 1.5
    return 0.0 if _node_match(a, b) else 1.0


def _approximate_similarity(reference_graph: nx.Graph, current_graph: nx.Graph) -> float:
    ref_counts = _component_type_counts(reference_graph)
    cur_counts = _component_type_counts(current_graph)
    types = ref_counts.keys() | cur_counts.keys()
    if not types and reference_graph.number_of_edges() == current_graph.number_of_edges():
        return 1.0
    type_score = 1.0 - sum(abs(ref_counts[t] - cur_counts[t]) for t in types) / (sum(ref_counts.values()) + sum(cur_counts.values()) or 1)
    ref_edges = Counter(_edge_signatures(reference_graph))
    cur_edges = Counter(_edge_signatures(current_graph))
    edges = ref_edges.keys() | cur_edges.keys()
    edge_score = 1.0 - sum(abs(ref_edges[e] - cur_edges[e]) for e in edges) / (sum(ref_edges.values()) + sum(cur_edges.values()) or 1)
    net_score = 1.0 - abs(_net_count(reference_graph) - _net_count(current_graph)) / max(_net_count(reference_graph), _net_count(current_graph), 1)
    return max(0.0, min(1.0, 0.45 * type_score + 0.4 * edge_score + 0.15 * net_score))


def _component_progress(reference_graph: nx.Graph, current_graph: nx.Graph) -> float:
    ref_counts = _component_type_counts(reference_graph)
    cur_counts = _component_type_counts(current_graph)
    return max(0.0, min(1.0, sum(min(count, cur_counts.get(ctype, 0)) for ctype, count in ref_counts.items()) / (sum(ref_counts.values()) or 1)))


def _component_type_counts(graph: nx.Graph) -> Counter[str]:
    return Counter(_component_type_key(data.get("ctype") or "UNKNOWN") for _, data in graph.nodes(data=True) if data.get("kind") == "comp")


def _component_count(graph: nx.Graph) -> int:
    return sum(1 for _, data in graph.nodes(data=True) if data.get("kind") == "comp")


def _net_count(graph: nx.Graph) -> int:
    return sum(1 for _, data in graph.nodes(data=True) if data.get("kind") == "net")


def _edge_signatures(graph: nx.Graph) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for u, v in graph.edges:
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        comp_data = u_data if u_data.get("kind") == "comp" else v_data
        net_data = v_data if u_data.get("kind") == "comp" else u_data
        out.append((_component_type_key(comp_data.get("ctype") or "UNKNOWN"), str(net_data.get("role") or "signal")))
    return sorted(out)


def _current_net_descriptor(net_id: Any, cur_net_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_id = str(net_id or "")
    net = cur_net_by_id.get(source_id, {})
    canonical = str(net.get("canonical_name") or net.get("role_label") or net.get("power_role") or source_id)
    aliases = [str(item) for item in net.get("aliases", []) or [] if str(item)]
    if canonical and canonical not in aliases:
        aliases.insert(0, canonical)
    return {"source_id": source_id, "canonical_name": canonical, "role": normalize_net_role(net.get("role") or net.get("manual_role") or canonical), "role_label": normalize_role_label(net.get("role_label") or net.get("power_role") or canonical), "aliases": list(dict.fromkeys(aliases)), "merged_source_ids": list(net.get("merged_source_ids", []) or [])}


def _dedupe_detailed_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = f"{item.get('error_code')}:{item.get('message')}"
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = (str(item.get("error_code")), str(item.get("message")))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out
