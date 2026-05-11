from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from app.domain.logical_reference import (
    CRITICAL_ROLE_LABELS,
    normalize_component_type,
    normalize_net_role,
    normalize_pin_role,
    normalize_role_label,
)


STRICT_NET_ROLES = {"ground", "power", "input", "output"}
PASSIVE_TWO_PIN_TYPES = {"Resistor", "Capacitor", "CapacitorCeramic", "Wire"}
STRICT_PIN_ROLE_TYPES = {"Transistor", "Potentiometer", "LED", "Diode", "CapacitorElectrolytic"}


def compare_logical_graphs(
    reference_graph: nx.Graph,
    current_graph: nx.Graph,
    *,
    ref_payload: dict[str, Any] | None = None,
    cur_netlist_v2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """比较参考电路图与当前电路图，返回包含差异报告的比较结果。

    当提供 ref_payload 和 cur_netlist_v2 时，会生成带有 expected/actual
    细节的 enriched error items，便于前端精确定位错误。
    """
    iso_mapping = _find_isomorphism(reference_graph, current_graph)
    if iso_mapping is not None:
        match_type = "full_isomorphism"
        if _mapping_uses_allowed_symmetry(iso_mapping, reference_graph, current_graph):
            match_type = "equivalent_with_allowed_symmetry"
        result = _result(
            logic_correct=True,
            similarity=1.0,
            progress=1.0,
            message="电路逻辑连接与参考电路一致",
            items=[],
            details={"match_type": match_type},
            ref_payload=ref_payload,
        )
        if ref_payload is not None and cur_netlist_v2 is not None:
            result = _enrich_result(
                result, reference_graph, current_graph, ref_payload, cur_netlist_v2
            )
        return result

    if _contains_subgraph(current_graph, reference_graph):
        items = _extra_items(reference_graph, current_graph)
        if not items:
            items = [
                _item(
                    "EXTRA_CONNECTION",
                    "extra_connection",
                    "warning",
                    "参考电路逻辑已存在，但当前电路包含额外连接。",
                    expected={},
                    actual={},
                    suggested_action="请检查是否有多余连接。",
                )
            ]
        result = _result(
            logic_correct=True,
            similarity=max(0.85, _approximate_similarity(reference_graph, current_graph)),
            progress=1.0,
            message="参考电路逻辑已存在，但当前电路包含额外元件或连接",
            items=items,
            details={"match_type": "equivalent_with_extra"},
            ref_payload=ref_payload,
        )
        if ref_payload is not None and cur_netlist_v2 is not None:
            result = _enrich_result(
                result, reference_graph, current_graph, ref_payload, cur_netlist_v2
            )
        return result

    if _contains_subgraph(reference_graph, current_graph):
        items = _missing_items(reference_graph, current_graph)
        items.append(
            _item(
                "INCOMPLETE_CIRCUIT",
                "incomplete_circuit",
                "error",
                "当前电路只匹配到参考电路的一部分。",
                expected={"reference_component_count": _component_count(reference_graph)},
                actual={"current_component_count": _component_count(current_graph)},
                suggested_action="请补齐缺失元件或连接后重新验证。",
            )
        )
        result = _result(
            logic_correct=False,
            similarity=_approximate_similarity(reference_graph, current_graph),
            progress=_component_progress(reference_graph, current_graph),
            message="当前电路未完整实现参考电路逻辑",
            items=_dedupe_items(items),
            details={"match_type": "current_subgraph_in_reference"},
            ref_payload=ref_payload,
        )
        if ref_payload is not None and cur_netlist_v2 is not None:
            result = _enrich_result(
                result, reference_graph, current_graph, ref_payload, cur_netlist_v2
            )
        return result

    similarity = _ged_similarity(reference_graph, current_graph)
    items = _difference_items(reference_graph, current_graph)
    result = _result(
        logic_correct=False,
        similarity=similarity,
        progress=_component_progress(reference_graph, current_graph),
        message="检测到元件连接关系与参考电路不一致，可能存在错接。",
        items=items,
        details={"match_type": "graph_edit_distance_or_fallback"},
        ref_payload=ref_payload,
    )
    if ref_payload is not None and cur_netlist_v2 is not None:
        result = _enrich_result(
            result, reference_graph, current_graph, ref_payload, cur_netlist_v2
        )
    return result


def _is_isomorphic(reference_graph: nx.Graph, current_graph: nx.Graph) -> bool:
    return _find_isomorphism(reference_graph, current_graph) is not None


def _find_isomorphism(reference_graph: nx.Graph, current_graph: nx.Graph) -> dict[Any, Any] | None:
    if reference_graph.number_of_nodes() != current_graph.number_of_nodes():
        return None
    if reference_graph.number_of_edges() != current_graph.number_of_edges():
        return None
    matcher = GraphMatcher(
        reference_graph,
        current_graph,
        node_match=_node_match,
        edge_match=_edge_match,
    )
    if not matcher.is_isomorphic():
        return None
    return next(matcher.isomorphisms_iter())


def _mapping_uses_allowed_symmetry(
    mapping: dict[Any, Any],
    reference_graph: nx.Graph,
    current_graph: nx.Graph,
) -> bool:
    for ref_node, cur_node in mapping.items():
        ref_data = reference_graph.nodes.get(ref_node, {})
        cur_data = current_graph.nodes.get(cur_node, {})
        if ref_data.get("kind") != "net" or cur_data.get("kind") != "net":
            continue
        ref_label = normalize_role_label(ref_data.get("role_label"))
        cur_label = normalize_role_label(cur_data.get("role_label"))
        if ref_label and cur_label and ref_label != cur_label and _role_labels_equivalent(ref_data, cur_data):
            return True
    return False


def _contains_subgraph(container: nx.Graph, pattern: nx.Graph) -> bool:
    if pattern.number_of_nodes() > container.number_of_nodes():
        return False
    if pattern.number_of_edges() > container.number_of_edges():
        return False
    return GraphMatcher(
        container,
        pattern,
        node_match=_node_match,
        edge_match=_edge_match,
    ).subgraph_is_isomorphic()


def _node_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("kind") != b.get("kind"):
        return False
    if a.get("kind") == "comp":
        return a.get("ctype") == b.get("ctype")
    role_a = str(a.get("role") or "signal")
    role_b = str(b.get("role") or "signal")
    if role_a in STRICT_NET_ROLES or role_b in STRICT_NET_ROLES:
        if role_a != role_b:
            return False
    label_a = normalize_role_label(a.get("role_label"))
    label_b = normalize_role_label(b.get("role_label"))
    if label_a in CRITICAL_ROLE_LABELS or label_b in CRITICAL_ROLE_LABELS:
        return _role_labels_equivalent(a, b)
    return True


def _edge_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    comp_type_a = a.get("comp_type", "")
    comp_type_b = b.get("comp_type", "")
    if comp_type_a in PASSIVE_TWO_PIN_TYPES and comp_type_b in PASSIVE_TWO_PIN_TYPES:
        return True
    pin_a = str(a.get("pin_role") or a.get("pin") or "")
    pin_b = str(b.get("pin_role") or b.get("pin") or "")
    if comp_type_a in STRICT_PIN_ROLE_TYPES or comp_type_b in STRICT_PIN_ROLE_TYPES:
        return pin_a == pin_b
    return pin_a == pin_b


def _role_labels_equivalent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    label_a = normalize_role_label(a.get("role_label"))
    label_b = normalize_role_label(b.get("role_label"))
    if not label_a or not label_b:
        return False
    if label_a == label_b:
        return True
    allowed_a = {normalize_role_label(value) for value in a.get("allowed_role_labels", []) or []}
    allowed_b = {normalize_role_label(value) for value in b.get("allowed_role_labels", []) or []}
    return label_b in allowed_a or label_a in allowed_b


def _result(
    *,
    logic_correct: bool,
    similarity: float,
    progress: float,
    message: str,
    items: list[dict[str, Any]],
    details: dict[str, Any],
    ref_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    items = _dedupe_items(items)
    summary: dict[str, Any] = {
        "total_item_count": len(items),
        "logic_correct": logic_correct,
        "similarity": round(float(max(0.0, min(1.0, similarity))), 3),
        "comparison_mode": "logical_graph",
        "ignore_component_id": True,
        "ignore_hole_id": True,
        "ignore_passive_pin_order": True,
        "strict_functional_pin_roles": True,
        "equivalence_rule": "logical_topology_with_port_semantics",
        "match_type": details.get("match_type"),
    }
    if ref_payload:
        summary["reference_id"] = ref_payload.get("reference_id")
        summary["reference_name"] = ref_payload.get("name")

    report = {
        "version": "validator_report_v2",
        "summary": summary,
        "items": items,
        "topology_errors": [
            item for item in items if item.get("error_family") in {"wiring_mismatch", "open_circuit", "extra_connection", "incomplete_circuit"}
        ],
        "node_errors": [],
        "hole_errors": [],
        "component_errors": [
            item for item in items if item.get("error_family") in {"missing_component", "extra_component"}
        ],
        "polarity_errors": [],
    }
    return {
        "logic_correct": logic_correct,
        "is_correct": logic_correct,
        "is_match": logic_correct,
        "message": message,
        "similarity": report["summary"]["similarity"],
        "progress": round(float(max(0.0, min(1.0, progress))), 3),
        "items": items,
        "report": report,
        "details": details,
    }


def _item(
    error_code: str,
    error_family: str,
    severity: str,
    message: str,
    *,
    expected: Any = None,
    actual: Any = None,
    suggested_action: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = {
        "error_code": error_code,
        "error_family": error_family,
        "severity": severity,
        "title": _default_title(error_code),
        "message": message,
        "expected": expected,
        "actual": actual,
        "component_ref": None,
        "component_actual": None,
        "evidence_refs": [],
        "suggested_action": suggested_action,
    }
    if evidence:
        item["evidence"] = evidence
    return item


def _default_title(error_code: str) -> str:
    return {
        "PIN_ROLE_MISMATCH": "功能引脚错误",
        "SHORT_CIRCUIT": "短路",
        "ROLE_LABEL_MISMATCH": "端口标签错误",
        "UNSUPPORTED_REFERENCE_FORMAT": "不支持的参考格式",
        "REFERENCE_NOT_SET": "未设置参考电路",
        "COMPONENT_MISSING": "缺元件",
        "COMPONENT_EXTRA": "多余元件",
        "OPEN_CIRCUIT": "断路",
        "WRONG_CONNECTION": "错接",
        "EXTRA_CONNECTION": "多余连接",
        "INCOMPLETE_CIRCUIT": "电路未完成",
        "ROLE_MISMATCH": "网络角色错误",
        "INPUT_NODE_MISMATCH": "输入节点错误",
        "OUTPUT_NODE_MISMATCH": "输出节点错误",
        "POWER_NODE_MISMATCH": "电源节点错误",
        "GROUND_NODE_MISMATCH": "地节点错误",
    }.get(error_code, "电路异常")


# ---------------------------------------------------------------------------
# Detailed enrichment
# ---------------------------------------------------------------------------

def _enrich_result(
    result: dict[str, Any],
    ref_graph: nx.Graph,
    cur_graph: nx.Graph,
    ref_payload: dict[str, Any],
    cur_netlist_v2: dict[str, Any],
) -> dict[str, Any]:
    """用原始 payload / netlist 数据对 error items 做精细化增强。"""
    match_type = result.get("details", {}).get("match_type")
    comp_map, net_map = _build_mappings(ref_graph, cur_graph, ref_payload, cur_netlist_v2)
    _attach_mappings(result, comp_map, net_map)
    detailed_items = _generate_detailed_items(
        ref_graph, cur_graph, ref_payload, cur_netlist_v2, match_type, comp_map, net_map
    )
    if not detailed_items:
        return result

    # Replace items with detailed versions
    result["items"] = detailed_items
    if any(item.get("severity") == "error" for item in detailed_items):
        result["logic_correct"] = False
        result["is_correct"] = False
        result["is_match"] = False
    report = dict(result.get("report", {}))
    report["items"] = detailed_items
    report["summary"] = dict(report.get("summary", {}))
    report["summary"]["total_item_count"] = len(detailed_items)
    report["summary"]["logic_correct"] = result["logic_correct"]
    report["topology_errors"] = [
        item for item in detailed_items
        if item.get("error_family") in {"wiring_mismatch", "open_circuit", "extra_connection", "incomplete_circuit"}
    ]
    report["component_errors"] = [
        item for item in detailed_items
        if item.get("error_family") in {"missing_component", "extra_component"}
    ]
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


def _generate_detailed_items(
    ref_graph: nx.Graph,
    cur_graph: nx.Graph,
    ref_payload: dict[str, Any],
    cur_netlist_v2: dict[str, Any],
    match_type: str | None,
    comp_map: dict[str, str] | None = None,
    net_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if comp_map is None or net_map is None:
        comp_map, net_map = _build_mappings(ref_graph, cur_graph, ref_payload, cur_netlist_v2)
    ref_comp_by_id = {c["ref_id"]: c for c in ref_payload.get("components", [])}
    cur_comp_by_id = {
        c["component_id"]: c
        for c in cur_netlist_v2.get("components", [])
        if c.get("component_type") != "Wire"
    }
    ref_net_roles = {
        n["net"]: normalize_net_role(n.get("role") or n.get("role_label") or n.get("label") or n.get("net"))
        for n in ref_payload.get("nets", [])
    }
    ref_net_labels = {
        n["net"]: normalize_role_label(n.get("role_label") or n.get("label") or n.get("net"))
        for n in ref_payload.get("nets", [])
    }

    items: list[dict[str, Any]] = []

    ref_comp_ids = set(ref_comp_by_id.keys())
    cur_comp_ids = set(cur_comp_by_id.keys())
    missing_refs = sorted(ref_comp_ids - set(comp_map.keys()))
    extra_curs = sorted(cur_comp_ids - set(comp_map.values()))

    # 1. COMPONENT_MISSING
    for ref_id in missing_refs:
        ref_comp = ref_comp_by_id[ref_id]
        items.append(_detailed_item(
            error_code="COMPONENT_MISSING",
            error_family="missing_component",
            severity="error",
            message=f"参考电路需要元件 {ref_id}（{ref_comp.get('type')}），但当前电路中未找到。",
            expected={"ref_id": ref_id, "type": ref_comp.get("type")},
            actual=None,
            component_ref={"ref_id": ref_id, "type": ref_comp.get("type")},
            component_actual=None,
            evidence_refs=[{"type": "reference_component", "ref_id": ref_id}],
            suggested_action=f"请添加元件 {ref_id}（{ref_comp.get('type')}）并完成其引脚连接。",
        ))

    # 2. COMPONENT_EXTRA
    for cur_id in extra_curs:
        cur_comp = cur_comp_by_id[cur_id]
        items.append(_detailed_item(
            error_code="COMPONENT_EXTRA",
            error_family="extra_component",
            severity="warning",
            message=f"当前电路包含多余的元件 {cur_id}（{cur_comp.get('component_type')}）。",
            expected=None,
            actual={"component_id": cur_id, "type": cur_comp.get("component_type")},
            component_ref=None,
            component_actual={"component_id": cur_id, "type": cur_comp.get("component_type")},
            evidence_refs=[{"type": "component", "component_id": cur_id}],
            suggested_action=f"请移除多余的元件 {cur_id}（{cur_comp.get('component_type')}）或确认是否需要它。",
        ))

    # 3. Pin-level checks for matched components
    wrong_connection_items: list[dict[str, Any]] = []
    open_circuit_items: list[dict[str, Any]] = []

    for ref_id, cur_id in comp_map.items():
        ref_comp = ref_comp_by_id[ref_id]
        cur_comp = cur_comp_by_id.get(cur_id, {})
        ctype = normalize_component_type(ref_comp.get("type"))

        ref_pins = {p["pin"]: p["net"] for p in ref_comp.get("pins", [])}
        ref_pin_roles = {p["pin"]: normalize_pin_role(ctype, p) for p in ref_comp.get("pins", [])}
        cur_pins = {p["pin_name"]: p for p in cur_comp.get("pins", [])}
        cur_pins_by_role = {normalize_pin_role(ctype, p): p for p in cur_comp.get("pins", [])}

        if ctype in PASSIVE_TWO_PIN_TYPES:
            # For passive two-pin components, treat pins as an unordered set.
            ref_nets = {p["net"] for p in ref_comp.get("pins", [])}
            cur_nets = {p.get("electrical_net_id") for p in cur_comp.get("pins", [])}
            mapped_ref_nets = {net_map.get(rn) for rn in ref_nets}
            # Only flag when all reference nets are mapped and the sets differ.
            if None not in mapped_ref_nets and mapped_ref_nets != cur_nets:
                wrong_connection_items.append(_detailed_item(
                    error_code="WRONG_CONNECTION",
                    error_family="wiring_mismatch",
                    severity="error",
                    message=f"{ref_id} 的连接网络与参考电路不一致。",
                    expected={
                        "ref_id": ref_id,
                        "nets": sorted(ref_nets),
                    },
                    actual={
                        "actual_component_id": cur_id,
                        "nets": sorted(cur_nets) if cur_nets else [],
                    },
                    component_ref={"ref_id": ref_id, "type": ref_comp.get("type")},
                    component_actual={"component_id": cur_id, "type": cur_comp.get("component_type")},
                    evidence_refs=[{"type": "component", "component_id": cur_id}],
                    suggested_action=f"请将 {cur_id} 改接到与 {ref_id} 对应的网络。",
                ))
            continue

        for pin_name, ref_net in ref_pins.items():
            ref_pin_role = ref_pin_roles.get(pin_name, normalize_pin_role(ctype, pin_name))
            cur_pin = cur_pins_by_role.get(ref_pin_role) or cur_pins.get(pin_name)
            if not cur_pin:
                if ctype in STRICT_PIN_ROLE_TYPES:
                    wrong_connection_items.append(_detailed_item(
                        error_code="PIN_ROLE_MISMATCH",
                        error_family="wiring_mismatch",
                        severity="error",
                        message=f"{ref_id}.{pin_name} 需要功能引脚 {ref_pin_role}，但当前元件 {cur_id} 未找到对应功能引脚。",
                        expected={"ref_pin": f"{ref_id}.{pin_name}", "pin_role": ref_pin_role},
                        actual={"actual_component_id": cur_id, "available_pin_roles": sorted(cur_pins_by_role)},
                        component_ref={"ref_id": ref_id, "type": ref_comp.get("type")},
                        component_actual={"component_id": cur_id, "type": cur_comp.get("component_type")},
                        evidence_refs=[{"type": "component", "component_id": cur_id}],
                        suggested_action=f"请重新标注 {cur_id} 的功能引脚，确保 {ref_pin_role} 接到正确网络。",
                    ))
                continue
            cur_net = cur_pin.get("electrical_net_id")
            mapped_cur_net = net_map.get(ref_net)

            if mapped_cur_net and cur_net != mapped_cur_net:
                actual_pin_role = normalize_pin_role(ctype, cur_pin)
                error_code = "PIN_ROLE_MISMATCH" if ctype in STRICT_PIN_ROLE_TYPES and actual_pin_role != ref_pin_role else "WRONG_CONNECTION"
                wrong_connection_items.append(_detailed_item(
                    error_code=error_code,
                    error_family="wiring_mismatch",
                    severity="error",
                    message=f"{ref_id}.{pin_name} 应连接到参考网络 {ref_net}，但当前实际连接到 {cur_net}。",
                    expected={
                        "ref_pin": f"{ref_id}.{pin_name}",
                        "pin_role": ref_pin_role,
                        "expected_net": ref_net,
                    },
                    actual={
                        "actual_component_id": cur_id,
                        "actual_pin": cur_pin.get("pin_name"),
                        "pin_role": actual_pin_role,
                        "actual_net": cur_net,
                        "hole_id": cur_pin.get("hole_id"),
                    },
                    component_ref={"ref_id": ref_id, "type": ref_comp.get("type")},
                    component_actual={"component_id": cur_id, "type": cur_comp.get("component_type")},
                    evidence_refs=[
                        {"type": "component", "component_id": cur_id},
                        {"type": "net", "electrical_net_id": cur_net},
                    ],
                    suggested_action=f"请将 {cur_id}.{pin_name} 从 {cur_net} 改接到与 {ref_net} 对应的网络。",
                ))

    # 3.5 Net role mismatch checks for mapped nets
    cur_net_by_id = {
        n.get("electrical_net_id"): n
        for n in cur_netlist_v2.get("nets", [])
        if n.get("electrical_net_id")
    }
    cur_net_roles = {
        net_id: normalize_net_role(
            n.get("role") or n.get("manual_role") or n.get("role_label") or n.get("power_role")
        )
        for net_id, n in cur_net_by_id.items()
    }
    cur_net_labels = {
        net_id: normalize_role_label(n.get("role_label") or n.get("power_role"))
        for net_id, n in cur_net_by_id.items()
    }

    role_error_code_map = {
        "input": "INPUT_NODE_MISMATCH",
        "output": "OUTPUT_NODE_MISMATCH",
        "power": "POWER_NODE_MISMATCH",
        "ground": "GROUND_NODE_MISMATCH",
    }

    for ref_net, ref_role in ref_net_roles.items():
        if ref_role == "signal":
            continue
        mapped_cur_net = net_map.get(ref_net)
        if not mapped_cur_net:
            continue
        cur_role = cur_net_roles.get(mapped_cur_net, "signal")
        if cur_role != ref_role:
            ref_pins = []
            for rc in ref_payload.get("components", []):
                for p in rc.get("pins", []):
                    if p.get("net") == ref_net:
                        ref_pins.append(f"{rc['ref_id']}.{p['pin']}")

            cur_connected_pins = []
            for cc in cur_netlist_v2.get("components", []):
                cid = cc.get("component_id")
                for p in cc.get("pins", []):
                    if p.get("electrical_net_id") == mapped_cur_net:
                        cur_connected_pins.append(f"{cid}.{p.get('pin_name')}")

            cur_net_obj = cur_net_by_id.get(mapped_cur_net, {})
            actual_data: dict[str, Any] = {
                "role": cur_role,
                "current_net": mapped_cur_net,
                "connected_pins": cur_connected_pins,
            }
            if cur_net_obj.get("role_label"):
                actual_data["role_label"] = cur_net_obj["role_label"]
            if cur_net_obj.get("member_hole_ids"):
                actual_data["member_hole_ids"] = cur_net_obj["member_hole_ids"]

            wrong_connection_items.append(_detailed_item(
                error_code=role_error_code_map.get(ref_role, "ROLE_MISMATCH"),
                error_family="wiring_mismatch",
                severity="error",
                message=f"参考电路中 {ref_net} 为 {ref_role} 节点，但当前映射网络 {mapped_cur_net} 实际为 {cur_role} 节点。",
                expected={
                    "role": ref_role,
                    "reference_net": ref_net,
                    "pins": ref_pins,
                },
                actual=actual_data,
                component_ref=None,
                component_actual=None,
                evidence_refs=[{"type": "net", "electrical_net_id": mapped_cur_net}],
                suggested_action=f"请在二维面包板图上重新点选正确的 {ref_role} 节点。",
            ))

        ref_label = ref_net_labels.get(ref_net, "")
        cur_label = cur_net_labels.get(mapped_cur_net, "")
        if ref_label in CRITICAL_ROLE_LABELS and cur_label and cur_label != ref_label:
            ref_node = ref_graph.nodes.get(f"ref_net:{ref_net}", {})
            cur_node = cur_graph.nodes.get(f"cur_net:{mapped_cur_net}", {})
            if not _role_labels_equivalent(ref_node, cur_node):
                wrong_connection_items.append(_detailed_item(
                    error_code="ROLE_LABEL_MISMATCH",
                    error_family="wiring_mismatch",
                    severity="error",
                    message=(
                        f"参考网络 {ref_net} 应匹配 role_label={ref_label} 的当前网络，"
                        f"但当前映射到了 role_label={cur_label}。"
                    ),
                    expected={"reference_net": ref_net, "role": ref_role, "role_label": ref_label},
                    actual={"current_net": mapped_cur_net, "role": cur_role, "role_label": cur_label},
                    component_ref=None,
                    component_actual=None,
                    evidence_refs=[{"type": "net", "electrical_net_id": mapped_cur_net}],
                    suggested_action=f"请将当前网络标注为 {ref_label}，或检查端口是否接反。",
                ))

    # 4. OPEN_CIRCUIT: ref pins that should share a net but are on different cur nets
    for ref_net_id in ref_net_roles:
        ref_pins_on_net: list[tuple[str, str]] = []
        for ref_comp in ref_payload.get("components", []):
            for pin in ref_comp.get("pins", []):
                if pin["net"] == ref_net_id:
                    ref_pins_on_net.append((ref_comp["ref_id"], pin["pin"]))

        if len(ref_pins_on_net) < 2:
            continue

        cur_nets_for_pins: list[tuple[str, str, str | None, str | None]] = []
        for ref_id, pin_name in ref_pins_on_net:
            if ref_id not in comp_map:
                continue
            cur_id = comp_map[ref_id]
            cur_comp = cur_comp_by_id.get(cur_id, {})
            for pin in cur_comp.get("pins", []):
                if pin.get("pin_name") == pin_name:
                    cur_nets_for_pins.append(
                        (cur_id, pin_name, pin.get("electrical_net_id"), pin.get("hole_id"))
                    )
                    break

        unique_cur_nets = list({n for _, _, n, _ in cur_nets_for_pins if n})
        if len(unique_cur_nets) > 1:
            open_circuit_items.append(_detailed_item(
                error_code="OPEN_CIRCUIT",
                error_family="wiring_mismatch",
                severity="error",
                message=f"参考电路中网络 {ref_net_id} 连接的引脚在当前电路中被分到了 {len(unique_cur_nets)} 个不同网络。",
                expected={
                    "shared_net": ref_net_id,
                    "pins": [f"{rid}.{pname}" for rid, pname in ref_pins_on_net],
                },
                actual={
                    "connected": False,
                    "pins": [
                        {
                            "actual_component_id": cid,
                            "actual_pin": pname,
                            "actual_net": cnet,
                            "hole_id": hid,
                        }
                        for cid, pname, cnet, hid in cur_nets_for_pins
                    ],
                },
                component_ref=None,
                component_actual=None,
                evidence_refs=[
                    {"type": "net", "electrical_net_id": cnet}
                    for _, _, cnet, _ in cur_nets_for_pins
                    if cnet
                ],
                suggested_action=f"请将上述引脚连接到同一电气网络，以构成 {ref_net_id} 连接。",
            ))

    ref_net_count = _net_count(ref_graph)
    cur_net_count = _net_count(cur_graph)
    short_circuit_items = _short_circuit_items(ref_net_roles, ref_net_labels, net_map)
    short_circuit_items.extend(
        _pin_level_short_circuit_items(ref_payload, cur_netlist_v2, comp_map)
    )
    short_circuit_items = _dedupe_detailed_items(short_circuit_items)

    # 5. Assemble items based on match_type
    if match_type == "current_subgraph_in_reference":
        items.extend(short_circuit_items)
        items.extend(open_circuit_items)
        items.append(_detailed_item(
            error_code="INCOMPLETE_CIRCUIT",
            error_family="incomplete_circuit",
            severity="error",
            message="当前电路只匹配到参考电路的一部分，电路尚未完整实现。",
            expected={
                "reference_component_count": len(ref_comp_ids),
                "reference_edge_count": ref_graph.number_of_edges(),
            },
            actual={
                "current_component_count": len(cur_comp_ids),
                "current_edge_count": cur_graph.number_of_edges(),
            },
            component_ref=None,
            component_actual=None,
            evidence_refs=[],
            suggested_action="请补齐缺失的元件和连接后重新验证。",
        ))
    elif match_type in {"reference_subgraph_in_current", "equivalent_with_extra"}:
        items.extend(short_circuit_items)
        if cur_net_count < ref_net_count:
            all_cur_nets = sorted({
                n.get("electrical_net_id")
                for n in cur_netlist_v2.get("nets", [])
                if n.get("electrical_net_id")
            })
            items.append(_detailed_item(
                error_code="EXTRA_CONNECTION",
                error_family="extra_connection",
                severity="warning",
                message=f"当前电路的电气网络数（{cur_net_count}）少于参考电路（{ref_net_count}），可能存在不应有的短接。",
                expected={"net_count": ref_net_count, "separate_nets": sorted(ref_net_roles.keys())},
                actual={"net_count": cur_net_count, "merged_nets": all_cur_nets},
                component_ref=None,
                component_actual=None,
                evidence_refs=[],
                suggested_action="请检查是否有不应相连的节点被错误地连接到了一起。",
            ))
    else:  # graph_edit_distance_or_fallback
        items.extend(short_circuit_items)
        items.extend(wrong_connection_items)
        items.extend(open_circuit_items)
        if cur_net_count < ref_net_count:
            all_cur_nets = sorted({
                n.get("electrical_net_id")
                for n in cur_netlist_v2.get("nets", [])
                if n.get("electrical_net_id")
            })
            items.append(_detailed_item(
                error_code="EXTRA_CONNECTION",
                error_family="extra_connection",
                severity="warning",
                message=f"当前电路的电气网络数（{cur_net_count}）少于参考电路（{ref_net_count}），可能存在不应有的短接。",
                expected={"net_count": ref_net_count, "separate_nets": sorted(ref_net_roles.keys())},
                actual={"net_count": cur_net_count, "merged_nets": all_cur_nets},
                component_ref=None,
                component_actual=None,
                evidence_refs=[],
                suggested_action="请检查是否有不应相连的节点被错误地连接到了一起。",
            ))

    return _dedupe_detailed_items(items)


def _short_circuit_items(
    ref_net_roles: dict[str, str],
    ref_net_labels: dict[str, str],
    net_map: dict[str, str],
) -> list[dict[str, Any]]:
    mapped_by_current: dict[str, list[str]] = defaultdict(list)
    for ref_net, cur_net in net_map.items():
        if cur_net:
            mapped_by_current[cur_net].append(ref_net)

    items: list[dict[str, Any]] = []
    for cur_net, ref_nets in mapped_by_current.items():
        if len(ref_nets) < 2:
            continue
        for i, left in enumerate(ref_nets):
            for right in ref_nets[i + 1:]:
                if not _is_harmful_merge(left, right, ref_net_roles, ref_net_labels):
                    continue
                left_label = ref_net_labels.get(left, normalize_role_label(left))
                right_label = ref_net_labels.get(right, normalize_role_label(right))
                items.append(_detailed_item(
                    error_code="SHORT_CIRCUIT",
                    error_family="extra_connection",
                    severity="error",
                    message=f"参考中应分离的关键网络 {left_label} 与 {right_label} 在当前电路中被合并到 {cur_net}。",
                    expected={"separate_nets": [left, right], "role_labels": [left_label, right_label]},
                    actual={"current_net": cur_net, "merged_reference_nets": ref_nets},
                    component_ref=None,
                    component_actual=None,
                    evidence_refs=[{"type": "net", "electrical_net_id": cur_net}],
                    suggested_action="请检查是否有多余导线或元件把两个关键网络短接在一起。",
                ))
    return _dedupe_detailed_items(items)


def _is_harmful_merge(
    left: str,
    right: str,
    ref_net_roles: dict[str, str],
    ref_net_labels: dict[str, str],
) -> bool:
    role_pair = {ref_net_roles.get(left, "signal"), ref_net_roles.get(right, "signal")}
    labels = {ref_net_labels.get(left, normalize_role_label(left)), ref_net_labels.get(right, normalize_role_label(right))}
    if role_pair == {"power", "ground"}:
        return True
    if {"VCC", "VEE"} <= labels:
        return True
    if {"input", "output"} <= role_pair:
        return True
    if {"UO1", "UO2"} <= labels:
        return True
    if {"UI1", "UI2"} <= labels:
        return True
    return False


def _pin_level_short_circuit_items(
    ref_payload: dict[str, Any],
    cur_netlist_v2: dict[str, Any],
    comp_map: dict[str, str],
) -> list[dict[str, Any]]:
    """通过元件引脚级别比对检测短路：若对应不同参考网络的引脚在当前电路中共享同一电气网络，则视为短路。"""
    cur_pin_to_net: dict[tuple[str, str], str] = {}
    for comp in cur_netlist_v2.get("components", []):
        cid = comp.get("component_id")
        for pin in comp.get("pins", []):
            cur_pin_to_net[(cid, pin.get("pin_name"))] = pin.get("electrical_net_id")

    ref_net_roles = {
        n["net"]: normalize_net_role(n.get("role") or n.get("role_label") or n.get("label") or n["net"])
        for n in ref_payload.get("nets", [])
    }
    ref_net_labels = {
        n["net"]: normalize_role_label(n.get("role_label") or n.get("label") or n["net"])
        for n in ref_payload.get("nets", [])
    }

    # 收集每个参考网络对应的引脚
    ref_net_pins: dict[str, list[tuple[str, str]]] = {}
    for comp in ref_payload.get("components", []):
        ref_id = comp["ref_id"]
        for pin in comp.get("pins", []):
            net = pin.get("net")
            if net:
                ref_net_pins.setdefault(net, []).append((ref_id, pin.get("pin")))

    items: list[dict[str, Any]] = []
    ref_nets = sorted(ref_net_pins.keys())
    for i, left in enumerate(ref_nets):
        for right in ref_nets[i + 1 :]:
            if not _is_harmful_merge(left, right, ref_net_roles, ref_net_labels):
                continue

            left_cur_nets: set[str] = set()
            for ref_id, pin_name in ref_net_pins.get(left, []):
                if ref_id in comp_map:
                    net_id = cur_pin_to_net.get((comp_map[ref_id], pin_name))
                    if net_id:
                        left_cur_nets.add(net_id)

            right_cur_nets: set[str] = set()
            for ref_id, pin_name in ref_net_pins.get(right, []):
                if ref_id in comp_map:
                    net_id = cur_pin_to_net.get((comp_map[ref_id], pin_name))
                    if net_id:
                        right_cur_nets.add(net_id)

            shared = left_cur_nets & right_cur_nets
            for cur_net in shared:
                left_label = ref_net_labels.get(left, normalize_role_label(left))
                right_label = ref_net_labels.get(right, normalize_role_label(right))
                items.append(
                    _detailed_item(
                        error_code="SHORT_CIRCUIT",
                        error_family="extra_connection",
                        severity="error",
                        message=f"参考中应分离的关键网络 {left_label} 与 {right_label} 在当前电路中被合并到 {cur_net}。",
                        expected={"separate_nets": [left, right], "role_labels": [left_label, right_label]},
                        actual={"current_net": cur_net, "merged_reference_nets": [left, right]},
                        component_ref=None,
                        component_actual=None,
                        evidence_refs=[{"type": "net", "electrical_net_id": cur_net}],
                        suggested_action="请检查是否有多余导线或元件把两个关键网络短接在一起。",
                    )
                )

    return items


def _detailed_item(
    *,
    error_code: str,
    error_family: str,
    severity: str,
    message: str,
    expected: Any,
    actual: Any,
    component_ref: dict[str, Any] | None,
    component_actual: dict[str, Any] | None,
    evidence_refs: list[dict[str, Any]],
    suggested_action: str,
    title: str = "",
) -> dict[str, Any]:
    return {
        "error_code": error_code,
        "error_family": error_family,
        "severity": severity,
        "title": title or _default_title(error_code),
        "message": message,
        "expected": expected,
        "actual": actual,
        "component_ref": component_ref,
        "component_actual": component_actual,
        "evidence_refs": evidence_refs,
        "suggested_action": suggested_action,
    }


def _dedupe_detailed_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = f"{item.get('error_code')}:{item.get('message')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _build_mappings(
    ref_graph: nx.Graph,
    cur_graph: nx.Graph,
    ref_payload: dict[str, Any],
    cur_netlist_v2: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """返回 (comp_map: ref_id -> cur_id, net_map: ref_net -> cur_net)。"""
    iso_mapping = _find_any_isomorphism_mapping(ref_graph, cur_graph)
    if iso_mapping is not None:
        comp_map = _extract_comp_mapping(iso_mapping, ref_graph, cur_graph)
        net_map = _extract_net_mapping(iso_mapping, ref_graph, cur_graph)
        return comp_map, net_map

    comp_map = _fallback_comp_mapping(ref_graph, cur_graph)
    net_map = _build_net_mapping(ref_graph, cur_graph, comp_map)
    return comp_map, net_map


def _find_any_isomorphism_mapping(
    ref_graph: nx.Graph, cur_graph: nx.Graph
) -> dict[Any, Any] | None:
    """尝试完整同构或子图同构，返回 ref_node -> cur_node 映射。"""
    matcher = GraphMatcher(ref_graph, cur_graph, node_match=_node_match, edge_match=_edge_match)
    if matcher.is_isomorphic():
        return next(matcher.isomorphisms_iter())

    # ref ⊂ cur
    matcher = GraphMatcher(cur_graph, ref_graph, node_match=_node_match, edge_match=_edge_match)
    if matcher.subgraph_is_isomorphic():
        mapping = next(matcher.subgraph_isomorphisms_iter())  # cur_node -> ref_node
        return {v: k for k, v in mapping.items()}

    # cur ⊂ ref
    matcher = GraphMatcher(ref_graph, cur_graph, node_match=_node_match, edge_match=_edge_match)
    if matcher.subgraph_is_isomorphic():
        return next(matcher.subgraph_isomorphisms_iter())

    return None


def _extract_net_mapping(
    ref_to_cur_mapping: dict[Any, Any],
    ref_graph: nx.Graph,
    cur_graph: nx.Graph,
) -> dict[str, str]:
    net_map: dict[str, str] = {}
    for ref_node, cur_node in ref_to_cur_mapping.items():
        ref_data = ref_graph.nodes.get(ref_node, {})
        cur_data = cur_graph.nodes.get(cur_node, {})
        if ref_data.get("kind") == "net" and cur_data.get("kind") == "net":
            net_map[ref_data["source_id"]] = cur_data["source_id"]
    return net_map


def _extract_comp_mapping(
    ref_to_cur_mapping: dict[Any, Any],
    ref_graph: nx.Graph,
    cur_graph: nx.Graph,
) -> dict[str, str]:
    comp_map: dict[str, str] = {}
    for ref_node, cur_node in ref_to_cur_mapping.items():
        ref_data = ref_graph.nodes.get(ref_node, {})
        cur_data = cur_graph.nodes.get(cur_node, {})
        if ref_data.get("kind") == "comp" and cur_data.get("kind") == "comp":
            comp_map[ref_data["source_id"]] = cur_data["source_id"]
    return comp_map


def _fallback_comp_mapping(
    ref_graph: nx.Graph,
    cur_graph: nx.Graph,
) -> dict[str, str]:
    """基于连接签名（邻居 net role、pin 标签、度数）进行贪心最大分数匹配。

    完全不依赖 ref_id / component_id 的相等性，只比较拓扑特征。
    """
    ref_comps = [
        (n, ref_graph.nodes[n])
        for n in ref_graph.nodes
        if ref_graph.nodes[n].get("kind") == "comp"
    ]
    cur_comps = [
        (n, cur_graph.nodes[n])
        for n in cur_graph.nodes
        if cur_graph.nodes[n].get("kind") == "comp"
    ]

    ref_by_type: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for n, data in ref_comps:
        ref_by_type[data.get("ctype", "UNKNOWN")].append((n, data))

    cur_by_type: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for n, data in cur_comps:
        cur_by_type[data.get("ctype", "UNKNOWN")].append((n, data))

    comp_map: dict[str, str] = {}
    for ctype, ref_list in ref_by_type.items():
        cur_list = cur_by_type.get(ctype, [])
        if not cur_list:
            continue

        ref_sigs = {n: _graph_neighbor_signature(n, ref_graph) for n, _ in ref_list}
        cur_sigs = {n: _graph_neighbor_signature(n, cur_graph) for n, _ in cur_list}

        matched = _greedy_match_by_score(
            [n for n, _ in ref_list],
            [n for n, _ in cur_list],
            ref_sigs,
            cur_sigs,
        )
        for ref_node, cur_node in matched.items():
            ref_source = ref_graph.nodes[ref_node].get("source_id", ref_node)
            cur_source = cur_graph.nodes[cur_node].get("source_id", cur_node)
            comp_map[ref_source] = cur_source

    return comp_map


def _graph_neighbor_signature(node: str, graph: nx.Graph) -> tuple:
    """返回元件节点的连接签名：(度数, 邻居 net role 元组, 边的 pin 标签元组)。"""
    neighbors = list(graph.neighbors(node))
    net_roles = tuple(sorted(
        graph.nodes[n].get("role", "signal")
        for n in neighbors
        if graph.nodes[n].get("kind") == "net"
    ))
    pin_labels = []
    for n in neighbors:
        edge_data = graph.get_edge_data(node, n)
        if edge_data:
            pin_labels.append(str(edge_data.get("pin", "")))
    pin_labels = tuple(sorted(pin_labels))
    return (len(neighbors), net_roles, pin_labels)


def _greedy_match_by_score(
    ref_nodes: list[str],
    cur_nodes: list[str],
    ref_sigs: dict[str, tuple],
    cur_sigs: dict[str, tuple],
) -> dict[str, str]:
    """按签名相似度贪心匹配，返回 comp_map (ref_node -> cur_node)。"""
    scores: list[tuple[float, str, str]] = []
    for rn in ref_nodes:
        for cn in cur_nodes:
            score = _neighbor_signature_similarity(ref_sigs[rn], cur_sigs[cn])
            scores.append((score, rn, cn))
    scores.sort(key=lambda x: x[0], reverse=True)

    matched: dict[str, str] = {}
    used_c: set[str] = set()
    for score, rn, cn in scores:
        if rn in matched or cn in used_c:
            continue
        if score > 0:
            matched[rn] = cn
            used_c.add(cn)
    return matched


def _neighbor_signature_similarity(ref_sig: tuple, cur_sig: tuple) -> float:
    """计算两个连接签名的相似度分数。"""
    score = 0.0
    # 度数匹配（pin 数量）
    if ref_sig[0] == cur_sig[0]:
        score += 10.0

    # 邻居 net role 交集
    ref_roles = set(ref_sig[1])
    cur_roles = set(cur_sig[1])
    role_overlap = len(ref_roles & cur_roles)
    score += role_overlap * 5.0

    # 边的 pin 标签交集
    ref_pins = set(ref_sig[2])
    cur_pins = set(cur_sig[2])
    pin_overlap = len(ref_pins & cur_pins)
    score += pin_overlap * 3.0

    return score


def _build_net_mapping(
    ref_graph: nx.Graph,
    cur_graph: nx.Graph,
    comp_map: dict[str, str],
) -> dict[str, str]:
    """基于元件映射，按共享元件数量最多的原则建立网络映射。

    使用贪心算法确保 1-to-1 映射，避免多个参考网络映射到同一当前网络。
    """
    ref_nets: list[tuple[str, dict[str, Any]]] = []
    for ref_node in ref_graph.nodes:
        ref_data = ref_graph.nodes[ref_node]
        if ref_data.get("kind") == "net":
            ref_nets.append((ref_node, ref_data))

    cur_nets: list[tuple[str, dict[str, Any]]] = []
    for cur_node in cur_graph.nodes:
        cur_data = cur_graph.nodes[cur_node]
        if cur_data.get("kind") == "net":
            cur_nets.append((cur_node, cur_data))

    scores: list[tuple[float, str, str]] = []
    for ref_node, ref_data in ref_nets:
        ref_net_id = ref_data["source_id"]
        ref_role = str(ref_data.get("role") or "signal")
        ref_label = normalize_role_label(ref_data.get("role_label"))

        ref_neighbors = {
            ref_graph.nodes[n]["source_id"]
            for n in ref_graph.neighbors(ref_node)
            if ref_graph.nodes[n].get("kind") == "comp"
        }

        for cur_node, cur_data in cur_nets:
            cur_net_id = cur_data["source_id"]
            cur_role = str(cur_data.get("role") or "signal")
            cur_label = normalize_role_label(cur_data.get("role_label"))

            cur_neighbors = {
                cur_graph.nodes[n]["source_id"]
                for n in cur_graph.neighbors(cur_node)
                if cur_graph.nodes[n].get("kind") == "comp"
            }

            score = float(sum(
                1 for ref_comp in ref_neighbors
                if ref_comp in comp_map and comp_map[ref_comp] in cur_neighbors
            ))
            if ref_role == cur_role:
                score += 2.0
            if ref_label and cur_label and ref_label == cur_label:
                score += 4.0
            elif ref_label in CRITICAL_ROLE_LABELS and cur_label in CRITICAL_ROLE_LABELS:
                score -= 4.0
            scores.append((score, ref_net_id, cur_net_id))

    scores.sort(key=lambda x: x[0], reverse=True)

    net_map: dict[str, str] = {}
    used_cur: set[str] = set()
    for score, ref_net_id, cur_net_id in scores:
        if ref_net_id in net_map or cur_net_id in used_cur:
            continue
        if score > 0:
            net_map[ref_net_id] = cur_net_id
            used_cur.add(cur_net_id)

    return net_map


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def _missing_items(reference_graph: nx.Graph, current_graph: nx.Graph) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    ref_counts = _component_type_counts(reference_graph)
    cur_counts = _component_type_counts(current_graph)
    for ctype in sorted(ref_counts.keys() | cur_counts.keys()):
        missing = ref_counts[ctype] - cur_counts[ctype]
        if missing > 0:
            items.append(
                _item(
                    "COMPONENT_MISSING",
                    "missing_component",
                    "error",
                    f"参考电路需要 {ref_counts[ctype]} 个 {ctype}，当前电路未匹配到 {missing} 个。",
                    expected={"component_type": ctype, "count": ref_counts[ctype]},
                    actual={"component_type": ctype, "count": cur_counts[ctype]},
                    suggested_action=f"请检查是否漏接 {ctype}。",
                )
            )

    if current_graph.number_of_edges() < reference_graph.number_of_edges():
        items.append(
            _item(
                "OPEN_CIRCUIT",
                "open_circuit",
                "error",
                "参考电路存在当前未完成的逻辑连接，可能有断路。",
                expected={"edge_count": reference_graph.number_of_edges()},
                actual={"edge_count": current_graph.number_of_edges()},
                suggested_action="请检查相关元件引脚是否接到同一个电气节点。",
            )
        )
    return items


def _extra_items(reference_graph: nx.Graph, current_graph: nx.Graph) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    ref_counts = _component_type_counts(reference_graph)
    cur_counts = _component_type_counts(current_graph)
    for ctype in sorted(ref_counts.keys() | cur_counts.keys()):
        extra = cur_counts[ctype] - ref_counts[ctype]
        if extra > 0:
            items.append(
                _item(
                    "COMPONENT_EXTRA",
                    "extra_component",
                    "warning",
                    f"当前电路比参考电路多出 {extra} 个 {ctype}。",
                    expected={"component_type": ctype, "count": ref_counts[ctype]},
                    actual={"component_type": ctype, "count": cur_counts[ctype]},
                    suggested_action=f"请检查是否误放了多余的 {ctype}。",
                )
            )
    if current_graph.number_of_edges() > reference_graph.number_of_edges():
        items.append(
            _item(
                "EXTRA_CONNECTION",
                "extra_connection",
                "warning",
                "当前电路包含参考电路之外的额外连接。",
                expected={"edge_count": reference_graph.number_of_edges()},
                actual={"edge_count": current_graph.number_of_edges()},
                suggested_action="请检查是否有多余导线或多余元件连接。",
            )
        )
    return items


def _difference_items(reference_graph: nx.Graph, current_graph: nx.Graph) -> list[dict[str, Any]]:
    items = _missing_items(reference_graph, current_graph) + _extra_items(reference_graph, current_graph)
    ref_net_count = _net_count(reference_graph)
    cur_net_count = _net_count(current_graph)
    if cur_net_count > ref_net_count:
        items.append(
            _item(
                "OPEN_CIRCUIT",
                "open_circuit",
                "error",
                "当前电路的电气网络比参考电路更多，可能存在应相连但未相连的断点。",
                expected={"net_count": ref_net_count},
                actual={"net_count": cur_net_count},
                suggested_action="请检查应共用节点的元件引脚是否断开。",
            )
        )
    if cur_net_count < ref_net_count:
        items.append(
            _item(
                "EXTRA_CONNECTION",
                "extra_connection",
                "error",
                "当前电路的电气网络比参考电路更少，可能存在额外短接。",
                expected={"net_count": ref_net_count},
                actual={"net_count": cur_net_count},
                suggested_action="请检查是否把不应相连的节点接在一起。",
            )
        )
    items.append(
        _item(
            "WRONG_CONNECTION",
            "wiring_mismatch",
            "error",
            "检测到元件连接关系与参考电路不一致，可能存在错接。",
            expected={"edge_signatures": _edge_signatures(reference_graph)},
            actual={"edge_signatures": _edge_signatures(current_graph)},
            suggested_action="请检查相关元件是否连接到正确的电气节点。",
        )
    )
    return _dedupe_items(items)


def _ged_similarity(reference_graph: nx.Graph, current_graph: nx.Graph) -> float:
    graph_size = max(
        reference_graph.number_of_nodes() + reference_graph.number_of_edges(),
        current_graph.number_of_nodes() + current_graph.number_of_edges(),
        1,
    )
    if reference_graph.number_of_nodes() > 30 or current_graph.number_of_nodes() > 30:
        return _approximate_similarity(reference_graph, current_graph)
    try:
        best = None
        for ged in nx.optimize_graph_edit_distance(
            reference_graph,
            current_graph,
            node_subst_cost=_node_subst_cost,
            node_del_cost=lambda _a: 1.0,
            node_ins_cost=lambda _a: 1.0,
            edge_subst_cost=lambda _a, _b: 0.0,
            edge_del_cost=lambda _a: 1.0,
            edge_ins_cost=lambda _a: 1.0,
            timeout=0.25,
        ):
            best = ged
            break
        if best is None:
            return _approximate_similarity(reference_graph, current_graph)
        return max(0.0, min(1.0, 1.0 - float(best) / graph_size))
    except Exception:
        return _approximate_similarity(reference_graph, current_graph)


def _node_subst_cost(a: dict[str, Any], b: dict[str, Any]) -> float:
    if a.get("kind") != b.get("kind"):
        return 2.0
    if a.get("kind") == "comp":
        return 0.0 if a.get("ctype") == b.get("ctype") else 1.5
    return 0.0 if _node_match(a, b) else 1.0


def _approximate_similarity(reference_graph: nx.Graph, current_graph: nx.Graph) -> float:
    ref_counts = _component_type_counts(reference_graph)
    cur_counts = _component_type_counts(current_graph)
    all_types = ref_counts.keys() | cur_counts.keys()
    if not all_types and reference_graph.number_of_edges() == current_graph.number_of_edges():
        return 1.0

    count_delta = sum(abs(ref_counts[t] - cur_counts[t]) for t in all_types)
    count_total = sum(ref_counts.values()) + sum(cur_counts.values()) or 1
    type_score = 1.0 - count_delta / count_total

    ref_edges = Counter(_edge_signatures(reference_graph))
    cur_edges = Counter(_edge_signatures(current_graph))
    all_edges = ref_edges.keys() | cur_edges.keys()
    edge_delta = sum(abs(ref_edges[e] - cur_edges[e]) for e in all_edges)
    edge_total = sum(ref_edges.values()) + sum(cur_edges.values()) or 1
    edge_score = 1.0 - edge_delta / edge_total

    ref_nets = _net_count(reference_graph)
    cur_nets = _net_count(current_graph)
    net_score = 1.0 - abs(ref_nets - cur_nets) / max(ref_nets, cur_nets, 1)

    return max(0.0, min(1.0, 0.45 * type_score + 0.4 * edge_score + 0.15 * net_score))


def _component_progress(reference_graph: nx.Graph, current_graph: nx.Graph) -> float:
    ref_counts = _component_type_counts(reference_graph)
    cur_counts = _component_type_counts(current_graph)
    total = sum(ref_counts.values()) or 1
    matched = sum(min(count, cur_counts.get(ctype, 0)) for ctype, count in ref_counts.items())
    return max(0.0, min(1.0, matched / total))


def _component_type_counts(graph: nx.Graph) -> Counter[str]:
    return Counter(
        str(data.get("ctype") or "UNKNOWN")
        for _node, data in graph.nodes(data=True)
        if data.get("kind") == "comp"
    )


def _component_count(graph: nx.Graph) -> int:
    return sum(1 for _node, data in graph.nodes(data=True) if data.get("kind") == "comp")


def _net_count(graph: nx.Graph) -> int:
    return sum(1 for _node, data in graph.nodes(data=True) if data.get("kind") == "net")


def _edge_signatures(graph: nx.Graph) -> list[tuple[str, str]]:
    signatures: list[tuple[str, str]] = []
    for u, v in graph.edges:
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        comp_data = u_data if u_data.get("kind") == "comp" else v_data
        net_data = v_data if u_data.get("kind") == "comp" else u_data
        signatures.append((str(comp_data.get("ctype") or "UNKNOWN"), str(net_data.get("role") or "signal")))
    return sorted(signatures)


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = (str(item.get("error_code")), str(item.get("message")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
