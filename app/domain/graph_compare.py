from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from app.domain.logical_reference import normalize_component_type


STRICT_NET_ROLES = {"ground", "power"}


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
    if _is_isomorphic(reference_graph, current_graph):
        result = _result(
            logic_correct=True,
            similarity=1.0,
            progress=1.0,
            message="电路逻辑连接与参考电路一致",
            items=[],
            details={"match_type": "full_isomorphism"},
            ref_payload=ref_payload,
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
            logic_correct=False,
            similarity=max(0.85, _approximate_similarity(reference_graph, current_graph)),
            progress=1.0,
            message="参考电路逻辑已存在，但当前电路包含额外元件或连接",
            items=items,
            details={"match_type": "reference_subgraph_in_current"},
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
    if reference_graph.number_of_nodes() != current_graph.number_of_nodes():
        return False
    if reference_graph.number_of_edges() != current_graph.number_of_edges():
        return False
    return GraphMatcher(
        reference_graph,
        current_graph,
        node_match=_node_match,
        edge_match=_edge_match,
    ).is_isomorphic()


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
        return role_a == role_b
    return True


def _edge_match(_a: dict[str, Any], _b: dict[str, Any]) -> bool:
    return True


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
        "message": message,
        "expected": expected,
        "actual": actual,
        "suggested_action": suggested_action,
    }
    if evidence:
        item["evidence"] = evidence
    return item


def _default_title(error_code: str) -> str:
    return {
        "COMPONENT_MISSING": "缺元件",
        "COMPONENT_EXTRA": "多余元件",
        "OPEN_CIRCUIT": "断路",
        "WRONG_CONNECTION": "错接",
        "EXTRA_CONNECTION": "多余连接",
        "INCOMPLETE_CIRCUIT": "电路未完成",
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
    detailed_items = _generate_detailed_items(
        ref_graph, cur_graph, ref_payload, cur_netlist_v2, match_type
    )
    if not detailed_items:
        return result

    # Replace items with detailed versions
    result["items"] = detailed_items
    report = dict(result.get("report", {}))
    report["items"] = detailed_items
    report["summary"] = dict(report.get("summary", {}))
    report["summary"]["total_item_count"] = len(detailed_items)
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


def _generate_detailed_items(
    ref_graph: nx.Graph,
    cur_graph: nx.Graph,
    ref_payload: dict[str, Any],
    cur_netlist_v2: dict[str, Any],
    match_type: str | None,
) -> list[dict[str, Any]]:
    comp_map, net_map = _build_mappings(ref_graph, cur_graph, ref_payload, cur_netlist_v2)
    ref_comp_by_id = {c["ref_id"]: c for c in ref_payload.get("components", [])}
    cur_comp_by_id = {
        c["component_id"]: c
        for c in cur_netlist_v2.get("components", [])
        if c.get("component_type") != "Wire"
    }
    ref_net_roles = {n["net"]: n.get("role", "signal") for n in ref_payload.get("nets", [])}

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

        ref_pins = {p["pin"]: p["net"] for p in ref_comp.get("pins", [])}
        cur_pins = {p["pin_name"]: p for p in cur_comp.get("pins", [])}

        for pin_name, ref_net in ref_pins.items():
            cur_pin = cur_pins.get(pin_name)
            if not cur_pin:
                continue
            cur_net = cur_pin.get("electrical_net_id")
            mapped_cur_net = net_map.get(ref_net)

            if mapped_cur_net and cur_net != mapped_cur_net:
                wrong_connection_items.append(_detailed_item(
                    error_code="WRONG_CONNECTION",
                    error_family="wiring_mismatch",
                    severity="error",
                    message=f"{ref_id}.{pin_name} 应连接到参考网络 {ref_net}，但当前实际连接到 {cur_net}。",
                    expected={
                        "ref_pin": f"{ref_id}.{pin_name}",
                        "expected_net": ref_net,
                    },
                    actual={
                        "actual_component_id": cur_id,
                        "actual_pin": pin_name,
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

    # 5. Assemble items based on match_type
    if match_type == "current_subgraph_in_reference":
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
    elif match_type == "reference_subgraph_in_current":
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
    comp_map = _build_component_mapping(ref_graph, cur_graph, ref_payload, cur_netlist_v2)
    net_map = _build_net_mapping(ref_graph, cur_graph, comp_map)
    return comp_map, net_map


def _build_component_mapping(
    ref_graph: nx.Graph,
    cur_graph: nx.Graph,
    ref_payload: dict[str, Any],
    cur_netlist_v2: dict[str, Any],
) -> dict[str, str]:
    """尝试用图同构建立 ref 与 current 的元件映射，失败则回退到类型匹配。"""
    # 1. 尝试完整同构
    matcher = GraphMatcher(ref_graph, cur_graph, node_match=_node_match, edge_match=_edge_match)
    if matcher.is_isomorphic():
        mapping = next(matcher.isomorphisms_iter())
        return _extract_comp_mapping(mapping, ref_graph, cur_graph)

    # 2. 尝试子图同构（ref ⊂ cur）
    matcher = GraphMatcher(cur_graph, ref_graph, node_match=_node_match, edge_match=_edge_match)
    if matcher.subgraph_is_isomorphic():
        mapping = next(matcher.subgraph_isomorphisms_iter())  # cur_node -> ref_node
        return _extract_comp_mapping({v: k for k, v in mapping.items()}, ref_graph, cur_graph)

    # 3. 尝试子图同构（cur ⊂ ref）
    matcher = GraphMatcher(ref_graph, cur_graph, node_match=_node_match, edge_match=_edge_match)
    if matcher.subgraph_is_isomorphic():
        mapping = next(matcher.subgraph_isomorphisms_iter())
        return _extract_comp_mapping(mapping, ref_graph, cur_graph)

    # 4. 回退：按类型顺序匹配
    return _fallback_comp_mapping(ref_payload, cur_netlist_v2)


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
    ref_payload: dict[str, Any],
    cur_netlist_v2: dict[str, Any],
) -> dict[str, str]:
    ref_by_type: dict[str, list[str]] = defaultdict(list)
    for c in ref_payload.get("components", []):
        ctype = normalize_component_type(c.get("type"))
        ref_by_type[ctype].append(c["ref_id"])

    cur_by_type: dict[str, list[str]] = defaultdict(list)
    for c in cur_netlist_v2.get("components", []):
        ctype = normalize_component_type(c.get("component_type") or c.get("type"))
        if ctype == "Wire":
            continue
        cur_by_type[ctype].append(c["component_id"])

    comp_map: dict[str, str] = {}
    for ctype, ref_ids in ref_by_type.items():
        cur_ids = cur_by_type.get(ctype, [])
        for i, ref_id in enumerate(ref_ids):
            if i < len(cur_ids):
                comp_map[ref_id] = cur_ids[i]
    return comp_map


def _build_net_mapping(
    ref_graph: nx.Graph,
    cur_graph: nx.Graph,
    comp_map: dict[str, str],
) -> dict[str, str]:
    """基于元件映射，按共享元件数量最多的原则建立网络映射。"""
    net_map: dict[str, str] = {}
    for ref_node in ref_graph.nodes:
        ref_data = ref_graph.nodes[ref_node]
        if ref_data.get("kind") != "net":
            continue
        ref_net_id = ref_data["source_id"]

        ref_neighbors = {
            ref_graph.nodes[n]["source_id"]
            for n in ref_graph.neighbors(ref_node)
            if ref_graph.nodes[n].get("kind") == "comp"
        }

        best_cur_net: str | None = None
        best_score = -1
        for cur_node in cur_graph.nodes:
            cur_data = cur_graph.nodes[cur_node]
            if cur_data.get("kind") != "net":
                continue
            cur_net_id = cur_data["source_id"]

            cur_neighbors = {
                cur_graph.nodes[n]["source_id"]
                for n in cur_graph.neighbors(cur_node)
                if cur_graph.nodes[n].get("kind") == "comp"
            }

            score = sum(
                1 for ref_comp in ref_neighbors
                if ref_comp in comp_map and comp_map[ref_comp] in cur_neighbors
            )
            if score > best_score:
                best_score = score
                best_cur_net = cur_net_id

        if best_cur_net and best_score > 0:
            net_map[ref_net_id] = best_cur_net

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
