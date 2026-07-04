from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from .diff_report import (
    _approximate_similarity,
    _component_count,
    _component_progress,
    _critical_extra_items,
    _dedupe_items,
    _difference_items,
    _enrich_result,
    _extra_items,
    _ged_similarity,
    _item,
    _missing_items,
    _result,
)
from .matcher import (
    _contains_subgraph,
    _find_isomorphism,
    _mapping_uses_allowed_symmetry,
    auto_detect_symmetries,
)
from .role_inference import (
    _attach_role_inferences,
    _infer_current_net_roles_from_reference,
)

log = logging.getLogger(__name__)


def _is_wire_component(data: dict[str, Any]) -> bool:
    ctype = str(data.get("ctype") or data.get("component_type") or "")
    return ctype in {"Wire", "Jumper"}


def _collapse_wire_components(graph: nx.Graph) -> nx.Graph:
    """Collapse Wire/Jumper components into direct net unions for compare.

    The visual detector may expose ordinary jumper wires as components, but
    logical comparison should treat them as conductors. We keep the original
    graph for critical bridge auditing, and compare against this collapsed
    graph so benign jumpers do not create extra-component noise.
    """
    wire_nodes = {
        node
        for node, data in graph.nodes(data=True)
        if data.get("kind") == "comp" and _is_wire_component(data)
    }
    if not wire_nodes:
        return graph

    parent: dict[Any, Any] = {}

    def find(node: Any) -> Any:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: Any, b: Any) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    net_nodes = {
        node
        for node, data in graph.nodes(data=True)
        if data.get("kind") == "net"
    }
    for node in net_nodes:
        parent[node] = node

    for wire_node in wire_nodes:
        nets = [
            neighbor
            for neighbor in graph.neighbors(wire_node)
            if graph.nodes[neighbor].get("kind") == "net"
        ]
        if len(nets) < 2:
            continue
        first = nets[0]
        for net in nets[1:]:
            union(first, net)

    groups: dict[Any, list[Any]] = {}
    for node in net_nodes:
        groups.setdefault(find(node), []).append(node)

    collapsed = nx.Graph()
    net_to_rep: dict[Any, Any] = {}
    for root, members in groups.items():
        rep = min(members, key=str)
        for member in members:
            net_to_rep[member] = rep
        collapsed.add_node(rep, **_merged_net_attrs(graph, members, rep))

    for node, data in graph.nodes(data=True):
        if node in wire_nodes or data.get("kind") == "net":
            continue
        collapsed.add_node(node, **dict(data))

    for left, right, data in graph.edges(data=True):
        if left in wire_nodes or right in wire_nodes:
            continue
        mapped_left = net_to_rep.get(left, left)
        mapped_right = net_to_rep.get(right, right)
        if mapped_left == mapped_right:
            continue
        collapsed.add_edge(mapped_left, mapped_right, **dict(data))

    collapsed.graph.update(graph.graph)
    return collapsed


def _merged_net_attrs(graph: nx.Graph, members: list[Any], rep: Any) -> dict[str, Any]:
    attrs = dict(graph.nodes[rep])
    if len(members) == 1:
        return attrs

    roles = [str(graph.nodes[node].get("role") or "signal") for node in members]
    labels = [
        str(graph.nodes[node].get("role_label") or graph.nodes[node].get("canonical_name") or "")
        for node in members
        if graph.nodes[node].get("role_label") or graph.nodes[node].get("canonical_name")
    ]
    source_ids = [
        str(graph.nodes[node].get("source_id") or str(node).split(":", 1)[-1])
        for node in members
    ]
    for role in ("ground", "power", "input", "output", "signal"):
        if role in roles:
            attrs["role"] = role
            break
    if labels:
        attrs["role_label"] = labels[0]
        attrs["canonical_name"] = labels[0]
    attrs["merged_source_ids"] = list(dict.fromkeys(source_ids))
    attrs["source_id"] = source_ids[0]
    return attrs


def _promote_critical_extras(
    result: dict[str, Any],
    reference_graph: nx.Graph,
    current_graph: nx.Graph,
    *,
    wire_only: bool = False,
) -> dict[str, Any]:
    """When a nominally equivalent circuit has critical extras, fail hard.

    If the rule path returns ``logic_correct=True`` via the
    ``equivalent_with_extra`` branch and the current graph has extra edges on
    a role-critical net (power / ground / input / output), promote to a hard
    fail.

    Layered after :func:`_enrich_result` so the detailed wiring-mismatch
    items it produces survive. Idempotent on circuits with no critical
    extras (no-op).

    Plan §一 invariant: when the rule path itself decides to flip
    ``logic_correct`` based on its own analysis, the rule owns the verdict.
    """

    critical = _critical_extra_items(reference_graph, current_graph)
    if wire_only:
        critical = [
            item for item in critical
            if "bridged_critical_nets" in ((item.get("actual") or {}))
        ]
    if not critical:
        return result

    # Append items to both result.items + report.items (mirroring R2 shape).
    result_items = list(result.get("items", []) or [])
    result_items.extend(critical)
    result["items"] = result_items
    result["logic_correct"] = False
    result["is_correct"] = False
    result["is_match"] = False
    result["message"] = (
        "参考电路逻辑已存在，但在关键网络（电源 / 地 / 输入 / 输出）上"
        "检测到多余连接，可能影响电路核心功能。"
    )
    # Update similarity floor (a circuit with critical extras is further
    # from ref than a clean equivalent_with_extra).
    result["similarity"] = min(
        result.get("similarity", 1.0),
        max(0.70, _approximate_similarity(reference_graph, current_graph)),
    )

    details = dict(result.get("details", {}))
    details["match_type"] = "extra_on_critical_net"
    critical_details: list[dict[str, Any]] = []
    for item in critical:
        expected = item.get("expected") or {}
        actual = item.get("actual") or {}
        if "role" in expected:
            critical_details.append(
                {
                    "role": expected["role"],
                    "extra_edges": (
                        actual["edge_count_on_role_nets"]
                        - expected["edge_count_on_role_nets"]
                    ),
                }
            )
        else:
            critical_details.append(
                {
                    "wire_component": actual.get("wire_component"),
                    "bridged_critical_nets": actual.get("bridged_critical_nets", []),
                }
            )
    details["critical_extras"] = critical_details
    result["details"] = details

    report = dict(result.get("report", {}))
    summary = dict(report.get("summary", {}))
    summary["logic_correct"] = False
    summary["match_type"] = "extra_on_critical_net"
    summary["similarity"] = round(float(result["similarity"]), 3)
    report["summary"] = summary

    report_items = list(report.get("items", []) or [])
    report_items.extend(critical)
    report["items"] = report_items
    summary["total_item_count"] = len(report_items)
    # Move critical items into the topology_errors bucket as the
    # validator_report_v2 convention.
    report["topology_errors"] = list(
        report.get("topology_errors", []) or []
    ) + critical
    result["report"] = report

    return result


def _apply_phase_e_propagation(
    ref_payload: dict[str, Any],
    cur_netlist_v2: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return no inferred role propagation records.

    Reference comparison now stays on the deterministic rule/DSL path only.
    """
    return []


def compare_logical_graphs(
    reference_graph: nx.Graph,
    current_graph: nx.Graph,
    *,
    ref_payload: dict[str, Any] | None = None,
    cur_netlist_v2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """比较参考电路图与当前电路图，返回包含差异报告的比较结果。

    当前比较路径保持确定性：先折叠普通跳线，再做结构匹配、角色补全和
    diff report enrichment。旧的模型辅助传播路径已移除。
    """
    raw_current_graph = current_graph
    current_graph = _collapse_wire_components(current_graph)
    role_inferences: list[dict[str, Any]] = []
    inference_applied = False
    auto_symmetry_groups: list[dict[str, Any]] = []

    # Keep the comparison path deterministic. The removed model-assisted
    # propagation path used to enrich net names here; the current demo flow
    # relies on explicit annotations and rule-based enrichment only.
    if ref_payload is not None and cur_netlist_v2 is not None:
        phase_e_records = _apply_phase_e_propagation(ref_payload, cur_netlist_v2)
        if phase_e_records:
            role_inferences.extend(phase_e_records)
            inference_applied = True

    if not reference_graph.graph.get("symmetry_groups"):
        auto_symmetry_groups = auto_detect_symmetries(reference_graph)
    iso_mapping = _find_isomorphism(reference_graph, current_graph)
    if iso_mapping is None and ref_payload is not None and cur_netlist_v2 is not None:
        # Legacy exact-isomorphism fallback. It rarely fires on real boards with
        # extra jumper wires, but remains useful for perfectly aligned graphs.
        inferred = _infer_current_net_roles_from_reference(
            reference_graph,
            current_graph,
            cur_netlist_v2,
        )
        if inferred is not None:
            inferred_graph, inferred_netlist, legacy_inferences = inferred
            inferred_mapping = _find_isomorphism(reference_graph, inferred_graph)
            if inferred_mapping is not None:
                current_graph = inferred_graph
                cur_netlist_v2 = inferred_netlist
                iso_mapping = inferred_mapping
                role_inferences.extend(legacy_inferences)
                inference_applied = True

    if iso_mapping is not None:
        match_type = "full_isomorphism"
        if _mapping_uses_allowed_symmetry(iso_mapping, reference_graph, current_graph):
            match_type = "equivalent_with_allowed_symmetry"
        if inference_applied:
            match_type = "full_isomorphism_with_inferred_roles"
        details = {"match_type": match_type}
        if auto_symmetry_groups:
            details["auto_symmetry_groups"] = auto_symmetry_groups
        result = _result(
            logic_correct=True,
            similarity=1.0,
            progress=1.0,
            message="电路逻辑连接与参考电路一致",
            items=[],
            details=details,
            ref_payload=ref_payload,
        )
        if ref_payload is not None and cur_netlist_v2 is not None:
            result = _enrich_result(
                result,
                reference_graph,
                current_graph,
                ref_payload,
                cur_netlist_v2,
            )
        if inference_applied:
            _attach_role_inferences(result, role_inferences)
        result = _promote_critical_extras(
            result, reference_graph, raw_current_graph, wire_only=True
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
                result,
                reference_graph,
                current_graph,
                ref_payload,
                cur_netlist_v2,
            )
        # R1 Position B (RULE_SEMANTICS §3) — promote extras on
        # role-critical nets (power / ground / input / output) to a
        # hard fail. Extras on signal / internal nets stay as warnings
        # under the unchanged ``equivalent_with_extra`` match type.
        # Layered AFTER enrich so detailed_items from enrich aren't lost.
        result = _promote_critical_extras(
            result, reference_graph, raw_current_graph
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
                result,
                reference_graph,
                current_graph,
                ref_payload,
                cur_netlist_v2,
            )
        result = _promote_critical_extras(
            result, reference_graph, raw_current_graph, wire_only=True
        )
        return result

    result = _result(
        logic_correct=False,
        similarity=_ged_similarity(reference_graph, current_graph),
        progress=_component_progress(reference_graph, current_graph),
        message="检测到元件连接关系与参考电路不一致，可能存在错接。",
        items=_difference_items(reference_graph, current_graph),
        details={"match_type": "graph_edit_distance_or_fallback"},
        ref_payload=ref_payload,
    )
    if ref_payload is not None and cur_netlist_v2 is not None:
        result = _enrich_result(
            result,
            reference_graph,
            current_graph,
            ref_payload,
            cur_netlist_v2,
        )
    result = _promote_critical_extras(
        result, reference_graph, raw_current_graph, wire_only=True
    )
    return result
