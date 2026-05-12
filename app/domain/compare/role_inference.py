from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from app.domain.logical_reference import CRITICAL_ROLE_LABELS, normalize_net_role, normalize_role_label
from .matcher import _component_types_equivalent, _edge_match, _node_match

STRICT_INFERRED_ROLES = {"ground", "power"}


def _infer_current_net_roles_from_reference(reference_graph: nx.Graph, current_graph: nx.Graph, cur_netlist_v2: dict[str, Any]) -> tuple[nx.Graph, dict[str, Any], list[dict[str, Any]]] | None:
    matcher = GraphMatcher(reference_graph, current_graph, node_match=_node_match_for_role_inference, edge_match=_edge_match)
    if not matcher.is_isomorphic():
        return None
    candidate_sets: list[list[dict[str, Any]]] = []
    checked_count = 0
    for mapping in matcher.isomorphisms_iter():
        checked_count += 1
        inferences = _role_inferences_for_mapping(mapping, reference_graph, current_graph)
        if inferences:
            candidate_sets.append(inferences)
        if checked_count >= 50:
            break
    selected_inferences = _select_role_inferences(candidate_sets, current_graph)
    if not selected_inferences:
        return None
    inferred_graph = current_graph.copy()
    inferred_netlist = copy.deepcopy(cur_netlist_v2)
    _apply_role_inferences_to_graph(inferred_graph, selected_inferences)
    _apply_role_inferences_to_netlist(inferred_netlist, selected_inferences)
    return inferred_graph, inferred_netlist, selected_inferences


def _node_match_for_role_inference(ref_data: dict[str, Any], cur_data: dict[str, Any]) -> bool:
    if ref_data.get("kind") != cur_data.get("kind"):
        return False
    if ref_data.get("kind") == "comp":
        return _component_types_equivalent(ref_data.get("ctype"), cur_data.get("ctype"))
    cur_role_source = str(cur_data.get("role_source") or "")
    cur_role = normalize_net_role(cur_data.get("role"))
    ref_role = normalize_net_role(ref_data.get("role"))
    ref_label = normalize_role_label(ref_data.get("role_label"))
    cur_label = normalize_role_label(cur_data.get("role_label"))
    if (
        ref_role in {"input", "output"}
        and ref_label in CRITICAL_ROLE_LABELS
        and cur_label in CRITICAL_ROLE_LABELS
        and cur_label != ref_label
    ):
        return _node_match(ref_data, cur_data)
    if cur_role_source != "default_signal" and cur_role in STRICT_INFERRED_ROLES:
        return _node_match(ref_data, cur_data)
    if cur_role_source != "manual_role":
        return True
    return _node_match(ref_data, cur_data)


def _role_inferences_for_mapping(mapping: dict[Any, Any], reference_graph: nx.Graph, current_graph: nx.Graph) -> list[dict[str, Any]]:
    inferences_by_current: dict[str, dict[str, Any]] = {}
    for ref_node, cur_node in mapping.items():
        ref_data = reference_graph.nodes.get(ref_node, {})
        cur_data = current_graph.nodes.get(cur_node, {})
        if ref_data.get("kind") != "net" or cur_data.get("kind") != "net":
            continue
        if str(cur_data.get("role_source") or "") == "manual_role":
            continue
        ref_role = normalize_net_role(ref_data.get("role"))
        ref_label = normalize_role_label(ref_data.get("role_label"))
        cur_label = normalize_role_label(cur_data.get("role_label"))
        if ref_role == "signal" and ref_label not in CRITICAL_ROLE_LABELS:
            continue
        current_net = str(cur_data.get("source_id") or "")
        record = {
            "reference_net": ref_data.get("source_id"),
            "current_net": current_net,
            "role": ref_role,
            "role_label": ref_label if not cur_label or cur_label == ref_label else "",
            "source": "inferred_from_reference",
        }
        existing = inferences_by_current.get(current_net)
        if existing and (existing.get("role") != record["role"] or existing.get("role_label") != record["role_label"]):
            return []
        inferences_by_current[current_net] = record
    return sorted(inferences_by_current.values(), key=lambda item: str(item.get("current_net") or ""))


def _select_role_inferences(
    candidate_sets: list[list[dict[str, Any]]],
    current_graph: nx.Graph,
) -> list[dict[str, Any]] | None:
    if not candidate_sets:
        return None

    by_current: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_sets:
        for item in candidate:
            by_current[str(item.get("current_net") or "")].append(item)

    selected: dict[str, dict[str, Any]] = {}
    for current_net, items in by_current.items():
        roles = {str(item.get("role") or "signal") for item in items}
        if len(roles) != 1:
            return None
        labels = {normalize_role_label(item.get("role_label")) for item in items}
        best = max(items, key=lambda item: _label_match_score(current_graph, current_net, item))
        selected[current_net] = {
            **best,
            "role": next(iter(roles)),
            "role_label": labels.pop() if len(labels) == 1 else "",
        }
    return sorted(selected.values(), key=lambda item: str(item.get("current_net") or ""))


def _label_match_score(
    current_graph: nx.Graph,
    current_net: str,
    inference: dict[str, Any],
) -> float:
    ref_label = normalize_role_label(inference.get("role_label"))
    if not ref_label:
        return 0.0
    cur_data = current_graph.nodes.get(f"cur_net:{current_net}", {})
    candidates = {
        normalize_role_label(cur_data.get("role_label")),
        normalize_role_label(cur_data.get("canonical_name")),
        *{normalize_role_label(alias) for alias in cur_data.get("aliases", []) or []},
    }
    candidates.discard("")
    if ref_label in candidates:
        return 1.0
    if any(ref_label in value or value in ref_label for value in candidates):
        return 0.5
    return 0.0


def _inference_signature(inferences: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple((item.get("reference_net"), item.get("current_net"), item.get("role"), item.get("role_label")) for item in inferences)


def _apply_role_inferences_to_graph(graph: nx.Graph, inferences: list[dict[str, Any]]) -> None:
    for item in inferences:
        net_node = f"cur_net:{item.get('current_net')}"
        if not graph.has_node(net_node):
            continue
        graph.nodes[net_node]["role"] = item.get("role") or "signal"
        graph.nodes[net_node]["role_label"] = item.get("role_label") or ""
        graph.nodes[net_node]["canonical_name"] = item.get("role_label") or item.get("reference_net") or item.get("current_net")
        graph.nodes[net_node]["role_source"] = "inferred_from_reference"
        graph.nodes[net_node]["inferred_reference_net"] = item.get("reference_net")


def _apply_role_inferences_to_netlist(netlist_v2: dict[str, Any], inferences: list[dict[str, Any]]) -> None:
    by_net = {str(item.get("current_net") or ""): item for item in inferences}
    for net in netlist_v2.get("nets", []) or []:
        if not isinstance(net, dict):
            continue
        net_id = str(net.get("electrical_net_id") or net.get("net_id") or "")
        item = by_net.get(net_id)
        if not item:
            continue
        role = str(item.get("role") or "signal")
        label = str(item.get("role_label") or "")
        net["role"] = role
        net["role_label"] = label
        net["canonical_name"] = label or str(item.get("reference_net") or net_id)
        net["aliases"] = list(dict.fromkeys([net["canonical_name"], net_id, *list(net.get("aliases") or [])]))
        net["role_source"] = "inferred_from_reference"
        net["inferred_reference_net"] = item.get("reference_net")
        if role == "power":
            net["power_role"] = label if label in {"VCC", "VEE", "VDD", "VSS"} else "VCC"
        elif role == "ground":
            net["power_role"] = "GND"


def _attach_role_inferences(result: dict[str, Any], inferences: list[dict[str, Any]]) -> None:
    details = dict(result.get("details", {}))
    details["role_inference_applied"] = True
    details["inferred_net_roles"] = inferences
    result["details"] = details
    report = dict(result.get("report", {}))
    summary = dict(report.get("summary", {}))
    summary["role_inference_applied"] = True
    summary["inferred_net_roles"] = inferences
    report["summary"] = summary
    result["report"] = report
