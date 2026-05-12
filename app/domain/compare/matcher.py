from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from app.domain.logical_reference import (
    CRITICAL_ROLE_LABELS,
    normalize_component_type,
    normalize_net_role,
    normalize_role_label,
)

PORT_NET_ROLES = {"input", "output"}
STRICT_NET_ROLES = {"ground", "power"}
PASSIVE_TWO_PIN_TYPES = {"Resistor", "Capacitor", "CapacitorCeramic", "Wire"}
STRICT_PIN_ROLE_TYPES = {"Transistor", "Potentiometer", "LED", "Diode", "CapacitorElectrolytic"}
NON_POLAR_CAPACITOR_TYPES = {"Capacitor", "CapacitorCeramic"}


def _is_isomorphic(reference_graph: nx.Graph, current_graph: nx.Graph) -> bool:
    return _find_isomorphism(reference_graph, current_graph) is not None


def _find_isomorphism(reference_graph: nx.Graph, current_graph: nx.Graph) -> dict[Any, Any] | None:
    if reference_graph.number_of_nodes() != current_graph.number_of_nodes():
        return None
    if reference_graph.number_of_edges() != current_graph.number_of_edges():
        return None
    matcher = GraphMatcher(reference_graph, current_graph, node_match=_node_match, edge_match=_edge_match)
    if not matcher.is_isomorphic():
        return None
    return next(matcher.isomorphisms_iter())


def _mapping_uses_allowed_symmetry(mapping: dict[Any, Any], reference_graph: nx.Graph, current_graph: nx.Graph) -> bool:
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
    if pattern.number_of_nodes() > container.number_of_nodes() or pattern.number_of_edges() > container.number_of_edges():
        return False
    return GraphMatcher(container, pattern, node_match=_node_match, edge_match=_edge_match).subgraph_is_isomorphic()


def _node_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("kind") != b.get("kind"):
        return False
    if a.get("kind") == "comp":
        return _component_types_equivalent(a.get("ctype"), b.get("ctype"))
    role_a = normalize_net_role(a.get("role"))
    role_b = normalize_net_role(b.get("role"))
    if (role_a in STRICT_NET_ROLES or role_b in STRICT_NET_ROLES) and role_a != role_b:
        return False
    label_a = normalize_role_label(a.get("role_label"))
    label_b = normalize_role_label(b.get("role_label"))

    if role_a in PORT_NET_ROLES and label_a in CRITICAL_ROLE_LABELS:
        if role_b != role_a:
            return False
        if not label_b:
            return True
        return _role_labels_equivalent(a, b)
    return True


def _component_types_equivalent(left: Any, right: Any) -> bool:
    left_type = normalize_component_type(left)
    right_type = normalize_component_type(right)
    return left_type == right_type or (left_type in NON_POLAR_CAPACITOR_TYPES and right_type in NON_POLAR_CAPACITOR_TYPES)


def _component_type_key(value: Any) -> str:
    ctype = normalize_component_type(value)
    return "Capacitor" if ctype in NON_POLAR_CAPACITOR_TYPES else ctype


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


def auto_detect_symmetries(reference_graph: nx.Graph) -> list[dict[str, Any]]:
    """Detect interchangeable reference nets with identical local topology."""
    by_signature: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for node, data in reference_graph.nodes(data=True):
        if data.get("kind") != "net":
            continue
        role = normalize_net_role(data.get("role"))
        if role in STRICT_NET_ROLES:
            continue
        label = normalize_role_label(data.get("role_label"))
        if not label:
            continue
        signature = (
            role,
            reference_graph.degree(node),
            tuple(sorted(_net_neighbor_signature(reference_graph, node))),
        )
        by_signature[signature].append(node)

    groups: list[dict[str, Any]] = []
    for nodes in by_signature.values():
        labels = [
            normalize_role_label(reference_graph.nodes[node].get("role_label"))
            for node in nodes
        ]
        labels = [label for label in labels if label]
        if len(labels) < 2:
            continue
        for node in nodes:
            node_label = normalize_role_label(reference_graph.nodes[node].get("role_label"))
            existing = {
                normalize_role_label(value)
                for value in reference_graph.nodes[node].get("allowed_role_labels", []) or []
            }
            existing.update(label for label in labels if label != node_label)
            reference_graph.nodes[node]["allowed_role_labels"] = sorted(existing)
        groups.append({"mode": "swap_allowed", "nets": [labels]})
    return groups


def _net_neighbor_signature(graph: nx.Graph, net_node: str) -> list[tuple[str, str]]:
    signature: list[tuple[str, str]] = []
    for comp_node in graph.neighbors(net_node):
        comp_data = graph.nodes[comp_node]
        if comp_data.get("kind") != "comp":
            continue
        edge_data = graph.get_edge_data(comp_node, net_node) or {}
        signature.append(
            (
                _component_type_key(comp_data.get("ctype") or "UNKNOWN"),
                str(edge_data.get("pin_role") or edge_data.get("pin") or ""),
            )
        )
    return signature


def _find_any_isomorphism_mapping(ref_graph: nx.Graph, cur_graph: nx.Graph) -> dict[Any, Any] | None:
    matcher = GraphMatcher(ref_graph, cur_graph, node_match=_node_match, edge_match=_edge_match)
    if matcher.is_isomorphic():
        return next(matcher.isomorphisms_iter())
    matcher = GraphMatcher(cur_graph, ref_graph, node_match=_node_match, edge_match=_edge_match)
    if matcher.subgraph_is_isomorphic():
        mapping = next(matcher.subgraph_isomorphisms_iter())
        return {v: k for k, v in mapping.items()}
    matcher = GraphMatcher(ref_graph, cur_graph, node_match=_node_match, edge_match=_edge_match)
    if matcher.subgraph_is_isomorphic():
        return next(matcher.subgraph_isomorphisms_iter())
    return None
