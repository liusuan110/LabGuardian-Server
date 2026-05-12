from __future__ import annotations

from typing import Any

import networkx as nx

from .diff_report import (
    _approximate_similarity,
    _component_count,
    _component_progress,
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
from .role_inference import _attach_role_inferences, _infer_current_net_roles_from_reference


def compare_logical_graphs(
    reference_graph: nx.Graph,
    current_graph: nx.Graph,
    *,
    ref_payload: dict[str, Any] | None = None,
    cur_netlist_v2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """比较参考电路图与当前电路图，返回包含差异报告的比较结果。"""
    role_inferences: list[dict[str, Any]] = []
    inference_applied = False
    auto_symmetry_groups: list[dict[str, Any]] = []
    if not reference_graph.graph.get("symmetry_groups"):
        auto_symmetry_groups = auto_detect_symmetries(reference_graph)
    iso_mapping = _find_isomorphism(reference_graph, current_graph)
    if iso_mapping is None and ref_payload is not None and cur_netlist_v2 is not None:
        inferred = _infer_current_net_roles_from_reference(reference_graph, current_graph, cur_netlist_v2)
        if inferred is not None:
            inferred_graph, inferred_netlist, role_inferences = inferred
            inferred_mapping = _find_isomorphism(reference_graph, inferred_graph)
            if inferred_mapping is not None:
                current_graph = inferred_graph
                cur_netlist_v2 = inferred_netlist
                iso_mapping = inferred_mapping
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
        result = _result(logic_correct=True, similarity=1.0, progress=1.0, message="电路逻辑连接与参考电路一致", items=[], details=details, ref_payload=ref_payload)
        if ref_payload is not None and cur_netlist_v2 is not None:
            result = _enrich_result(result, reference_graph, current_graph, ref_payload, cur_netlist_v2)
        if inference_applied:
            _attach_role_inferences(result, role_inferences)
        return result

    if _contains_subgraph(current_graph, reference_graph):
        items = _extra_items(reference_graph, current_graph)
        if not items:
            items = [_item("EXTRA_CONNECTION", "extra_connection", "warning", "参考电路逻辑已存在，但当前电路包含额外连接。", expected={}, actual={}, suggested_action="请检查是否有多余连接。")]
        result = _result(logic_correct=True, similarity=max(0.85, _approximate_similarity(reference_graph, current_graph)), progress=1.0, message="参考电路逻辑已存在，但当前电路包含额外元件或连接", items=items, details={"match_type": "equivalent_with_extra"}, ref_payload=ref_payload)
        if ref_payload is not None and cur_netlist_v2 is not None:
            result = _enrich_result(result, reference_graph, current_graph, ref_payload, cur_netlist_v2)
        return result

    if _contains_subgraph(reference_graph, current_graph):
        items = _missing_items(reference_graph, current_graph)
        items.append(_item("INCOMPLETE_CIRCUIT", "incomplete_circuit", "error", "当前电路只匹配到参考电路的一部分。", expected={"reference_component_count": _component_count(reference_graph)}, actual={"current_component_count": _component_count(current_graph)}, suggested_action="请补齐缺失元件或连接后重新验证。"))
        result = _result(logic_correct=False, similarity=_approximate_similarity(reference_graph, current_graph), progress=_component_progress(reference_graph, current_graph), message="当前电路未完整实现参考电路逻辑", items=_dedupe_items(items), details={"match_type": "current_subgraph_in_reference"}, ref_payload=ref_payload)
        if ref_payload is not None and cur_netlist_v2 is not None:
            result = _enrich_result(result, reference_graph, current_graph, ref_payload, cur_netlist_v2)
        return result

    result = _result(logic_correct=False, similarity=_ged_similarity(reference_graph, current_graph), progress=_component_progress(reference_graph, current_graph), message="检测到元件连接关系与参考电路不一致，可能存在错接。", items=_difference_items(reference_graph, current_graph), details={"match_type": "graph_edit_distance_or_fallback"}, ref_payload=ref_payload)
    if ref_payload is not None and cur_netlist_v2 is not None:
        result = _enrich_result(result, reference_graph, current_graph, ref_payload, cur_netlist_v2)
    return result
