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

# P4.1 R2 — disagreement warning thresholds (see RISK_REGISTER.md §5 R2).
# A wrong-edge call needs `p_correct < _GNN_WRONG_EDGE_P_FLOOR` for the
# disagreement warning to fire; using a stricter floor than the SEAL
# decision threshold (0.5) keeps low-noise warnings on borderline edges.
_GNN_WRONG_EDGE_P_FLOOR = 0.3
_GNN_DISAGREE_ERROR_CODE = "WARN_GNN_DISAGREES_WITH_RULE"

log = logging.getLogger(__name__)


def _promote_critical_extras(
    result: dict[str, Any],
    reference_graph: nx.Graph,
    current_graph: nx.Graph,
) -> dict[str, Any]:
    """**R1 Position B** (RULE_SEMANTICS §3) — when the rule path has
    returned ``logic_correct=True`` via the ``equivalent_with_extra``
    branch **and** the cur graph has extra edges on a role-critical net
    (power / ground / input / output), promote to a hard fail.

    Layered after :func:`_enrich_result` so the detailed wiring-mismatch
    items it produces survive. Idempotent on circuits with no critical
    extras (no-op).

    Plan §一 invariant: when the rule path itself decides to flip
    ``logic_correct`` based on its own analysis, that's not a violation —
    the rule is owning the verdict. (GNN cannot do this, only the rule
    can. R2's warning items still remain advisory.)
    """

    critical = _critical_extra_items(reference_graph, current_graph)
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
    details["critical_extras"] = [
        {
            "role": item["expected"]["role"],
            "extra_edges": (
                item["actual"]["edge_count_on_role_nets"]
                - item["expected"]["edge_count_on_role_nets"]
            ),
        }
        for item in critical
    ]
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


def _maybe_attach_gnn_advice(
    result: dict[str, Any],
    ref_payload: dict[str, Any] | None,
    cur_netlist_v2: dict[str, Any] | None,
    ref_graph: nx.Graph,
    cur_graph: nx.Graph,
) -> dict[str, Any]:
    """**P4 hook (plan §六)** — annotate ``report.summary.gnn`` and
    ``details.gnn`` with advisory output from :class:`GNNAdvisor`.

    Hard rules (plan §一 "GNN 永远不直接决定 pass/fail"):
        - **Never** mutate ``logic_correct`` / ``is_correct`` / ``is_match``.
        - On any failure (import, missing checkpoint, model exception),
          silently return ``result`` unchanged so the rule path stays the
          authoritative comparator.
        - The added ``gnn`` field is purely additive — existing items /
          mappings / summary keys are untouched.
    """

    if ref_payload is None or cur_netlist_v2 is None:
        return result
    try:
        from app.domain.gnn import GNNAdvisor, should_use_gnn
        from app.domain.gnn.port_graph import (
            build_from_logical_reference,
            build_from_netlist_v2,
        )
    except ImportError:
        return result

    # Build a minimal context for the §七 trigger logic.
    n_total = ref_graph.number_of_nodes() + cur_graph.number_of_nodes()
    match_type = (result.get("details") or {}).get("match_type")
    ctx = {
        "node_count_total": n_total,
        "match_type_so_far": match_type,
        "full_isomorphism_failed": match_type != "full_isomorphism",
    }
    if not should_use_gnn(ctx):
        return result

    try:
        advisor = GNNAdvisor.get()
        ref_hcg = build_from_logical_reference(ref_payload)
        cur_hcg = build_from_netlist_v2(cur_netlist_v2)
        advice = advisor.advise(ref_hcg, cur_hcg, timeout_ms=300)
    except (RuntimeError, FileNotFoundError) as e:
        # Expected: no checkpoint / no torch / no model on this box.
        log.debug("gnn_advisor_unavailable: %s", e)
        return result
    except Exception as e:  # noqa: BLE001 — never let GNN crash rule path
        log.warning(
            "gnn_advisor_failed: %s — keeping rule-only report",
            type(e).__name__,
            exc_info=e,
        )
        return result
    if advice is None:
        return result

    advice_dict = advice.to_report_dict()

    # P4.1 R2 (RISK_REGISTER §5) — only the rule_pass + GNN_flags_wrong_edge
    # direction is actionable for the false_pass red line. Compute it here
    # rather than in the advisor so the rule verdict (which the advisor
    # never sees) is the source of truth.
    rule_pass = bool(result.get("logic_correct", False))
    suspicious_edges = [
        e for e in advice_dict.get("edge_predictions", [])
        if float(e.get("p_correct", 1.0)) < _GNN_WRONG_EDGE_P_FLOOR
    ]
    disagreement = rule_pass and len(suspicious_edges) > 0
    advice_dict["disagreement_with_rule"] = disagreement

    # Stuff into report.summary.gnn (plan §六 validator_report_v2 schema)
    report = dict(result.get("report", {}))
    summary = dict(report.get("summary", {}))
    summary["gnn"] = advice_dict
    report["summary"] = summary

    if disagreement:
        # Plan §一: NEVER flip logic_correct. severity="warning" keeps the
        # _dedupe / promotion logic in _enrich_result inert (it only promotes
        # to logic_correct=False on "error" severity).
        worst = min(suspicious_edges, key=lambda e: float(e["p_correct"]))
        warning_item = _item(
            _GNN_DISAGREE_ERROR_CODE,
            "gnn_advisory",
            "warning",
            (
                "规则比较器认为电路通过，但 GNN 怀疑有 "
                f"{len(suspicious_edges)} 条引脚连接异常 "
                f"(最低 P(edge_correct)={float(worst['p_correct']):.2f})。"
                "请人工复核高亮的引脚。"
            ),
            expected={"rule_logic_correct": True},
            actual={
                "gnn_suspicious_edges": [
                    {
                        "edge": e["edge"],
                        "p_correct": float(e["p_correct"]),
                    }
                    for e in suspicious_edges
                ],
                "gnn_model_version": advice_dict["model_version"],
            },
            suggested_action=(
                "GNN 给出的建议是 advisory，最终判定仍以规则为准。"
                "若复核发现确为错接，请补充规则或重新训练模型。"
            ),
            evidence={
                "p_floor": _GNN_WRONG_EDGE_P_FLOOR,
                "inference_ms": advice_dict["inference_ms"],
            },
        )
        report_items = list(report.get("items", []) or [])
        report_items.append(warning_item)
        report["items"] = report_items
        # Bump summary total_item_count so consumers reading report
        # standalone see the correct count.
        summary["total_item_count"] = len(report_items)

        result_items = list(result.get("items", []) or [])
        result_items.append(warning_item)
        result["items"] = result_items

    result["report"] = report
    # Mirror to details so non-report consumers can see it too
    details = dict(result.get("details", {}))
    details["gnn"] = {
        "enabled": True,
        "model_version": advice_dict["model_version"],
        "inference_ms": advice_dict["inference_ms"],
        "n_edges_scored": advice_dict["n_edges_scored"],
        "disagreement_with_rule": disagreement,
        "n_suspicious_edges": len(suspicious_edges),
    }
    result["details"] = details
    return result


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
        inferred = _infer_current_net_roles_from_reference(
            reference_graph,
            current_graph,
            cur_netlist_v2,
        )
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
            result = _maybe_attach_gnn_advice(
                result,
                ref_payload,
                cur_netlist_v2,
                reference_graph,
                current_graph,
            )
        if inference_applied:
            _attach_role_inferences(result, role_inferences)
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
            result = _maybe_attach_gnn_advice(
                result,
                ref_payload,
                cur_netlist_v2,
                reference_graph,
                current_graph,
            )
        # R1 Position B (RULE_SEMANTICS §3) — promote extras on
        # role-critical nets (power / ground / input / output) to a
        # hard fail. Extras on signal / internal nets stay as warnings
        # under the unchanged ``equivalent_with_extra`` match type.
        # Layered AFTER enrich so detailed_items from enrich aren't lost.
        result = _promote_critical_extras(
            result, reference_graph, current_graph
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
            result = _maybe_attach_gnn_advice(
                result,
                ref_payload,
                cur_netlist_v2,
                reference_graph,
                current_graph,
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
        result = _maybe_attach_gnn_advice(
            result,
            ref_payload,
            cur_netlist_v2,
            reference_graph,
            current_graph,
        )
    return result
