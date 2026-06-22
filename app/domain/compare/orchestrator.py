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
from .role_propagation import propagate_canonical_via_alignment

# P4.1 R2 — disagreement warning thresholds (see RISK_REGISTER.md §5 R2).
# A wrong-edge call needs `p_correct < _GNN_WRONG_EDGE_P_FLOOR` for the
# disagreement warning to fire; using a stricter floor than the SEAL
# decision threshold (0.5) keeps low-noise warnings on borderline edges.
_GNN_WRONG_EDGE_P_FLOOR = 0.3
_GNN_DISAGREE_ERROR_CODE = "WARN_GNN_DISAGREES_WITH_RULE"

# Diagnostic codes written to ``report.summary.gnn_disabled_reason`` (and the
# ``details.gnn_disabled_reason`` mirror) when the GNN hook silently bails
# out. Plan §一 keeps the advisor invisible to ``logic_correct``, but having
# **why** the advisor sat out the call observable in the JSON output saves
# hours of guesswork ("is torch missing?" / "is the checkpoint mounted?").
_GNN_REASON_RUNTIME_UNAVAILABLE = "runtime_unavailable"   # torch / pyg ImportError
_GNN_REASON_CHECKPOINT_MISSING = "checkpoint_missing"     # no .pt on disk
_GNN_REASON_TINY_CIRCUIT = "tiny_circuit"                  # ≥ 8-node trigger
_GNN_REASON_TRIGGER_SKIPPED = "trigger_predicate_skipped"  # safety / polarity etc.
_GNN_REASON_MODEL_FAILED = "model_failed"                  # advisor.advise crashed
# Phase C Stage 5 (2026-05-20) — when the GNN is clearly OOD on the
# current circuit (most observed edges flagged AND minimum p_correct is
# essentially zero), suppress R2 warnings and surface this reason so the
# user / UI knows GNN sat out *because* it couldn't form a useful
# opinion. Prevents the v3-era "GNN suspects 9/9 edges with p<0.001"
# pseudo-alarm from reaching the screen.
_GNN_REASON_OOD_DISAGREEMENT = "ood_disagreement_too_broad"
# GNN 路径整体弃用:项目决定逻辑比对只走 DSL 确定性比对,不再附加任何 GNN advisory。
_GNN_REASON_DEPRECATED = "deprecated_dsl_only"
# (`payload_missing` intentionally not emitted — the caller is signalling "I
# don't want GNN this time" by withholding ref_payload / cur_netlist_v2.)


# CADx Phase 0 (2026-05-22) — version tag for ``details.template_match`` so
# the frontend / report consumers can negotiate schema changes later.
_TEMPLATE_MATCH_VERSION = "cadx_phase0_v1"
# Number of top template hypotheses surfaced. Keep small for UI density.
_TEMPLATE_MATCH_TOP_K = 3

# Phase C Stage 5 — OOD self-suppression thresholds.
_GNN_OOD_SUSPICIOUS_RATIO_FLOOR = 0.5   # > 50% of observed edges flagged
_GNN_OOD_WORST_P_FLOOR = 0.05           # worst p_correct < this → noise, not signal

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


def _attach_template_match(
    result: dict[str, Any],
    current_graph: nx.Graph,
) -> dict[str, Any]:
    """CADx Phase 0 — run topology template matching as a read-only side
    channel and attach top-K results to ``details.template_match``.

    This function is **exception-safe**: any failure inside the template
    layer is swallowed (logged + surfaced as ``template_match_error`` in
    details). It must never change ``logic_correct`` / ``similarity`` /
    ``items`` — Phase 0 keeps the legacy Phase E verdict authoritative
    and only adds a parallel hypothesis for UI comparison.

    Args:
        result: The mutable result dict produced by ``compare_logical_graphs``.
            ``details`` is created if missing.
        current_graph: The bipartite student graph (post wire-collapse).

    Returns:
        The same ``result`` dict, mutated in place and returned for
        fluent chaining.
    """
    try:
        from app.domain.templates import (
            get_template_registry,
            match_all_templates,
        )

        registry = get_template_registry()
        all_results = match_all_templates(current_graph, registry)
        top_k = [r.to_dict() for r in all_results[:_TEMPLATE_MATCH_TOP_K]]
        details = result.setdefault("details", {})
        details["template_match"] = {
            "version": _TEMPLATE_MATCH_VERSION,
            "top_k": top_k,
        }
    except Exception as exc:  # noqa: BLE001 — must never break compare
        log.warning(
            "template_match_failed err=%s",
            type(exc).__name__,
            exc_info=exc,
        )
        details = result.setdefault("details", {})
        details["template_match_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
    return result


def _set_gnn_disabled_reason(result: dict[str, Any], reason: str) -> None:
    """Annotate ``report.summary.gnn_disabled_reason`` (+ ``details``
    mirror) so the JSON output explains why the GNN hook sat out the
    call. Plan §一 still applies — this is metadata only, ``logic_correct``
    untouched. See module-level ``_GNN_REASON_*`` codes."""

    report = dict(result.get("report", {}))
    summary = dict(report.get("summary", {}))
    summary["gnn_disabled_reason"] = reason
    report["summary"] = summary
    result["report"] = report
    details = dict(result.get("details", {}))
    details["gnn_disabled_reason"] = reason
    result["details"] = details


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
          silently return ``result`` with a ``gnn_disabled_reason`` field
          explaining why — rule path stays the authoritative comparator,
          but the JSON now carries enough info to diagnose
          "why is .gnn missing?" without reading server logs.
        - The added ``gnn`` field is purely additive — existing items /
          mappings / summary keys are untouched.
    """

    # GNN 路径已弃用(逻辑比对只用 DSL 确定性比对)。直接 no-op:不构建 advisor、
    # 不附加 edge_predictions / graph_similarity,避免误导性的相似度噪声进入报告。
    # rule path(确定性比对)是唯一权威判定。
    _set_gnn_disabled_reason(result, _GNN_REASON_DEPRECATED)
    return result

    if ref_payload is None or cur_netlist_v2 is None:
        # Caller is intentional (debug script, fast-path bypass). Do
        # not pollute output with a reason code.
        return result
    try:
        from app.domain.gnn import GNNAdvisor, should_use_gnn
        from app.domain.gnn.port_graph import (
            build_from_logical_reference,
            build_from_netlist_v2,
        )
    except ImportError:
        _set_gnn_disabled_reason(result, _GNN_REASON_RUNTIME_UNAVAILABLE)
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
        # Differentiate why so deploy-time / data-time issues are
        # distinguishable from genuine "small circuit, skip GNN" cases.
        if not GNNAdvisor.runtime_available():
            reason = _GNN_REASON_RUNTIME_UNAVAILABLE
        elif not GNNAdvisor.checkpoint_available():
            reason = _GNN_REASON_CHECKPOINT_MISSING
        elif n_total > 0 and n_total < 8:
            reason = _GNN_REASON_TINY_CIRCUIT
        else:
            reason = _GNN_REASON_TRIGGER_SKIPPED
        _set_gnn_disabled_reason(result, reason)
        return result

    try:
        advisor = GNNAdvisor.get()
        ref_hcg = build_from_logical_reference(ref_payload)
        cur_hcg = build_from_netlist_v2(cur_netlist_v2)
        advice = advisor.advise(ref_hcg, cur_hcg, timeout_ms=300)
    except FileNotFoundError as e:
        log.debug("gnn_advisor_unavailable (no checkpoint): %s", e)
        _set_gnn_disabled_reason(result, _GNN_REASON_CHECKPOINT_MISSING)
        return result
    except RuntimeError as e:
        # _ensure_gnn_runtime / model load failure
        log.debug("gnn_advisor_unavailable (runtime): %s", e)
        _set_gnn_disabled_reason(result, _GNN_REASON_RUNTIME_UNAVAILABLE)
        return result
    except Exception as e:  # noqa: BLE001 — never let GNN crash rule path
        log.warning(
            "gnn_advisor_failed: %s — keeping rule-only report",
            type(e).__name__,
            exc_info=e,
        )
        _set_gnn_disabled_reason(result, _GNN_REASON_MODEL_FAILED)
        return result
    if advice is None:
        _set_gnn_disabled_reason(result, _GNN_REASON_MODEL_FAILED)
        return result

    advice_dict = advice.to_report_dict()

    # ----- 显示文案注入（demo / 人类可读） -----------------------------
    # ``_enrich_result`` 在前面已经把 ref↔cur 映射写进了 summary。借这些
    # 映射 + ref_payload 的 role/role_label 给每个 cur_port / cur_net
    # 加 ``*_display`` 字段，让 JSON 在演示场景下不再充斥 NET_004 / IC1.pin4
    # 这种内部 id。raw id 保留以便前端高亮联动。
    summary_for_display = (result.get("report") or {}).get("summary") or {}
    try:
        from .gnn_display import enrich_advice_with_display
        advice_dict = enrich_advice_with_display(
            advice_dict, ref_payload, cur_netlist_v2, summary_for_display
        )
    except Exception as e:  # noqa: BLE001
        # 显示层失败永远不能阻塞 advisory 主流程
        log.debug("gnn_display_enrich_failed: %s", type(e).__name__)

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

    # Phase C Stage 5 · OOD 自抑制门 ----------------------------------------
    # 如果**多数边都被怀疑**（>50%）**且**最低分接近 0（<0.05），那不是
    # "GNN 挑出可疑边"，而是"GNN 在该电路上完全 OOD，每条边都乱叫"。这
    # 种 R2 警告毫无信息量，强制静音并把原因写进 gnn_disabled_reason 供
    # 调试。原始的 edge_predictions 仍在 summary.gnn 里供开发查看。
    n_scored = advice_dict.get("n_edges_scored", 0) or 0
    if (
        disagreement
        and n_scored > 0
        and len(suspicious_edges) / n_scored > _GNN_OOD_SUSPICIOUS_RATIO_FLOOR
    ):
        worst_p = min(
            float(e["p_correct"]) for e in suspicious_edges
        )
        if worst_p < _GNN_OOD_WORST_P_FLOOR:
            disagreement = False
            log.info(
                "gnn_advise: OOD self-suppression engaged "
                "(suspicious=%d/%d, worst_p=%.4f) — R2 warning silenced",
                len(suspicious_edges), n_scored, worst_p,
            )
            _set_gnn_disabled_reason(result, _GNN_REASON_OOD_DISAGREEMENT)

    advice_dict["disagreement_with_rule"] = disagreement

    # Stuff into report.summary.gnn (plan §六 validator_report_v2 schema)
    report = dict(result.get("report", {}))
    summary = dict(report.get("summary", {}))
    summary["gnn"] = advice_dict
    report["summary"] = summary

    # Build {port_id: suggested_targets_entry} index so the warning item
    # can show "you wired pin X to net A, but GNN thinks net B (p=0.92)".
    # Same index also exposes floating REQUIRED suggestions to consumers.
    suggestions_by_port: dict[str, dict[str, Any]] = {}
    for entry in advice_dict.get("suggested_targets", []) or []:
        port_id = entry.get("port")
        if isinstance(port_id, str):
            suggestions_by_port[port_id] = entry

    if disagreement:
        # Plan §一: NEVER flip logic_correct. severity="warning" keeps the
        # _dedupe / promotion logic in _enrich_result inert (it only promotes
        # to logic_correct=False on "error" severity).
        worst = min(suspicious_edges, key=lambda e: float(e["p_correct"]))

        suspicious_actual: list[dict[str, Any]] = []
        for e in suspicious_edges:
            edge = e.get("edge") or []
            port_id_for_edge = edge[0] if edge else None
            suggestion = suggestions_by_port.get(port_id_for_edge) if isinstance(port_id_for_edge, str) else None
            entry: dict[str, Any] = {
                "edge": edge,
                "p_correct": float(e["p_correct"]),
            }
            if suggestion is not None:
                entry["suggested_targets"] = suggestion.get("candidates", [])
            suspicious_actual.append(entry)

        # Enrich the warning's actual with display labels (mirrors what
        # got attached to advice_dict above). Best-effort; on failure we
        # keep the raw IDs.
        try:
            from .gnn_display import enrich_suspicious_edges_for_warning
            suspicious_actual = enrich_suspicious_edges_for_warning(
                suspicious_actual,
                advice_dict,
                ref_payload,
                cur_netlist_v2,
                summary_for_display,
            )
        except Exception as e:  # noqa: BLE001
            log.debug("gnn_display_enrich_warning_failed: %s", type(e).__name__)

        # Hint text: include the top-1 suggestion for the lowest-p edge
        # when we have one, so the user sees a concrete "接到哪" instead
        # of just "有 N 条可疑". Use display labels where available so
        # the on-screen message reads "U1 · pin2 (反相输入) 改接到
        # VOUT (输出)" instead of raw IDs.
        worst_edge = worst.get("edge") or []
        worst_port = worst_edge[0] if worst_edge else None
        worst_suggestion = (
            suggestions_by_port.get(worst_port)
            if isinstance(worst_port, str) else None
        )
        if worst_suggestion and worst_suggestion.get("candidates"):
            top1 = worst_suggestion["candidates"][0]
            # advice_dict was already enriched; pull display for worst_port + top1.net
            enriched_targets = {
                t.get("port"): t for t in advice_dict.get("suggested_targets", []) or []
            }
            enriched_for_worst = enriched_targets.get(worst_port)
            worst_port_display = (
                (enriched_for_worst or {}).get("port_display") or worst_port
            )
            top1_net_display = (
                ((enriched_for_worst or {}).get("candidates") or [{}])[0].get("net_display")
                or top1.get("net")
            )
            hint_tail = (
                f"。GNN 建议把 {worst_port_display} 改接到 "
                f"{top1_net_display} (P(connect)={float(top1['p_connect']):.2f})"
            )
        else:
            hint_tail = "。请人工复核高亮的引脚"

        warning_item = _item(
            _GNN_DISAGREE_ERROR_CODE,
            "gnn_advisory",
            "warning",
            (
                "规则比较器认为电路通过，但 GNN 怀疑有 "
                f"{len(suspicious_edges)} 条引脚连接异常 "
                f"(最低 P(edge_correct)={float(worst['p_correct']):.2f})"
                f"{hint_tail}。"
            ),
            expected={"rule_logic_correct": True},
            actual={
                "gnn_suspicious_edges": suspicious_actual,
                "gnn_model_version": advice_dict["model_version"],
            },
            suggested_action=(
                "GNN 给出的建议是 advisory，最终判定仍以规则为准。"
                "若复核发现确为错接，请按 suggested_targets 调整接线或"
                "补充规则。"
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
        "n_suggestion_candidates_scored": advice_dict.get(
            "n_suggestion_candidates_scored", 0
        ),
        "disagreement_with_rule": disagreement,
        "n_suspicious_edges": len(suspicious_edges),
        "n_suggested_targets": len(suggestions_by_port),
    }
    result["details"] = details
    return result


def _apply_phase_e_propagation(
    ref_payload: dict[str, Any],
    cur_netlist_v2: dict[str, Any],
) -> list[dict[str, Any]]:
    """Phase E · S3 — fuzzy alignment + canonical name propagation.

    Runs the new alignment-based role propagation pipeline. Mutates
    ``cur_netlist_v2["nets"][...]`` in place with derived ``canonical_name`` /
    ``role`` / ``role_label`` / ``role_source`` fields, respecting existing
    user-provided labels (``manual_role`` / ``port_annotation`` /
    ``power_role``) via the protection rules in ``role_propagation``.

    Returns the list of applied propagation records (one per cur net updated).
    Safe to call regardless of whether downstream isomorphism matching
    succeeds — its job is to enrich net semantics for ALL downstream
    consumers (GNN advisor, semantic analysis, diff report).

    Returns ``[]`` on any failure (logged at WARN; caller continues with the
    legacy isomorphism-based fallback path).
    """
    try:
        from app.domain.gnn.alignment_fuzzy import align_components_by_signature
        from app.domain.gnn.port_graph import (
            build_from_logical_reference,
            build_from_netlist_v2,
        )
    except ImportError as exc:
        log.warning("phase_e_imports_failed: %s — skipping propagation", exc)
        return []

    try:
        ref_hcg = build_from_logical_reference(ref_payload)
        cur_hcg = build_from_netlist_v2(cur_netlist_v2)
        alignment = align_components_by_signature(ref_hcg, cur_hcg)
        return propagate_canonical_via_alignment(
            ref_hcg, cur_hcg, alignment, cur_netlist_v2,
        )
    except Exception as exc:  # noqa: BLE001 — defensive: never crash compare
        log.warning(
            "phase_e_propagation_failed: %s — falling back to legacy path",
            type(exc).__name__,
            exc_info=exc,
        )
        return []


def compare_logical_graphs(
    reference_graph: nx.Graph,
    current_graph: nx.Graph,
    *,
    ref_payload: dict[str, Any] | None = None,
    cur_netlist_v2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """比较参考电路图与当前电路图，返回包含差异报告的比较结果。

    **Phase E (2026-05-21)**: 在 isomorphism 检查前先跑 fuzzy 组件对齐 +
    canonical name 传播，让 cur 的匿名 net 拿到 ref 的语义（INV/VOUT 等）。
    这样 GNN advisor / 语义分析等下游消费者都能看到富语义。如果传播失败
    或全 net 已被 ``manual_role`` / ``port_annotation`` 保护，回退到原本的
    is_isomorphic 路径（实践中几乎永远 None，但留作 defense-in-depth）。
    """
    raw_current_graph = current_graph
    current_graph = _collapse_wire_components(current_graph)
    role_inferences: list[dict[str, Any]] = []
    inference_applied = False
    auto_symmetry_groups: list[dict[str, Any]] = []

    # ★ Phase E · 新 fuzzy alignment 路径 ★
    # 在 isomorphism 之前先把 ref 语义传播到 cur 的匿名 net 上。
    #
    # **关键：不重建 current_graph**。Phase E 的职责是为下游消费者
    # (GNN advisor / 语义分析 / diff report enrichment) 提供富 net 语义，
    # **不应改变 iso path 的结构匹配行为**。把 propagated canonical_names
    # 注入 iso 会引起 tie-break 漂移（例如 R2 vs R3 谁是 "extra component"
    # 时挑反），所以 iso 仍在结构图上跑，只有 cur_netlist_v2 被 mutate。
    if ref_payload is not None and cur_netlist_v2 is not None:
        phase_e_records = _apply_phase_e_propagation(ref_payload, cur_netlist_v2)
        if phase_e_records:
            role_inferences.extend(phase_e_records)
            inference_applied = True

    if not reference_graph.graph.get("symmetry_groups"):
        auto_symmetry_groups = auto_detect_symmetries(reference_graph)
    iso_mapping = _find_isomorphism(reference_graph, current_graph)
    if iso_mapping is None and ref_payload is not None and cur_netlist_v2 is not None:
        # **Legacy fallback** (Phase E S3 kept this for defense-in-depth).
        # _infer_current_net_roles_from_reference uses GraphMatcher.is_isomorphic
        # which in practice returns None on any cur board with extra jumper
        # wires — so it rarely fires after Phase E propagation has done its job.
        # Plan: remove once production data confirms Phase E covers all cases.
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
            result = _maybe_attach_gnn_advice(
                result,
                ref_payload,
                cur_netlist_v2,
                reference_graph,
                current_graph,
            )
        if inference_applied:
            _attach_role_inferences(result, role_inferences)
        result = _promote_critical_extras(
            result, reference_graph, raw_current_graph, wire_only=True
        )
        result = _attach_template_match(result, current_graph)
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
            result, reference_graph, raw_current_graph
        )
        result = _attach_template_match(result, current_graph)
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
        result = _promote_critical_extras(
            result, reference_graph, raw_current_graph, wire_only=True
        )
        result = _attach_template_match(result, current_graph)
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
    result = _promote_critical_extras(
        result, reference_graph, raw_current_graph, wire_only=True
    )
    result = _attach_template_match(result, current_graph)
    return result
