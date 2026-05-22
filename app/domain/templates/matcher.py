"""Topology template matcher — Phase 0 of CADx.

This module produces :class:`TemplateMatchResult` for each
:class:`TopologyTemplate` against a student's bipartite (component, net)
graph (the one already built by
``app/domain/compare/orchestrator.py::compare_logical_graphs``).

## Matching algorithm (Phase 0, deliberately simple)

For each template:
  1. For the base spec, build a tiny bipartite ``nx.Graph`` representing
     the canonical (required) structure: one comp node per
     :class:`ComponentSlot`, one net node per :class:`NetSlot`, edges
     per :class:`EdgeSpec`.
  2. For each :class:`TopologyVariant`, build a richer graph by appending
     ``additional_components`` + ``additional_edges`` to the base spec.
  3. For each spec graph (base + variants), use
     ``networkx.algorithms.isomorphism.GraphMatcher.subgraph_isomorphisms_iter``
     to find an embedding of the spec into the student graph, matching
     comp nodes by ``component_type`` and net nodes by ``role``.
  4. Take the embedding with the most required edges satisfied.
     ``structural_score`` = ``matched_required_edges / total_required_edges``.
  5. Check ``forbidden_components``: any forbidden component type observed
     in the student graph (regardless of its connections) accumulates a
     penalty. Phase 0 uses a flat ``0.3`` penalty per violation.
  6. ``confidence = min(structural_score, role_score) - 0.3 * len(forbidden_violations)``
     clamped to ``[0.0, 1.0]``.

## Deliberate design choices

* **Read-only, exception-safe**: the caller (``compare_logical_graphs``)
  must be able to wrap calls in ``try/except`` without worrying about
  state mutation.
* **No reliance on Phase E alignment**: matching consumes the raw
  bipartite graph the orchestrator already has — no preprocessing.
* **Subgraph isomorphism, not full isomorphism**: the student may have
  extra wires, decorations, or duplicates that the canonical template
  doesn't mention. We only require the canonical structure to be
  *embeddable*.
* **No partial matching of variants** (Phase 0): each variant is matched
  in full or skipped. Phase 1 will introduce partial-match scoring.

## Frequently asked

* *Why not reuse* ``compare/matcher.py::_node_match``? Because that
  matcher enforces strict role-label equivalence (UI1==UI1 etc.) which
  is unnecessarily restrictive for template matching — the student's
  canonical labels may not be propagated yet. Phase 0 uses a looser
  ``ctype`` / ``role`` match. Phase 1 will optionally tighten this when
  Phase E has run successfully.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Iterable

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from app.domain.compare.matcher import (
    NON_POLAR_CAPACITOR_TYPES,
    PASSIVE_TWO_PIN_TYPES,
)
from app.domain.logical_reference import (
    normalize_component_type,
    normalize_net_role,
)

from .base import (
    ComponentSlot,
    EdgeSpec,
    NetSlot,
    TopologyTemplate,
    TopologyVariant,
)
from .result import ForbiddenViolation, TemplateMatchResult


log = logging.getLogger(__name__)

# Penalty per forbidden-component observation. Empirically picked: a single
# forbidden violation drops a 0.9 base confidence to 0.6 (warning territory),
# two violations to 0.3 (visibly rejected).
FORBIDDEN_PENALTY = 0.3

# When the student graph is unrealistically small (no comp or no net nodes),
# skip matching entirely to avoid pathological zero-division.
MIN_GRAPH_NODES_FOR_MATCH = 2

# Tie-breaking weight: templates that explain more of the student's
# components (excluding wires/decorations) win over templates that only
# match a small subgraph. Without this, ``inverting_amp`` and
# ``summing_amp`` both score 1.0 on a summing-amp board (since
# inverting_amp's 3-comp spec embeds into the 4-comp summing student
# graph), and tie-break falls to registry insertion order.
COVERAGE_WEIGHT = 0.3

# Component types that count toward "explainable components" in the
# student graph. Wires and other decorations are excluded — they should
# not penalize a template that legitimately doesn't model them.
ELIGIBLE_STUDENT_COMP_TYPES = {
    "Resistor",
    "Capacitor",
    "CapacitorCeramic",
    "CapacitorElectrolytic",
    "IC",
    "Transistor",
    "Diode",
    "LED",
    "Potentiometer",
}


# --- internal: spec-graph construction ----------------------------------------


def _spec_node_id(kind: str, identifier: str) -> str:
    """Internal node id used in the synthetic spec graph.

    For comp nodes ``identifier`` is the :class:`ComponentSlot.role`.
    For net nodes ``identifier`` is the :class:`NetSlot.canonical_name`
    (unique within a template, unlike role which may repeat for
    multiple signal nets).
    """
    return f"spec::{kind}::{identifier}"


def _build_spec_graph(
    components: Iterable[ComponentSlot],
    nets: Iterable[NetSlot],
    edges: Iterable[EdgeSpec],
) -> nx.Graph:
    """Build a bipartite spec graph in the same shape as the student graph.

    Node identity:
      * Comp nodes are keyed by :class:`ComponentSlot.role` (assumed unique).
      * Net nodes are keyed by :class:`NetSlot.canonical_name` (assumed
        unique within a template). This is critical: keying by role would
        collapse multiple ``signal`` or ``power`` nets into one node and
        cause IC pins (pin4=VEE, pin7=VCC) to share a single edge —
        breaking matching.

    Edge resolution:
      * :class:`EdgeSpec.net_role` is looked up first as a NetSlot's
        ``canonical_name`` (preferred). If not found, falls back to
        matching by ``role`` — but only when exactly one slot has that
        role (else error to avoid ambiguity).

    Node attributes:
      * comp nodes: ``kind="comp"``, ``ctype``, ``subtype``,
        ``slot_role=<ComponentSlot.role>``.
      * net nodes: ``kind="net"``, ``role=<NetSlot.role>``,
        ``canonical_name=<NetSlot.canonical_name>``,
        ``slot_role=<NetSlot.canonical_name>``.

    Edge attributes:
      * ``pin=<EdgeSpec.pin>``, ``comp_type``, ``is_required``.
    """
    g = nx.Graph()

    # Pre-build slot lookups (we may not need to materialize all of them
    # as graph nodes — only those actually touched by `edges`).
    comp_slots_by_role: dict[str, ComponentSlot] = {
        slot.role: slot for slot in components
    }
    net_slots_by_canonical: dict[str, NetSlot] = {
        slot.canonical_name: slot for slot in nets
    }
    net_slots_by_role: dict[str, list[NetSlot]] = {}
    for slot in nets:
        net_slots_by_role.setdefault(slot.role, []).append(slot)

    used_comp_roles: set[str] = set()
    used_net_canonicals: set[str] = set()
    resolved_edges: list[tuple[str, str, str, bool]] = []  # (comp_node, net_node, pin, is_required)

    def _resolve_net(net_role_or_name: str) -> NetSlot | None:
        slot = net_slots_by_canonical.get(net_role_or_name)
        if slot is not None:
            return slot
        candidates = net_slots_by_role.get(net_role_or_name, [])
        if len(candidates) == 1:
            return candidates[0]
        return None

    for edge in edges:
        comp_slot = comp_slots_by_role.get(edge.component_role)
        net_slot = _resolve_net(edge.net_role)
        if comp_slot is None or net_slot is None:
            log.warning(
                "spec_edge_unresolved component_role=%s net_role=%s",
                edge.component_role,
                edge.net_role,
            )
            continue
        used_comp_roles.add(comp_slot.role)
        used_net_canonicals.add(net_slot.canonical_name)
        comp_node = _spec_node_id("comp", comp_slot.role)
        net_node = _spec_node_id("net", net_slot.canonical_name)
        resolved_edges.append((comp_node, net_node, edge.pin, edge.is_required))

    # Materialize only the comp slots actually touched by edges.
    for role in used_comp_roles:
        slot = comp_slots_by_role[role]
        node = _spec_node_id("comp", slot.role)
        g.add_node(
            node,
            kind="comp",
            ctype=slot.component_type,
            subtype=slot.component_subtype,
            slot_role=slot.role,
        )

    # Materialize only the net slots actually touched by edges.
    for canonical in used_net_canonicals:
        slot = net_slots_by_canonical[canonical]
        node = _spec_node_id("net", slot.canonical_name)
        g.add_node(
            node,
            kind="net",
            role=slot.role,
            canonical_name=slot.canonical_name,
            slot_role=slot.canonical_name,
        )

    for comp_node, net_node, pin, is_required in resolved_edges:
        comp_ctype = g.nodes[comp_node]["ctype"]
        g.add_edge(
            comp_node,
            net_node,
            pin=pin,
            comp_type=comp_ctype,
            is_required=is_required,
        )

    return g


def _spec_for_base(template: TopologyTemplate) -> nx.Graph:
    return _build_spec_graph(
        template.required_components,
        list(template.required_nets) + list(template.optional_nets),
        template.required_edges,
    )


def _spec_for_variant(
    template: TopologyTemplate, variant: TopologyVariant
) -> nx.Graph:
    """Base spec + variant additions."""
    components = list(template.required_components) + list(
        variant.additional_components
    )
    nets = list(template.required_nets) + list(template.optional_nets)
    edges = list(template.required_edges) + list(variant.additional_edges)
    return _build_spec_graph(components, nets, edges)


# --- internal: node match predicates ------------------------------------------


def _comp_types_compatible(spec_ctype: Any, student_ctype: Any) -> bool:
    """Match component types with the same passive-capacitor leniency the
    legacy compare/matcher uses (Ceramic ≡ Electrolytic ≡ generic Capacitor).
    """
    spec_norm = normalize_component_type(spec_ctype)
    student_norm = normalize_component_type(student_ctype)
    if spec_norm == student_norm:
        return True
    if (
        spec_norm in NON_POLAR_CAPACITOR_TYPES
        and student_norm in NON_POLAR_CAPACITOR_TYPES
    ):
        return True
    return False


def _net_roles_compatible(spec_role: Any, student_role: Any) -> bool:
    """Match net roles. Templates' ``"signal"`` is treated as a wildcard
    (matches any non-power, non-ground net) since student net role inference
    may not have propagated yet at the time templates are matched.
    """
    spec_norm = normalize_net_role(spec_role)
    student_norm = normalize_net_role(student_role)
    if spec_norm == student_norm:
        return True
    # Spec "signal" is a wildcard accepting any non-power, non-ground role.
    # (We deliberately do NOT do the reverse: a spec "ground" requirement
    # must match a student "ground" exactly.)
    if spec_norm == "signal" and student_norm not in {"ground", "power"}:
        return True
    return False


def _node_match_factory(spec_graph: nx.Graph):
    """Return a node_match function bound to the given spec graph."""

    def _match(spec_data: dict[str, Any], student_data: dict[str, Any]) -> bool:
        if spec_data.get("kind") != student_data.get("kind"):
            return False
        if spec_data.get("kind") == "comp":
            if not _comp_types_compatible(
                spec_data.get("ctype"), student_data.get("ctype")
            ):
                return False
            # Subtype check is strict only when spec sets it (e.g. UA741).
            spec_subtype = spec_data.get("subtype")
            if spec_subtype:
                student_subtype = (student_data.get("subtype") or "").upper()
                if spec_subtype.upper() not in student_subtype and student_subtype not in spec_subtype.upper():
                    return False
            return True
        # net node
        return _net_roles_compatible(spec_data.get("role"), student_data.get("role"))

    # The factory pattern keeps `spec_graph` available in closure if we want
    # cross-edge constraints later (Phase 1).
    _ = spec_graph
    return _match


def _normalize_pin_label(pin: str) -> str:
    """Normalize pin labels for cross-fixture matching.

    The student fixtures and DSL references are not consistent about
    IC pin naming: some use ``"pin2"``, some use just ``"2"``. Strip
    the ``pin`` prefix when present so both forms compare equal.
    """
    s = str(pin or "").strip()
    if s.startswith("pin") and s[3:].isdigit():
        return s[3:]
    return s


def _edge_match(spec_edge_data: dict[str, Any], student_edge_data: dict[str, Any]) -> bool:
    """Match edges by pin label, with the same passive-leniency as the
    legacy matcher (passive 2-pin components do not require pin order).

    IC pin labels are normalized via :func:`_normalize_pin_label` so
    ``"pin2"`` matches ``"2"``.
    """
    spec_comp_type = normalize_component_type(spec_edge_data.get("comp_type"))
    student_comp_type = normalize_component_type(student_edge_data.get("comp_type"))
    if (
        spec_comp_type in PASSIVE_TWO_PIN_TYPES
        and student_comp_type in PASSIVE_TWO_PIN_TYPES
    ):
        return True
    spec_pin = _normalize_pin_label(spec_edge_data.get("pin"))
    student_pin = _normalize_pin_label(
        student_edge_data.get("pin_role") or student_edge_data.get("pin")
    )
    return spec_pin == student_pin


# --- internal: scoring --------------------------------------------------------


def _count_required_edges(spec_graph: nx.Graph) -> int:
    return sum(
        1
        for _, _, data in spec_graph.edges(data=True)
        if data.get("is_required", True)
    )


def _build_assignments_from_mapping(
    mapping: dict[str, str],
    spec_graph: nx.Graph,
    student_graph: nx.Graph,
) -> tuple[dict[str, str], dict[str, str]]:
    """From a spec→student node mapping, extract human-friendly assignments.

    Returns:
        ``(role_assignments, net_assignments)`` where:
        * ``role_assignments``: ``student_comp_id -> ComponentSlot.role``
        * ``net_assignments``: ``student_net_id -> NetSlot.role``
    """
    role_assignments: dict[str, str] = {}
    net_assignments: dict[str, str] = {}
    for spec_node, student_node in mapping.items():
        spec_data = spec_graph.nodes[spec_node]
        slot_role = spec_data.get("slot_role")
        if not slot_role:
            continue
        kind = spec_data.get("kind")
        student_id = str(student_node)
        if kind == "comp":
            role_assignments[student_id] = slot_role
        elif kind == "net":
            net_assignments[student_id] = slot_role
    return role_assignments, net_assignments


def _collect_forbidden_violations(
    template: TopologyTemplate,
    student_graph: nx.Graph,
) -> list[ForbiddenViolation]:
    """A forbidden component is flagged if the student graph contains a
    component of that type (with optional subtype filter)."""
    if not template.forbidden_components:
        return []
    out: list[ForbiddenViolation] = []
    for slot in template.forbidden_components:
        for node, data in student_graph.nodes(data=True):
            if data.get("kind") != "comp":
                continue
            if not _comp_types_compatible(slot.component_type, data.get("ctype")):
                continue
            if slot.component_subtype:
                student_subtype = (data.get("subtype") or "").upper()
                spec_subtype = slot.component_subtype.upper()
                if (
                    spec_subtype not in student_subtype
                    and student_subtype not in spec_subtype
                ):
                    continue
            out.append(
                ForbiddenViolation(
                    component_role=slot.role,
                    student_component_id=str(node),
                    reason=(
                        f"forbidden component {slot.component_type}"
                        + (
                            f" (subtype={slot.component_subtype})"
                            if slot.component_subtype
                            else ""
                        )
                        + " observed in student graph"
                    ),
                )
            )
    return out


def _missing_roles(
    component_slots: Iterable[ComponentSlot],
    assigned_roles: set[str],
) -> list[str]:
    return [slot.role for slot in component_slots if slot.role not in assigned_roles]


def _try_match_spec(
    spec_graph: nx.Graph,
    student_graph: nx.Graph,
) -> dict[str, str] | None:
    """Run a subgraph-isomorphism search of spec into student. Returns the
    first mapping found or ``None``.

    Networkx ``GraphMatcher.subgraph_isomorphisms_iter`` looks for an
    isomorphism of a subgraph of ``G1`` (first arg, the larger graph)
    matching ``G2`` (second arg, the smaller pattern). We want the *spec*
    embedded into the *student*, so pass student first.
    """
    if spec_graph.number_of_nodes() == 0:
        return None
    if student_graph.number_of_nodes() < spec_graph.number_of_nodes():
        return None
    matcher = GraphMatcher(
        student_graph,
        spec_graph,
        node_match=_node_match_factory(spec_graph),
        edge_match=_edge_match,
    )
    try:
        first = next(matcher.subgraph_isomorphisms_iter())
    except StopIteration:
        return None
    # ``first`` maps student_node -> spec_node; invert to spec -> student.
    return {spec_node: student_node for student_node, spec_node in first.items()}


# --- public API ---------------------------------------------------------------


def match_template(
    student_graph: nx.Graph,
    template: TopologyTemplate,
) -> TemplateMatchResult:
    """Match a single template against a student bipartite graph.

    Always returns a :class:`TemplateMatchResult` (even on no-match — with
    ``confidence=0.0``). Never raises.

    Args:
        student_graph: Bipartite ``nx.Graph`` with nodes carrying
            ``kind="comp"|"net"`` and edges carrying ``pin``/``comp_type``.
            This is the same shape produced by ``compare_logical_graphs``'s
            ``current_graph`` argument.
        template: The :class:`TopologyTemplate` to match.

    Returns:
        A :class:`TemplateMatchResult`. ``matched_variant`` is the
        variant_id of the highest-scoring spec, or ``None`` if the base
        spec (no variant) won.
    """
    result = TemplateMatchResult(
        template_id=template.template_id,
        template_name=template.name,
        topology_label=template.topology_label,
        reference_id=template.reference_id,
    )

    if student_graph.number_of_nodes() < MIN_GRAPH_NODES_FOR_MATCH:
        return result

    forbidden_violations = _collect_forbidden_violations(template, student_graph)
    result.forbidden_violations = forbidden_violations

    # Precompute eligible student component count for coverage scoring.
    eligible_student_comps = sum(
        1
        for _, data in student_graph.nodes(data=True)
        if data.get("kind") == "comp"
        and normalize_component_type(data.get("ctype")) in ELIGIBLE_STUDENT_COMP_TYPES
    )

    # Candidates: (variant_id_or_None, spec_graph)
    candidates: list[tuple[str | None, nx.Graph]] = [
        (None, _spec_for_base(template))
    ]
    for variant in template.variants:
        candidates.append((variant.variant_id, _spec_for_variant(template, variant)))

    best_confidence = -1.0
    best_payload: dict[str, Any] | None = None

    for variant_id, spec_graph in candidates:
        mapping = _try_match_spec(spec_graph, student_graph)
        if mapping is None:
            continue
        total_required = _count_required_edges(spec_graph)
        # All required spec edges that are in the mapping are by construction
        # satisfied (subgraph_isomorphisms_iter already required edge_match).
        # In Phase 0 we count this as "all required edges matched" when the
        # subgraph iso succeeds — partial spec-matching is a Phase 1 feature.
        matched_required = total_required
        structural_score = 1.0 if total_required == 0 else matched_required / total_required

        role_assignments, net_assignments = _build_assignments_from_mapping(
            mapping, spec_graph, student_graph
        )

        assigned_required_roles = {
            role for role in role_assignments.values() if role
        }
        required_roles = {slot.role for slot in template.required_components}
        if required_roles:
            role_score = (
                len(required_roles & assigned_required_roles) / len(required_roles)
            )
        else:
            role_score = 1.0

        # Coverage: how much of the student's component inventory does
        # this template explain? Tie-breaker that prefers richer templates
        # (e.g. summing over inverting on a summing board, LPF over
        # integrator on an LPF board with both R_f and C_f).
        matched_student_comps = len(role_assignments)
        if eligible_student_comps > 0:
            coverage = matched_student_comps / eligible_student_comps
            coverage = min(coverage, 1.0)
        else:
            coverage = 0.0

        penalty = FORBIDDEN_PENALTY * len(forbidden_violations)
        base = min(structural_score, role_score)
        # Blend in coverage: dominant structural+role match, with a small
        # coverage bonus. Capped at 1.0 so a perfectly-matched template
        # still reads as ``100% confidence`` in the UI.
        confidence = max(
            0.0,
            min(1.0, base * (1.0 - COVERAGE_WEIGHT) + coverage * COVERAGE_WEIGHT) - penalty,
        )

        # ``>=`` deliberately: when a variant ties with the base spec,
        # the variant is the more specific topology description and
        # should win the assignment (e.g. CE amp ``direct_grounded_emitter``
        # variant tying with base — the variant carries the emitter→GND
        # edge that documents the actual wiring, even if scoring is equal).
        # Iteration order is base first, then variants, so >= naturally
        # prefers later (more specific) candidates on ties.
        if confidence >= best_confidence:
            best_confidence = confidence
            best_payload = {
                "variant_id": variant_id,
                "structural_score": structural_score,
                "role_score": role_score,
                "confidence": confidence,
                "role_assignments": role_assignments,
                "net_assignments": net_assignments,
            }

    if best_payload is not None:
        result.matched_variant = best_payload["variant_id"]
        result.structural_score = best_payload["structural_score"]
        result.role_score = best_payload["role_score"]
        result.confidence = best_payload["confidence"]
        result.role_assignments = best_payload["role_assignments"]
        result.net_assignments = best_payload["net_assignments"]
        assigned = set(result.role_assignments.values())
        result.missing_required = _missing_roles(
            template.required_components, assigned
        )
        result.missing_optional = _missing_roles(
            template.optional_components, assigned
        )
    else:
        # No isomorphism — all required roles are missing.
        result.missing_required = [s.role for s in template.required_components]
        result.missing_optional = [s.role for s in template.optional_components]

    return result


def match_all_templates(
    student_graph: nx.Graph,
    registry: dict[str, TopologyTemplate] | None = None,
) -> list[TemplateMatchResult]:
    """Match a student graph against every template in the registry.

    Args:
        student_graph: See :func:`match_template`.
        registry: Optional override; defaults to
            :func:`app.domain.templates.registry.get_template_registry`.

    Returns:
        Results sorted by ``confidence`` descending.
    """
    if registry is None:
        # Lazy import to avoid circular dependency at module load time.
        from .registry import get_template_registry

        registry = get_template_registry()

    results: list[TemplateMatchResult] = []
    for template in registry.values():
        try:
            results.append(match_template(student_graph, template))
        except Exception as exc:  # noqa: BLE001 — match must never crash compare
            log.warning(
                "template_match_failed template_id=%s err=%s",
                template.template_id,
                type(exc).__name__,
                exc_info=exc,
            )
            results.append(
                TemplateMatchResult(
                    template_id=template.template_id,
                    template_name=template.name,
                    topology_label=template.topology_label,
                    reference_id=template.reference_id,
                )
            )
    results.sort(key=lambda r: r.confidence, reverse=True)
    return results


__all__ = ["match_template", "match_all_templates"]


# Silence unused import warnings for itertools (kept for variant-permutation
# extension planned in Phase 1).
_ = itertools
