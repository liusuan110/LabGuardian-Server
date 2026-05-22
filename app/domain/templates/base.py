"""TopologyTemplate spec — Phase 0 of CADx (Canonical-Anchored Diagnosis).

A ``TopologyTemplate`` describes the *intent* of a canonical analog circuit
topology (e.g. "UA741 inverting amplifier"). Unlike a raw reference netlist,
a template expresses **partial specifications** with three semantic tiers:

* ``required_*``   — must be present, else the template does not match.
* ``optional_*``   — may be present without penalty; their absence is recorded
                     but never breaks the match (e.g. emitter bypass C_E in CE).
* ``forbidden_*``  — if observed, downgrade or reject the match
                     (e.g. feedback C in an inverting amp would suggest the
                     student is actually building an integrator).

Templates also carry ``variants`` (alternative implementations of the same
topology, e.g. CE with voltage-divider vs fixed bias) and
``parametric_invariants`` (math constraints like ``R_C1 ≈ R_C2`` for diff pair).

The Phase 0 matcher consumes templates and emits
:class:`app.domain.templates.result.TemplateMatchResult`. The matcher is
deliberately read-only with respect to existing pipeline state — its output is
attached to ``compare_logical_graphs(...)['details']['template_match']`` for
side-by-side comparison with the legacy Phase E verdict, **not** to replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ComponentSlot:
    """One canonical component role in a template.

    Attributes:
        role: Canonical role identifier (e.g. ``"feedback_resistor"``,
            ``"opamp"``, ``"input_R"``). Used as key in
            ``TemplateMatchResult.role_assignments``.
        component_type: DSL component type string (e.g. ``"Resistor"``,
            ``"CapacitorCeramic"``, ``"IC"``, ``"Transistor"``). Must match
            ``ctype`` on the bipartite comp nodes (see
            ``app/domain/compare/matcher.py::_component_types_equivalent``).
        component_subtype: Optional, e.g. ``"UA741"`` or ``"NPN/BJT"``.
        is_required: When ``False`` the slot is treated as optional. The
            matcher records absence in ``missing_optional`` without
            decreasing ``structural_score``.
        multiplicity: ``(min, max)`` allowed count of student components
            playing this role. Used for summing amps where N input
            resistors are all valid (e.g. ``(2, 5)``).
    """

    role: str
    component_type: str
    component_subtype: str | None = None
    is_required: bool = True
    multiplicity: tuple[int, int] = (1, 1)


@dataclass(frozen=True)
class NetSlot:
    """One canonical net role in a template.

    Attributes:
        role: Canonical role string. Must align with the
            ``normalize_net_role`` set: ``input`` / ``output`` / ``power``
            / ``ground`` / ``signal``.
        canonical_name: Pretty name used in UI / Edit Script
            (e.g. ``"INV"``, ``"VOUT"``, ``"GND"``).
        role_label: Optional canonical role label (UI1/UO1/VCC/VEE/GND/...)
            for stricter matching against existing reference circuits.
    """

    role: str
    canonical_name: str
    role_label: str | None = None


@dataclass(frozen=True)
class EdgeSpec:
    """One canonical connection between a component and a net.

    Attributes:
        component_role: References a :class:`ComponentSlot.role` in the
            same template.
        pin: DSL pin identifier (``"pin1"``, ``"pin2"``, or for ICs the
            absolute pin number ``"2"``, ``"3"``, ``"6"``, ...).
        net_role: References a :class:`NetSlot.role` in the same template.
        is_required: ``False`` for connections that may be absent without
            invalidating the topology (e.g. R_p compensation in inverting
            amp would have its connecting edges marked optional).
    """

    component_role: str
    pin: str
    net_role: str
    is_required: bool = True


@dataclass(frozen=True)
class TopologyVariant:
    """An alternative implementation of the same topology.

    Each variant adds components / edges on top of the base template's
    ``required_*`` set. The matcher tries the base alone *and* each variant,
    picking the highest-scoring combination. Use this for cases like
    "CE amp with vs without emitter bypass cap" or
    "diff pair with tail resistor vs with current source".

    Attributes:
        variant_id: Short id stored in :class:`TemplateMatchResult.matched_variant`.
        description: Human-readable description for UI tooltips.
        additional_components: Components present in this variant only.
            These are still considered required *for this variant*.
        additional_edges: Edges present in this variant only.
    """

    variant_id: str
    description: str
    additional_components: tuple[ComponentSlot, ...] = ()
    additional_edges: tuple[EdgeSpec, ...] = ()


@dataclass(frozen=True)
class ParametricInvariant:
    """A math constraint on component values for this topology.

    Phase 0 ships the framework but does **not** execute these (visual
    pipeline currently does not extract component values). Phase 2 will
    enable execution once values are available.

    Attributes:
        name: Stable identifier for the invariant (used in test asserts).
        formula: Human-readable formula referencing component roles
            (e.g. ``"abs(R_C1.value - R_C2.value)/R_C1.value < 0.1"``).
        severity: ``"error"`` or ``"warning"`` when violated.
        requires_values: ``True`` when the formula needs numeric component
            values to evaluate. Phase 0 leaves all invariants with
            ``requires_values=True`` (they all need values).
        violation_msg: User-facing message shown when the invariant
            evaluates to False. May contain ``{placeholders}`` for
            future formatter.
    """

    name: str
    formula: str
    severity: Severity
    requires_values: bool
    violation_msg: str


@dataclass(frozen=True)
class TopologyTemplate:
    """A complete canonical-topology specification for diagnosis.

    A template binds:
      * **structural skeleton**: components + nets + edges across three
        tiers (required / optional / forbidden);
      * **variants**: alternative realizations of the same topology;
      * **parametric invariants**: math constraints on component values
        (Phase 2 will evaluate; Phase 0 only declares).

    Attributes:
        template_id: Stable unique id (e.g. ``"inverting_amp_ua741_v1"``).
        name: Chinese / human-readable name shown in UI.
        topology_label: 7-class label consumed by the future GNN-A classifier.
            One of:
            ``rc_first_order`` / ``common_emitter`` / ``differential_pair``
            / ``inverting_amp_ua741`` / ``summing_amp_ua741``
            / ``integrator_ua741``  (+ ``unknown`` reserved for fallback).
        reference_id: Optional link to a DSL reference circuit in
            ``knowledge/references/``. ``None`` allowed if no reference
            netlist exists yet.
        required_components / optional_components / forbidden_components:
            Three tiers of component slots.
        required_nets / optional_nets:
            Net slots (no ``forbidden_nets`` — net roles are
            structurally implied by edges).
        required_edges / optional_edges:
            Edge slots.
        variants: Alternative implementations.
        parametric_invariants: Math constraints (Phase 0 declarative-only).
    """

    template_id: str
    name: str
    topology_label: str
    reference_id: str | None
    required_components: tuple[ComponentSlot, ...]
    optional_components: tuple[ComponentSlot, ...] = ()
    forbidden_components: tuple[ComponentSlot, ...] = ()
    required_nets: tuple[NetSlot, ...] = ()
    optional_nets: tuple[NetSlot, ...] = ()
    required_edges: tuple[EdgeSpec, ...] = ()
    optional_edges: tuple[EdgeSpec, ...] = ()
    variants: tuple[TopologyVariant, ...] = ()
    parametric_invariants: tuple[ParametricInvariant, ...] = ()
    description: str = ""

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid).

        Checks:
          * Every ``EdgeSpec.component_role`` references some
            ``ComponentSlot.role`` in this template (across all tiers + variants).
          * Every ``EdgeSpec.net_role`` references either a
            ``NetSlot.canonical_name`` (preferred — unambiguous) or a
            ``NetSlot.role`` (legacy fallback, only valid when the role
            is unique across all NetSlots in this template).
          * Required component roles are unique.
          * Variant ids are unique.
          * Net canonical names are unique.
        """
        errors: list[str] = []

        all_component_roles = {slot.role for slot in self.required_components}
        all_component_roles.update(slot.role for slot in self.optional_components)
        all_component_roles.update(slot.role for slot in self.forbidden_components)
        for variant in self.variants:
            all_component_roles.update(slot.role for slot in variant.additional_components)

        all_nets = list(self.required_nets) + list(self.optional_nets)
        all_canonical_names = {slot.canonical_name for slot in all_nets}
        net_roles_count: dict[str, int] = {}
        for slot in all_nets:
            net_roles_count[slot.role] = net_roles_count.get(slot.role, 0) + 1

        seen_canonical: set[str] = set()
        for slot in all_nets:
            if slot.canonical_name in seen_canonical:
                errors.append(
                    f"duplicate net canonical_name: {slot.canonical_name!r}"
                )
            seen_canonical.add(slot.canonical_name)

        seen_required = set()
        for slot in self.required_components:
            if slot.role in seen_required:
                errors.append(f"duplicate required component role: {slot.role!r}")
            seen_required.add(slot.role)

        variant_ids = [v.variant_id for v in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            errors.append("variant ids must be unique within a template")

        all_edges = list(self.required_edges) + list(self.optional_edges)
        for variant in self.variants:
            all_edges.extend(variant.additional_edges)
        for edge in all_edges:
            if edge.component_role not in all_component_roles:
                errors.append(
                    f"edge references unknown component role: {edge.component_role!r}"
                )
            # net_role may be a canonical_name (preferred) or a unique role.
            if edge.net_role in all_canonical_names:
                continue
            count = net_roles_count.get(edge.net_role, 0)
            if count == 0:
                errors.append(
                    f"edge references unknown net (neither canonical_name "
                    f"nor role): {edge.net_role!r}"
                )
            elif count > 1:
                errors.append(
                    f"edge references ambiguous net role {edge.net_role!r} "
                    f"({count} NetSlots share this role; use canonical_name "
                    "to disambiguate)"
                )

        return errors
