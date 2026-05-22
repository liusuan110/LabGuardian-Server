"""Result types emitted by the Phase 0 template matcher.

All result objects are JSON-serializable via :meth:`TemplateMatchResult.to_dict`
so they can be embedded into the ``compare_logical_graphs`` response under
``details.template_match``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ForbiddenViolation:
    """Record of a forbidden component / connection that disqualifies a match."""

    component_role: str
    student_component_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_role": self.component_role,
            "student_component_id": self.student_component_id,
            "reason": self.reason,
        }


@dataclass
class TemplateMatchResult:
    """Outcome of matching one :class:`TopologyTemplate` against a student graph.

    Designed to be JSON-friendly so the frontend can consume it directly via
    ``details.template_match.top_3[]``.

    Attributes:
        template_id: The matched template's stable id.
        template_name: Chinese / human-readable name.
        topology_label: 7-class topology label (used by future GNN-A).
        reference_id: Linked DSL reference (or None).
        structural_score: Fraction of ``required_edges`` satisfied.
            Range ``[0.0, 1.0]``. Primary ranking signal.
        role_score: Fraction of ``required_components`` whose role is
            assigned to some student component. Range ``[0.0, 1.0]``.
        confidence: Combined score in ``[0.0, 1.0]`` used for sorting
            (currently ``min(structural, role) - 0.5 * forbidden_penalty``).
        matched_variant: ``variant_id`` of the best-matching variant, or
            ``None`` if the base template (no variant) won.
        role_assignments: Map ``student_component_id -> ComponentSlot.role``.
            Empty when no isomorphic mapping was found.
        net_assignments: Map ``student_net_id -> NetSlot.role``.
            Empty when no isomorphic mapping was found.
        missing_required: List of required component roles not assigned.
        missing_optional: List of optional component roles not assigned
            (informational only — does not reduce score).
        forbidden_violations: List of forbidden components observed.
    """

    template_id: str
    template_name: str
    topology_label: str
    reference_id: str | None
    structural_score: float = 0.0
    role_score: float = 0.0
    confidence: float = 0.0
    matched_variant: str | None = None
    role_assignments: dict[str, str] = field(default_factory=dict)
    net_assignments: dict[str, str] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    forbidden_violations: list[ForbiddenViolation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "topology_label": self.topology_label,
            "reference_id": self.reference_id,
            "structural_score": round(self.structural_score, 4),
            "role_score": round(self.role_score, 4),
            "confidence": round(self.confidence, 4),
            "matched_variant": self.matched_variant,
            "role_assignments": dict(self.role_assignments),
            "net_assignments": dict(self.net_assignments),
            "missing_required": list(self.missing_required),
            "missing_optional": list(self.missing_optional),
            "forbidden_violations": [v.to_dict() for v in self.forbidden_violations],
        }
