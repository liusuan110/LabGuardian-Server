"""CADx Phase 0 — Topology template matching package.

Public API:
    * :class:`TopologyTemplate` — declarative spec for a canonical analog
      topology (required / optional / forbidden components + edges, variants,
      parametric invariants).
    * :class:`TemplateMatchResult` — output of matching a template against a
      student bipartite graph; JSON-serializable.
    * :func:`match_template` / :func:`match_all_templates` — pure functions
      that run subgraph-isomorphism-based matching.
    * :func:`get_template_registry` — returns all 6 canonical demo templates
      keyed by ``template_id``.

The package is consumed in two places:
    1. ``app/domain/compare/orchestrator.py::compare_logical_graphs`` — runs
       :func:`match_all_templates` and attaches results to
       ``details.template_match`` (Phase 0: read-only side channel,
       does not influence verdict).
    2. Future: ``app/api/v1/topology/suggest`` (Phase 1) — exposes the same
       results as an HTTP endpoint for the frontend ReferenceSelector.
"""

from app.domain.templates.base import (
    ComponentSlot,
    EdgeSpec,
    NetSlot,
    ParametricInvariant,
    Severity,
    TopologyTemplate,
    TopologyVariant,
)
from app.domain.templates.matcher import match_all_templates, match_template
from app.domain.templates.registry import get_template_registry
from app.domain.templates.result import ForbiddenViolation, TemplateMatchResult


__all__ = [
    "ComponentSlot",
    "EdgeSpec",
    "ForbiddenViolation",
    "NetSlot",
    "ParametricInvariant",
    "Severity",
    "TemplateMatchResult",
    "TopologyTemplate",
    "TopologyVariant",
    "get_template_registry",
    "match_all_templates",
    "match_template",
]
