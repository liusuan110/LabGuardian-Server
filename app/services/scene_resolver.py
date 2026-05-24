"""Resolve a teaching scene_id from station / validator state.

WP-1 (2026-05-24): this module replaces the hardcoded
``scene_id="exp_first_order_rc"`` defaults that previously lived in
``rag_service.py``, ``agent/tools.py``, ``agent/tool_runner.py`` and
``agent/nodes/react_observe.py``. The retrieval contract (see
``docs/retrieval-contract.md``) requires that non-RC questions never
silently pull RC fault cases.

## Resolution order

The resolver checks the following sources in order, returning the first
non-empty match:

  1. ``station["scene_id"]`` — explicit override set by upstream
     (e.g. teacher tool, classroom-side manual scene selection).
  2. ``station["topology_label"]`` — set by the topology classifier
     pipeline (see ``TopologyClassifierService.suggest``).
  3. ``comparison_report["topology_label"]`` — same value carried in
     validator output.

If none of the above is present **OR** the topology label maps to
``unknown``, the resolver returns ``None``. Callers MUST treat ``None``
as "skip fault_case retrieval"; they MUST NOT fall back to a default
scene_id (that is exactly the bug WP-1 is fixing).

## Fail-open by design

The resolver does not call the GNN classifier inline. Inline inference
would (a) add latency to every agent turn and (b) silently mask
upstream wiring bugs. Pipeline is responsible for stamping
``topology_label`` into station when classification is desired; this
resolver only consumes what's already there.
"""

from __future__ import annotations

from typing import Any, Final


# 6 demo scenes — keep in sync with knowledge/teaching_scenes/*.json
# and app/domain/topology/labels.py::TOPOLOGY_LABELS.
TOPOLOGY_LABEL_TO_SCENE_ID: Final[dict[str, str]] = {
    "rc_first_order": "exp_first_order_rc",
    "common_emitter": "exp_common_emitter_amplifier",
    "differential_pair": "exp_differential_amplifier",
    "inverting_amp_ua741": "exp_ua741_inverting_amplifier",
    "summing_amp_ua741": "exp_ua741_summing_amplifier",
    "integrator_ua741": "exp_ua741_integrator",
}

# Reverse map for scene_id → human-friendly name (used in citations).
SCENE_ID_TO_DISPLAY_NAME: Final[dict[str, str]] = {
    "exp_first_order_rc": "一阶 RC 滤波器",
    "exp_common_emitter_amplifier": "共射放大电路",
    "exp_differential_amplifier": "BJT 差分放大器",
    "exp_ua741_inverting_amplifier": "UA741 反相放大器",
    "exp_ua741_summing_amplifier": "UA741 反相加法器",
    "exp_ua741_integrator": "UA741 反相积分器",
}

VALID_SCENE_IDS: Final[frozenset[str]] = frozenset(
    TOPOLOGY_LABEL_TO_SCENE_ID.values()
)


# WP-3 v2 (2026-05-24): per-scene whitelist of allowed datasheet document_ids.
# Used by ``DatasheetKbService.search(scene_id=...)`` to prevent cross-chip
# leakage — e.g. a UA741 inverter turn must not surface NE555 / 74LS74 chunks
# even when the query keyword matches. NE555, 74LS74, LM324 are kept in the
# corpus for future scenes (and admin lookups) but are out-of-scope for the
# current 6 demo topologies.
#
# ``passive.capacitor_polarity`` is shared by all scenes — every demo uses
# capacitors at some point (decoupling, bypass, coupling, integration), and
# its content is topology-neutral.
SCENE_TO_ALLOWED_DATASHEETS: Final[dict[str, frozenset[str]]] = {
    "exp_first_order_rc": frozenset(["passive.capacitor_polarity"]),
    "exp_common_emitter_amplifier": frozenset(["bjt_8050", "passive.capacitor_polarity"]),
    "exp_differential_amplifier": frozenset(["bjt_8050", "passive.capacitor_polarity"]),
    "exp_ua741_inverting_amplifier": frozenset(["ua741", "passive.capacitor_polarity"]),
    "exp_ua741_summing_amplifier": frozenset(["ua741", "passive.capacitor_polarity"]),
    "exp_ua741_integrator": frozenset(["ua741", "passive.capacitor_polarity"]),
}


def allowed_datasheets_for_scene(scene_id: str | None) -> frozenset[str] | None:
    """Return the document_id whitelist for the given scene_id.

    Returns:
        - ``None`` if scene_id is empty / not one of the 6 demo scenes
          (caller MUST allow all documents — concept_tutor / lab_guidance
          without topology context, or admin tools).
        - ``frozenset()`` (empty) is never returned — every demo scene
          has at least one allowed document.
        - A frozenset of allowed document_ids otherwise.
    """
    if not scene_id:
        return None
    return SCENE_TO_ALLOWED_DATASHEETS.get(scene_id)


def resolve_scene_id(
    *,
    station: dict[str, Any] | None = None,
    comparison_report: dict[str, Any] | None = None,
) -> str | None:
    """Resolve a scene_id from upstream classifier / explicit override.

    Args:
        station: Classroom station snapshot. May contain explicit
            ``scene_id`` or ``topology_label`` set by upstream.
        comparison_report: validator_report_v2 dict. May carry
            ``topology_label`` written by the pipeline.

    Returns:
        One of the 6 canonical scene_ids, or ``None`` if topology is
        unknown / unmapped. Callers MUST treat ``None`` as "skip
        fault_case retrieval" — never fall back to a default scene.
    """
    station = station or {}
    comparison_report = comparison_report or {}

    # 1) Explicit override.
    explicit = str(station.get("scene_id") or "").strip()
    if explicit in VALID_SCENE_IDS:
        return explicit

    # 2) station["topology_label"].
    station_label = str(station.get("topology_label") or "").strip()
    mapped = TOPOLOGY_LABEL_TO_SCENE_ID.get(station_label)
    if mapped:
        return mapped

    # 3) comparison_report["topology_label"].
    report_label = str(comparison_report.get("topology_label") or "").strip()
    mapped = TOPOLOGY_LABEL_TO_SCENE_ID.get(report_label)
    if mapped:
        return mapped

    # Fail-open: caller must skip fault_case retrieval entirely.
    return None


def scene_display_name(scene_id: str | None) -> str:
    """Human-readable name for a scene_id, or empty for None / unknown."""
    if not scene_id:
        return ""
    return SCENE_ID_TO_DISPLAY_NAME.get(scene_id, scene_id)
