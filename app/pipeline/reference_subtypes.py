"""Reference-driven component subtype backfill.

The vision pipeline can detect that a component is an ``IC`` and infer its
package, but it generally cannot know whether the chip is UA741, LM358, NE555,
etc. When the user has selected a reference experiment, that reference is the
authoritative source for the expected IC subtype. Fill missing current-side
``part_subtype`` values from it before topology/GNN conversion.
"""

from __future__ import annotations

from typing import Any


def apply_reference_ic_subtypes(
    components: list[dict[str, Any]],
    reference_circuit: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Fill blank ``part_subtype`` on current IC components from reference.

    The function mutates ``components`` in place and returns a list of records
    describing what was applied. Existing ``part_subtype`` / ``subtype`` values
    are never overwritten.

    Matching policy:
    - exact component id / ref_id match first;
    - if the reference has exactly one IC subtype, use it for every blank IC
      (common production case: reference ``U1`` detected as current ``IC1``);
    - if all reference ICs use the same subtype, use that subtype for blank ICs;
    - otherwise leave ambiguous ICs blank.
    """

    if not isinstance(reference_circuit, dict):
        return []

    ref_ics: list[tuple[str, str]] = []
    for comp in reference_circuit.get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        if str(comp.get("type") or comp.get("component_type") or "") != "IC":
            continue
        subtype = str(comp.get("subtype") or comp.get("part_subtype") or "").strip()
        if not subtype:
            continue
        ref_id = str(comp.get("ref_id") or comp.get("component_id") or "").strip()
        ref_ics.append((ref_id, subtype))

    if not ref_ics:
        return []

    by_ref_id = {ref_id: subtype for ref_id, subtype in ref_ics if ref_id}
    unique_subtypes = sorted({subtype for _ref_id, subtype in ref_ics})
    unambiguous_subtype = unique_subtypes[0] if len(unique_subtypes) == 1 else None

    applied: list[dict[str, Any]] = []
    for comp in components:
        if not isinstance(comp, dict):
            continue
        if str(comp.get("component_type") or comp.get("type") or "") != "IC":
            continue
        existing = str(comp.get("part_subtype") or comp.get("subtype") or "").strip()
        if existing:
            comp["part_subtype"] = existing
            continue

        component_id = str(comp.get("component_id") or comp.get("ref_id") or "").strip()
        subtype = by_ref_id.get(component_id) or unambiguous_subtype
        if not subtype:
            continue

        comp["part_subtype"] = subtype
        metadata = dict(comp.get("metadata") or {})
        metadata["part_subtype_source"] = "reference_circuit"
        metadata["part_subtype_reference_id"] = component_id if component_id in by_ref_id else None
        comp["metadata"] = metadata
        applied.append({
            "component_id": component_id,
            "part_subtype": subtype,
            "source": "reference_circuit",
            "matched_by": "component_id" if component_id in by_ref_id else "single_reference_ic",
        })

    return applied
