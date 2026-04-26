from __future__ import annotations

from typing import Any

HIGHLIGHT_PROTOCOL_VERSION = "labguardian_highlight_v1"
DEFAULT_VIEW_ID = "top"


def build_highlight_protocol(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    targets: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for diagnostic in diagnostics:
        for target in build_highlight_targets_for_diagnostic(diagnostic):
            key = _target_key(target)
            if key in seen:
                continue
            seen.add(key)
            targets.append(target)
    return {
        "version": HIGHLIGHT_PROTOCOL_VERSION,
        "targets": targets,
    }


def build_highlight_targets_for_diagnostic(
    diagnostic: dict[str, Any],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for ref in diagnostic.get("evidence_refs", []):
        if not isinstance(ref, dict):
            continue
        kind = ref.get("kind")
        if kind == "component_bbox_ref":
            target = _component_bbox_target(ref, diagnostic)
        elif kind == "pin_keypoint_ref":
            target = _pin_keypoint_target(ref, diagnostic)
        elif kind == "hole_candidate_ref":
            target = _hole_candidate_target(ref, diagnostic)
        else:
            target = None
        if target:
            targets.append(target)
    return targets


def _component_bbox_target(
    ref: dict[str, Any],
    diagnostic: dict[str, Any],
) -> dict[str, Any] | None:
    bbox = ref.get("bbox")
    if not bbox:
        return None
    component_id = ref.get("component_id") or diagnostic.get("current_component_id")
    return {
        "kind": "component_bbox_ref",
        "render": "box",
        "target_type": "component",
        "component_id": component_id,
        "view_id": ref.get("view_id") or DEFAULT_VIEW_ID,
        "bbox": bbox,
        "source_ref_id": ref.get("ref_id"),
        "diagnostic": _diagnostic_summary(diagnostic),
    }


def _pin_keypoint_target(
    ref: dict[str, Any],
    diagnostic: dict[str, Any],
) -> dict[str, Any] | None:
    keypoint = ref.get("keypoint")
    if not keypoint:
        return None
    return {
        "kind": "pin_keypoint_ref",
        "render": "point",
        "target_type": "pin",
        "component_id": ref.get("component_id") or diagnostic.get("current_component_id"),
        "pin_name": ref.get("pin_name") or diagnostic.get("current_pin_name"),
        "view_id": ref.get("view_id") or DEFAULT_VIEW_ID,
        "keypoint": keypoint,
        "radius_px": 8,
        "source_ref_id": ref.get("ref_id"),
        "diagnostic": _diagnostic_summary(diagnostic),
    }


def _hole_candidate_target(
    ref: dict[str, Any],
    diagnostic: dict[str, Any],
) -> dict[str, Any] | None:
    current_holes = _as_list(ref.get("current_hole_id"))
    target_holes = _as_list(ref.get("target_hole_id"))
    candidate_holes = _as_list(ref.get("candidate_hole_ids"))
    holes = _dedupe(current_holes + target_holes + candidate_holes)
    if not holes:
        return None
    return {
        "kind": "hole_candidate_ref",
        "render": "hole",
        "target_type": "hole",
        "component_id": ref.get("component_id") or diagnostic.get("current_component_id"),
        "pin_name": ref.get("pin_name") or diagnostic.get("current_pin_name"),
        "current_hole_ids": current_holes,
        "target_hole_ids": target_holes,
        "candidate_hole_ids": candidate_holes,
        "hole_ids": holes,
        "source_ref_id": ref.get("ref_id"),
        "diagnostic": _diagnostic_summary(diagnostic),
    }


def _diagnostic_summary(diagnostic: dict[str, Any]) -> dict[str, Any]:
    return {
        "error_code": diagnostic.get("error_code", ""),
        "category": diagnostic.get("category", ""),
        "severity": diagnostic.get("severity", ""),
        "message": diagnostic.get("message", ""),
    }


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [item for item in value if item not in (None, "")]
    return [value]


def _dedupe(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _target_key(target: dict[str, Any]) -> tuple:
    return (
        target.get("kind"),
        target.get("component_id"),
        _hashable(target.get("pin_name")),
        _hashable(target.get("bbox") or []),
        _hashable(target.get("keypoint") or []),
        _hashable(target.get("hole_ids") or []),
        target.get("source_ref_id"),
    )


def _hashable(value: Any):
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    return value
