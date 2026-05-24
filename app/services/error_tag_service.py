"""Maps validator_report_v2 items to scene-agnostic teaching error tags.

WP-1 (2026-05-24): the tag vocabulary was previously RC-flavored
(``missing_rc_component`` / ``incomplete_rc_circuit`` / ``rc_output_node``
/ ``rc_component_set``), which leaked RC framing into non-RC topologies
(common-emitter, differential-pair, UA741 family) — a silent
distillation-data contamination. The tags below are now topology-agnostic
and carry the same semantics across all 6 demo scenes. See
``docs/retrieval-contract.md``.
"""

from __future__ import annotations

from typing import Any


class ErrorTagService:
    """Maps validator_report_v2 items to scene-agnostic teaching error tags."""

    _CODE_TO_TAG: dict[str, str] = {
        "NODE_MISMATCH": "wrong_node_connection",
        "HOLE_MISMATCH": "wrong_hole_connection",
        "FLOATING_PIN": "floating_connection",
        "COMPONENT_SHORTED_SAME_NET": "scope_ground_or_short_risk",
        "COMPONENT_MISSING": "missing_required_component",
        "COMPONENT_INSTANCE_MISSING": "missing_required_component",
        "PIN_MISSING": "floating_connection",
        "PIN_EXTRA": "unexpected_connection",
        "TOPOLOGY_VALID_SUBSET": "incomplete_circuit",
        "MULTIPLE_DISCONNECTED_SUBGRAPHS": "incomplete_circuit",
    }

    _TAG_FOCUS: dict[str, list[str]] = {
        "wrong_node_connection": ["expected_output_node", "breadboard_node"],
        "wrong_hole_connection": ["breadboard_hole", "reference_circuit_compare"],
        "floating_connection": ["open_circuit", "pin_contact"],
        "scope_ground_or_short_risk": ["scope_ground", "reference_ground", "short_risk"],
        "missing_required_component": ["required_component_set", "reference_circuit_compare"],
        "unexpected_connection": ["extra_pin", "reference_circuit_compare"],
        "incomplete_circuit": ["closed_loop", "reference_circuit_compare"],
    }

    def extract_tags(self, comparison_report: dict[str, Any]) -> list[dict[str, Any]]:
        tags: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for item in self._iter_report_items(comparison_report):
            code = item.get("error_code")
            if not isinstance(code, str):
                continue
            tag = self._CODE_TO_TAG.get(code)
            if not tag:
                continue
            component_id = str(
                item.get("component_id")
                or item.get("current_component_id")
                or item.get("expected")
                or ""
            )
            pin_name = str(item.get("pin_name") or "")
            dedupe_key = (tag, component_id, pin_name)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            tags.append(
                {
                    "error_tag": tag,
                    "source_error_code": code,
                    "severity": item.get("severity", "warning"),
                    "component_id": component_id,
                    "pin_name": pin_name,
                    "expected": item.get("expected"),
                    "actual": item.get("actual"),
                    "suggested_action": item.get("suggested_action", ""),
                    "teaching_focus": self._TAG_FOCUS.get(tag, []),
                    "evidence_refs": item.get("evidence_refs", []),
                }
            )
        return tags

    def _iter_report_items(self, comparison_report: dict[str, Any]) -> list[dict[str, Any]]:
        raw_items: list[Any] = list(comparison_report.get("items", []))
        for key in (
            "topology_errors",
            "node_errors",
            "hole_errors",
            "polarity_errors",
            "component_errors",
        ):
            value = comparison_report.get(key, [])
            if isinstance(value, list):
                raw_items.extend(value)
        return [item for item in raw_items if isinstance(item, dict)]
