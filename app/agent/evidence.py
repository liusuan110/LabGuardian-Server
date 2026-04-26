from __future__ import annotations

from typing import Any

from app.agent.contracts import DiagnosticFinding, EvidenceRef, RuntimeEvidence
from app.services.error_tag_service import ErrorTagService

REPORT_ITEM_BUCKETS = (
    "items",
    "topology_errors",
    "node_errors",
    "hole_errors",
    "polarity_errors",
    "component_errors",
)


def build_runtime_evidence_from_station(
    *,
    station_id: str,
    station: dict[str, Any],
    error_tag_service: ErrorTagService | None = None,
) -> RuntimeEvidence:
    """Normalize classroom station state into a compact evidence contract."""

    comparison_report = station.get("comparison_report", {}) or {}
    findings = [
        _finding_from_item(item, idx)
        for idx, item in enumerate(_iter_report_items(comparison_report))
    ]
    evidence_refs = _dedupe_refs([ref for finding in findings for ref in finding.evidence_refs])
    error_codes = _dedupe([finding.error_code for finding in findings if finding.error_code])

    tags_payload = (
        error_tag_service.extract_tags(comparison_report)
        if error_tag_service
        else station.get("error_tags", [])
    )
    error_tags = _dedupe(
        [
            str(tag.get("error_tag") if isinstance(tag, dict) else tag)
            for tag in tags_payload
            if tag
        ]
    )

    return RuntimeEvidence(
        station_id=station_id,
        risk_level=_normalize_risk_level(station.get("risk_level")),
        diagnostics=[str(item) for item in station.get("diagnostics", [])],
        risk_reasons=[str(item) for item in station.get("risk_reasons", [])],
        error_codes=error_codes,
        error_tags=error_tags,
        findings=findings,
        evidence_refs=evidence_refs,
        netlist_v2=station.get("netlist_v2", {}) or {},
        validator_report_v2=comparison_report,
        circuit_snapshot=str(station.get("circuit_snapshot", "") or ""),
        runtime_metadata=station.get("runtime_metadata", {}) or {},
    )


def build_runtime_evidence_from_classroom(
    *,
    station_id: str,
    stations: dict[str, dict[str, Any]],
    error_tag_service: ErrorTagService | None = None,
) -> RuntimeEvidence:
    return build_runtime_evidence_from_station(
        station_id=station_id,
        station=stations.get(station_id, {}),
        error_tag_service=error_tag_service,
    )


def _finding_from_item(item: dict[str, Any], index: int) -> DiagnosticFinding:
    code = str(item.get("error_code") or "")
    refs = _refs_from_item(item=item, index=index, error_code=code)
    return DiagnosticFinding(
        error_code=code,
        severity=str(item.get("severity") or "warning"),
        component_id=str(
            item.get("current_component_id")
            or item.get("component_id")
            or item.get("expected")
            or ""
        ),
        pin_name=str(item.get("current_pin_name") or item.get("pin_name") or ""),
        expected=item.get("expected"),
        actual=item.get("actual"),
        suggested_action=str(item.get("suggested_action") or ""),
        evidence_refs=refs,
        payload=dict(item),
    )


def _refs_from_item(item: dict[str, Any], index: int, error_code: str) -> list[EvidenceRef]:
    explicit_refs = item.get("evidence_refs", [])
    refs: list[EvidenceRef] = []
    if isinstance(explicit_refs, list):
        for ref_idx, raw_ref in enumerate(explicit_refs):
            if isinstance(raw_ref, dict):
                refs.append(
                    EvidenceRef(
                        ref_id=str(raw_ref.get("ref_id") or f"{error_code}:{index}:{ref_idx}"),
                        source=str(raw_ref.get("source") or "validator_report_v2"),
                        component_id=str(
                            raw_ref.get("component_id")
                            or item.get("current_component_id")
                            or item.get("component_id")
                            or ""
                        ),
                        pin_name=str(
                            raw_ref.get("pin_name")
                            or item.get("current_pin_name")
                            or item.get("pin_name")
                            or ""
                        ),
                        hole_id=_scalar_ref_value(
                            raw_ref.get("hole_id")
                            or raw_ref.get("current_hole_id")
                            or raw_ref.get("target_hole_id")
                            or item.get("current_hole_id")
                        ),
                        electrical_node_id=_scalar_ref_value(
                            raw_ref.get("electrical_node_id")
                            or raw_ref.get("current_node_id")
                            or raw_ref.get("target_node_id")
                            or item.get("current_node_id")
                        ),
                        summary=str(raw_ref.get("summary") or raw_ref.get("kind") or error_code),
                        payload=dict(raw_ref),
                    )
                )
            elif raw_ref:
                refs.append(
                    EvidenceRef(
                        ref_id=str(raw_ref),
                        component_id=str(
                            item.get("current_component_id")
                            or item.get("component_id")
                            or ""
                        ),
                        pin_name=str(item.get("current_pin_name") or item.get("pin_name") or ""),
                        summary=error_code,
                    )
                )

    if refs:
        return refs

    component_id = str(item.get("current_component_id") or item.get("component_id") or "")
    pin_name = str(item.get("current_pin_name") or item.get("pin_name") or "")
    ref_id = ":".join(part for part in (error_code or "finding", component_id, pin_name) if part)
    return [
        EvidenceRef(
            ref_id=ref_id or f"finding:{index}",
            component_id=component_id,
            pin_name=pin_name,
            hole_id=_scalar_ref_value(item.get("current_hole_id")),
            electrical_node_id=_scalar_ref_value(item.get("current_node_id")),
            summary=str(item.get("suggested_action") or error_code or "validator finding"),
            payload=dict(item),
        )
    ]


def _scalar_ref_value(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if value is None:
        return ""
    return str(value)


def _iter_report_items(comparison_report: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items: list[Any] = []
    for key in REPORT_ITEM_BUCKETS:
        value = comparison_report.get(key, [])
        if isinstance(value, list):
            raw_items.extend(value)
    return [item for item in raw_items if isinstance(item, dict)]


def _normalize_risk_level(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower()
    if normalized in {"safe", "warning", "danger"}:
        return normalized
    return "unknown"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _dedupe_refs(values: list[EvidenceRef]) -> list[EvidenceRef]:
    result: list[EvidenceRef] = []
    seen: set[str] = set()
    for value in values:
        if value.ref_id in seen:
            continue
        seen.add(value.ref_id)
        result.append(value)
    return result
