from __future__ import annotations

from app.agent.contracts import ContextPack, RuntimeEvidence
from app.agent.tools import (
    BoardSchemaLookupInput,
    DatasheetLookupInput,
    FaultCaseLookupInput,
    NetlistTraceInput,
    SafetyRuleLookupInput,
    ToolResult,
    board_schema_lookup_tool,
    datasheet_lookup_tool,
    fault_case_lookup_tool,
    netlist_trace_tool,
    safety_rule_lookup_tool,
)


def run_diagnostic_tools(
    *,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    query: str,
    top_k: int,
) -> list[ToolResult]:
    results: list[ToolResult] = []
    tool_names = [tool.name for tool in context_pack.allowed_tools]
    first_finding = evidence.findings[0] if evidence.findings else None

    if "netlist_trace_tool" in tool_names:
        results.append(
            netlist_trace_tool(
                evidence,
                NetlistTraceInput(
                    component_id=first_finding.component_id if first_finding else "",
                    pin_name=first_finding.pin_name if first_finding else "",
                ),
            )
        )

    if "board_schema_lookup_tool" in tool_names:
        results.append(board_schema_lookup_tool(_board_lookup_input_from_evidence(evidence)))

    if "fault_case_lookup_tool" in tool_names:
        # WP-1 (2026-05-24): scene_id is injected from runtime evidence so
        # non-RC topologies don't silently pull RC fault cases. Empty
        # current_scene_id → tool skips retrieval (see fault_case_lookup_tool).
        # WP-1 v4 (2026-05-24): error_codes is the canonical KB recall
        # signal — without it the agent path matches only on tags that
        # don't exist in the KB vocabulary, returning empty fault_cases
        # even when validator clearly identified the fault.
        results.append(
            fault_case_lookup_tool(
                FaultCaseLookupInput(
                    query=query,
                    error_tags=evidence.error_tags,
                    error_codes=evidence.error_codes,
                    scene_id=evidence.current_scene_id,
                    top_k=min(top_k, 5),
                )
            )
        )

    if "datasheet_lookup_tool" in tool_names:
        component_id = first_finding.component_id if first_finding else ""
        # WP-3 v2 (2026-05-24): inject scene_id so DISTILL_MODE can restrict
        # the datasheet candidate set to the current scene's allowed chips.
        results.append(
            datasheet_lookup_tool(
                DatasheetLookupInput(
                    component_id=component_id,
                    component_type=_component_type_from_evidence(evidence),
                    part_number=_part_number_from_evidence(evidence, component_id),
                    package_type=_package_type_from_evidence(evidence, component_id),
                    query=query,
                    error_family=context_pack.error_family,
                    scene_id=evidence.current_scene_id,
                )
            )
        )

    if "safety_rule_lookup_tool" in tool_names:
        results.append(
            safety_rule_lookup_tool(
                SafetyRuleLookupInput(
                    risk_level=evidence.risk_level,
                    error_family=context_pack.error_family,
                )
            )
        )

    return results


def _component_type_from_evidence(evidence: RuntimeEvidence) -> str:
    first_finding = evidence.findings[0] if evidence.findings else None
    component_id = first_finding.component_id if first_finding else ""
    for component in evidence.netlist_v2.get("components", []):
        if not isinstance(component, dict):
            continue
        if component_id and component.get("component_id") != component_id:
            continue
        return str(component.get("component_type") or component.get("type") or "")
    for finding in evidence.findings:
        component_type = finding.payload.get("component_type")
        if component_type:
            return str(component_type)
    return ""


def _part_number_from_evidence(evidence: RuntimeEvidence, component_id: str) -> str:
    cid = str(component_id or "").strip()
    for component in evidence.netlist_v2.get("components", []):
        if not isinstance(component, dict):
            continue
        if cid and str(component.get("component_id") or "").strip() != cid:
            continue
        subtype = str(component.get("part_subtype") or "").strip()
        if subtype:
            return subtype
        meta = component.get("metadata", {}) if isinstance(component.get("metadata"), dict) else {}
        for key in ("part_number", "model", "chip", "ic", "label", "name"):
            value = str(meta.get(key) or "").strip()
            if value:
                return value
    return ""


def _package_type_from_evidence(evidence: RuntimeEvidence, component_id: str) -> str:
    cid = str(component_id or "").strip()
    for component in evidence.netlist_v2.get("components", []):
        if not isinstance(component, dict):
            continue
        if cid and str(component.get("component_id") or "").strip() != cid:
            continue
        pkg = str(component.get("package_type") or "").strip()
        if pkg:
            return pkg
    return ""


def _board_lookup_input_from_evidence(evidence: RuntimeEvidence) -> BoardSchemaLookupInput:
    for ref in evidence.evidence_refs:
        if ref.hole_id or ref.electrical_node_id:
            return BoardSchemaLookupInput(
                hole_id=ref.hole_id,
                node_id=ref.electrical_node_id,
            )
    for finding in evidence.findings:
        actual = str(finding.actual or "")
        expected = str(finding.expected or "")
        if actual:
            if actual.startswith("ROW_") or actual.startswith("TRACK_"):
                return BoardSchemaLookupInput(node_id=actual)
            return BoardSchemaLookupInput(hole_id=actual)
        if expected:
            if expected.startswith("ROW_") or expected.startswith("TRACK_"):
                return BoardSchemaLookupInput(node_id=expected)
            return BoardSchemaLookupInput(hole_id=expected)
    return BoardSchemaLookupInput()
