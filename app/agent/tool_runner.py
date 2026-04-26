from __future__ import annotations

from app.agent.contracts import ContextPack, RuntimeEvidence
from app.agent.tools import (
    BoardSchemaLookupInput,
    FaultCaseLookupInput,
    NetlistTraceInput,
    SafetyRuleLookupInput,
    ToolResult,
    board_schema_lookup_tool,
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
        results.append(
            fault_case_lookup_tool(
                FaultCaseLookupInput(
                    query=query,
                    error_tags=evidence.error_tags,
                    top_k=min(top_k, 5),
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

