from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["safe", "warning", "danger", "unknown"]
ErrorFamily = Literal[
    "short_circuit",
    "wiring_mismatch",
    "polarity_error",
    "missing_protection",
    "missing_component",
    "incomplete_circuit",
    "measurement_error",
    "unknown",
]
ToolName = Literal[
    "netlist_trace_tool",
    "board_schema_lookup_tool",
    "fault_case_lookup_tool",
    "datasheet_lookup_tool",
    "heatmap_overlay_tool",
    "safety_rule_lookup_tool",
]


class EvidenceRef(BaseModel):
    """A compact pointer back to a validator/netlist fact."""

    ref_id: str
    source: str = "validator_report_v2"
    component_id: str = ""
    pin_name: str = ""
    hole_id: str = ""
    electrical_node_id: str = ""
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class DiagnosticFinding(BaseModel):
    """Normalized validator finding for agent routing."""

    error_code: str
    severity: str = "warning"
    component_id: str = ""
    pin_name: str = ""
    expected: Any | None = None
    actual: Any | None = None
    suggested_action: str = ""
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class RuntimeEvidence(BaseModel):
    """Minimal structured facts pushed from the deterministic pipeline."""

    station_id: str
    risk_level: RiskLevel = "unknown"
    diagnostics: list[str] = Field(default_factory=list)
    risk_reasons: list[str] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    error_tags: list[str] = Field(default_factory=list)
    findings: list[DiagnosticFinding] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    netlist_v2: dict[str, Any] = Field(default_factory=dict)
    validator_report_v2: dict[str, Any] = Field(default_factory=dict)
    circuit_snapshot: str = ""
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)


class AllowedTool(BaseModel):
    """Tool declaration that can be pushed into an agent prompt."""

    name: ToolName
    reason: str
    required: bool = False


class ContextPack(BaseModel):
    """Push-Based Context Management payload for one diagnostic turn."""

    pack_id: str
    error_family: ErrorFamily
    risk_level: RiskLevel
    pushed_facts: list[str] = Field(default_factory=list)
    allowed_tools: list[AllowedTool] = Field(default_factory=list)
    prompt_rules: list[str] = Field(default_factory=list)
    citation_requirements: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


class DiagnosticState(BaseModel):
    """LangGraph state object for deterministic PCM diagnostic flow."""

    query: str = ""
    top_k: int = 5
    runtime_evidence: RuntimeEvidence
    error_family: ErrorFamily = "unknown"
    context_pack: ContextPack | None = None
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    draft_answer: str = ""
    verification_report: VerificationReport | None = None
    final_answer: str = ""


class VerificationReport(BaseModel):
    """Rule-based critic result before an answer is shown to users."""

    passed: bool
    issues: list[str] = Field(default_factory=list)
    required_rewrite_hint: str = ""


DiagnosticState.model_rebuild()
