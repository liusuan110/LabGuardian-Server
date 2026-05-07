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
    history_facts: list[str] = Field(default_factory=list)
    history_summary: str = ""


class AllowedTool(BaseModel):
    """Tool declaration that can be pushed into an agent prompt."""

    name: ToolName
    reason: str
    required: bool = False


class ContextPackMetrics(BaseModel):
    """Lightweight context size estimates for PCM ablation and edge metrics."""

    pushed_facts_count: int = 0
    allowed_tool_count: int = 0
    evidence_ref_count: int = 0
    history_facts_count: int = 0
    history_char_count: int = 0
    history_estimated_tokens: int = 0
    char_count: int = 0
    estimated_tokens: int = 0


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
    history_facts: list[str] = Field(default_factory=list)
    history_summary: str = ""
    metrics: ContextPackMetrics | None = None


class VerificationReport(BaseModel):
    """Rule-based critic result before an answer is shown to users."""

    passed: bool
    issues: list[str] = Field(default_factory=list)
    required_rewrite_hint: str = ""
    # Phase 6 — white-box gate for VLM micro-defect inspection.
    # Set True ONLY when validator confidence is low and error context
    # plausibly maps to a micro defect (BURN_MARK / UNSTRIPPED_WIRE / COLD_SOLDER).
    needs_micro_inspection: bool = False
    suspected_defect_types: list[str] = Field(default_factory=list)


class VlmFinding(BaseModel):
    """One VLM-attributed finding produced by `vlm_explain_node`."""

    defect_type: str
    provider: str = "template"
    status: str = "completed"
    conclusion: str = ""
    evidence: str = ""
    fix_steps: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class GraphNodeMetric(BaseModel):
    """Per-node telemetry emitted by the PCM LangGraph shell."""

    node_name: str
    duration_ms: float = 0.0
    status: str = "ok"
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A single planned tool invocation emitted by the LLM/template planner.

    The planner is restricted to `ContextPack.allowed_tools` to prevent the
    agent from inventing tool names (white-box defense).
    """

    tool_name: ToolName | str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ReActStep(BaseModel):
    """One iteration of the Plan → Act → Observe → Reflect ReAct loop.

    The shape is provider-agnostic: a deterministic template provider and
    a future OpenVINO-backed text LLM both produce the same schema.
    """

    iteration: int
    thought: str = ""
    tool_call: ToolCall | None = None
    observation: dict[str, Any] = Field(default_factory=dict)
    reflection: str = ""
    terminate: bool = False
    duration_ms: float = 0.0


class ReflectionResult(BaseModel):
    """Critic outcome that decides whether the ReAct loop should terminate."""

    passed: bool
    reason: str = ""
    next_hint: str = ""


class DiagnosticState(BaseModel):
    """LangGraph state object for deterministic PCM diagnostic flow."""

    query: str = ""
    user_message: str = ""
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    top_k: int = 5
    runtime_evidence: RuntimeEvidence
    error_family: ErrorFamily = "unknown"
    context_pack: ContextPack | None = None
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    draft_answer: str = ""
    verification_report: VerificationReport | None = None
    final_answer: str = ""
    graph_metrics: list[GraphNodeMetric] = Field(default_factory=list)
    # Phase 4 ReAct trace (Plan → Act → Observe → Reflect)
    react_trace: list[ReActStep] = Field(default_factory=list)
    react_iterations: int = 0
    max_react_iterations: int = 4
    react_terminate_reason: str = ""
    # Phase 6 — VLM micro-defect findings (only populated when verifier gates allow it)
    vlm_findings: list[VlmFinding] = Field(default_factory=list)


DiagnosticState.model_rebuild()
