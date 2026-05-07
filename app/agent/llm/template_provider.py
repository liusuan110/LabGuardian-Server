"""Deterministic template LLM provider.

This provider does not call any model. It emulates a Plan/Reflect agent
using:
- `error_family` and `ContextPack.allowed_tools` to choose the next tool
- the existing rule-based `verify_draft_answer` as the reflection critic

The goal is to keep the ReAct loop **shape** identical to a real LLM-driven
agent so we can swap in `openvino_genai_text` later without touching the
graph topology.
"""

from __future__ import annotations

from typing import Any

from app.agent.contracts import (
    ContextPack,
    ReActStep,
    ReflectionResult,
    RuntimeEvidence,
    ToolCall,
)
from app.agent.llm.base import LLMProvider, PlanRequest, ReflectRequest
from app.agent.verification import verify_draft_answer


# Rough preference order: cheap deterministic lookups first, semantic
# / heuristic ones last. The provider never calls a tool twice in a row.
_TOOL_PRIORITY: list[str] = [
    "netlist_trace_tool",
    "board_schema_lookup_tool",
    "fault_case_lookup_tool",
    "datasheet_lookup_tool",
    "safety_rule_lookup_tool",
    "heatmap_overlay_tool",
]


class TemplateLLMProvider(LLMProvider):
    """Rule-driven planner + reflector.

    Plan strategy:
    - On iteration 0, prefer `netlist_trace_tool` if allowed (root-cause first)
    - Then walk `_TOOL_PRIORITY` in order, skipping tools already called
    - If no allowed tool remains, return `None` to end the Act phase

    Reflect strategy:
    - Re-use `verify_draft_answer`. If it passes, terminate. If it fails,
      surface the rewrite hint as `next_hint`.
    """

    name = "template"

    def plan(self, request: PlanRequest) -> ToolCall | None:
        allowed = {tool.name for tool in request.context_pack.allowed_tools}
        if not allowed:
            return None

        already_called = {
            step.tool_call.tool_name
            for step in request.prior_steps
            if step.tool_call is not None
        }
        candidate: str | None = None
        for tool_name in _TOOL_PRIORITY:
            if tool_name in allowed and tool_name not in already_called:
                candidate = tool_name
                break

        if candidate is None:
            return None

        first_finding = (
            request.evidence.findings[0] if request.evidence.findings else None
        )
        arguments: dict[str, Any] = {}
        if candidate == "netlist_trace_tool" and first_finding is not None:
            arguments = {
                "component_id": first_finding.component_id,
                "pin_name": first_finding.pin_name,
            }
        rationale = self._rationale_for(candidate, request)
        return ToolCall(
            tool_name=candidate,
            arguments=arguments,
            rationale=rationale,
        )

    def reflect(self, request: ReflectRequest) -> ReflectionResult:
        report = verify_draft_answer(
            evidence=request.evidence,
            context_pack=request.context_pack,
            draft_answer=request.draft_answer,
        )
        if report.passed:
            return ReflectionResult(
                passed=True,
                reason="verifier_passed",
                next_hint="",
            )
        return ReflectionResult(
            passed=False,
            reason="; ".join(report.issues) or "verification_failed",
            next_hint=report.required_rewrite_hint,
        )

    def _rationale_for(self, tool_name: str, request: PlanRequest) -> str:
        family = request.context_pack.error_family
        if tool_name == "netlist_trace_tool":
            return f"trace netlist for {family} root-cause"
        if tool_name == "board_schema_lookup_tool":
            return "verify hole-to-node mapping"
        if tool_name == "fault_case_lookup_tool":
            return "recall similar teaching case"
        if tool_name == "datasheet_lookup_tool":
            return "consult component datasheet"
        if tool_name == "safety_rule_lookup_tool":
            return "check safety rule for current risk level"
        return f"call {tool_name}"
