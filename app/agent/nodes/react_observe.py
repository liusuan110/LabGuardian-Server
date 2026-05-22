"""react_observe node: executes the planned tool and records the observation.

Only invokes tools listed in `ContextPack.allowed_tools` (whitelist already
enforced by `react_plan_node`). If the planner produced no tool call, this
node is a no-op so the loop can advance to reflection.
"""

from __future__ import annotations

import logging
from time import perf_counter

from app.agent.contracts import DiagnosticState, ReActStep
from app.agent.nodes._metrics import append_metric, require_context_pack
from app.agent.tools import (
    BoardSchemaLookupInput,
    CircuitLookupInput,
    DatasheetLookupInput,
    FaultCaseLookupInput,
    NetlistTraceInput,
    SafetyRuleLookupInput,
    TeachingConceptLookupInput,
    ToolResult,
    board_schema_lookup_tool,
    circuit_lookup_tool,
    datasheet_lookup_tool,
    fault_case_lookup_tool,
    netlist_trace_tool,
    safety_rule_lookup_tool,
    teaching_concept_lookup_tool,
)

logger = logging.getLogger(__name__)


def _ensure_iteration_index(state: DiagnosticState) -> int:
    """Index of the step we are about to observe (= last appended by react_plan)."""
    return len(state.react_trace) - 1


def react_observe_node(state: DiagnosticState) -> dict:
    started_at = perf_counter()
    context_pack = require_context_pack(state)
    iteration = state.react_iterations
    idx = _ensure_iteration_index(state)
    if idx < 0:
        return {
            "graph_metrics": append_metric(
                state,
                node_name=f"react_observe_{iteration}",
                started_at=started_at,
                payload={"skipped": "no_plan_step"},
                status="skipped",
            ),
        }

    raw_step = state.react_trace[idx]
    step = raw_step if isinstance(raw_step, ReActStep) else ReActStep.model_validate(raw_step)
    if step.tool_call is None:
        # Planner declined to act this iteration — leave observation empty and
        # let reflect decide whether to terminate.
        return {
            "graph_metrics": append_metric(
                state,
                node_name=f"react_observe_{iteration}",
                started_at=started_at,
                payload={"iteration": iteration, "skipped": "no_tool_call"},
                status="skipped",
            ),
        }

    result = _dispatch_tool(step.tool_call.tool_name, step.tool_call.arguments, state)
    observation = {
        "tool_name": result.tool_name,
        "status": result.status,
        "summary": result.summary,
        "payload_keys": sorted(result.payload.keys()),
    }
    step.observation = observation

    react_trace = [
        (item.model_dump() if isinstance(item, ReActStep) else dict(item))
        for item in state.react_trace
    ]
    react_trace[idx] = step.model_dump()

    tool_results = list(state.tool_results)
    tool_results.append(result.model_dump())

    return {
        "react_trace": react_trace,
        "tool_results": tool_results,
        "graph_metrics": append_metric(
            state,
            node_name=f"react_observe_{iteration}",
            started_at=started_at,
            payload={
                "iteration": iteration,
                "tool_name": result.tool_name,
                "status": result.status,
            },
        ),
    }


def _dispatch_tool(tool_name: str, arguments: dict, state: DiagnosticState) -> ToolResult:
    """Look up and invoke a single tool. Falls back to an error ToolResult."""
    evidence = state.runtime_evidence
    args = arguments or {}
    try:
        if tool_name == "netlist_trace_tool":
            return netlist_trace_tool(
                evidence,
                NetlistTraceInput(**args),
            )
        if tool_name == "board_schema_lookup_tool":
            return board_schema_lookup_tool(BoardSchemaLookupInput(**args))
        if tool_name == "fault_case_lookup_tool":
            payload = {
                "query": args.get("query", state.query),
                "error_tags": args.get("error_tags", evidence.error_tags),
                "top_k": args.get("top_k", min(state.top_k, 5)),
            }
            return fault_case_lookup_tool(FaultCaseLookupInput(**payload))
        if tool_name == "datasheet_lookup_tool":
            payload = {
                "component_id": args.get("component_id", ""),
                "component_type": args.get("component_type", ""),
                "part_number": args.get("part_number", ""),
                "package_type": args.get("package_type", ""),
                "query": args.get("query", state.query),
                "error_family": args.get("error_family", state.error_family),
            }
            return datasheet_lookup_tool(DatasheetLookupInput(**payload))
        if tool_name == "circuit_lookup_tool":
            payload = {
                "query": args.get("query", state.query),
                "circuit_id": args.get("circuit_id", ""),
                "top_k": args.get("top_k", min(state.top_k, 3)),
            }
            return circuit_lookup_tool(CircuitLookupInput(**payload))
        if tool_name == "safety_rule_lookup_tool":
            payload = {
                "risk_level": args.get("risk_level", evidence.risk_level),
                "error_family": args.get("error_family", state.error_family),
            }
            return safety_rule_lookup_tool(SafetyRuleLookupInput(**payload))
        if tool_name == "teaching_concept_lookup_tool":
            payload = {
                "query": args.get("query", state.user_message or state.query),
                "concept_id": args.get("concept_id", ""),
                "error_family": args.get("error_family", state.error_family),
            }
            return teaching_concept_lookup_tool(
                TeachingConceptLookupInput(**payload)
            )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("ReAct tool %s failed: %s", tool_name, exc)
        return ToolResult(
            tool_name=tool_name,
            status="error",
            summary=f"tool execution failed: {exc}",
            payload={},
        )

    return ToolResult(
        tool_name=tool_name,
        status="unsupported",
        summary=f"tool {tool_name!r} is not wired into ReAct dispatcher",
        payload={},
    )
