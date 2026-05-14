"""Phase 5 — chip-parameter questions answered without LLM.

Pre-Phase-5 flow: `_run_datasheet_kb_job` short-circuited inside agent_service
and asked an LLM to summarize Chroma/pypdf hits. Without a configured LLM the
answer came back empty, which is what the user was seeing.

Post-Phase-5 flow: outer gate is gone, all queries go through the ReAct chain.
`build_diagnostic_template_answer` detects `datasheet_lookup_tool` chunk hits
and renders the answer deterministically — chunk text + chunk_id citation
inline, no LLM, no Ollama, no Chroma. Verifier passes because the chunk_id
appears in the answer body.
"""

from __future__ import annotations

import pytest

from app.agent.answering import (
    build_datasheet_answer,
    build_diagnostic_template_answer,
)
from app.agent.context_pack import build_context_pack
from app.agent.evidence import build_runtime_evidence_from_station
from app.agent.tools import DatasheetLookupInput, ToolResult, datasheet_lookup_tool
from app.agent.verification import verify_draft_answer
from app.services.error_tag_service import ErrorTagService


def _empty_evidence_pack():
    """User asks about NE555 without any station context. This is exactly
    the case the old outer gate diverted to `_run_datasheet_kb_job`."""
    evidence = build_runtime_evidence_from_station(
        station_id="NO_STATION",
        station={"comparison_report": {"items": []}},
        error_tag_service=ErrorTagService(),
    )
    return evidence, build_context_pack(evidence, user_message="NE555 引脚分布")


def test_ne555_pinout_query_produces_chunk_cited_answer() -> None:
    evidence, context_pack = _empty_evidence_pack()

    tool_result = datasheet_lookup_tool(
        DatasheetLookupInput(query="NE555 引脚分布", part_number="NE555")
    )
    assert tool_result.payload["provider"] == "local_datasheet_v2"
    expected_chunk_id = tool_result.payload["hits"][0]["chunk_id"]
    assert expected_chunk_id.startswith("ne555.")

    answer = build_diagnostic_template_answer(
        station_id="NO_STATION",
        query="NE555 引脚分布",
        user_message="NE555 引脚分布",
        evidence=evidence,
        context_pack=context_pack,
        tool_results=[tool_result],
    )

    # 1) Real chunk_id appears (otherwise verifier would reject under Phase 1 rules).
    assert expected_chunk_id in answer
    # 2) Deterministic provenance line — confirms no LLM was invoked.
    assert "无 LLM 合成" in answer
    # 3) Verifier passes the answer with the datasheet rule active.
    report = verify_draft_answer(
        evidence=evidence,
        context_pack=context_pack,
        draft_answer=answer,
        intent="diagnostic",
        tool_results=[tool_result],
    )
    assert report.passed, report.issues


def test_datasheet_answer_returns_none_without_datasheet_tool() -> None:
    """If no datasheet_lookup_tool ran, the builder must yield None so the
    caller falls back to the standard diagnostic template."""

    evidence, context_pack = _empty_evidence_pack()
    other = ToolResult(tool_name="netlist_trace_tool", summary="x", payload={})

    assert (
        build_datasheet_answer(
            evidence=evidence,
            context_pack=context_pack,
            tool_results=[other],
            user_message="something else",
        )
        is None
    )


def test_fallback_rule_path_cites_rule_id_no_llm() -> None:
    """LED with no matching v2 doc / KB → local_fallback rules. Answer must
    cite rule_id (verifier requirement) and surface the rules to the user
    without any LLM call."""

    evidence, context_pack = _empty_evidence_pack()

    tool_result = datasheet_lookup_tool(
        DatasheetLookupInput(component_type="LED", component_id="D1")
    )
    assert tool_result.payload["provider"] == "local_fallback"
    rule_id = tool_result.payload["structured_rules"][0]["rule_id"]
    assert rule_id.startswith("fallback.led.")

    answer = build_diagnostic_template_answer(
        station_id="NO_STATION",
        query="LED 怎么用",
        user_message="LED 怎么用",
        evidence=evidence,
        context_pack=context_pack,
        tool_results=[tool_result],
    )
    assert rule_id in answer
    assert "无 datasheet 命中" in answer
    assert "无 LLM 合成" in answer


def test_phase4_router_enables_datasheet_tool_for_ne555_pinout_query() -> None:
    """Sanity check: the SemanticRouter must add datasheet_lookup_tool to
    allowed_tools for this exact query, which is what unlocks the chain.
    """
    evidence = build_runtime_evidence_from_station(
        station_id="NO_STATION",
        station={"comparison_report": {"items": []}},
        error_tag_service=ErrorTagService(),
    )
    pack = build_context_pack(evidence, user_message="NE555 引脚分布")
    tool_names = {tool.name for tool in pack.allowed_tools}
    assert "datasheet_lookup_tool" in tool_names


@pytest.mark.parametrize(
    "query",
    [
        "NE555 引脚分布",
        "555 输出脉冲宽度怎么算",
        "LM324 的供电范围",
    ],
)
def test_router_admits_chip_queries_into_diagnostic_pack(query: str) -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="NO_STATION",
        station={"comparison_report": {"items": []}},
        error_tag_service=ErrorTagService(),
    )
    pack = build_context_pack(evidence, user_message=query)
    tool_names = {tool.name for tool in pack.allowed_tools}
    assert "datasheet_lookup_tool" in tool_names, query
