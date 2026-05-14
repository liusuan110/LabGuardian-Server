from __future__ import annotations

from app.agent.context_pack import build_context_pack
from app.agent.evidence import build_runtime_evidence_from_station
from app.agent.tools import (
    DatasheetLookupInput,
    ToolResult,
    datasheet_lookup_tool,
)
from app.agent.verification import verify_draft_answer
from app.services.error_tag_service import ErrorTagService


def test_datasheet_tool_prefers_local_v2_with_chunk_ids() -> None:
    result = datasheet_lookup_tool(
        DatasheetLookupInput(part_number="NE555", query="NE555 引脚 pinout")
    )

    assert result.payload["provider"] == "local_datasheet_v2"
    hits = result.payload["hits"]
    assert hits, "expected v2 hits for NE555 pinout"
    for hit in hits:
        assert hit["chunk_id"]
        assert hit["modality"] in {"text", "table", "figure", "schematic", "waveform"}
        assert hit["document_id"]


def test_datasheet_tool_local_fallback_emits_rule_ids() -> None:
    result = datasheet_lookup_tool(
        DatasheetLookupInput(component_type="LED", component_id="D1")
    )

    assert result.payload["provider"] == "local_fallback"
    structured = result.payload["structured_rules"]
    assert structured
    assert all(rule.get("rule_id", "").startswith("fallback.led.") for rule in structured)
    assert result.payload["component_type"] == "LED"


def _evidence_and_pack():
    evidence = build_runtime_evidence_from_station(
        station_id="DS01",
        station={"comparison_report": {"items": []}},
        error_tag_service=ErrorTagService(),
    )
    return evidence, build_context_pack(evidence)


def test_verifier_fails_when_chunk_id_missing_after_v2_hit() -> None:
    evidence, pack = _evidence_and_pack()
    tool_result = datasheet_lookup_tool(
        DatasheetLookupInput(part_number="NE555", query="NE555 引脚 pinout")
    )

    report = verify_draft_answer(
        evidence=evidence,
        context_pack=pack,
        draft_answer="NE555 是一个常见的定时器芯片。",
        intent="concept_tutor",
        tool_results=[tool_result],
    )

    assert not report.passed
    assert any("chunk_id" in issue for issue in report.issues)


def test_verifier_passes_when_chunk_id_cited() -> None:
    evidence, pack = _evidence_and_pack()
    tool_result = datasheet_lookup_tool(
        DatasheetLookupInput(part_number="NE555", query="NE555 引脚 pinout")
    )
    cited_chunk_id = tool_result.payload["hits"][0]["chunk_id"]

    report = verify_draft_answer(
        evidence=evidence,
        context_pack=pack,
        draft_answer=(
            "知识来源：NE555 引脚 1=GND, 8=VCC（参见 "
            f"{cited_chunk_id}）。"
        ),
        intent="concept_tutor",
        tool_results=[tool_result],
    )

    assert report.passed, report.issues


def test_verifier_passes_when_rule_id_cited_for_fallback() -> None:
    evidence, pack = _evidence_and_pack()
    tool_result = datasheet_lookup_tool(
        DatasheetLookupInput(component_type="LED", component_id="D1")
    )
    rule_id = tool_result.payload["structured_rules"][0]["rule_id"]

    report = verify_draft_answer(
        evidence=evidence,
        context_pack=pack,
        draft_answer=(
            "原理：LED 必须串联限流电阻，且必须先断电再检查电源轨。"
            f"依据规则 {rule_id}。"
        ),
        intent="concept_tutor",
        tool_results=[tool_result],
    )

    assert report.passed, report.issues


def test_verifier_skips_when_no_datasheet_tool_result() -> None:
    evidence, pack = _evidence_and_pack()
    other = ToolResult(tool_name="netlist_trace_tool", summary="x", payload={})

    report = verify_draft_answer(
        evidence=evidence,
        context_pack=pack,
        draft_answer="原理：电路分析方法。",
        intent="concept_tutor",
        tool_results=[other],
    )

    assert report.passed, report.issues
