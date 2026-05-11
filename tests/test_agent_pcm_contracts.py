from app.agent.context_pack import build_context_pack, classify_error_family
from app.agent.evidence import build_runtime_evidence_from_station
from app.agent.tools import (
    BoardSchemaLookupInput,
    DatasheetLookupInput,
    FaultCaseLookupInput,
    NetlistTraceInput,
    board_schema_lookup_tool,
    datasheet_lookup_tool,
    fault_case_lookup_tool,
    netlist_trace_tool,
)
from app.agent.verification import verify_draft_answer
from app.services.error_tag_service import ErrorTagService


def test_runtime_evidence_extracts_validator_codes_tags_and_refs() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="S01",
        station={
            "risk_level": "warning",
            "diagnostics": ["C1.pin1 节点不匹配"],
            "comparison_report": {
                "items": [
                    {
                        "error_code": "NODE_MISMATCH",
                        "severity": "error",
                        "component_id": "C1",
                        "pin_name": "pin1",
                        "expected": "ROW_10_L",
                        "actual": "ROW_11_L",
                    }
                ]
            },
        },
        error_tag_service=ErrorTagService(),
    )

    assert evidence.station_id == "S01"
    assert evidence.error_codes == ["NODE_MISMATCH"]
    assert evidence.error_tags == ["wrong_node_connection"]
    assert evidence.evidence_refs[0].component_id == "C1"
    assert evidence.evidence_refs[0].pin_name == "pin1"


def test_context_pack_routes_short_circuit_to_required_tools() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="S02",
        station={
            "risk_level": "danger",
            "comparison_report": {
                "items": [
                    {
                        "error_code": "COMPONENT_SHORTED_SAME_NET",
                        "component_id": "R1",
                    }
                ]
            },
        },
        error_tag_service=ErrorTagService(),
    )

    pack = build_context_pack(evidence, query="为什么冒烟")
    required_tools = {tool.name for tool in pack.allowed_tools if tool.required}

    assert classify_error_family(evidence) == "short_circuit"
    assert pack.pack_id == "pcm_short_circuit_v1"
    assert "netlist_trace_tool" in required_tools
    assert "safety_rule_lookup_tool" in required_tools
    assert any("断电" in rule for rule in pack.prompt_rules)
    assert pack.metrics is not None
    assert pack.metrics.pushed_facts_count == len(pack.pushed_facts)
    assert pack.metrics.allowed_tool_count == len(pack.allowed_tools)
    assert pack.metrics.char_count > 0
    assert pack.metrics.estimated_tokens > 0


def test_context_pack_routes_node_mismatch_to_board_lookup() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="S03",
        station={
            "risk_level": "warning",
            "comparison_report": {
                "items": [
                    {
                        "error_code": "NODE_MISMATCH",
                        "component_id": "C1",
                        "pin_name": "pin1",
                    }
                ]
            },
        },
    )

    pack = build_context_pack(evidence)
    required_tools = {tool.name for tool in pack.allowed_tools if tool.required}

    assert pack.error_family == "wiring_mismatch"
    assert "board_schema_lookup_tool" in required_tools


def test_context_pack_routes_polarity_error_to_datasheet_lookup() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="S06",
        station={
            "risk_level": "warning",
            "comparison_report": {
                "items": [
                    {
                        "error_code": "POLARITY_REVERSED",
                        "component_id": "D1",
                        "component_type": "LED",
                    }
                ]
            },
        },
    )

    pack = build_context_pack(evidence)
    required_tools = {tool.name for tool in pack.allowed_tools if tool.required}

    assert pack.error_family == "polarity_error"
    assert "datasheet_lookup_tool" in required_tools


def test_board_schema_lookup_tool_explains_rail_segments() -> None:
    result = board_schema_lookup_tool(BoardSchemaLookupInput(hole_id="LP32"))

    assert result.tool_name == "board_schema_lookup_tool"
    assert result.payload["hole"]["electrical_node_id"] == "TRACK_LP_SEG2"


def test_fault_case_lookup_tool_uses_teaching_kb() -> None:
    result = fault_case_lookup_tool(
        FaultCaseLookupInput(error_tags=["wrong_node_connection"])
    )

    assert result.payload["fault_cases"]
    knowledge_ids = {case["knowledge_id"] for case in result.payload["fault_cases"]}
    assert "rc_wrong_output_node_for_integrator" in knowledge_ids


def test_datasheet_lookup_tool_uses_local_fallback() -> None:
    result = datasheet_lookup_tool(
        DatasheetLookupInput(component_type="LED", component_id="D1")
    )

    assert result.tool_name == "datasheet_lookup_tool"
    assert result.payload["provider"] == "local_fallback"
    assert result.payload["component_type"] == "LED"
    assert any("限流" in rule for rule in result.payload["safety_rules"])


def test_netlist_trace_tool_reads_runtime_netlist() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="S04",
        station={
            "netlist_v2": {
                "components": [
                    {
                        "component_id": "R1",
                        "pins": [{"pin_name": "pin1", "electrical_net_id": "N1"}],
                    }
                ],
                "nets": [{"net_id": "N1", "members": ["R1.pin1", "ROW_1_L"]}],
            },
        },
    )

    result = netlist_trace_tool(evidence, NetlistTraceInput(component_id="R1"))

    assert result.payload["components"][0]["component_id"] == "R1"
    assert result.payload["nets"][0]["net_id"] == "N1"


def test_verifier_requires_runtime_citation_and_safety_hint() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="S05",
        station={
            "risk_level": "danger",
            "comparison_report": {
                "items": [
                    {
                        "error_code": "COMPONENT_SHORTED_SAME_NET",
                        "component_id": "R1",
                    }
                ]
            },
        },
    )
    pack = build_context_pack(evidence)

    failed = verify_draft_answer(
        evidence=evidence,
        context_pack=pack,
        draft_answer="这个电路可能有问题。",
    )
    passed = verify_draft_answer(
        evidence=evidence,
        context_pack=pack,
        draft_answer="COMPONENT_SHORTED_SAME_NET 显示 R1 两端短路，请先断电复查。",
    )

    assert not failed.passed
    assert failed.issues
    assert passed.passed


def test_evidence_extracts_visual_uncertainty_from_netlist_v2() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="S10",
        station={
            "risk_level": "warning",
            "netlist_v2": {
                "components": [
                    {
                        "component_id": "R1",
                        "confidence": 0.4,
                        "pins": [
                            {
                                "pin_name": "1",
                                "is_ambiguous": True,
                                "metadata": {
                                    "source": "heuristic_fallback",
                                    "snap_confidence": 0.2,
                                },
                            },
                            {"pin_name": "2", "is_ambiguous": False, "metadata": {}},
                        ],
                    },
                    {
                        "component_id": "C1",
                        "confidence": 0.9,
                        "pins": [{"pin_name": "a", "metadata": {}}],
                    },
                ]
            },
        },
    )

    assert evidence.ambiguous_pin_count == 1
    assert evidence.fallback_pin_count == 1
    assert evidence.snap_conflict_count == 1
    assert evidence.low_confidence_component_count == 1


def test_evidence_visual_uncertainty_defaults_zero_without_signal() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="S11",
        station={
            "netlist_v2": {
                "components": [
                    {"component_id": "R1", "confidence": 1.0, "pins": [{"pin_name": "1"}]},
                ]
            },
        },
    )

    assert evidence.ambiguous_pin_count == 0
    assert evidence.fallback_pin_count == 0
    assert evidence.snap_conflict_count == 0
    assert evidence.low_confidence_component_count == 0


def test_verifier_requires_reshoot_hint_when_visual_uncertain() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="S12",
        station={
            "risk_level": "warning",
            "netlist_v2": {
                "components": [
                    {
                        "component_id": "R1",
                        "confidence": 1.0,
                        "pins": [{"pin_name": "1", "is_ambiguous": True}],
                    }
                ]
            },
            "comparison_report": {
                "items": [{"error_code": "NODE_MISMATCH", "component_id": "R1"}]
            },
        },
    )
    pack = build_context_pack(evidence)

    failed = verify_draft_answer(
        evidence=evidence,
        context_pack=pack,
        draft_answer="R1 的引脚接到了错误的电气节点，请按图纸更正接线。",
    )
    passed = verify_draft_answer(
        evidence=evidence,
        context_pack=pack,
        draft_answer=(
            "NODE_MISMATCH 显示 R1 引脚可能接错，"
            "但识别置信度较低，建议复拍后人工确认孔位。"
        ),
    )

    assert not failed.passed
    assert any("复拍" in issue or "孔位" in issue for issue in failed.issues)
    assert passed.passed
