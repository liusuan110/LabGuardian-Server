from app.agent.answering import build_verified_diagnostic_answer, extract_fix_steps
from app.agent.context_pack import build_context_pack
from app.agent.evidence import build_runtime_evidence_from_station
from app.agent.tools import ToolResult


def test_build_verified_diagnostic_answer_passes_danger_short_circuit() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="S01",
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
    pack = build_context_pack(evidence, query="为什么危险")

    answer, passed, issues = build_verified_diagnostic_answer(
        station_id="S01",
        query="为什么危险",
        user_message="为什么危险",
        evidence=evidence,
        context_pack=pack,
        tool_results=[],
    )

    assert passed is True
    assert issues == []
    assert "R1" in answer
    assert "断电" in answer


def test_extract_fix_steps_dedupes_tool_steps() -> None:
    steps = extract_fix_steps(
        [
            ToolResult(
                tool_name="fault_case_lookup_tool",
                payload={
                    "fault_cases": [
                        {
                            "fix_steps": [
                                "断电后检查连接",
                                "重新跨行插接元件",
                            ]
                        }
                    ]
                },
            ),
            ToolResult(
                tool_name="safety_rule_lookup_tool",
                payload={"rules": ["断电后检查连接"]},
            ),
        ]
    )

    assert steps == ["断电后检查连接", "重新跨行插接元件"]

