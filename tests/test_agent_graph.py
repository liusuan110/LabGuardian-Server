from app.agent.evidence import build_runtime_evidence_from_station
from app.agent.graph import run_diagnostic_graph
from app.services.error_tag_service import ErrorTagService


def test_diagnostic_graph_runs_white_box_short_circuit_path() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="S01",
        station={
            "risk_level": "danger",
            "diagnostics": ["R1 两端短路"],
            "risk_reasons": ["R1 两端落在同一节点"],
            "comparison_report": {
                "items": [
                    {
                        "error_code": "COMPONENT_SHORTED_SAME_NET",
                        "severity": "danger",
                        "component_id": "R1",
                        "suggested_action": "断电后重新跨行插接 R1",
                    }
                ]
            },
            "netlist_v2": {
                "components": [{"component_id": "R1", "pins": []}],
                "nets": [{"net_id": "N1", "members": ["R1.pin1", "R1.pin2"]}],
            },
        },
        error_tag_service=ErrorTagService(),
    )

    state = run_diagnostic_graph(
        evidence=evidence,
        query="为什么电路危险",
        top_k=5,
    )

    assert state.error_family == "short_circuit"
    assert state.context_pack is not None
    assert state.context_pack.pack_id == "pcm_short_circuit_v1"
    tool_names = {item["tool_name"] for item in state.tool_results}
    assert "netlist_trace_tool" in tool_names
    assert "fault_case_lookup_tool" in tool_names
    assert "safety_rule_lookup_tool" in tool_names
    assert "COMPONENT_SHORTED_SAME_NET" in state.final_answer
    assert "R1" in state.final_answer
    assert "断电" in state.final_answer
    assert state.verification_report is not None
    assert state.verification_report.passed is True
    metric_names = [metric.node_name for metric in state.graph_metrics]
    assert metric_names == [
        "classify_error",
        "build_context_pack",
        "run_tools",
        "generate_draft",
        "verify_answer",
        "finalize_answer",
    ]
    run_tools_metric = next(
        metric for metric in state.graph_metrics if metric.node_name == "run_tools"
    )
    assert run_tools_metric.payload["tool_count"] == len(tool_names)
    context_metric = next(
        metric for metric in state.graph_metrics if metric.node_name == "build_context_pack"
    )
    assert context_metric.payload["context_char_count"] > 0
    assert context_metric.payload["estimated_tokens"] > 0


def test_diagnostic_graph_runs_without_station_findings() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="missing",
        station={},
    )

    state = run_diagnostic_graph(
        evidence=evidence,
        query="现在状态如何",
        top_k=3,
    )

    assert state.error_family == "unknown"
    assert state.context_pack is not None
    assert state.context_pack.error_family == "unknown"
    assert state.final_answer
    assert state.verification_report is not None
    assert state.verification_report.passed is True


def test_diagnostic_graph_routes_failed_verification_to_repair(monkeypatch) -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="S02",
        station={
            "risk_level": "danger",
            "comparison_report": {
                "items": [
                    {
                        "error_code": "COMPONENT_SHORTED_SAME_NET",
                        "severity": "danger",
                        "component_id": "R1",
                    }
                ]
            },
        },
        error_tag_service=ErrorTagService(),
    )

    monkeypatch.setattr(
        "app.agent.graph.build_diagnostic_template_answer",
        lambda **_kwargs: "这个电路可能有问题。",
    )

    state = run_diagnostic_graph(
        evidence=evidence,
        query="为什么危险",
        top_k=5,
    )

    metric_names = [metric.node_name for metric in state.graph_metrics]
    assert "repair_answer" in metric_names
    assert "COMPONENT_SHORTED_SAME_NET" in state.final_answer
    assert "断电" in state.final_answer
    assert state.verification_report is not None
    assert state.verification_report.passed is True
