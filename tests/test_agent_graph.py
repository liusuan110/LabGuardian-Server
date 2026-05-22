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
    assert "R1" in state.final_answer
    assert "断电" in state.final_answer
    assert state.verification_report is not None
    assert state.verification_report.passed is True
    metric_names = [metric.node_name for metric in state.graph_metrics]
    # Phase 4 ReAct loop: classify → context → (plan/observe/reflect)*N → verify → finalize.
    # We assert presence + ordering of the deterministic boundary nodes; ReAct iterations
    # vary by tool count, so we only assert the loop ran at least once.
    assert metric_names[:2] == ["classify_error", "build_context_pack"]
    assert "react_plan_0" in metric_names
    assert "react_observe_0" in metric_names
    assert "react_reflect_0" in metric_names
    assert metric_names[-2:] == ["verify_answer", "finalize_answer"]
    # ReAct trace should record at least one step and terminate cleanly.
    assert state.react_iterations >= 1
    assert state.react_iterations <= state.max_react_iterations
    assert state.react_terminate_reason in {
        "no_more_tools",
        "verifier_passed_no_more_tools",
        "max_iterations_reached",
    }
    assert len(state.react_trace) >= 1
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


def test_diagnostic_graph_answers_circuit_inventory_question_from_kb() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="LG-DEMO-01",
        station={
            "risk_level": "warning",
            "diagnostics": ["RC1 缺失或连接异常", "RC2 缺失或连接异常"],
            "comparison_report": {
                "summary": {
                    "reference_id": "diff_pair_current_source_ref_split_potentiometer",
                    "reference_name": "差分放大器 + VT3 恒流源参考电路",
                },
                "items": [
                    {
                        "error_code": "COMPONENT_MISSING",
                        "severity": "warning",
                        "expected": {"ref_id": "RC1", "type": "Resistor"},
                        "suggested_action": "请添加 RC1。",
                    },
                    {
                        "error_code": "COMPONENT_MISSING",
                        "severity": "warning",
                        "expected": {"ref_id": "RC2", "type": "Resistor"},
                        "suggested_action": "请添加 RC2。",
                    },
                ],
            },
            "netlist_v2": {
                "components": [{"component_id": "Q1", "component_type": "Transistor"}],
                "nets": [],
            },
        },
        error_tag_service=ErrorTagService(),
    )

    state = run_diagnostic_graph(
        evidence=evidence,
        query="那差分电路一共需要几个电阻",
        user_message="那差分电路一共需要几个电阻",
        top_k=5,
        max_react_iterations=4,
    )

    tool_names = {item["tool_name"] for item in state.tool_results}
    assert "circuit_lookup_tool" in tool_names
    assert "电阻类元件一共是 6 个" in state.final_answer
    assert "RC1、RC2、RP、R1、R2、RE" in state.final_answer
    assert "RP_LEFT 和 RP_RIGHT" in state.final_answer
    assert "微观缺陷复检" not in state.final_answer
    assert state.verification_report is not None
    assert state.verification_report.needs_micro_inspection is False


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

    # Phase 4: build_diagnostic_template_answer is now invoked from react_reflect_node.
    monkeypatch.setattr(
        "app.agent.nodes.react_reflect.build_diagnostic_template_answer",
        lambda **_kwargs: "这个电路可能有问题。",
    )

    state = run_diagnostic_graph(
        evidence=evidence,
        query="为什么危险",
        top_k=5,
    )

    metric_names = [metric.node_name for metric in state.graph_metrics]
    assert "repair_answer" in metric_names
    assert "断电" in state.final_answer
    assert state.verification_report is not None
    assert state.verification_report.passed is True
