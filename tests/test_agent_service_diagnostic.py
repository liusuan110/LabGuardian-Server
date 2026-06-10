from app.schemas.angnt import AngntAskRequest
from app.services.agent_service import AgentService
from app.services.classroom_state import ClassroomState
from app.services.error_tag_service import ErrorTagService
from app.services.rag_service import RagService


def _submit_diagnostic(
    service: AgentService,
    classroom: ClassroomState,
    *,
    station_id: str = "S01",
    query: str = "现在怎么办",
):
    accepted = service.submit(
        AngntAskRequest(
            station_id=station_id,
            query=query,
            mode="diagnostic_agent",
        ),
        classroom,
    )
    status = service.get_status(accepted.job_id)
    assert status.result is not None
    return status.result


def _station_payload(
    *,
    station_id: str = "S01",
    risk_level: str,
    error_code: str,
    component_id: str = "R1",
) -> dict:
    return {
        "station_id": station_id,
        "risk_level": risk_level,
        "diagnostics": [f"{component_id} {error_code}"],
        "risk_reasons": [error_code],
        "comparison_report": {
            "items": [
                {
                    "error_code": error_code,
                    "severity": "danger" if risk_level == "danger" else "warning",
                    "component_id": component_id,
                    "suggested_action": f"按 {error_code} 修复 {component_id}",
                }
            ]
        },
    }


def test_diagnostic_agent_mode_builds_template_answer_and_verifies() -> None:
    classroom = ClassroomState()
    classroom.update_station(
        {
            "station_id": "S01",
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
        }
    )
    service = AgentService(
        rag_service=RagService(error_tag_service=ErrorTagService())
    )

    accepted = service.submit(
        AngntAskRequest(
            station_id="S01",
            query="为什么电路危险",
            mode="diagnostic_agent",
        ),
        classroom,
    )
    status = service.get_status(accepted.job_id)

    assert status.result is not None
    assert status.result.mode == "diagnostic_agent"
    assert "R1" in status.result.answer
    assert "断电" in status.result.answer
    evidence_types = {item.evidence_type for item in status.result.evidence}
    assert "runtime_evidence" in evidence_types
    assert "context_pack" in evidence_types
    assert "tool_results" in evidence_types
    assert "verification_report" in evidence_types
    assert "graph_metrics" in evidence_types
    verification = next(
        item for item in status.result.evidence if item.evidence_type == "verification_report"
    )
    assert verification.payload["passed"] is True
    graph_metrics = next(
        item for item in status.result.evidence if item.evidence_type == "graph_metrics"
    )
    metric_names = [item["node_name"] for item in graph_metrics.payload["metrics"]]
    # Phase 4 ReAct loop emits per-iteration metrics; we only assert the boundary nodes.
    assert any(name.startswith("react_observe_") for name in metric_names)
    assert "verify_answer" in metric_names
    assert "react_trace" in evidence_types
    react_trace = next(
        item for item in status.result.evidence if item.evidence_type == "react_trace"
    )
    assert react_trace.payload["iterations"] >= 1
    assert isinstance(react_trace.payload["steps"], list)
    assert react_trace.payload["steps"]
    assert all("iteration" in step for step in react_trace.payload["steps"])


def test_diagnostic_agent_first_sentence_prioritizes_circuit_facts() -> None:
    classroom = ClassroomState()
    classroom.update_station(
        {
            "station_id": "S01",
            "risk_level": "warning",
            "diagnostics": ["R1 连接节点与参考不符"],
            "comparison_report": {
                "summary": {
                    "reference_id": "rc_first_order_v1",
                    "reference_name": "一阶 RC 电路",
                },
                "items": [
                    {
                        "error_code": "WRONG_CONNECTION",
                        "severity": "warning",
                        "component_id": "R1",
                        "suggested_action": "请将 R1 改接到参考电路对应网络。",
                    }
                ],
            },
            "netlist_v2": {
                "components": [
                    {"component_id": "R1", "component_type": "Resistor", "pins": []},
                    {"component_id": "C1", "component_type": "Capacitor", "pins": []},
                ],
                "nets": [],
            },
        }
    )
    service = AgentService(rag_service=RagService(error_tag_service=ErrorTagService()))

    result = _submit_diagnostic(service, classroom, query="请根据当前诊断结果给出演示用诊断解释和下一步建议。")
    first_sentence = result.answer.split("。", 1)[0]

    assert first_sentence.startswith("先看这个电路本身")
    assert "WRONG_CONNECTION" in first_sentence
    assert "一阶 RC 电路(rc_first_order_v1)" in first_sentence
    assert "R1(Resistor)" in first_sentence
    assert "C1(Capacitor)" in first_sentence


def test_diagnostic_agent_mode_works_without_station_state() -> None:
    classroom = ClassroomState()
    service = AgentService(rag_service=RagService())

    accepted = service.submit(
        AngntAskRequest(
            station_id="missing",
            query="现在状态如何",
            mode="diagnostic_agent",
        ),
        classroom,
    )
    status = service.get_status(accepted.job_id)

    assert status.result is not None
    assert status.result.answer
    # 证据链首位现在是 intent 记录，station_id 按 payload 键定位而非硬编码下标
    station_evidence = next(
        item for item in status.result.evidence if "station_id" in item.payload
    )
    assert station_evidence.payload["station_id"] == "missing"


def test_diagnostic_agent_uses_history_for_repeated_error() -> None:
    classroom = ClassroomState()
    service = AgentService(rag_service=RagService(error_tag_service=ErrorTagService()))
    classroom.update_station(
        _station_payload(
            risk_level="danger",
            error_code="COMPONENT_SHORTED_SAME_NET",
        )
    )
    _submit_diagnostic(service, classroom, query="第一次诊断")

    classroom.update_station(
        _station_payload(
            risk_level="danger",
            error_code="COMPONENT_SHORTED_SAME_NET",
        )
    )
    result = _submit_diagnostic(service, classroom, query="第二次诊断")

    assert "这个问题仍然存在" in result.answer
    assert "上一轮也检测到" in result.answer
    timeline = next(item for item in result.evidence if item.evidence_type == "context_timeline")
    assert any(
        "repeated_error_codes=COMPONENT_SHORTED_SAME_NET" in fact
        for fact in timeline.payload["history_facts"]
    )


def test_diagnostic_agent_mentions_risk_decrease() -> None:
    classroom = ClassroomState()
    service = AgentService(rag_service=RagService(error_tag_service=ErrorTagService()))
    classroom.update_station(
        _station_payload(
            risk_level="danger",
            error_code="COMPONENT_SHORTED_SAME_NET",
        )
    )
    _submit_diagnostic(service, classroom)

    classroom.update_station(
        _station_payload(
            risk_level="warning",
            error_code="NODE_MISMATCH",
            component_id="C1",
        )
    )
    result = _submit_diagnostic(service, classroom)

    assert "比上一轮有所改善" in result.answer
    timeline = next(item for item in result.evidence if item.evidence_type == "context_timeline")
    assert any(
        "risk_level_decreased=danger->warning" in fact
        for fact in timeline.payload["history_facts"]
    )


def test_diagnostic_agent_history_facts_record_error_change() -> None:
    classroom = ClassroomState()
    service = AgentService(rag_service=RagService(error_tag_service=ErrorTagService()))
    classroom.update_station(
        _station_payload(
            risk_level="warning",
            error_code="NODE_MISMATCH",
            component_id="C1",
        )
    )
    _submit_diagnostic(service, classroom)

    classroom.update_station(
        _station_payload(
            risk_level="warning",
            error_code="POLARITY_REVERSED",
            component_id="D1",
        )
    )
    result = _submit_diagnostic(service, classroom)

    timeline = next(item for item in result.evidence if item.evidence_type == "context_timeline")
    assert any(
        "error_codes_changed=NODE_MISMATCH->POLARITY_REVERSED" in fact
        for fact in timeline.payload["history_facts"]
    )
    assert "当前主要问题已从 NODE_MISMATCH 变为 POLARITY_REVERSED" in result.answer


def test_diagnostic_agent_memory_keeps_last_five_records() -> None:
    classroom = ClassroomState()
    service = AgentService(rag_service=RagService(error_tag_service=ErrorTagService()))

    for idx in range(6):
        classroom.update_station(
            _station_payload(
                risk_level="warning",
                error_code="NODE_MISMATCH",
                component_id=f"C{idx}",
            )
        )
        _submit_diagnostic(service, classroom)

    assert len(service._get_station_memory("S01")) == 5
