from app.schemas.angnt import AngntAskRequest
from app.services.agent_service import AgentService
from app.services.classroom_state import ClassroomState
from app.services.error_tag_service import ErrorTagService
from app.services.rag_service import RagService


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
    assert "COMPONENT_SHORTED_SAME_NET" in status.result.answer
    assert "R1" in status.result.answer
    assert "断电" in status.result.answer
    evidence_types = {item.evidence_type for item in status.result.evidence}
    assert "runtime_evidence" in evidence_types
    assert "context_pack" in evidence_types
    assert "tool_results" in evidence_types
    assert "verification_report" in evidence_types
    verification = next(
        item for item in status.result.evidence if item.evidence_type == "verification_report"
    )
    assert verification.payload["passed"] is True


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
    assert status.result.evidence[0].payload["station_id"] == "missing"

