from app.services.classroom_state import ClassroomState
from app.services.error_tag_service import ErrorTagService
from app.services.rag_service import RagService
from app.services.teaching_kb_service import TeachingKbService


def test_teaching_kb_lists_all_demo_topology_scenes():
    """The KB carries one scene per demo topology (6 demos total: RC + 5 amps).

    Updated when the analog-circuit demo scenes were added; the original
    assertion `== {"exp_first_order_rc"}` would now spuriously fail.
    """
    service = TeachingKbService()

    scene_ids = {scene["scene_id"] for scene in service.list_scenes()}

    expected = {
        "exp_first_order_rc",
        "exp_common_emitter_amplifier",
        "exp_differential_amplifier",
        "exp_ua741_inverting_amplifier",
        "exp_ua741_summing_amplifier",
        "exp_ua741_integrator",
    }
    assert scene_ids == expected


def test_teaching_kb_matches_validator_error_code():
    service = TeachingKbService()

    hits = service.search(query="为什么这几个孔不是同一节点", error_codes=["NODE_MISMATCH"])

    assert hits
    assert hits[0]["matching_faults"]
    assert any(
        "NODE_MISMATCH" in fault["related_error_codes"]
        for fault in hits[0]["matching_faults"]
    )


def test_teaching_kb_matches_error_tag():
    service = TeachingKbService()

    hits = service.search(query="", error_tags=["probe_mode_error"])

    assert hits
    assert hits[0]["scene_id"] == "exp_first_order_rc"
    assert hits[0]["matching_faults"][0]["fault_id"] == "probe_x10_not_accounted"


def test_teaching_kb_lists_rc_fault_cases():
    service = TeachingKbService()

    cases = service.list_fault_cases(scene_id="exp_first_order_rc")

    assert {case["knowledge_id"] for case in cases} == {
        "rc_scope_ground_not_reference_ground",
        "rc_wrong_output_node_for_integrator",
        "rc_probe_x10_not_accounted",
        "rc_wrong_signal_offset",
        "rc_capacitor_value_mismatch",
    }


def test_teaching_kb_searches_fault_case_by_error_tag():
    service = TeachingKbService()

    cases = service.search_fault_cases(error_tags=["probe_mode_error"])

    assert cases
    assert cases[0]["knowledge_id"] == "rc_probe_x10_not_accounted"
    assert cases[0]["reference_waveforms"]


def test_teaching_kb_builds_rc_knowledge_pack():
    service = TeachingKbService()

    pack = service.build_knowledge_pack(error_tags=["wrong_node_connection"])

    assert pack["scene_id"] == "exp_first_order_rc"
    assert pack["fault_cases"]
    assert pack["references"]["texts"]
    assert pack["fix_steps"]


def test_teaching_kb_matches_rc_measurement_question():
    service = TeachingKbService()

    hits = service.search(query="示波器 X10 档为什么读数要乘以 10")

    assert hits
    assert hits[0]["scene_id"] == "exp_first_order_rc"


def test_error_tag_service_maps_rc_validator_report():
    service = ErrorTagService()
    report = {
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
    }

    tags = service.extract_tags(report)

    assert tags == [
        {
            "error_tag": "wrong_node_connection",
            "source_error_code": "NODE_MISMATCH",
            "severity": "error",
            "component_id": "C1",
            "pin_name": "pin1",
            "expected": "ROW_10_L",
            "actual": "ROW_11_L",
            "suggested_action": "",
            "teaching_focus": ["rc_output_node", "breadboard_node"],
            "evidence_refs": [],
        }
    ]


def test_rag_context_includes_rc_error_tags_and_scene():
    classroom = ClassroomState()
    classroom.update_station(
        {
            "station_id": "S01",
            "risk_level": "warning",
            "progress": 0.5,
            "diagnostics": ["C1.pin1 节点不匹配"],
            "comparison_report": {
                "items": [
                    {
                        "error_code": "NODE_MISMATCH",
                        "severity": "error",
                        "component_id": "C1",
                        "pin_name": "pin1",
                    }
                ]
            },
        }
    )
    service = RagService(
        teaching_kb_service=TeachingKbService(),
        error_tag_service=ErrorTagService(),
    )

    context = service.build_context(
        classroom=classroom,
        station_id="S01",
        query="为什么积分电路输出波形不对",
    )

    evidence_types = [item.evidence_type for item in context["evidence"]]
    assert "error_tags" in evidence_types
    assert "teaching_scene" in evidence_types
    assert "fault_case_pack" in evidence_types


def test_rag_context_can_use_fault_case_from_query_without_error_tag():
    classroom = ClassroomState()
    classroom.update_station(
        {
            "station_id": "S02",
            "risk_level": "safe",
            "progress": 0.2,
            "diagnostics": [],
            "comparison_report": {"items": []},
        }
    )
    service = RagService(teaching_kb_service=TeachingKbService())

    context = service.build_context(
        classroom=classroom,
        station_id="S02",
        query="示波器 X10 档为什么读数要乘以 10",
    )

    evidence_types = [item.evidence_type for item in context["evidence"]]
    assert "fault_case_pack" in evidence_types
