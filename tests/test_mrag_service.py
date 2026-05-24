from app.services.classroom_state import ClassroomState
from app.services.error_tag_service import ErrorTagService
from app.services.mrag_service import MragService
from app.services.rag_service import RagService
from app.services.teaching_kb_service import TeachingKbService


def test_mrag_builds_versioned_rc_pack_from_error_tag():
    service = MragService(teaching_kb_service=TeachingKbService())

    pack = service.build_pack(
        scene_id="exp_first_order_rc",  # WP-1: scene_id now required
        error_tags=["probe_mode_error"],
    )

    assert pack["pack_version"] == "mrag_pack_v1"
    assert pack["scene"]["scene_id"] == "exp_first_order_rc"
    assert pack["fault_cases"][0]["knowledge_id"] == "rc_probe_x10_not_accounted"
    assert pack["references"]["waveforms"]
    assert pack["fix_steps"]


def test_mrag_preserves_structured_context_for_vlm_next_stage():
    service = MragService(teaching_kb_service=TeachingKbService())

    pack = service.build_pack(
        scene_id="exp_first_order_rc",  # WP-1: scene_id now required
        error_tags=["wrong_node_connection"],
        structured_context={
            "error_codes": ["NODE_MISMATCH"],
            "component_id": "C1",
            "pin_name": "pin1",
        },
    )

    assert pack["structured_context"]["error_codes"] == ["NODE_MISMATCH"]
    assert pack["structured_context"]["component_id"] == "C1"
    assert pack["fault_cases"]


def test_mrag_pack_stays_v1_when_no_retrieved_supplied():
    service = MragService(teaching_kb_service=TeachingKbService())

    pack = service.build_pack(
        scene_id="exp_first_order_rc",  # WP-1
        error_tags=["probe_mode_error"],
        retrieved=None,
    )

    assert pack["pack_version"] == "mrag_pack_v1"
    assert "retrieved" not in pack


def test_mrag_pack_bumps_to_v2_when_retrieved_present():
    service = MragService(teaching_kb_service=TeachingKbService())
    retrieved = {
        "datasheet_chunks": [
            {"chunk_id": "ne555.pinout.text.1", "modality": "text", "title": "NE555 引脚"}
        ],
        "figures": [],
        "tables": [],
    }

    pack = service.build_pack(
        scene_id="exp_first_order_rc",  # WP-1
        error_tags=["probe_mode_error"],
        retrieved=retrieved,
    )

    assert pack["pack_version"] == "mrag_pack_v2"
    assert pack["retrieved"]["datasheet_chunks"][0]["chunk_id"] == "ne555.pinout.text.1"
    # Other top-level fields unchanged.
    assert pack["scene"]["scene_id"] == "exp_first_order_rc"
    assert pack["fault_cases"]


def test_mrag_pack_empty_retrieved_falls_back_to_v1():
    service = MragService(teaching_kb_service=TeachingKbService())

    pack = service.build_pack(
        scene_id="exp_first_order_rc",  # WP-1
        error_tags=["probe_mode_error"],
        retrieved={"datasheet_chunks": [], "figures": [], "tables": []},
    )

    assert pack["pack_version"] == "mrag_pack_v1"
    assert "retrieved" not in pack


def test_rag_context_uses_mrag_pack_payload():
    # WP-1 (2026-05-24): topology_label required to surface fault_case_pack.
    classroom = ClassroomState()
    classroom.update_station(
        {
            "station_id": "S03",
            "topology_label": "rc_first_order",
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
    teaching_kb = TeachingKbService()
    service = RagService(
        teaching_kb_service=teaching_kb,
        error_tag_service=ErrorTagService(),
        mrag_service=MragService(teaching_kb_service=teaching_kb),
    )

    context = service.build_context(
        classroom=classroom,
        station_id="S03",
        query="积分电路输出波形不对",
    )

    packs = [item for item in context["evidence"] if item.evidence_type == "fault_case_pack"]
    assert packs
    assert packs[0].payload["pack_version"] == "mrag_pack_v1"
    assert packs[0].payload["structured_context"]["error_codes"] == ["NODE_MISMATCH"]
