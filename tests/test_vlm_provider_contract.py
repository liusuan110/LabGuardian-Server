"""Phase 6 VLM contract & white-box gate tests.

Covers:
- `vlm_explanation_v1` schema is identical across providers
- `analyze_micro_defect` returns the same shape + adds `defect_type`
- `suggest_defect_types` matches by tag overlap
- `verify_draft_answer` only sets `needs_micro_inspection=True` on the
  white-box-gated families and never on short_circuit / polarity
- `vlm_explain_node` is a no-op when the gate is closed
- `_route_after_verification` directs to `vlm_explain` only when both
  passed=True and needs_micro_inspection=True
"""

from __future__ import annotations

from app.agent.contracts import (
    AllowedTool,
    ContextPack,
    DiagnosticState,
    EvidenceRef,
    RuntimeEvidence,
    VerificationReport,
)
from app.agent.evidence import build_runtime_evidence_from_station
from app.agent.graph import _route_after_verification
from app.agent.nodes.vlm_explain import vlm_explain_node
from app.agent.verification import verify_draft_answer
from app.services.error_tag_service import ErrorTagService
from app.services.vlm import (
    DEFECT_TYPE_PROMPTS,
    MicroDefectType,
    analyze_micro_defect,
    suggest_defect_types,
)
from app.services.vlm_service import VlmService


_REQUIRED_KEYS = {
    "result_version",
    "provider",
    "model",
    "status",
    "inputs",
    "prompt",
    "answer",
    "raw_response",
}


def _minimal_pack() -> dict:
    return {
        "scene": {"scene_id": "exp_first_order_rc", "scene_name": "first-order RC"},
        "fault_cases": [
            {"title": "导线未剥皮导致悬空", "reference_text": "请检查跳线 W1"},
        ],
        "fix_steps": ["剥皮 5mm", "重新插入孔位"],
        "error_tags": ["unstripped_wire"],
        "structured_context": {"error_codes": ["FLOATING_PIN"]},
        "references": {"images": [], "waveforms": [], "schematics": []},
    }


def test_template_provider_returns_v1_schema() -> None:
    service = VlmService(provider="template")
    result = service.explain_rc_pack(mrag_pack=_minimal_pack(), user_query="为什么 W1 悬空")
    assert isinstance(result, dict)
    assert _REQUIRED_KEYS.issubset(result.keys())
    assert result["result_version"] == "vlm_explanation_v1"
    assert result["provider"] == "template"
    answer = result["answer"]
    assert "conclusion" in answer
    assert "evidence" in answer
    assert "fix_steps" in answer


def test_analyze_micro_defect_preserves_v1_schema_and_tags_defect() -> None:
    service = VlmService(provider="template")
    result = analyze_micro_defect(
        vlm_service=service,
        defect_type=MicroDefectType.UNSTRIPPED_WIRE,
        mrag_pack=_minimal_pack(),
        user_query="W1 是否未剥皮",
    )
    assert _REQUIRED_KEYS.issubset(result.keys())
    assert result["defect_type"] == MicroDefectType.UNSTRIPPED_WIRE.value
    assert result["result_version"] == "vlm_explanation_v1"


def test_defect_type_prompts_have_one_entry_per_enum() -> None:
    assert set(DEFECT_TYPE_PROMPTS.keys()) == set(MicroDefectType)
    for prompt in DEFECT_TYPE_PROMPTS.values():
        assert prompt and isinstance(prompt, str)


def test_suggest_defect_types_matches_by_tag() -> None:
    assert suggest_defect_types(["burn_mark"]) == [MicroDefectType.BURN_MARK]
    # Chinese alias should match too.
    assert suggest_defect_types(["焦黑"]) == [MicroDefectType.BURN_MARK]
    assert suggest_defect_types(["unstripped_wire", "未剥皮"]) == [
        MicroDefectType.UNSTRIPPED_WIRE,
    ]
    assert suggest_defect_types(["random_tag"]) == []


# ---- White-box gate ----

def _make_evidence(*, family_codes: list[str], risk: str = "warning", error_tags: list[str] | None = None) -> RuntimeEvidence:
    """Build a synthetic RuntimeEvidence for gating tests."""
    items = [
        {
            "error_code": code,
            "severity": risk,
            "component_id": "X1",
        }
        for code in family_codes
    ]
    return build_runtime_evidence_from_station(
        station_id="ST_GATE",
        station={
            "risk_level": risk,
            "comparison_report": {"items": items},
        },
        error_tag_service=ErrorTagService(),
    )


def _make_pack(family: str) -> ContextPack:
    return ContextPack(
        pack_id="gate_test",
        error_family=family,  # type: ignore[arg-type]
        risk_level="warning",
        pushed_facts=[],
        allowed_tools=[],
        evidence_refs=[],
    )


def test_gate_does_not_fire_for_short_circuit() -> None:
    evidence = _make_evidence(family_codes=["COMPONENT_SHORTED_SAME_NET"])
    pack = _make_pack("short_circuit")
    report = verify_draft_answer(
        evidence=evidence,
        context_pack=pack,
        draft_answer="X1 短路，请断电后复查电源接线。",
    )
    assert report.passed is True
    assert report.needs_micro_inspection is False
    assert report.suspected_defect_types == []


def test_gate_fires_for_missing_component_family() -> None:
    evidence = _make_evidence(family_codes=["COMPONENT_MISSING"])
    pack = _make_pack("missing_component")
    report = verify_draft_answer(
        evidence=evidence,
        context_pack=pack,
        draft_answer="缺少 X1。",
    )
    assert report.passed is True
    assert report.needs_micro_inspection is True
    assert MicroDefectType.BURN_MARK.value in report.suspected_defect_types
    assert MicroDefectType.UNSTRIPPED_WIRE.value in report.suspected_defect_types
    assert MicroDefectType.COLD_SOLDER.value in report.suspected_defect_types


def test_gate_fires_when_error_tag_explicitly_signals_defect() -> None:
    evidence = build_runtime_evidence_from_station(
        station_id="ST_BURN",
        station={
            "risk_level": "danger",
            "comparison_report": {
                "items": [{"error_code": "POLARITY_REVERSED", "severity": "danger", "component_id": "D1"}]
            },
        },
    )
    # Inject the suspicious tag manually (in production it comes from validator).
    evidence_with_tag = evidence.model_copy(update={"error_tags": [*evidence.error_tags, "burn_mark"]})
    pack = _make_pack("polarity_error")
    report = verify_draft_answer(
        evidence=evidence_with_tag,
        context_pack=pack,
        draft_answer="D1 极性可能接反，请断电复查电源。",
    )
    assert report.needs_micro_inspection is True
    assert report.suspected_defect_types == [MicroDefectType.BURN_MARK.value]


# ---- vlm_explain node ----

def _state_with_report(report: VerificationReport) -> DiagnosticState:
    evidence = _make_evidence(family_codes=["COMPONENT_MISSING"])
    pack = _make_pack("missing_component")
    return DiagnosticState(
        runtime_evidence=evidence,
        context_pack=pack,
        error_family="missing_component",
        verification_report=report,
        draft_answer="缺少 X1。",
    )


def test_vlm_explain_node_skipped_when_gate_closed() -> None:
    state = _state_with_report(VerificationReport(passed=True, needs_micro_inspection=False))
    update = vlm_explain_node(state)
    assert "vlm_findings" not in update  # node returned no findings
    assert any(
        m["payload"].get("skipped") == "gate_closed"
        for m in update["graph_metrics"]
    )


def test_vlm_explain_node_skipped_when_no_defect_types() -> None:
    state = _state_with_report(
        VerificationReport(passed=True, needs_micro_inspection=True, suspected_defect_types=[])
    )
    update = vlm_explain_node(state)
    assert "vlm_findings" not in update
    assert any(
        m["payload"].get("skipped") == "no_defect_types"
        for m in update["graph_metrics"]
    )


# ---- Routing ----

def test_route_after_verification_three_way() -> None:
    base_evidence = _make_evidence(family_codes=["COMPONENT_MISSING"])
    base_pack = _make_pack("missing_component")
    common = dict(
        runtime_evidence=base_evidence,
        context_pack=base_pack,
        draft_answer="x",
    )
    failing = DiagnosticState(
        verification_report=VerificationReport(passed=False, issues=["x"]),
        **common,
    )
    passing_clean = DiagnosticState(
        verification_report=VerificationReport(passed=True, needs_micro_inspection=False),
        **common,
    )
    passing_needs_vlm = DiagnosticState(
        verification_report=VerificationReport(
            passed=True,
            needs_micro_inspection=True,
            suspected_defect_types=[MicroDefectType.BURN_MARK.value],
        ),
        **common,
    )
    assert _route_after_verification(failing) == "repair_answer"
    assert _route_after_verification(passing_clean) == "finalize_answer"
    assert _route_after_verification(passing_needs_vlm) == "vlm_explain"
