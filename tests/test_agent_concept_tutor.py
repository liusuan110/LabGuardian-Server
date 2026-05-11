from app.agent.concepts import CONCEPT_LIBRARY, lookup_concept
from app.agent.contracts import (
    ContextPack,
    EvidenceRef,
    RuntimeEvidence,
)
from app.agent.intent import classify_intent
from app.agent.tools import (
    TeachingConceptLookupInput,
    teaching_concept_lookup_tool,
)
from app.agent.verification import verify_draft_answer
from app.schemas.angnt import AngntAskRequest
from app.services.agent_service import AgentService
from app.services.classroom_state import ClassroomState
from app.services.rag_service import RagService


def _service() -> AgentService:
    return AgentService(rag_service=RagService())


def _submit(service: AgentService, classroom: ClassroomState, *, mode: str, query: str):
    accepted = service.submit(
        AngntAskRequest(station_id="S01", query=query, mode=mode),
        classroom,
    )
    status = service.get_status(accepted.job_id)
    assert status.result is not None
    return status.result


# ---------- intent classifier ----------

def test_intent_classifier_routes_led_question_to_concept_tutor() -> None:
    assert classify_intent("为什么 LED 要串联电阻", evidence=None) == "concept_tutor"


def test_intent_classifier_routes_current_circuit_led_question_to_mixed() -> None:
    assert classify_intent("我这个电路为什么 LED 要串联电阻", evidence=None) == "mixed"


def test_intent_classifier_routes_rc_question_to_concept_tutor() -> None:
    assert classify_intent("什么是 RC 时间常数", evidence=None) == "concept_tutor"


def test_intent_classifier_routes_diagnosis_to_diagnostic() -> None:
    assert classify_intent("我这个电路哪里错了", evidence=None) == "diagnostic"


def test_intent_classifier_routes_multimeter_to_lab_guidance() -> None:
    assert classify_intent("怎么用万用表检查短路", evidence=None) == "lab_guidance"


def test_intent_classifier_returns_mixed_when_concept_with_findings() -> None:
    evidence = RuntimeEvidence(station_id="S", findings=[])
    # Inject a fake finding by appending after construction.
    from app.agent.contracts import DiagnosticFinding

    evidence.findings.append(DiagnosticFinding(error_code="NODE_MISMATCH"))
    assert classify_intent("RC 时间常数和我现在的实验有什么关系", evidence=evidence) == "mixed"


# ---------- teaching_concept_lookup_tool ----------

def test_teaching_concept_lookup_tool_returns_local_pack_by_keyword() -> None:
    result = teaching_concept_lookup_tool(TeachingConceptLookupInput(query="为什么 LED 要限流"))
    assert result.status == "ok"
    assert result.payload["concept"]["concept_id"] == "led_current_limit"
    assert result.payload["provider"] == "local_concept_library"


def test_teaching_concept_lookup_tool_supports_all_six_concepts() -> None:
    expected = {
        "breadboard_basics",
        "ohms_law",
        "led_current_limit",
        "rc_time_constant",
        "voltage_divider",
        "capacitor_filtering",
    }
    assert set(CONCEPT_LIBRARY.keys()) == expected
    for cid in expected:
        result = teaching_concept_lookup_tool(TeachingConceptLookupInput(concept_id=cid))
        assert result.status == "ok"
        assert result.payload["concept"]["concept_id"] == cid


def test_teaching_concept_lookup_tool_returns_not_found_on_miss() -> None:
    result = teaching_concept_lookup_tool(TeachingConceptLookupInput(query="量子隧穿"))
    assert result.status == "not_found"
    assert "available_concepts" in result.payload


# ---------- verifier per intent ----------

def _stub_pack() -> ContextPack:
    return ContextPack(pack_id="p", error_family="unknown", risk_level="unknown")


def test_verifier_concept_mode_does_not_require_error_code() -> None:
    evidence = RuntimeEvidence(
        station_id="S",
        risk_level="safe",
        error_codes=["LED_MISSING_RESISTOR"],
    )
    concept = lookup_concept("LED 限流")
    draft = (
        "直接回答：LED 需要串联限流电阻。\n"
        "原理：欧姆定律 V=IR 用于估算限流。\n"
        "安全提醒：先断电再操作。\n"
        "知识来源：led_current_limit"
    )
    report = verify_draft_answer(
        evidence=evidence,
        context_pack=_stub_pack(),
        draft_answer=draft,
        intent="concept_tutor",
        concept=concept,
    )
    assert report.passed, report.issues


def test_verifier_diagnostic_requires_error_code_when_present() -> None:
    evidence = RuntimeEvidence(
        station_id="S",
        risk_level="safe",
        error_codes=["NODE_MISMATCH"],
    )
    report = verify_draft_answer(
        evidence=evidence,
        context_pack=_stub_pack(),
        draft_answer="R1 的连接节点与参考不符，请对照参考电路检查。",
        intent="diagnostic",
    )
    assert not report.passed
    assert any("error_code" in issue for issue in report.issues)


def test_verifier_diagnostic_requires_evidence_ref_when_present() -> None:
    evidence = RuntimeEvidence(
        station_id="S",
        risk_level="safe",
        evidence_refs=[
            EvidenceRef(
                ref_id="validator:1",
                component_id="R1",
                pin_name="A",
                hole_id="A1",
            )
        ],
    )
    report = verify_draft_answer(
        evidence=evidence,
        context_pack=_stub_pack(),
        draft_answer="当前连接与参考不一致，请重新核对。",
        intent="diagnostic",
    )
    assert not report.passed
    assert any(
        "evidence_ref" in issue or "component_id" in issue
        for issue in report.issues
    )


def test_verifier_diagnostic_passes_with_error_code_and_evidence_ref() -> None:
    evidence = RuntimeEvidence(
        station_id="S",
        risk_level="safe",
        error_codes=["NODE_MISMATCH"],
        evidence_refs=[
            EvidenceRef(
                ref_id="validator:1",
                component_id="R1",
                pin_name="A",
                hole_id="A1",
            )
        ],
    )
    report = verify_draft_answer(
        evidence=evidence,
        context_pack=_stub_pack(),
        draft_answer="NODE_MISMATCH 显示 R1 的 A 引脚连接与参考不一致，请核对 A1 孔位。",
        intent="diagnostic",
    )
    assert report.passed, report.issues


def test_verifier_concept_mode_blocks_fabricated_hole_when_no_evidence() -> None:
    evidence = RuntimeEvidence(station_id="S", risk_level="safe")
    concept = lookup_concept("LED 限流")
    bad_draft = (
        "原理：LED 限流。当前电路中 R1 接到了 ROW_5_L。\n"
        "安全提醒：先断电再操作。\n"
        "知识来源：led_current_limit"
    )
    report = verify_draft_answer(
        evidence=evidence,
        context_pack=_stub_pack(),
        draft_answer=bad_draft,
        intent="concept_tutor",
        concept=concept,
    )
    assert not report.passed
    assert any("没有 evidence" in issue or "孔位" in issue for issue in report.issues)


def test_verifier_lab_guidance_requires_steps_and_safety() -> None:
    evidence = RuntimeEvidence(station_id="S", risk_level="safe")
    fail = verify_draft_answer(
        evidence=evidence,
        context_pack=_stub_pack(),
        draft_answer="用万用表检查就行。",
        intent="lab_guidance",
    )
    assert not fail.passed
    good = verify_draft_answer(
        evidence=evidence,
        context_pack=_stub_pack(),
        draft_answer="1. 先断电。\n2. 用万用表通断挡测电源轨。",
        intent="lab_guidance",
    )
    assert good.passed, good.issues


def test_verifier_concept_mode_requires_safety_for_led_topic() -> None:
    evidence = RuntimeEvidence(station_id="S", risk_level="safe")
    concept = lookup_concept("LED 限流")
    no_safety = (
        "直接回答：LED 需要串联电阻。\n"
        "原理：欧姆定律。\n"
        "知识来源：led_current_limit"
    )
    report = verify_draft_answer(
        evidence=evidence,
        context_pack=_stub_pack(),
        draft_answer=no_safety,
        intent="concept_tutor",
        concept=concept,
    )
    assert not report.passed
    assert any("安全" in issue for issue in report.issues)


def test_verifier_diagnostic_intent_default_preserves_existing_behavior() -> None:
    """Regression: omitting intent keeps the diagnostic rule set."""
    evidence = RuntimeEvidence(station_id="S", risk_level="danger")
    report = verify_draft_answer(
        evidence=evidence,
        context_pack=_stub_pack(),
        draft_answer="电路有问题。",
    )
    assert not report.passed
    assert any("danger" in issue for issue in report.issues)


# ---------- agent_auto mode end-to-end ----------

def test_agent_auto_routes_led_question_to_concept_tutor() -> None:
    classroom = ClassroomState()
    result = _submit(_service(), classroom, mode="agent_auto", query="为什么 LED 要串联电阻")

    evidence_types = {item.evidence_type for item in result.evidence}
    assert "intent" in evidence_types
    intent_item = next(item for item in result.evidence if item.evidence_type == "intent")
    assert intent_item.payload["intent"] == "concept_tutor"
    assert "concept_pack" in evidence_types
    assert "知识来源" in result.answer
    assert result.mode == "agent_auto"


def test_agent_auto_routes_rc_question_to_concept_tutor() -> None:
    classroom = ClassroomState()
    result = _submit(_service(), classroom, mode="agent_auto", query="什么是 RC 时间常数")

    intent_item = next(item for item in result.evidence if item.evidence_type == "intent")
    assert intent_item.payload["intent"] == "concept_tutor"
    concept_item = next(item for item in result.evidence if item.evidence_type == "concept_pack")
    assert concept_item.payload["concept_id"] == "rc_time_constant"


def test_agent_auto_routes_multimeter_to_lab_guidance() -> None:
    classroom = ClassroomState()
    result = _submit(_service(), classroom, mode="agent_auto", query="怎么用万用表检查短路")

    intent_item = next(item for item in result.evidence if item.evidence_type == "intent")
    assert intent_item.payload["intent"] == "lab_guidance"
    # Step markers and safety word must be in the answer (verifier-enforced).
    assert "1." in result.answer
    assert any(w in result.answer for w in ("断电", "电源", "短路"))


def test_agent_auto_routes_diagnostic_question_to_diagnostic_path() -> None:
    classroom = ClassroomState()
    classroom.update_station(
        {
            "station_id": "S01",
            "risk_level": "danger",
            "diagnostics": ["R1 两端短路"],
            "comparison_report": {
                "items": [
                    {
                        "error_code": "COMPONENT_SHORTED_SAME_NET",
                        "component_id": "R1",
                        "severity": "danger",
                    }
                ]
            },
        }
    )
    result = _submit(_service(), classroom, mode="agent_auto", query="我这个电路哪里错了")

    evidence_types = {item.evidence_type for item in result.evidence}
    # Diagnostic path emits runtime_evidence + context_pack + react_trace.
    assert "context_pack" in evidence_types
    assert "react_trace" in evidence_types
    assert "断电" in result.answer


def test_agent_auto_mixed_returns_diagnostic_answer_and_attaches_concept() -> None:
    classroom = ClassroomState()
    classroom.update_station(
        {
            "station_id": "S01",
            "risk_level": "warning",
            "diagnostics": ["R1 NODE_MISMATCH"],
            "comparison_report": {
                "items": [
                    {
                        "error_code": "NODE_MISMATCH",
                        "component_id": "R1",
                        "severity": "warning",
                    }
                ]
            },
        }
    )
    result = _submit(
        _service(),
        classroom,
        mode="agent_auto",
        query="RC 时间常数和我现在的实验有什么关系",
    )

    evidence_types = {item.evidence_type for item in result.evidence}
    assert "context_pack" in evidence_types  # diagnostic main path
    assert "intent" in evidence_types
    intent_item = next(item for item in result.evidence if item.evidence_type == "intent")
    assert intent_item.payload["intent"] == "mixed"
    # Concept attached but final_answer is the diagnostic answer, not the concept template.
    assert "concept_pack" in evidence_types
    assert "知识来源" not in result.answer


# ---------- direct concept_tutor / lab_guidance modes ----------

def test_explicit_concept_tutor_mode_runs_without_findings() -> None:
    classroom = ClassroomState()
    result = _submit(_service(), classroom, mode="concept_tutor", query="什么是欧姆定律")

    assert "知识来源" in result.answer
    concept_item = next(item for item in result.evidence if item.evidence_type == "concept_pack")
    assert concept_item.payload["concept_id"] == "ohms_law"


def test_explicit_lab_guidance_mode_returns_steps_and_safety() -> None:
    classroom = ClassroomState()
    result = _submit(_service(), classroom, mode="lab_guidance", query="怎么用万用表检查短路")

    assert "1." in result.answer
    assert any(w in result.answer for w in ("断电", "电源", "短路"))


# ---------- regression: diagnostic_agent unchanged ----------

def test_diagnose_mode_still_falls_back_to_rag() -> None:
    """Regression: mode="diagnose" must NOT be diverted into concept routing."""
    classroom = ClassroomState()
    classroom.update_station({"station_id": "S01", "risk_level": "safe"})
    result = _submit(_service(), classroom, mode="diagnose", query="为什么 LED 要串联电阻")
    # The diagnose/RAG path does NOT emit intent / concept_pack evidence items.
    evidence_types = {item.evidence_type for item in result.evidence}
    assert "intent" not in evidence_types
    assert "concept_pack" not in evidence_types
