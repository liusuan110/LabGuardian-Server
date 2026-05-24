"""WP-1 integration tests: assert no RC scene leakage for non-RC topologies.

This pins the contract in ``docs/retrieval-contract.md`` end-to-end:

  - **Wrong-Scene Rate = 0**: for every non-RC topology, the resolved
    ``scene_id`` in evidence MUST equal the expected one (never RC).
  - **No-topology behavior**: when topology cannot be resolved, fault_case
    evidence is skipped entirely (no silent fallback to RC).
  - **Tool-level skip**: ``fault_case_lookup_tool`` skips on empty
    scene_id without raising.

Together with ``test_scene_resolver.py`` these tests form the WP-1
regression net — if any of them fails, the legacy RC default has crept
back in somewhere.
"""

from __future__ import annotations

import pytest

from app.agent.contracts import RuntimeEvidence
from app.agent.evidence import build_runtime_evidence_from_station
from app.agent.tools import FaultCaseLookupInput, fault_case_lookup_tool
from app.services.classroom_state import ClassroomState
from app.services.error_tag_service import ErrorTagService
from app.services.rag_service import RagService
from app.services.scene_resolver import TOPOLOGY_LABEL_TO_SCENE_ID
from app.services.teaching_kb_service import TeachingKbService


# ---------------------------------------------------------------------------
# 1. Evidence builder propagates scene_id without leaking RC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topology_label,expected_scene_id",
    list(TOPOLOGY_LABEL_TO_SCENE_ID.items()),
)
def test_evidence_builder_stamps_correct_scene_id(
    topology_label: str, expected_scene_id: str
) -> None:
    """For each of the 6 topologies, evidence.current_scene_id must match."""
    evidence = build_runtime_evidence_from_station(
        station_id="S_test",
        station={
            "station_id": "S_test",
            "topology_label": topology_label,
            "risk_level": "safe",
            "diagnostics": [],
            "comparison_report": {"items": []},
        },
    )
    assert isinstance(evidence, RuntimeEvidence)
    assert evidence.current_scene_id == expected_scene_id


def test_evidence_builder_leaves_scene_id_empty_when_topology_unknown() -> None:
    """No topology hint → evidence.current_scene_id is empty string (not RC)."""
    evidence = build_runtime_evidence_from_station(
        station_id="S_unknown",
        station={
            "station_id": "S_unknown",
            "risk_level": "safe",
            "diagnostics": [],
            "comparison_report": {"items": []},
        },
    )
    assert evidence.current_scene_id == ""
    # Specifically — must not default to RC.
    assert evidence.current_scene_id != "exp_first_order_rc"


# ---------------------------------------------------------------------------
# 2. fault_case_lookup_tool fails open on empty scene_id
# ---------------------------------------------------------------------------


def test_fault_case_tool_skips_on_empty_scene_id() -> None:
    """Empty scene_id → ``status='skipped'``, no service call."""
    result = fault_case_lookup_tool(
        FaultCaseLookupInput(
            query="任何问题",
            error_tags=["wrong_pin_assignment"],
            scene_id="",  # explicit empty
            top_k=3,
        )
    )
    assert result.status == "skipped"
    assert result.payload["fault_cases"] == []
    assert result.payload["skip_reason"] == "no_scene_context"


def test_fault_case_tool_runs_when_scene_id_present() -> None:
    """A valid scene_id → tool actually queries TeachingKbService."""
    result = fault_case_lookup_tool(
        FaultCaseLookupInput(
            query="UA741 输出饱和",
            error_tags=[],
            scene_id="exp_ua741_inverting_amplifier",
            top_k=3,
        )
    )
    # Service was called; status defaults to "ok" and scene_id is echoed.
    assert result.status == "ok"
    assert result.payload.get("scene_id") == "exp_ua741_inverting_amplifier"


# ---------------------------------------------------------------------------
# 3. RagService.build_context: no RC scene_id in any non-RC evidence source_id
# ---------------------------------------------------------------------------


NON_RC_TOPOLOGIES = [
    label for label in TOPOLOGY_LABEL_TO_SCENE_ID if label != "rc_first_order"
]


@pytest.mark.parametrize("topology_label", NON_RC_TOPOLOGIES)
def test_rag_context_has_no_rc_leakage_for_non_rc_topology(topology_label: str) -> None:
    """For every non-RC topology, no evidence source_id may contain
    ``exp_first_order_rc``. This is the wrong-scene-rate=0 gate."""
    expected_scene_id = TOPOLOGY_LABEL_TO_SCENE_ID[topology_label]
    classroom = ClassroomState()
    classroom.update_station(
        {
            "station_id": "S_topo",
            "topology_label": topology_label,
            "risk_level": "warning",
            "progress": 0.5,
            "diagnostics": ["示例诊断"],
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
        station_id="S_topo",
        query="为什么该电路输出不正常",
    )

    rc_hits = [
        item.source_id
        for item in context["evidence"]
        if "exp_first_order_rc" in (item.source_id or "")
    ]
    assert rc_hits == [], (
        f"WP-1 wrong-scene leakage: topology={topology_label} produced evidence "
        f"with exp_first_order_rc source_id ({rc_hits}). "
        f"Expected scene_id={expected_scene_id}."
    )


@pytest.mark.parametrize(
    "topology_label,expected_scene_id",
    list(TOPOLOGY_LABEL_TO_SCENE_ID.items()),
)
def test_rag_context_fault_case_pack_is_scene_keyed(
    topology_label: str, expected_scene_id: str
) -> None:
    """If a fault_case_pack surfaces, its source_id MUST contain the
    expected scene_id and not any other scene."""
    classroom = ClassroomState()
    classroom.update_station(
        {
            "station_id": "S_pack",
            "topology_label": topology_label,
            "risk_level": "warning",
            "diagnostics": ["示例"],
            "comparison_report": {
                "items": [
                    {"error_code": "NODE_MISMATCH", "severity": "error"},
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
        station_id="S_pack",
        query="任意诊断问题",
    )

    packs = [item for item in context["evidence"] if item.evidence_type == "fault_case_pack"]
    for pack in packs:
        assert expected_scene_id in (pack.source_id or ""), (
            f"WP-1: fault_case_pack source_id={pack.source_id!r} does not "
            f"contain expected scene_id={expected_scene_id!r}"
        )
        # And critically — no other scene_id may appear in this source_id.
        for other_scene in TOPOLOGY_LABEL_TO_SCENE_ID.values():
            if other_scene == expected_scene_id:
                continue
            assert other_scene not in (pack.source_id or ""), (
                f"WP-1 cross-scene leakage: pack source_id={pack.source_id!r} "
                f"contains {other_scene!r} but expected {expected_scene_id!r}"
            )


def test_rag_context_skips_pack_entirely_when_no_topology() -> None:
    """No topology → no fault_case_pack. Confirms the resolver's None branch
    is honored end-to-end through RagService."""
    classroom = ClassroomState()
    classroom.update_station(
        {
            "station_id": "S_none",
            # No topology_label, no scene_id, no validator topology_label.
            "risk_level": "safe",
            "diagnostics": [],
            "comparison_report": {"items": []},
        }
    )
    service = RagService(
        teaching_kb_service=TeachingKbService(),
        error_tag_service=ErrorTagService(),
    )
    context = service.build_context(
        classroom=classroom,
        station_id="S_none",
        query="一般性问题",
    )
    evidence_types = [item.evidence_type for item in context["evidence"]]
    assert "fault_case_pack" not in evidence_types


# ---------------------------------------------------------------------------
# 4. Error tags are scene-agnostic — no RC vocabulary in non-RC turns
#    (User-reported P1-A: missing_rc_component / incomplete_rc_circuit /
#    rc_output_node / rc_component_set were leaking into all scenes.)
# ---------------------------------------------------------------------------


_RC_FLAVORED_TAG_TOKENS = (
    "missing_rc_component",
    "incomplete_rc_circuit",
    "rc_output_node",
    "rc_component_set",
)


@pytest.mark.parametrize("topology_label", list(TOPOLOGY_LABEL_TO_SCENE_ID.keys()))
def test_error_tags_have_no_rc_specific_vocabulary(topology_label: str) -> None:
    """ErrorTagService output MUST be scene-agnostic. The legacy tag
    vocabulary (``missing_rc_component`` etc.) is forbidden across all
    topologies — including RC itself, where the renamed tag is correct."""
    # Synthetic validator report covering each code that previously produced
    # an RC-flavored tag.
    report = {
        "items": [
            {"error_code": "COMPONENT_MISSING", "severity": "error", "component_id": "R1"},
            {"error_code": "TOPOLOGY_VALID_SUBSET", "severity": "warning"},
            {"error_code": "NODE_MISMATCH", "severity": "error", "component_id": "C1", "pin_name": "pin1"},
        ]
    }
    tags = ErrorTagService().extract_tags(report)
    rendered_parts: list[str] = []
    for tag in tags:
        rendered_parts.append(str(tag.get("error_tag", "")))
        rendered_parts.extend(str(focus) for focus in tag.get("teaching_focus", []))
    rendered = " ".join(rendered_parts)
    for forbidden in _RC_FLAVORED_TAG_TOKENS:
        assert forbidden not in rendered, (
            f"WP-1 P1-A leak: topology={topology_label} produced RC-flavored "
            f"tag token {forbidden!r}. Full tag output: {tags!r}"
        )


# ---------------------------------------------------------------------------
# 5. Planner cannot override scene_id via tool args (P1-B defense)
# ---------------------------------------------------------------------------


def test_react_observe_ignores_planner_scene_id_override() -> None:
    """A planner ToolCall that emits scene_id != resolved scene MUST be
    overridden by ``runtime_evidence.current_scene_id``. Otherwise an
    over-confident planner could re-introduce RC fault cases on UA741 turns."""
    from app.agent.contracts import (
        AllowedTool,
        ContextPack,
        DiagnosticState,
        RuntimeEvidence,
    )
    from app.agent.nodes.react_observe import _dispatch_tool

    state = DiagnosticState(
        query="UA741 输出饱和",
        top_k=3,
        runtime_evidence=RuntimeEvidence(
            station_id="S_planner",
            current_scene_id="exp_ua741_inverting_amplifier",
            error_tags=[],
        ),
        context_pack=ContextPack(
            pack_id="t",
            error_family="unknown",
            risk_level="safe",
            allowed_tools=[
                AllowedTool(name="fault_case_lookup_tool", reason="test")
            ],
        ),
    )
    # Planner attempts to override with RC.
    malicious_args = {"scene_id": "exp_first_order_rc", "query": state.query}
    result = _dispatch_tool("fault_case_lookup_tool", malicious_args, state)
    # Tool MUST have queried with the resolved scene, not the planner's.
    assert result.payload.get("scene_id") == "exp_ua741_inverting_amplifier", (
        f"WP-1 P1-B defense failed: planner override slipped through; "
        f"tool payload scene_id={result.payload.get('scene_id')!r}"
    )


# ---------------------------------------------------------------------------
# 6. Recall guarantee — each of 6 scenes always surfaces its own
#    teaching_scene evidence when the topology is resolved (P2-A)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. v3 fixes — main-path wiring + error_code recall + scene-gated tool
# ---------------------------------------------------------------------------


def test_pipeline_stamps_topology_label_into_station(monkeypatch) -> None:
    """WP-1 v3 P0: pipeline_service.sync_result_to_classroom MUST call the
    GNN-A classifier and write ``topology_label`` into the station. Without
    this wiring the entire WP-1 contract is hollow — scene_resolver never
    gets the topology hint, current_scene_id always stays empty."""
    from app.services import pipeline_service as ps_mod

    captured: dict[str, str] = {}

    class _StubConsensus:
        recommended_label = "common_emitter"
        confidence_band = "high"

    class _StubSuggestion:
        enabled = True
        consensus = _StubConsensus()

    def _fake_suggest(netlist_v2):
        captured["netlist"] = "called"
        return _StubSuggestion()

    monkeypatch.setattr(
        "app.services.topology_classifier_service.suggest_from_netlist_v2",
        _fake_suggest,
    )

    label = ps_mod._resolve_topology_label_from_netlist(
        {"components": [{"component_id": "Q1", "pins": []}]}
    )
    assert label == "common_emitter"
    assert captured.get("netlist") == "called"


def test_pipeline_topology_resolver_returns_empty_on_unknown(monkeypatch) -> None:
    """``unknown`` / low-confidence MUST return '' so resolver fails open."""
    from app.services import pipeline_service as ps_mod

    class _LowConsensus:
        recommended_label = "common_emitter"
        confidence_band = "low"

    class _LowSuggestion:
        enabled = True
        consensus = _LowConsensus()

    monkeypatch.setattr(
        "app.services.topology_classifier_service.suggest_from_netlist_v2",
        lambda netlist_v2: _LowSuggestion(),
    )
    assert ps_mod._resolve_topology_label_from_netlist({"x": 1}) == ""


def test_search_fault_cases_recalls_by_error_code() -> None:
    """WP-1 v3 P1-A: validator error_codes are the primary KB bridge.
    Without this, the renamed scene-agnostic error_tags would never match
    fault_case JSONs (which use domain-specific vocab like
    ``missing_power_connection``)."""
    service = TeachingKbService()
    # UA741 inverting + FLOATING_PIN should hit ``vee_pin_not_connected``.
    cases = service.search_fault_cases(
        scene_id="exp_ua741_inverting_amplifier",
        error_tags=[],  # deliberately empty → only error_code matching
        error_codes=["FLOATING_PIN"],
    )
    assert cases, (
        "WP-1 v3 P1-A: error_code recall failed — UA741 + FLOATING_PIN "
        "should match vee_pin_not_connected via related_error_codes."
    )
    knowledge_ids = {c.get("knowledge_id") for c in cases}
    assert "inv_vee_pin_not_connected" in knowledge_ids


def test_context_pack_omits_fault_case_tool_when_scene_unresolved() -> None:
    """WP-1 v3 P1-B: when topology is not resolved, fault_case_lookup_tool
    MUST NOT be added to allowed_tools. Otherwise it burns a ReAct
    iteration on a guaranteed-skip call and can crowd out safety_rule."""
    from app.agent.context_pack import build_context_pack
    from app.agent.contracts import RuntimeEvidence

    # No topology_label → resolver returns None → current_scene_id == "".
    evidence = RuntimeEvidence(
        station_id="S_noscene",
        current_scene_id="",  # explicit
        error_codes=["COMPONENT_SHORTED_SAME_NET"],
        risk_level="danger",
    )
    pack = build_context_pack(evidence, query="为什么短路", intent="diagnostic")
    tool_names = {t.name for t in pack.allowed_tools}
    assert "fault_case_lookup_tool" not in tool_names, (
        f"WP-1 v3 P1-B: fault_case_lookup_tool surfaced in allowed_tools "
        f"despite empty current_scene_id. Tools: {tool_names}"
    )


def test_context_pack_includes_fault_case_tool_when_scene_resolved() -> None:
    """Mirror of the above: with a resolved scene, the tool must be present."""
    from app.agent.context_pack import build_context_pack
    from app.agent.contracts import RuntimeEvidence

    evidence = RuntimeEvidence(
        station_id="S_withscene",
        current_scene_id="exp_ua741_inverting_amplifier",
        error_codes=["FLOATING_PIN"],
        risk_level="warning",
    )
    pack = build_context_pack(evidence, query="UA741 输出饱和", intent="diagnostic")
    tool_names = {t.name for t in pack.allowed_tools}
    assert "fault_case_lookup_tool" in tool_names


# ---------------------------------------------------------------------------
# 8. Dynamic ReAct cap — required tools always fit (WP-1 v5, P1-A)
# ---------------------------------------------------------------------------


def test_dangerous_circuit_calls_safety_rule_on_production_default_cap() -> None:
    """WP-1 v5: with prod default ``REACT_MAX_ITERATIONS=4`` and a
    short-circuit + 'circuit' keyword query, the 6 allowed tools used to
    overflow the budget and ``safety_rule_lookup_tool`` (required) was
    not called. Dynamic cap = max(default, len(allowed_tools)) fixes it."""
    from app.agent.evidence import build_runtime_evidence_from_station
    from app.agent.graph import run_diagnostic_graph

    evidence = build_runtime_evidence_from_station(
        station_id="S_safety",
        station={
            "topology_label": "rc_first_order",
            "risk_level": "danger",
            "diagnostics": ["R1 两端短路"],
            "comparison_report": {
                "items": [
                    {
                        "error_code": "COMPONENT_SHORTED_SAME_NET",
                        "severity": "danger",
                        "component_id": "R1",
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
    # Production callsite (agent_service.py:647) does NOT pass
    # max_react_iterations — uses settings default. This test pins that
    # the safety_rule tool runs *without* the caller overriding the cap.
    state = run_diagnostic_graph(
        evidence=evidence,
        query="为什么电路危险",
        top_k=5,
    )
    required = {t.name for t in state.context_pack.allowed_tools if t.required}
    called = {r["tool_name"] for r in state.tool_results}
    missing = required - called
    assert "safety_rule_lookup_tool" in called, (
        f"WP-1 v5 P1-A regression: dangerous-circuit turn did not call "
        f"safety_rule_lookup_tool. tool_results={sorted(called)}, "
        f"cap={state.max_react_iterations}, "
        f"react_iterations={state.react_iterations}"
    )
    assert not missing, f"required tools still skipped: {missing}"


def test_react_cap_does_not_shrink_below_caller_explicit_value() -> None:
    """WP-1 v5: dynamic cap takes the MAX of caller cap and tools count.
    A caller that explicitly set cap=10 must not be downsized to a smaller
    tools-based value."""
    from app.agent.evidence import build_runtime_evidence_from_station
    from app.agent.graph import run_diagnostic_graph

    evidence = build_runtime_evidence_from_station(
        station_id="S_cap",
        station={
            # Concept-tutor-style turn: few allowed tools.
            "topology_label": "rc_first_order",
            "risk_level": "safe",
            "diagnostics": [],
            "comparison_report": {"items": []},
        },
        error_tag_service=ErrorTagService(),
    )
    state = run_diagnostic_graph(
        evidence=evidence,
        query="什么是 RC 时间常数",
        top_k=3,
        max_react_iterations=10,
        intent="concept_tutor",
    )
    # cap stays at the explicit 10 even though only ~2-3 tools are allowed.
    assert state.max_react_iterations >= 10


def test_react_cap_dynamic_for_normal_diagnostic_turns() -> None:
    """WP-1 v5: cap auto-grows to fit allowed_tools when caller didn't
    override. Verifies the telemetry field for audit visibility."""
    from app.agent.evidence import build_runtime_evidence_from_station
    from app.agent.graph import run_diagnostic_graph

    evidence = build_runtime_evidence_from_station(
        station_id="S_telemetry",
        station={
            "topology_label": "common_emitter",
            "risk_level": "warning",
            "diagnostics": [],
            "comparison_report": {
                "items": [{"error_code": "NODE_MISMATCH", "component_id": "Q1"}]
            },
        },
        error_tag_service=ErrorTagService(),
    )
    state = run_diagnostic_graph(
        evidence=evidence,
        query="共射放大输出异常",
        top_k=5,
    )
    ctx_metric = next(
        m for m in state.graph_metrics if m.node_name == "build_context_pack"
    )
    payload = ctx_metric.payload
    # Telemetry must record the cap decision for distillation audit logs.
    assert "react_cap_initial" in payload
    assert "react_cap_applied" in payload
    assert payload["react_cap_applied"] >= payload["allowed_tool_count"], (
        f"WP-1 v5: cap {payload['react_cap_applied']} smaller than "
        f"allowed_tool_count {payload['allowed_tool_count']} — required "
        f"tools may be skipped."
    )


# ---------------------------------------------------------------------------
# 7. error_code recall through the agent graph path (WP-1 v4)
#    User-reported: validator emits COMPONENT_MISSING, fault_case_pack
#    returns [] because tool_runner/react_observe weren't passing
#    error_codes to fault_case_lookup_tool.
# ---------------------------------------------------------------------------


def test_static_tool_runner_passes_error_codes_to_fault_case_tool() -> None:
    """WP-1 v4: tool_runner.run_diagnostic_tools must inject error_codes
    so the KB recall actually works in the production agent path."""
    from app.agent.contracts import (
        AllowedTool,
        ContextPack,
        DiagnosticFinding,
        RuntimeEvidence,
    )
    from app.agent.tool_runner import run_diagnostic_tools

    evidence = RuntimeEvidence(
        station_id="S_runner",
        current_scene_id="exp_ua741_inverting_amplifier",
        error_codes=["FLOATING_PIN", "COMPONENT_MISSING"],
        error_tags=["floating_connection"],
        findings=[
            DiagnosticFinding(
                error_code="FLOATING_PIN",
                component_id="U1",
                pin_name="pin4",
            )
        ],
    )
    context_pack = ContextPack(
        pack_id="t",
        error_family="incomplete_circuit",
        risk_level="warning",
        allowed_tools=[
            AllowedTool(name="fault_case_lookup_tool", reason="t", required=True),
        ],
    )
    results = run_diagnostic_tools(
        evidence=evidence,
        context_pack=context_pack,
        query="UA741 输出固定在 +13V 不变",
        top_k=3,
    )
    fault_results = [r for r in results if r.tool_name == "fault_case_lookup_tool"]
    assert fault_results, "fault_case_lookup_tool was not invoked"
    payload = fault_results[0].payload
    # The whole point of WP-1 v4: error_codes drive recall, so this MUST
    # find inv_vee_pin_not_connected (which declares related_error_codes
    # = [FLOATING_PIN, COMPONENT_MISSING]).
    assert payload["fault_cases"], (
        f"WP-1 v4 contract violation: fault_case_pack empty even though "
        f"validator emitted FLOATING_PIN + COMPONENT_MISSING and scene "
        f"is resolved. error_codes pipe is broken. Payload: {payload!r}"
    )
    knowledge_ids = {c["knowledge_id"] for c in payload["fault_cases"]}
    assert "inv_vee_pin_not_connected" in knowledge_ids


def test_react_observe_passes_error_codes_to_fault_case_tool() -> None:
    """WP-1 v4: the ReAct dynamic dispatcher must also inject error_codes
    so planner-driven invocations get KB recall too."""
    from app.agent.contracts import (
        AllowedTool,
        ContextPack,
        DiagnosticState,
        RuntimeEvidence,
    )
    from app.agent.nodes.react_observe import _dispatch_tool

    state = DiagnosticState(
        query="共射放大器静态工作点漂移",
        top_k=3,
        runtime_evidence=RuntimeEvidence(
            station_id="S_react",
            current_scene_id="exp_common_emitter_amplifier",
            error_codes=["NODE_MISMATCH"],
            error_tags=["wrong_node_connection"],
        ),
        context_pack=ContextPack(
            pack_id="t",
            error_family="wiring_mismatch",
            risk_level="warning",
            allowed_tools=[
                AllowedTool(name="fault_case_lookup_tool", reason="t", required=True),
            ],
        ),
    )
    # Planner emits an empty/no error_codes — dispatcher must inject from
    # evidence to keep recall working.
    result = _dispatch_tool("fault_case_lookup_tool", {"query": state.query}, state)
    assert result.payload["fault_cases"], (
        "WP-1 v4 contract violation: ReAct dispatcher did not inject "
        "error_codes; fault_case recall returns empty on a scene with "
        "a clear validator code."
    )


def test_rag_no_mrag_fallback_passes_error_codes_to_kb_service() -> None:
    """WP-1 v6 (P2): when ``RagService`` is constructed without a
    ``MragService`` (legacy tests / downgraded runs), the fallback to
    ``TeachingKbService.build_knowledge_pack`` must still pass
    ``error_codes`` — otherwise validator codes can't drive recall and
    ``fault_case_pack`` comes back empty even on a valid scene."""
    classroom = ClassroomState()
    classroom.update_station(
        {
            "station_id": "S_fallback",
            "topology_label": "inverting_amp_ua741",
            "risk_level": "danger",
            "diagnostics": ["UA741 输出固定在 +13V"],
            "comparison_report": {
                "items": [
                    {
                        "error_code": "FLOATING_PIN",
                        "severity": "danger",
                        "component_id": "U1",
                        "pin_name": "pin4",
                    }
                ]
            },
        }
    )
    # NOTE: NO mrag_service is injected — this is the fallback path.
    service = RagService(
        teaching_kb_service=TeachingKbService(),
        error_tag_service=ErrorTagService(),
    )
    context = service.build_context(
        classroom=classroom,
        station_id="S_fallback",
        query="UA741 输出饱和怎么办",
        top_k=5,
    )
    packs = [
        item for item in context["evidence"] if item.evidence_type == "fault_case_pack"
    ]
    assert packs, (
        "WP-1 v6 P2 regression: fallback path produced no fault_case_pack — "
        "error_codes not flowing into build_knowledge_pack."
    )
    fault_ids = {c.get("knowledge_id") for c in packs[0].payload.get("fault_cases", [])}
    assert "inv_vee_pin_not_connected" in fault_ids, (
        f"WP-1 v6 P2: expected inv_vee_pin_not_connected (FLOATING_PIN match) "
        f"in fallback path; got {fault_ids}"
    )


def test_react_observe_filters_planner_error_codes_to_validator_subset() -> None:
    """Defense: planner cannot widen the error_codes set beyond what the
    validator actually emitted. If planner names a code that's not in
    evidence.error_codes, we fall back to evidence's full list."""
    from app.agent.contracts import (
        AllowedTool,
        ContextPack,
        DiagnosticState,
        RuntimeEvidence,
    )
    from app.agent.nodes.react_observe import _dispatch_tool

    state = DiagnosticState(
        query="UA741 反相饱和",
        top_k=3,
        runtime_evidence=RuntimeEvidence(
            station_id="S_filter",
            current_scene_id="exp_ua741_inverting_amplifier",
            error_codes=["FLOATING_PIN"],  # validator only saw this
            error_tags=[],
        ),
        context_pack=ContextPack(
            pack_id="t",
            error_family="wiring_mismatch",
            risk_level="warning",
            allowed_tools=[
                AllowedTool(name="fault_case_lookup_tool", reason="t", required=True),
            ],
        ),
    )
    # Planner tries to inject a code validator never emitted.
    args = {"error_codes": ["POLARITY_REVERSED"]}
    result = _dispatch_tool("fault_case_lookup_tool", args, state)
    # Still gets results because we fell back to validator's FLOATING_PIN.
    assert result.payload["fault_cases"], (
        "WP-1 v4 defense failed: planner's bogus error_codes wiped out "
        "validator's real codes instead of being rejected."
    )


@pytest.mark.parametrize(
    "topology_label,expected_scene_id",
    list(TOPOLOGY_LABEL_TO_SCENE_ID.items()),
)
def test_resolved_topology_always_surfaces_its_own_teaching_scene(
    topology_label: str, expected_scene_id: str
) -> None:
    """When topology is resolved, the corresponding teaching_scene MUST
    appear in evidence — regardless of how the cross-scene ranker would
    have scored the query. This pins the P2-A recall guarantee."""
    classroom = ClassroomState()
    classroom.update_station(
        {
            "station_id": "S_recall",
            "topology_label": topology_label,
            "risk_level": "safe",
            "diagnostics": [],
            "comparison_report": {"items": [{"error_code": "NODE_MISMATCH"}]},
        }
    )
    service = RagService(
        teaching_kb_service=TeachingKbService(),
        error_tag_service=ErrorTagService(),
    )
    # Deliberately generic query that the ranker may not associate with
    # this specific topology — the recall guarantee MUST not depend on it.
    context = service.build_context(
        classroom=classroom,
        station_id="S_recall",
        query="输出异常",
        top_k=5,
    )
    scenes_in_evidence = [
        item.source_id
        for item in context["evidence"]
        if item.evidence_type == "teaching_scene"
    ]
    assert expected_scene_id in scenes_in_evidence, (
        f"WP-1 P2-A recall miss: topology={topology_label} resolved to "
        f"{expected_scene_id} but evidence had teaching_scene source_ids "
        f"{scenes_in_evidence}. The cross-scene ranker likely truncated the "
        f"correct scene below scene_cap."
    )
    # And — no other scene may appear as teaching_scene.
    for other_scene in TOPOLOGY_LABEL_TO_SCENE_ID.values():
        if other_scene == expected_scene_id:
            continue
        assert other_scene not in scenes_in_evidence, (
            f"WP-1 cross-scene leak: topology={topology_label} surfaced "
            f"teaching_scene {other_scene!r} (expected only {expected_scene_id!r})"
        )
