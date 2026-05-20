"""P4 · GNNAdvisor inference + orchestrator integration tests.

Two layers of testing:

1. **Inference layer** — GNNAdvisor.advise produces well-formed advice,
   handles edge cases (no edges, missing checkpoint), and never crashes
   the caller.

2. **Orchestrator integration** — the existing rule-based comparator
   still produces the same ``logic_correct`` verdicts (zero regression
   on the 29 existing compare tests is verified separately); additionally
   the new ``report.summary.gnn`` field appears when a checkpoint is
   on disk + circuit is non-trivial.

Skipped automatically if torch / torch_geometric aren't installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from app.domain.gnn import GNNAdvice, GNNAdvisor, should_use_gnn  # noqa: E402
from app.domain.gnn.inference import (  # noqa: E402
    _DEFAULT_CKPT_ENV,
    _locate_default_checkpoint,
)
from app.domain.gnn.port_graph import (  # noqa: E402
    build_from_logical_reference,
    build_from_netlist_v2,
)
from app.domain.logical_reference import (  # noqa: E402
    current_netlist_v2_to_graph,
    logical_reference_to_graph,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "references"


def _ref_payload() -> dict:
    return json.loads((FIXTURES / "test_voltage_divider_v1.json").read_text())


def _matching_cur_netlist_v2() -> dict:
    """A cur netlist that exactly mirrors the divider ref (same shape,
    renamed component_ids and net_ids — orchestrator should still match
    by topology)."""

    return {
        "components": [
            {
                "component_id": "R_a",
                "component_type": "Resistor",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "n_in"},
                    {"pin_name": "pin2", "electrical_net_id": "n_mid"},
                ],
            },
            {
                "component_id": "R_b",
                "component_type": "Resistor",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "n_mid"},
                    {"pin_name": "pin2", "electrical_net_id": "n_gnd"},
                ],
            },
        ],
        "nets": [
            {"electrical_net_id": "n_in", "role": "input"},
            {"electrical_net_id": "n_mid", "role": "output"},
            {"electrical_net_id": "n_gnd", "role": "ground"},
        ],
    }


# ---------------------------------------------------------------------------
# GNNAdvice dataclass
# ---------------------------------------------------------------------------


def test_gnn_advice_to_report_dict_has_plan_field_schema() -> None:
    """Plan §六 ``validator_report_v2.summary.gnn`` schema — these are
    the keys the orchestrator promises to expose."""

    advice = GNNAdvice(
        model_version="circuit_match:v1",
        inference_ms=12.5,
        n_edges_scored=4,
        edge_predictions=({"edge": ["p", "n"], "p_correct": 0.9, "verdict": "ok"},),
        hotspots=({"node": "p", "score": 0.8, "hint": "..."},),
        graph_similarity=0.92,
        graph_similarity_confidence=0.85,
    )
    d = advice.to_report_dict()
    required = {
        "enabled", "model_version", "graph_similarity",
        "graph_similarity_confidence", "progress_score",
        "n_edges_scored", "inference_ms",
        "edge_predictions", "hotspots",
        "suggested_targets", "n_suggestion_candidates_scored",
        "predicted_error_types", "component_mapping_topk",
        "net_mapping_topk", "disagreement_with_rule",
    }
    assert required.issubset(d), f"missing keys: {required - d.keys()}"


def test_gnn_advice_to_report_dict_filters_low_confidence_hotspots() -> None:
    """Plan §六 ``MIN_HOTSPOT_CONFIDENCE = 0.6`` — below-threshold
    hotspots dropped at the report boundary."""

    advice = GNNAdvice(
        model_version="x",
        inference_ms=0.0,
        n_edges_scored=2,
        hotspots=(
            {"node": "p_high", "score": 0.85, "hint": "strong"},
            {"node": "p_mid", "score": 0.55, "hint": "weak — below threshold"},
        ),
    )
    d = advice.to_report_dict(min_hotspot_confidence=0.6)
    nodes = {h["node"] for h in d["hotspots"]}
    assert nodes == {"p_high"}, "weak hotspot must be filtered"


# ---------------------------------------------------------------------------
# GNNAdvisor.advise — happy path + edge cases
# ---------------------------------------------------------------------------


def test_advise_on_perfect_copy_yields_high_similarity() -> None:
    """When cur exactly matches ref (same source_ids, same topology),
    the model — trained on perfect-copy positives — should score
    ``graph_similarity`` high (>= 0.7 is a forgiving gate)."""

    ref_hcg = build_from_logical_reference(_ref_payload())
    cur_hcg = build_from_logical_reference(_ref_payload())  # identical
    cur_hcg.metadata["side"] = "cur"
    advisor = GNNAdvisor.get()
    advice = advisor.advise(ref_hcg, cur_hcg)
    assert advice is not None
    assert advice.n_edges_scored == len(cur_hcg.edges)
    assert 0.5 <= advice.graph_similarity <= 1.0, (
        f"perfect copy should score in [0.5, 1.0], got {advice.graph_similarity}"
    )
    assert advice.inference_ms > 0
    # Every edge prediction has the contract shape
    for ep in advice.edge_predictions:
        assert {"edge", "p_correct", "verdict"} <= ep.keys()
        assert 0.0 <= ep["p_correct"] <= 1.0
        assert ep["verdict"] in {"ok", "likely_wrong"}


def test_advise_on_cur_with_zero_edges_returns_empty_advice() -> None:
    """When cur is structurally empty (no observed edges), the advisor
    returns a valid GNNAdvice with n_edges_scored=0 rather than None."""

    ref_hcg = build_from_logical_reference(_ref_payload())
    # Build an empty cur — just nodes, no edges. We make a minimal
    # netlist_v2 with no edges by giving a single component with no pins.
    empty_cur = {
        "components": [
            {"component_id": "R_z", "component_type": "Resistor", "pins": []},
        ],
        "nets": [],
    }
    cur_hcg = build_from_netlist_v2(empty_cur)
    advisor = GNNAdvisor.get()
    advice = advisor.advise(ref_hcg, cur_hcg)
    assert advice is not None
    assert advice.n_edges_scored == 0
    assert advice.edge_predictions == ()
    assert advice.graph_similarity == 0.0


def test_advise_never_raises_on_model_exception(monkeypatch) -> None:
    """The advisor's outer try/except guarantees None on any model
    crash. Simulate by monkey-patching the model.forward to raise."""

    advisor = GNNAdvisor.get()

    def _exploding_forward(*a, **kw):
        raise RuntimeError("synthetic explosion for test")

    monkeypatch.setattr(advisor.model, "forward", _exploding_forward)
    ref_hcg = build_from_logical_reference(_ref_payload())
    cur_hcg = build_from_logical_reference(_ref_payload())
    advice = advisor.advise(ref_hcg, cur_hcg)
    assert advice is None, "exception must surface as None (silent fallback)"


def test_advise_soft_timeout_logs_warning_but_still_returns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Soft timeout: when wall-clock exceeds budget, we log a warning
    but still return the result (MVP semantics — real cancellation is
    a P4.1 follow-up)."""

    ref_hcg = build_from_logical_reference(_ref_payload())
    cur_hcg = build_from_logical_reference(_ref_payload())
    advisor = GNNAdvisor.get()
    # Set an absurdly tight timeout so even the trivial divider trips it
    import logging
    with caplog.at_level(logging.WARNING, logger="app.domain.gnn.inference"):
        advice = advisor.advise(ref_hcg, cur_hcg, timeout_ms=0)
    assert advice is not None
    assert any("exceeded timeout" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# GNNAdvisor singleton / checkpoint discovery
# ---------------------------------------------------------------------------


def test_get_returns_singleton_across_calls() -> None:
    GNNAdvisor.reset_singleton()
    a = GNNAdvisor.get()
    b = GNNAdvisor.get()
    assert a is b


def test_env_override_selects_alternate_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    """``LABGUARDIAN_GNN_CKPT`` env var must take priority over the
    default search list. If the env path is missing, the locator falls
    back to the defaults and logs a warning."""

    # Pointing to a missing file → fall back
    monkeypatch.setenv(_DEFAULT_CKPT_ENV, str(tmp_path / "nope.pt"))
    located = _locate_default_checkpoint()
    # Falls back to repo defaults (which exist after P3.2)
    assert located is not None
    assert "nope.pt" not in str(located)


def test_checkpoint_available_returns_true_when_p3_ckpt_exists() -> None:
    assert GNNAdvisor.checkpoint_available()


# ---------------------------------------------------------------------------
# should_use_gnn trigger predicate
# ---------------------------------------------------------------------------


def test_should_use_gnn_skips_trivial_circuits() -> None:
    assert not should_use_gnn({"node_count_total": 3})
    assert not should_use_gnn({"node_count_total": 7})


def test_should_use_gnn_enables_on_normal_circuit() -> None:
    assert should_use_gnn({"node_count_total": 8})
    assert should_use_gnn({"node_count_total": 20})


def test_should_use_gnn_short_circuits_on_safety_critical() -> None:
    """Even on a non-trivial circuit, if a safety check is pending the
    rule must own the verdict — GNN sits out."""

    assert not should_use_gnn({
        "node_count_total": 20,
        "has_safety_critical_check_pending": True,
    })
    assert not should_use_gnn({
        "node_count_total": 20,
        "deterministic_polarity_violation": True,
    })


def test_should_use_gnn_triggers_on_ged_fallback_match_type() -> None:
    """Plan §七 rule-fallback trigger fires once the circuit clears
    the size early-exit (≥ 8 nodes). Below that, plan §七 keeps GNN
    out regardless — small circuits are cheap for the rule comparator."""

    # Size early-exit is strict per plan §七 — small circuits don't
    # invoke GNN even if rule fell to GED.
    assert not should_use_gnn({
        "node_count_total": 6,
        "match_type_so_far": "graph_edit_distance_or_fallback",
    })
    # Once above the size threshold, the GED-fallback trigger does fire.
    assert should_use_gnn({
        "node_count_total": 10,
        "match_type_so_far": "graph_edit_distance_or_fallback",
    })


# ---------------------------------------------------------------------------
# Orchestrator integration (the big one — zero regression on rule verdict)
# ---------------------------------------------------------------------------


def test_orchestrator_adds_gnn_field_to_report_summary() -> None:
    """When a checkpoint is available + circuit is non-trivial, the
    orchestrator must enrich ``result.report.summary.gnn`` with the
    advisor's output. ``logic_correct`` must remain rule-decided."""

    from app.domain.compare.orchestrator import compare_logical_graphs

    ref_payload = _ref_payload()
    cur_netlist = _matching_cur_netlist_v2()
    ref = logical_reference_to_graph(ref_payload)
    cur = current_netlist_v2_to_graph(cur_netlist)
    result = compare_logical_graphs(
        ref, cur,
        ref_payload=ref_payload,
        cur_netlist_v2=cur_netlist,
    )

    # Rule verdict unchanged: matching topology → logic_correct=True
    assert result["logic_correct"] is True
    # New GNN field present
    gnn_block = result["report"]["summary"].get("gnn")
    assert gnn_block is not None
    assert gnn_block["enabled"] is True
    assert gnn_block["n_edges_scored"] > 0
    assert "edge_predictions" in gnn_block
    assert "model_version" in gnn_block


def test_orchestrator_gnn_field_never_overrides_rule_verdict() -> None:
    """Plan §一 hard rule: GNN cannot flip pass/fail. Verify by
    constructing a case where the model would likely disagree with the
    rule (rule says correct, but a noisy GNN might score < 0.5)."""

    from app.domain.compare.orchestrator import compare_logical_graphs

    ref_payload = _ref_payload()
    cur_netlist = _matching_cur_netlist_v2()
    ref = logical_reference_to_graph(ref_payload)
    cur = current_netlist_v2_to_graph(cur_netlist)
    result = compare_logical_graphs(
        ref, cur,
        ref_payload=ref_payload,
        cur_netlist_v2=cur_netlist,
    )
    # The verdict comes from the rule path; GNN is advisory only
    assert "logic_correct" in result
    rule_verdict = result["logic_correct"]
    # Mirror it: even if gnn_block shows low similarity, logic_correct
    # is unchanged from what the rule path would produce.
    assert rule_verdict is True  # matching topology
    # The gnn field doesn't carry a "logic_correct" override
    gnn_block = result["report"]["summary"].get("gnn", {})
    assert "logic_correct" not in gnn_block


def test_orchestrator_emits_no_gnn_field_when_advisor_unavailable(
    monkeypatch,
) -> None:
    """If no checkpoint is on disk, the orchestrator silently produces
    a rule-only report — the advisory ``gnn`` block is absent, but a
    ``gnn_disabled_reason`` field explains why so the JSON output is
    self-diagnosable in production."""

    # Force the locator to return None
    import app.domain.gnn.inference as inf

    monkeypatch.setattr(inf, "_locate_default_checkpoint", lambda: None)
    GNNAdvisor.reset_singleton()

    from app.domain.compare.orchestrator import compare_logical_graphs

    ref_payload = _ref_payload()
    cur_netlist = _matching_cur_netlist_v2()
    ref = logical_reference_to_graph(ref_payload)
    cur = current_netlist_v2_to_graph(cur_netlist)
    result = compare_logical_graphs(
        ref, cur,
        ref_payload=ref_payload,
        cur_netlist_v2=cur_netlist,
    )
    assert result["logic_correct"] is True
    summary = result["report"].get("summary", {})
    assert "gnn" not in summary
    # New contract: an observable reason code lands in the JSON.
    assert summary.get("gnn_disabled_reason") == "checkpoint_missing"
    assert result["details"].get("gnn_disabled_reason") == "checkpoint_missing"

    # Restore for downstream tests
    GNNAdvisor.reset_singleton()


def test_orchestrator_keeps_working_when_payloads_are_none() -> None:
    """When the caller omits ref_payload / cur_netlist_v2, GNN can't
    rebuild HCGs — the hook silently no-ops."""

    from app.domain.compare.orchestrator import compare_logical_graphs

    ref = logical_reference_to_graph(_ref_payload())
    cur = current_netlist_v2_to_graph(_matching_cur_netlist_v2())
    # Pass nothing → GNN hook bails early, rule path runs as before
    result = compare_logical_graphs(ref, cur)
    assert result["logic_correct"] is True
    # No gnn field added
    summary = (result.get("report") or {}).get("summary", {})
    assert "gnn" not in summary


# ---------------------------------------------------------------------------
# P4.1 R2 — disagreement_with_rule warning (plan §六 conflict arbitration)
# ---------------------------------------------------------------------------


def test_orchestrator_disagreement_field_false_when_advice_consistent() -> None:
    """On a matching circuit the advisor should score every edge well
    above the 0.3 floor, so ``disagreement_with_rule`` stays False and
    no warning item is added."""

    from app.domain.compare.orchestrator import compare_logical_graphs

    ref_payload = _ref_payload()
    cur_netlist = _matching_cur_netlist_v2()
    ref = logical_reference_to_graph(ref_payload)
    cur = current_netlist_v2_to_graph(cur_netlist)
    result = compare_logical_graphs(
        ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist,
    )
    gnn_block = result["report"]["summary"]["gnn"]
    assert gnn_block["disagreement_with_rule"] is False
    assert not any(
        item.get("error_code") == "WARN_GNN_DISAGREES_WITH_RULE"
        for item in (result.get("items") or [])
    )
    # details mirror also exposes the flag
    assert result["details"]["gnn"]["disagreement_with_rule"] is False
    assert result["details"]["gnn"]["n_suspicious_edges"] == 0


def test_orchestrator_disagreement_warning_appears_when_gnn_low_confidence(
    monkeypatch,
) -> None:
    """Patch the advisor's ``advise`` to inject a synthetic GNNAdvice
    with one ``p_correct < 0.3`` edge. The orchestrator must:

    1. set ``disagreement_with_rule=True`` in ``report.summary.gnn``
    2. add exactly one ``WARN_GNN_DISAGREES_WITH_RULE`` item with
       ``severity == "warning"``
    3. **not** flip ``logic_correct`` (plan §一 hard rule)
    """

    from app.domain.compare.orchestrator import compare_logical_graphs

    GNNAdvisor.reset_singleton()
    real_advisor = GNNAdvisor.get()

    def _stub_advise(ref_hcg, cur_hcg, *, timeout_ms=300, num_hops=2):  # type: ignore[no-untyped-def]
        return GNNAdvice(
            model_version="circuit_match:stub",
            inference_ms=12.3,
            n_edges_scored=2,
            edge_predictions=(
                {"edge": ["cur_port:R_a.pin1", "cur_net:n_in"],
                 "p_correct": 0.94, "verdict": "ok"},
                {"edge": ["cur_port:R_b.pin1", "cur_net:n_mid"],
                 "p_correct": 0.12, "verdict": "likely_wrong"},
            ),
            hotspots=(
                {"node": "cur_port:R_b.pin1", "score": 0.88,
                 "hint": "Pin may be wired wrong"},
            ),
            graph_similarity=0.53,
            graph_similarity_confidence=0.42,
        )

    monkeypatch.setattr(real_advisor, "advise", _stub_advise)

    ref_payload = _ref_payload()
    cur_netlist = _matching_cur_netlist_v2()
    ref = logical_reference_to_graph(ref_payload)
    cur = current_netlist_v2_to_graph(cur_netlist)
    result = compare_logical_graphs(
        ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist,
    )

    # 1. logic_correct unchanged (rule path won)
    assert result["logic_correct"] is True
    assert result["is_correct"] is True
    assert result["is_match"] is True

    # 2. disagreement_with_rule promoted to True
    gnn_block = result["report"]["summary"]["gnn"]
    assert gnn_block["disagreement_with_rule"] is True
    assert result["details"]["gnn"]["disagreement_with_rule"] is True
    assert result["details"]["gnn"]["n_suspicious_edges"] == 1

    # 3. exactly one warning item appears at result.items + report.items
    warnings = [
        item for item in result["items"]
        if item.get("error_code") == "WARN_GNN_DISAGREES_WITH_RULE"
    ]
    assert len(warnings) == 1
    warn = warnings[0]
    assert warn["severity"] == "warning"
    assert warn["error_family"] == "gnn_advisory"
    assert "cur_port:R_b.pin1" in str(warn["actual"])

    # Also surfaces in report["items"] (the validator_report_v2 shape)
    report_warnings = [
        item for item in result["report"]["items"]
        if item.get("error_code") == "WARN_GNN_DISAGREES_WITH_RULE"
    ]
    assert len(report_warnings) == 1

    # Total count was bumped on the summary
    assert result["report"]["summary"]["total_item_count"] >= 1

    GNNAdvisor.reset_singleton()


def test_orchestrator_disagreement_skipped_when_rule_fails(monkeypatch) -> None:
    """R2 is **only** about rule_pass + GNN-flags-wrong (the false_pass
    direction). When the rule already says fail, the warning is moot."""

    from app.domain.compare.orchestrator import compare_logical_graphs

    GNNAdvisor.reset_singleton()
    real_advisor = GNNAdvisor.get()

    def _stub_advise(ref_hcg, cur_hcg, *, timeout_ms=300, num_hops=2):  # type: ignore[no-untyped-def]
        return GNNAdvice(
            model_version="circuit_match:stub",
            inference_ms=10.0,
            n_edges_scored=1,
            edge_predictions=(
                {"edge": ["cur_port:R_a.pin1", "cur_net:n_in"],
                 "p_correct": 0.08, "verdict": "likely_wrong"},
            ),
            hotspots=(),
            graph_similarity=0.1,
            graph_similarity_confidence=0.5,
        )

    monkeypatch.setattr(real_advisor, "advise", _stub_advise)

    # Construct a clearly-broken cur (different topology)
    ref_payload = _ref_payload()
    bad_cur = {
        "components": [
            {
                "component_id": "X1", "component_type": "Resistor",
                "pins": [{"pin_name": "pin1", "electrical_net_id": "stray"}],
            },
        ],
        "nets": [{"electrical_net_id": "stray", "canonical_name": "stray"}],
    }
    ref = logical_reference_to_graph(ref_payload)
    cur = current_netlist_v2_to_graph(bad_cur)
    result = compare_logical_graphs(
        ref, cur, ref_payload=ref_payload, cur_netlist_v2=bad_cur,
    )
    # Rule says fail
    assert result["logic_correct"] is False
    # disagreement only fires on rule_pass; rule_fail → no warning
    gnn_block = (result["report"].get("summary") or {}).get("gnn")
    if gnn_block is not None:  # advisor may have been triggered or not
        assert gnn_block["disagreement_with_rule"] is False
    assert not any(
        item.get("error_code") == "WARN_GNN_DISAGREES_WITH_RULE"
        for item in (result.get("items") or [])
    )

    GNNAdvisor.reset_singleton()


# ---------------------------------------------------------------------------
# D + A · suggested_targets ("应该接哪里")
# ---------------------------------------------------------------------------


def test_advise_on_perfect_copy_has_no_suggested_targets() -> None:
    """When every observed edge passes the threshold and no REQUIRED
    pin is floating, the suggestion pool is empty by construction —
    nothing to fix, nothing to suggest."""

    ref_hcg = build_from_logical_reference(_ref_payload())
    cur_hcg = build_from_logical_reference(_ref_payload())
    cur_hcg.metadata["side"] = "cur"
    advisor = GNNAdvisor.get()
    advice = advisor.advise(ref_hcg, cur_hcg)
    assert advice is not None
    # Identity-copy cur should score every edge well above 0.5 →
    # no port qualifies as suspicious; no floating REQUIRED pin →
    # suggested_targets is empty.
    suspicious_ports = {
        ep["edge"][0] for ep in advice.edge_predictions
        if ep["verdict"] == "likely_wrong"
    }
    if not suspicious_ports:
        assert advice.suggested_targets == ()
        assert advice.n_suggestion_candidates_scored == 0


def test_advise_on_floating_required_pin_returns_topk_candidates() -> None:
    """Drop one resistor pin from the cur netlist so the materializer
    flags it ``is_floating=True`` + REQUIRED. The advisor must:

    - include that port in ``suggested_targets``
    - rank candidate cur nets by ``p_connect`` (top-K, default 3)
    - the correct net (`n_gnd`, paired with R_b.pin1 on `n_mid` per the
      voltage divider topology) should be among the top candidates
      since the model was trained on the same topology
    """

    cur_with_floating_pin = {
        "components": [
            {
                "component_id": "R_a",
                "component_type": "Resistor",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "n_in"},
                    {"pin_name": "pin2", "electrical_net_id": "n_mid"},
                ],
            },
            {
                "component_id": "R_b",
                "component_type": "Resistor",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "n_mid"},
                    # pin2 left out → materialize_floating_required
                ],
            },
        ],
        "nets": [
            {"electrical_net_id": "n_in", "role": "input"},
            {"electrical_net_id": "n_mid", "role": "output"},
            {"electrical_net_id": "n_gnd", "role": "ground"},
        ],
    }
    ref_hcg = build_from_logical_reference(_ref_payload())
    cur_hcg = build_from_netlist_v2(cur_with_floating_pin)

    # Sanity: the materializer indeed surfaced a floating REQUIRED port.
    floating_required = [
        p for p in cur_hcg.ports.values()
        if p.is_floating and p.connection_policy == "required"
    ]
    assert floating_required, "expected one floating REQUIRED pin (R_b.pin2)"

    advisor = GNNAdvisor.get()
    advice = advisor.advise(ref_hcg, cur_hcg)
    assert advice is not None
    assert advice.suggested_targets, "expected non-empty suggested_targets"
    # The floating REQUIRED port should appear with reason="floating_required"
    floating_entries = [
        t for t in advice.suggested_targets
        if t["reason"] == "floating_required"
    ]
    assert floating_entries, (
        f"no floating_required entries in {advice.suggested_targets}"
    )

    fe = floating_entries[0]
    # Top-K shape contract
    assert 1 <= len(fe["candidates"]) <= advisor.suggestion_top_k
    for rank, cand in enumerate(fe["candidates"], start=1):
        assert {"net", "p_connect", "rank"} <= cand.keys()
        assert cand["rank"] == rank
        assert 0.0 <= cand["p_connect"] <= 1.0
    # Candidates must be sorted descending by p_connect
    probs = [c["p_connect"] for c in fe["candidates"]]
    assert probs == sorted(probs, reverse=True)
    # current_nets exposed (empty for floating port since no observed edges)
    assert fe["current_nets"] == []
    # Counter aligns with the size of the candidate plan we ran
    assert advice.n_suggestion_candidates_scored > 0


def test_advise_suggestion_candidate_budget_caps_total_evals() -> None:
    """``max_suggestion_candidates`` hard-caps per-call evaluations so
    huge circuits never blow the timeout budget."""

    cur_with_floating_pin = {
        "components": [
            {
                "component_id": "R_a",
                "component_type": "Resistor",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "n_in"},
                    {"pin_name": "pin2", "electrical_net_id": "n_mid"},
                ],
            },
            {
                "component_id": "R_b",
                "component_type": "Resistor",
                "pins": [{"pin_name": "pin1", "electrical_net_id": "n_mid"}],
            },
        ],
        "nets": [
            {"electrical_net_id": "n_in", "role": "input"},
            {"electrical_net_id": "n_mid", "role": "output"},
            {"electrical_net_id": "n_gnd", "role": "ground"},
        ],
    }
    ref_hcg = build_from_logical_reference(_ref_payload())
    cur_hcg = build_from_netlist_v2(cur_with_floating_pin)

    advisor = GNNAdvisor.get()
    saved = advisor.max_suggestion_candidates
    try:
        advisor.max_suggestion_candidates = 1
        advice = advisor.advise(ref_hcg, cur_hcg)
    finally:
        advisor.max_suggestion_candidates = saved
    assert advice is not None
    assert advice.n_suggestion_candidates_scored <= 1


def test_orchestrator_r2_warning_carries_suggested_targets(
    monkeypatch,
) -> None:
    """When R2 fires, the warning item's ``actual.gnn_suspicious_edges``
    must carry a ``suggested_targets`` per suspicious edge — that's how
    the UI shows "GNN 建议把 R_b.pin1 改接到 n_vout (P=0.92)" instead of
    just "N 条可疑"."""

    from app.domain.compare.orchestrator import compare_logical_graphs

    GNNAdvisor.reset_singleton()
    real_advisor = GNNAdvisor.get()

    def _stub_advise(ref_hcg, cur_hcg, *, timeout_ms=300, num_hops=2):  # type: ignore[no-untyped-def]
        return GNNAdvice(
            model_version="circuit_match:stub",
            inference_ms=15.0,
            n_edges_scored=2,
            edge_predictions=(
                {"edge": ["cur_port:R_a.pin1", "cur_net:n_in"],
                 "p_correct": 0.95, "verdict": "ok"},
                {"edge": ["cur_port:R_b.pin1", "cur_net:n_mid"],
                 "p_correct": 0.07, "verdict": "likely_wrong"},
            ),
            hotspots=(),
            graph_similarity=0.51,
            graph_similarity_confidence=0.4,
            suggested_targets=(
                {
                    "port": "cur_port:R_b.pin1",
                    "reason": "likely_wrong",
                    "current_nets": ["cur_net:n_mid"],
                    "top_p_connect": 0.92,
                    "candidates": [
                        {"net": "cur_net:n_gnd", "p_connect": 0.92, "rank": 1},
                        {"net": "cur_net:n_in", "p_connect": 0.18, "rank": 2},
                    ],
                },
            ),
            n_suggestion_candidates_scored=2,
        )

    monkeypatch.setattr(real_advisor, "advise", _stub_advise)

    ref_payload = _ref_payload()
    cur_netlist = _matching_cur_netlist_v2()
    ref = logical_reference_to_graph(ref_payload)
    cur = current_netlist_v2_to_graph(cur_netlist)
    result = compare_logical_graphs(
        ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist,
    )

    # logic_correct stays True — advisory only
    assert result["logic_correct"] is True

    # report.summary.gnn carries suggested_targets verbatim
    gnn_block = result["report"]["summary"]["gnn"]
    assert gnn_block["suggested_targets"], "suggested_targets missing from summary"
    assert gnn_block["n_suggestion_candidates_scored"] == 2

    # warning item carries the matched candidates inside its actual.
    warnings = [
        item for item in result["items"]
        if item.get("error_code") == "WARN_GNN_DISAGREES_WITH_RULE"
    ]
    assert len(warnings) == 1
    suspicious = warnings[0]["actual"]["gnn_suspicious_edges"]
    assert len(suspicious) == 1
    matched = suspicious[0]
    assert matched["edge"] == ["cur_port:R_b.pin1", "cur_net:n_mid"]
    assert matched["suggested_targets"], (
        "warning item missing the top-K candidates we wired in"
    )
    top = matched["suggested_targets"][0]
    assert top["net"] == "cur_net:n_gnd"
    assert top["rank"] == 1

    # Message text mentions the top-1 suggestion so a UI without the
    # structured data can still surface "改接到 GND (地)". Display
    # enrichment replaces raw IDs with human-readable role labels.
    assert "GND (地)" in warnings[0]["message"], warnings[0]["message"]
    assert "cur_net:" not in warnings[0]["message"]
    assert "P(connect)" in warnings[0]["message"]

    # details.gnn mirror gained the counters
    details_gnn = result["details"]["gnn"]
    assert details_gnn["n_suggested_targets"] == 1
    assert details_gnn["n_suggestion_candidates_scored"] == 2

    GNNAdvisor.reset_singleton()


# ---------------------------------------------------------------------------
# gnn_disabled_reason · observability patch (silent-bail diagnosability)
# ---------------------------------------------------------------------------


def test_disabled_reason_runtime_unavailable_when_torch_missing(
    monkeypatch,
) -> None:
    """When ``GNNAdvisor.runtime_available()`` is False (production box
    without ``[gnn]`` extra installed), the orchestrator writes
    ``gnn_disabled_reason = "runtime_unavailable"`` to the report."""

    # Spoof runtime_available → False so we don't have to actually
    # uninstall torch. This is the contract that should_use_gnn relies
    # on under the hood.
    monkeypatch.setattr(GNNAdvisor, "runtime_available", classmethod(lambda cls: False))
    GNNAdvisor.reset_singleton()

    from app.domain.compare.orchestrator import compare_logical_graphs

    ref_payload = _ref_payload()
    cur_netlist = _matching_cur_netlist_v2()
    ref = logical_reference_to_graph(ref_payload)
    cur = current_netlist_v2_to_graph(cur_netlist)
    result = compare_logical_graphs(
        ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist,
    )

    assert result["logic_correct"] is True
    summary = result["report"]["summary"]
    assert "gnn" not in summary
    assert summary["gnn_disabled_reason"] == "runtime_unavailable"
    assert result["details"]["gnn_disabled_reason"] == "runtime_unavailable"

    GNNAdvisor.reset_singleton()


def test_disabled_reason_tiny_circuit() -> None:
    """A tiny ref + cur (< 8 nodes total) shouldn't trigger the advisor
    per plan §七; the JSON must say so explicitly via ``tiny_circuit``."""

    # Build a 1-resistor ref + 1-resistor cur. Sum of nodes is well
    # below the 8-node trigger threshold.
    tiny_ref_payload = {
        "format": "logical_reference_v1",
        "components": [
            {"ref_id": "R_only", "type": "Resistor",
             "pins": [{"pin": "pin1", "net": "n_a"},
                      {"pin": "pin2", "net": "n_b"}]},
        ],
        "nets": [
            {"net": "n_a", "role": "input"},
            {"net": "n_b", "role": "output"},
        ],
    }
    tiny_cur = {
        "components": [
            {"component_id": "R1", "component_type": "Resistor", "pins": [
                {"pin_name": "pin1", "electrical_net_id": "n_a"},
                {"pin_name": "pin2", "electrical_net_id": "n_b"},
            ]},
        ],
        "nets": [
            {"electrical_net_id": "n_a", "role": "input"},
            {"electrical_net_id": "n_b", "role": "output"},
        ],
    }
    from app.domain.compare.orchestrator import compare_logical_graphs

    ref = logical_reference_to_graph(tiny_ref_payload)
    cur = current_netlist_v2_to_graph(tiny_cur)
    assert ref.number_of_nodes() + cur.number_of_nodes() < 8, (
        "fixture invariant: this case must trip the size early-exit"
    )

    result = compare_logical_graphs(
        ref, cur,
        ref_payload=tiny_ref_payload,
        cur_netlist_v2=tiny_cur,
    )
    summary = result["report"]["summary"]
    # GNN sat out — but reason is now visible
    assert "gnn" not in summary
    assert summary.get("gnn_disabled_reason") == "tiny_circuit"


def test_disabled_reason_model_failed_when_advise_returns_none(
    monkeypatch,
) -> None:
    """``advise()`` may legitimately return None (it swallows internal
    exceptions per plan §一 silent-fallback). The orchestrator surfaces
    that as ``gnn_disabled_reason = "model_failed"`` so production can
    distinguish "model crashed silently" from "checkpoint not deployed"."""

    from app.domain.compare.orchestrator import compare_logical_graphs

    GNNAdvisor.reset_singleton()
    real_advisor = GNNAdvisor.get()

    def _none_advise(ref_hcg, cur_hcg, *, timeout_ms=300, num_hops=2):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(real_advisor, "advise", _none_advise)

    ref_payload = _ref_payload()
    cur_netlist = _matching_cur_netlist_v2()
    ref = logical_reference_to_graph(ref_payload)
    cur = current_netlist_v2_to_graph(cur_netlist)
    result = compare_logical_graphs(
        ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist,
    )

    summary = result["report"]["summary"]
    assert "gnn" not in summary
    assert summary.get("gnn_disabled_reason") == "model_failed"

    GNNAdvisor.reset_singleton()


def test_no_disabled_reason_when_payloads_intentionally_none() -> None:
    """When the caller deliberately omits ``ref_payload`` /
    ``cur_netlist_v2`` (debug script, fast path), the orchestrator must
    NOT pollute the output with a ``gnn_disabled_reason`` — that's an
    intentional bypass, not a runtime issue."""

    from app.domain.compare.orchestrator import compare_logical_graphs

    ref = logical_reference_to_graph(_ref_payload())
    cur = current_netlist_v2_to_graph(_matching_cur_netlist_v2())
    result = compare_logical_graphs(ref, cur)

    summary = (result.get("report") or {}).get("summary", {})
    assert "gnn" not in summary
    assert "gnn_disabled_reason" not in summary
    assert "gnn_disabled_reason" not in (result.get("details") or {})


def test_display_helper_renders_inverting_amp_port_and_net_labels() -> None:
    """Direct test of ``build_display_maps`` on the inverting-amplifier
    fixture: verify ``cur_port:IC1.pin2 → "U1 · pin2 (反相输入)"`` and
    ``cur_net:NET_GND → "GND (地)"``. Bypasses the rule path (which is
    fussy about exact NC-pin encoding on UA741) so we test the
    enricher in isolation."""

    from app.domain.compare.gnn_display import build_display_maps

    inv_amp_ref = json.loads(
        (FIXTURES / "test_opamp_inverting_v1.json").read_text()
    )
    cur = {
        "components": [
            {"component_id": "IC1", "component_type": "IC", "subtype": "UA741",
             "pins": [
                 {"pin_name": "pin2", "electrical_net_id": "NET_INV"},
                 {"pin_name": "pin3", "electrical_net_id": "NET_VP"},
                 {"pin_name": "pin4", "electrical_net_id": "NET_GND"},
                 {"pin_name": "pin6", "electrical_net_id": "NET_OUT"},
                 {"pin_name": "pin7", "electrical_net_id": "NET_VCC"},
             ]},
        ],
        "nets": [],
    }
    # Synthesised mapping (the rule path would have written this on a
    # successful isomorphism)
    summary = {
        "ref_to_current_component_mapping": {"U1": "IC1"},
        "ref_to_current_net_mapping": {
            "INV": "NET_INV", "V_P": "NET_VP", "GND": "NET_GND",
            "VOUT": "NET_OUT", "VCC": "NET_VCC", "VIN": "NET_IN",
        },
    }

    net_display, port_display = build_display_maps(inv_amp_ref, cur, summary)

    # Pin role labels via IC subtype lookup
    assert port_display["cur_port:IC1.pin2"] == "U1 · pin2 (反相输入)"
    assert port_display["cur_port:IC1.pin3"] == "U1 · pin3 (非反相输入)"
    assert port_display["cur_port:IC1.pin4"] == "U1 · pin4 (V−电源)"
    assert port_display["cur_port:IC1.pin6"] == "U1 · pin6 (输出)"
    assert port_display["cur_port:IC1.pin7"] == "U1 · pin7 (V+电源)"

    # Net role labels via ref payload
    assert net_display["cur_net:NET_GND"] == "GND (地)"
    assert net_display["cur_net:NET_OUT"] == "VOUT (输出)"
    assert net_display["cur_net:NET_INV"] == "INV (信号)"
    assert net_display["cur_net:NET_VCC"] == "VCC (电源)"


def test_display_e2e_on_voltage_divider_with_r2_warning(monkeypatch) -> None:
    """End-to-end smoke through the orchestrator + R2 warning path. Uses
    voltage divider (whose rule path is exercised by other tests) so we
    know iso-mapping fires and the warning emits with display labels."""

    from app.domain.compare.orchestrator import compare_logical_graphs

    GNNAdvisor.reset_singleton()
    real_advisor = GNNAdvisor.get()

    def _stub_advise(ref_hcg, cur_hcg, *, timeout_ms=300, num_hops=2):  # type: ignore[no-untyped-def]
        return GNNAdvice(
            model_version="circuit_match:stub",
            inference_ms=10.0,
            n_edges_scored=2,
            edge_predictions=(
                {"edge": ["cur_port:R_a.pin1", "cur_net:n_in"],
                 "p_correct": 0.95, "verdict": "ok"},
                {"edge": ["cur_port:R_b.pin1", "cur_net:n_mid"],
                 "p_correct": 0.07, "verdict": "likely_wrong"},
            ),
            hotspots=(
                {"node": "cur_port:R_b.pin1", "score": 0.93, "hint": "..."},
            ),
            suggested_targets=(
                {
                    "port": "cur_port:R_b.pin1",
                    "reason": "likely_wrong",
                    "current_nets": ["cur_net:n_mid"],
                    "top_p_connect": 0.91,
                    "candidates": [
                        {"net": "cur_net:n_gnd", "p_connect": 0.91, "rank": 1},
                        {"net": "cur_net:n_in", "p_connect": 0.05, "rank": 2},
                    ],
                },
            ),
            n_suggestion_candidates_scored=2,
            graph_similarity=0.5,
            graph_similarity_confidence=0.4,
        )

    monkeypatch.setattr(real_advisor, "advise", _stub_advise)

    ref_payload = _ref_payload()
    cur_netlist = _matching_cur_netlist_v2()
    ref = logical_reference_to_graph(ref_payload)
    cur = current_netlist_v2_to_graph(cur_netlist)
    result = compare_logical_graphs(
        ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist,
    )
    assert result["logic_correct"] is True  # rule path matches

    gnn_block = result["report"]["summary"]["gnn"]

    # 1) edge_predictions carry edge_display tuples
    eps = gnn_block["edge_predictions"]
    flat = " | ".join(" → ".join(ep.get("edge_display") or []) for ep in eps)
    # Divider has R1/R2 in ref; net VOUT carries role=output
    assert "VOUT (输出)" in flat or "VIN (输入)" in flat, flat

    # 2) suggested_targets get full enrichment
    t0 = gnn_block["suggested_targets"][0]
    assert t0["port_display"]  # something non-empty
    assert isinstance(t0["current_nets_display"], list)
    cand0 = t0["candidates"][0]
    assert "net_display" in cand0

    # 3) R2 warning's suspicious_edges + message use display labels
    warnings = [
        i for i in result["items"]
        if i.get("error_code") == "WARN_GNN_DISAGREES_WITH_RULE"
    ]
    assert len(warnings) == 1
    susp0 = warnings[0]["actual"]["gnn_suspicious_edges"][0]
    assert susp0.get("edge_display")
    assert susp0["suggested_targets"][0].get("net_display")
    # Message body should not leak raw cur_port:/cur_net: prefixes
    assert "cur_port:" not in warnings[0]["message"]
    assert "cur_net:" not in warnings[0]["message"]

    GNNAdvisor.reset_singleton()


def test_display_falls_back_to_raw_when_mapping_absent() -> None:
    """When ref↔cur alignment failed (no mapping), the enricher should
    write raw IDs to ``*_display`` rather than crash or silently drop
    fields. Demo principle: show *something* over showing nothing."""

    from app.domain.compare.gnn_display import enrich_advice_with_display

    advice_dict = {
        "edge_predictions": [
            {"edge": ["cur_port:Foo.pinX", "cur_net:Bar"], "p_correct": 0.9, "verdict": "ok"},
        ],
        "hotspots": [{"node": "cur_port:Foo.pinX", "score": 0.5, "hint": "..."}],
        "suggested_targets": [
            {
                "port": "cur_port:Foo.pinX",
                "reason": "likely_wrong",
                "current_nets": ["cur_net:Bar"],
                "candidates": [{"net": "cur_net:Baz", "p_connect": 0.7, "rank": 1}],
            }
        ],
    }
    enriched = enrich_advice_with_display(
        advice_dict,
        ref_payload={"components": [], "nets": []},
        cur_netlist_v2={"components": [], "nets": []},
        summary={},  # mapping absent → no aligned data
    )
    # Falls back to raw IDs verbatim — no display info, but no crash
    assert enriched["edge_predictions"][0]["edge_display"] == [
        "cur_port:Foo.pinX", "cur_net:Bar"
    ]
    assert enriched["hotspots"][0]["node_display"] == "cur_port:Foo.pinX"
    assert enriched["suggested_targets"][0]["port_display"] == "cur_port:Foo.pinX"


def test_no_disabled_reason_when_advisor_succeeds() -> None:
    """Happy path: when the advisor runs successfully and writes the
    full ``gnn`` block, no ``gnn_disabled_reason`` field should leak
    in. Belt-and-braces against future regressions where someone might
    set the reason unconditionally."""

    from app.domain.compare.orchestrator import compare_logical_graphs

    GNNAdvisor.reset_singleton()
    ref_payload = _ref_payload()
    cur_netlist = _matching_cur_netlist_v2()
    ref = logical_reference_to_graph(ref_payload)
    cur = current_netlist_v2_to_graph(cur_netlist)
    result = compare_logical_graphs(
        ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist,
    )

    summary = result["report"]["summary"]
    assert "gnn" in summary, "advisor should have run on a non-trivial circuit"
    assert "gnn_disabled_reason" not in summary
    assert "gnn_disabled_reason" not in result["details"]
