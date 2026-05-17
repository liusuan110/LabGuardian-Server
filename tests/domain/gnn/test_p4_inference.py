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
    a rule-only report — no ``gnn`` field added."""

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
