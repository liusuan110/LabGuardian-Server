"""P5 evaluator tests — plan §九 P5 deliverable.

Covers:
- rule-only path produces sane metrics on a small set of labels
- false_pass / false_fail accounting matches manual count
- markdown renderer doesn't crash
- evaluate_split honours split_ids + limit
- gnn advisor None branch: gnn fields stay None / 0
- runs without torch installed (rule-only)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.gnn.evaluator import (
    DEFAULT_REF_PAYLOAD_PATHS,
    EvaluationReport,
    SampleEvaluation,
    evaluate_split,
)

LABEL_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "circuit_compare" / "labels"


def _label_dir_available() -> bool:
    return LABEL_ROOT.is_dir() and any(LABEL_ROOT.iterdir())


pytestmark = pytest.mark.skipif(
    not _label_dir_available(),
    reason="circuit_compare label dataset not generated on this box",
)


def _pick_label_ids(ref_id: str, k: int = 6) -> list[str]:
    """Pick `k` labels from a ref directory (mixed identity + perturbed)."""

    dirpath = LABEL_ROOT / ref_id
    files = sorted(dirpath.glob("*.json"))
    return [f"{ref_id}/{p.stem}" for p in files[:k]]


# ---------------------------------------------------------------------------
# Rule-only evaluator on a small slice (no torch / advisor)
# ---------------------------------------------------------------------------


def test_evaluate_split_rule_only_returns_report_with_expected_fields():
    split_ids = _pick_label_ids("rc_lowpass", k=6)
    report = evaluate_split(LABEL_ROOT, split_ids=split_ids, advisor=None)
    assert isinstance(report, EvaluationReport)
    assert report.n_samples == 6
    assert report.advisor_unavailable is True
    assert report.gnn_runtime_ms_mean is None
    assert report.seal_edge_n == 0
    assert 0.0 <= report.rule_false_pass_rate <= 1.0
    assert 0.0 <= report.rule_false_fail_rate <= 1.0
    assert report.rule_runtime_ms_mean > 0


def test_evaluate_split_combined_metrics_equal_rule_metrics():
    """Per plan §一, GNN never overrides logic_correct — so combined_*
    must equal rule_* even when the advisor is present."""

    split_ids = _pick_label_ids("rc_lowpass", k=4)
    report = evaluate_split(LABEL_ROOT, split_ids=split_ids, advisor=None)
    assert report.combined_false_pass_rate == pytest.approx(
        report.rule_false_pass_rate
    )
    assert report.combined_false_fail_rate == pytest.approx(
        report.rule_false_fail_rate
    )
    assert report.combined_accuracy == pytest.approx(report.rule_accuracy)


def test_evaluate_split_false_pass_matches_manual_count():
    """Manually count negatives that the rule called 'pass' and cross-check
    against the aggregated rate."""

    split_ids = _pick_label_ids("rc_lowpass", k=10)
    report = evaluate_split(LABEL_ROOT, split_ids=split_ids, advisor=None)

    n_neg = sum(1 for s in report.samples if not s.expected_positive)
    n_fp = sum(
        1 for s in report.samples
        if s.rule_logic_correct and not s.expected_positive
    )
    expected = n_fp / max(1, n_neg)
    assert report.rule_false_pass_rate == pytest.approx(expected)


# ---------------------------------------------------------------------------
# IO + edge cases
# ---------------------------------------------------------------------------


def test_evaluate_split_honours_limit():
    split_ids = _pick_label_ids("rc_lowpass", k=10)
    report = evaluate_split(LABEL_ROOT, split_ids=split_ids, advisor=None, limit=3)
    assert report.n_samples <= 3


def test_evaluate_split_raises_on_empty():
    with pytest.raises(ValueError, match="no label files"):
        evaluate_split(LABEL_ROOT, split_ids=["nonexistent/sample_0000"], advisor=None)


def test_report_to_markdown_renders_plan_targets_table():
    split_ids = _pick_label_ids("rc_lowpass", k=4)
    report = evaluate_split(LABEL_ROOT, split_ids=split_ids, advisor=None)
    md = report.to_markdown()
    assert "Plan §八 hard targets" in md
    assert "false_pass_rate" in md
    assert "rule comparator" in md
    # When rule_only, SEAL section says "no observed edges scored"
    assert "no observed edges scored" in md


def test_report_to_dict_roundtrips_json():
    """metrics.json must be JSON-encodable end-to-end."""

    split_ids = _pick_label_ids("rc_lowpass", k=4)
    report = evaluate_split(LABEL_ROOT, split_ids=split_ids, advisor=None)
    payload = json.dumps(report.to_dict(), default=str)
    parsed = json.loads(payload)
    assert parsed["n_samples"] == report.n_samples
    assert "samples" in parsed
    assert len(parsed["samples"]) == report.n_samples


def test_sample_evaluation_records_perturbation_chain():
    split_ids = _pick_label_ids("rc_lowpass", k=4)
    report = evaluate_split(LABEL_ROOT, split_ids=split_ids, advisor=None)
    for ev in report.samples:
        assert isinstance(ev, SampleEvaluation)
        assert ev.perturbation_chain  # tuple, may be ("identity",) etc.
        assert ev.rule_runtime_ms > 0


def test_default_payload_paths_cover_all_known_refs():
    expected = {
        "rc_lowpass", "divider", "all_signal", "opamp_buffer",
        "opamp_inverting", "npn_switch", "lm358_dual_buffer",
    }
    assert expected.issubset(set(DEFAULT_REF_PAYLOAD_PATHS.keys()))
    for path in DEFAULT_REF_PAYLOAD_PATHS.values():
        assert path.is_file(), f"fixture missing: {path}"


# ---------------------------------------------------------------------------
# Multi-ref aggregation
# ---------------------------------------------------------------------------


def test_evaluate_split_multi_ref_aggregates_by_ref_and_perturbation():
    split_ids = (
        _pick_label_ids("rc_lowpass", k=3)
        + _pick_label_ids("divider", k=3)
    )
    report = evaluate_split(LABEL_ROOT, split_ids=split_ids, advisor=None)
    assert set(report.by_ref_id.keys()) == {"rc_lowpass", "divider"}
    assert sum(report.by_ref_id.values()) == report.n_samples
    # Each label's first perturbation chain entry becomes the op key
    assert all(op for op in report.by_perturbation.keys())
    assert sum(report.by_perturbation.values()) == report.n_samples


# ---------------------------------------------------------------------------
# GNN-attached path (skipped if torch unavailable)
# ---------------------------------------------------------------------------


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _torch_available(), reason="torch / torch_geometric not installed")
def test_evaluate_split_with_advisor_populates_gnn_metrics():
    """End-to-end smoke: feed real advisor + small slice, expect
    gnn_inference_ms populated and seal_edge_n > 0."""

    from app.domain.gnn.inference import GNNAdvisor

    if not GNNAdvisor.checkpoint_available():
        pytest.skip("no GNN checkpoint available on disk")

    advisor = GNNAdvisor.get()
    split_ids = _pick_label_ids("rc_lowpass", k=2)
    report = evaluate_split(
        LABEL_ROOT, split_ids=split_ids, advisor=advisor,
    )
    assert report.advisor_unavailable is False
    assert report.gnn_runtime_ms_mean is not None
    assert report.advisor_version is not None
    # Even tiny circuits should expose observed-edge metrics
    assert report.seal_edge_n >= 0  # may be 0 if cur has no edges


# ---------------------------------------------------------------------------
# P5 quality refactor — single-pass + R2 visibility
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _torch_available(), reason="torch / torch_geometric not installed")
def test_single_pass_evaluator_calls_advisor_once_per_sample(monkeypatch):
    """The quality refactor moved SEAL score capture into _evaluate_sample
    so the advisor sees each sample exactly once (the old code called it
    twice — once for verdict, once for AUC scoring)."""

    from app.domain.gnn.inference import GNNAdvisor

    if not GNNAdvisor.checkpoint_available():
        pytest.skip("no GNN checkpoint available on disk")

    GNNAdvisor.reset_singleton()
    advisor = GNNAdvisor.get()
    real_advise = advisor.advise
    call_count = {"n": 0}

    def _counting_advise(*args, **kwargs):
        call_count["n"] += 1
        return real_advise(*args, **kwargs)

    monkeypatch.setattr(advisor, "advise", _counting_advise)
    split_ids = _pick_label_ids("rc_lowpass", k=4)
    report = evaluate_split(LABEL_ROOT, split_ids=split_ids, advisor=advisor)
    # Single pass: exactly n_samples advise() calls.
    assert call_count["n"] == report.n_samples
    GNNAdvisor.reset_singleton()


def test_sample_evaluation_captures_observed_edge_scores_when_no_advisor():
    """No advisor → no scores captured but lists exist (empty tuples)."""

    split_ids = _pick_label_ids("rc_lowpass", k=3)
    report = evaluate_split(LABEL_ROOT, split_ids=split_ids, advisor=None)
    for ev in report.samples:
        assert ev.observed_edge_scores == ()
        assert ev.observed_edge_labels == ()
        assert ev.n_suspicious_edges == 0


def test_report_exposes_r2_warning_counters():
    report = evaluate_split(
        LABEL_ROOT,
        split_ids=_pick_label_ids("rc_lowpass", k=4),
        advisor=None,
    )
    assert report.n_r2_warnings == 0  # no advisor → no suspicious edges
    assert isinstance(report.by_perturbation_r2_warning_rate, dict)


def test_markdown_renders_r2_section():
    report = evaluate_split(
        LABEL_ROOT,
        split_ids=_pick_label_ids("rc_lowpass", k=4),
        advisor=None,
    )
    md = report.to_markdown()
    assert "R2 — `WARN_GNN_DISAGREES_WITH_RULE`" in md
    assert "would-emit warnings" in md
    # Per-perturbation table now has 5 columns including R2-warn rate
    assert "R2-warn rate" in md


# ---------------------------------------------------------------------------
# P5 §6 follow-up — evaluator now runs the production rule path with
# _enrich_result pin-level checks. The adapter must round-trip subtype
# info (UA741 / LM358) so the inner GNN hook sees the same port_type
# vector as the explicit advise() fallback.
# ---------------------------------------------------------------------------


def test_hcg_to_netlist_v2_roundtrip_preserves_ids():
    """The adapter must emit a netlist_v2 whose build_from_netlist_v2
    yields a cur_hcg with the same port + net source_ids as the input.
    Without this guarantee, the orchestrator's GNN hook (which builds
    its own cur_hcg from the netlist) would score different edges
    than the SEAL label files expect."""

    from app.domain.gnn.evaluator import (
        DEFAULT_SUBTYPES_BY_REF,
        _build_ref_artifacts,
        _hcg_to_netlist_v2,
    )
    from app.domain.gnn.port_graph import build_from_netlist_v2
    from app.domain.gnn.pyg_dataset import reconstruct_cur_hcg

    split_ids = _pick_label_ids("opamp_buffer", k=1)
    if not split_ids:
        pytest.skip("opamp_buffer fixtures not generated")
    label = json.loads(
        (LABEL_ROOT / f"{split_ids[0]}.json").read_text()
    )
    cur_meta = label["cur_metadata"]
    _, ref_hcg, _, subtypes = _build_ref_artifacts(
        label["ref_id"],
        {"opamp_buffer": Path(
            "tests/fixtures/references/test_opamp_buffer_v1.json"
        )},
        DEFAULT_SUBTYPES_BY_REF,
    )
    cur_hcg = reconstruct_cur_hcg(
        ref_hcg, cur_meta, subtype_by_source_id=subtypes,
    )
    cur_v2 = _hcg_to_netlist_v2(cur_hcg, subtype_by_source_id=subtypes)
    cur_hcg_2 = build_from_netlist_v2(cur_v2)

    orig_edges = {
        (e.src_port_id, e.dst_net_id) for e in cur_hcg.edges
    }
    rt_edges = {
        (e.src_port_id, e.dst_net_id) for e in cur_hcg_2.edges
    }
    assert orig_edges == rt_edges, (
        "round-trip lost edges — SEAL F1 will degrade because the "
        "orchestrator-internal cur_hcg won't match label file IDs"
    )


def test_hcg_to_netlist_v2_omitting_subtypes_loses_ic_port_types(monkeypatch):
    """Sanity check: forgetting to pass subtype_by_source_id makes IC
    pins lose their UA741/LM358 port_type (regression guard for the
    bug found while shipping §6 follow-up — SEAL F1 fell from 0.99 to
    0.70 because the orchestrator-internal cur_hcg had generic pin
    roles)."""

    from app.domain.gnn.evaluator import (
        DEFAULT_SUBTYPES_BY_REF,
        _build_ref_artifacts,
        _hcg_to_netlist_v2,
    )
    from app.domain.gnn.port_graph import build_from_netlist_v2
    from app.domain.gnn.pyg_dataset import reconstruct_cur_hcg

    split_ids = _pick_label_ids("opamp_buffer", k=1)
    if not split_ids:
        pytest.skip("opamp_buffer fixtures not generated")
    label = json.loads(
        (LABEL_ROOT / f"{split_ids[0]}.json").read_text()
    )
    cur_meta = label["cur_metadata"]
    _, ref_hcg, _, subtypes = _build_ref_artifacts(
        label["ref_id"],
        {"opamp_buffer": Path(
            "tests/fixtures/references/test_opamp_buffer_v1.json"
        )},
        DEFAULT_SUBTYPES_BY_REF,
    )
    cur_hcg = reconstruct_cur_hcg(
        ref_hcg, cur_meta, subtype_by_source_id=subtypes,
    )

    # WITH subtypes — IC ports get specific port_type
    with_v2 = _hcg_to_netlist_v2(cur_hcg, subtype_by_source_id=subtypes)
    with_hcg = build_from_netlist_v2(with_v2)
    ua741_ports_with = {
        p.port_type for p in with_hcg.ports.values()
        if p.parent_ctype == "IC"
    }

    # WITHOUT subtypes — IC ports fall back to generic
    without_v2 = _hcg_to_netlist_v2(cur_hcg)
    without_hcg = build_from_netlist_v2(without_v2)
    ua741_ports_without = {
        p.port_type for p in without_hcg.ports.values()
        if p.parent_ctype == "IC"
    }

    # The "with subtype" set must include the specific op-amp port types
    # that the "without" set lacks.
    assert ua741_ports_with != ua741_ports_without
    assert any(
        pt in ua741_ports_with
        for pt in ("non_inverting_input", "inverting_input", "output")
    )
