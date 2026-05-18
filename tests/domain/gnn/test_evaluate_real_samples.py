"""``evaluate_real_samples`` end-to-end on the simulated-real corpus
(plan §十 R6 Phase 3).

The committed fixtures under ``tests/fixtures/real_student_simulated/``
cover three patterns:
- positive (identity) — must pass
- positive after pin-swap on symmetric passive — must pass
- positive after vision component-id rename — must pass
- negative (wire bridge VIN→GND) — must fail
- negative (missing R2) — must fail

This test is the regression guard against Phase 3 plumbing drift —
when real student data lands, this same scaffold scores it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.gnn.evaluator import EvaluationReport, evaluate_real_samples

REAL_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "fixtures" / "real_student_simulated"
)


def test_evaluate_real_samples_runs_end_to_end_rule_only():
    """Rule path on the 5 simulated samples must score 100%."""

    report = evaluate_real_samples(REAL_FIXTURE_ROOT, advisor=None)
    assert isinstance(report, EvaluationReport)
    assert report.n_samples == 5
    # All 5 land on correct verdicts
    assert report.rule_accuracy == 1.0
    assert report.rule_false_pass_rate == 0.0
    assert report.rule_false_fail_rate == 0.0
    # Phase 3 has no SEAL labels yet
    assert report.seal_edge_n == 0
    assert report.seal_edge_f1 is None
    assert report.advisor_unavailable is True
    # By-ref breakdown
    assert report.by_ref_id == {"divider": 3, "rc_lowpass": 2}


def test_evaluate_real_samples_records_match_types():
    """Each negative sample should hit a specific failure branch —
    pin to the patterns documented in the fixture .meta.json notes."""

    report = evaluate_real_samples(REAL_FIXTURE_ROOT, advisor=None)
    by_id = {s.sample_id: s for s in report.samples}

    # Stray wire bridging VIN↔GND → R8 critical-extra branch
    assert by_id["student_0002_short_to_gnd"].rule_match_type == "extra_on_critical_net"
    assert by_id["student_0002_short_to_gnd"].rule_logic_correct is False

    # Missing R2 → student_subgraph_of_ref branch
    assert by_id["student_0003_missing_r2"].rule_match_type == "current_subgraph_in_reference"
    assert by_id["student_0003_missing_r2"].rule_logic_correct is False

    # Positives all go through full_isomorphism
    for sid in ("student_0001_correct", "student_0004_correct_renamed", "student_0005_pin_swap"):
        assert by_id[sid].rule_logic_correct is True


def test_evaluate_real_samples_raises_on_empty_corpus(tmp_path: Path):
    """Empty corpus should fail loud with a descriptive message that
    cites the load_stats counters."""

    with pytest.raises(ValueError, match="no usable real samples"):
        evaluate_real_samples(tmp_path)


def test_evaluate_real_samples_honours_limit():
    report = evaluate_real_samples(REAL_FIXTURE_ROOT, advisor=None, limit=2)
    assert report.n_samples == 2


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _torch_available(), reason="torch / torch_geometric not installed"
)
def test_evaluate_real_samples_with_advisor_populates_gnn_block():
    """With a trained advisor available, per-sample GNN scores should
    be captured (no SEAL F1 — those need labels — but inference_ms
    and edge_predictions live in rule_result.report.summary.gnn)."""

    from app.domain.gnn.inference import GNNAdvisor

    if not GNNAdvisor.checkpoint_available():
        pytest.skip("no GNN checkpoint available on disk")

    GNNAdvisor.reset_singleton()
    advisor = GNNAdvisor.get()
    report = evaluate_real_samples(REAL_FIXTURE_ROOT, advisor=advisor)
    assert report.advisor_unavailable is False
    assert report.advisor_version is not None
    # Most samples should have non-None inference_ms
    n_with_gnn = sum(
        1 for s in report.samples if s.gnn_inference_ms is not None
    )
    assert n_with_gnn >= 1
    GNNAdvisor.reset_singleton()


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def test_cli_real_dir_runs_end_to_end(tmp_path: Path):
    """``python -m scripts.gnn_eval --real-dir ...`` must write
    metrics.json + report.md without needing a label_dir."""

    from scripts.gnn_eval import main

    rc = main([
        "--real-dir", str(REAL_FIXTURE_ROOT),
        "--output", str(tmp_path),
        "--false-pass-gate", "0.005",
    ])
    assert rc == 0
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "report.md").is_file()


def test_cli_rejects_real_dir_plus_netlist_dir(tmp_path: Path, capsys):
    """The two ingest modes are mutually exclusive — CLI must reject
    the combo with a clear error."""

    from scripts.gnn_eval import main

    with pytest.raises(SystemExit) as exc_info:
        main([
            "--real-dir", str(REAL_FIXTURE_ROOT),
            "--netlist-dir", str(tmp_path),
            "--output", str(tmp_path / "out"),
        ])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err
