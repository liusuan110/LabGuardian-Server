"""Tests for ``app.domain.gnn.real_netlist_loader`` (plan §十 R6 Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.gnn.real_netlist_loader import (
    ALLOWED_OUTCOMES,
    RealSample,
    load_real_samples,
)

REAL_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "fixtures" / "real_student_simulated"
)


def _minimal_netlist(component_id: str = "R1") -> dict:
    return {
        "components": [
            {
                "component_id": component_id,
                "component_type": "Resistor",
                "pins": [
                    {"pin_id": 0, "pin_name": "pin1",
                     "electrical_net_id": "n_a"},
                    {"pin_id": 1, "pin_name": "pin2",
                     "electrical_net_id": "n_b"},
                ],
            },
        ],
        "nets": [
            {"electrical_net_id": "n_a"},
            {"electrical_net_id": "n_b"},
        ],
    }


def _minimal_meta(
    *,
    sample_id: str = "smoke",
    ref_id: str = "divider",
    expected: str = "positive",
) -> dict:
    return {
        "sample_id": sample_id,
        "ref_id": ref_id,
        "expected_outcome": expected,
        "annotation_source": "test",
    }


# ---------------------------------------------------------------------------
# Schema permissiveness + happy path
# ---------------------------------------------------------------------------


def test_loads_committed_simulated_real_fixtures():
    """The 5 hand-rolled fixtures must all load with no skips."""

    samples, stats = load_real_samples(REAL_FIXTURE_ROOT)
    assert stats.n_skipped_no_meta == 0
    assert stats.n_skipped_bad_outcome == 0
    assert stats.n_skipped_invalid_schema == 0
    assert stats.n_skipped_other == 0
    assert stats.n_loaded == 5
    assert len(samples) == 5
    refs = {s.ref_id for s in samples}
    assert refs == {"divider", "rc_lowpass"}


def test_load_returns_real_sample_with_expected_fields():
    samples, _ = load_real_samples(REAL_FIXTURE_ROOT)
    s = next(s for s in samples if s.sample_id == "student_0001_correct")
    assert isinstance(s, RealSample)
    assert s.ref_id == "divider"
    assert s.expected_outcome == "positive"
    assert s.annotation_source == "teacher"
    assert s.netlist_v2["components"]
    assert s.netlist_path.is_file()
    assert s.meta_path.is_file()


def test_samples_sorted_by_ref_then_sample():
    samples, _ = load_real_samples(REAL_FIXTURE_ROOT)
    keys = [(s.ref_id, s.sample_id) for s in samples]
    assert keys == sorted(keys)


def test_load_respects_limit():
    samples, stats = load_real_samples(REAL_FIXTURE_ROOT, limit=2)
    assert len(samples) == 2
    assert stats.n_loaded == 2


# ---------------------------------------------------------------------------
# Skip behaviour — every failure mode must skip rather than crash
# ---------------------------------------------------------------------------


def test_skips_netlist_without_meta(tmp_path: Path):
    ref_dir = tmp_path / "divider"
    ref_dir.mkdir()
    (ref_dir / "orphan.json").write_text(json.dumps(_minimal_netlist()), encoding="utf-8")

    samples, stats = load_real_samples(tmp_path)
    assert samples == []
    assert stats.n_skipped_no_meta == 1
    assert "orphan.json" in stats.skipped_paths[0]


def test_skips_meta_with_unknown_outcome(tmp_path: Path):
    ref_dir = tmp_path / "divider"
    ref_dir.mkdir()
    (ref_dir / "bad.json").write_text(json.dumps(_minimal_netlist()), encoding="utf-8")
    (ref_dir / "bad.meta.json").write_text(json.dumps(
        _minimal_meta(expected="garbage")
    ), encoding="utf-8")

    samples, stats = load_real_samples(tmp_path)
    assert samples == []
    assert stats.n_skipped_bad_outcome == 1


def test_skips_meta_with_no_ref_id(tmp_path: Path):
    ref_dir = tmp_path / "divider"
    ref_dir.mkdir()
    (ref_dir / "no_ref.json").write_text(json.dumps(_minimal_netlist()), encoding="utf-8")
    meta = _minimal_meta()
    meta["ref_id"] = ""
    (ref_dir / "no_ref.meta.json").write_text(json.dumps(meta), encoding="utf-8")

    samples, stats = load_real_samples(tmp_path)
    assert samples == []
    assert stats.n_skipped_bad_outcome == 1


def test_skips_netlist_with_invalid_schema(tmp_path: Path):
    ref_dir = tmp_path / "divider"
    ref_dir.mkdir()
    # missing components.component_id
    bad = {
        "components": [
            {"component_type": "Resistor", "pins": []},
        ],
        "nets": [{"electrical_net_id": "n_a"}],
    }
    (ref_dir / "broken.json").write_text(json.dumps(bad), encoding="utf-8")
    (ref_dir / "broken.meta.json").write_text(json.dumps(_minimal_meta()), encoding="utf-8")

    samples, stats = load_real_samples(tmp_path)
    assert samples == []
    assert stats.n_skipped_invalid_schema == 1


def test_skips_netlist_with_non_object_top_level(tmp_path: Path):
    ref_dir = tmp_path / "divider"
    ref_dir.mkdir()
    (ref_dir / "bad.json").write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    (ref_dir / "bad.meta.json").write_text(json.dumps(_minimal_meta()), encoding="utf-8")

    samples, stats = load_real_samples(tmp_path)
    assert samples == []
    assert stats.n_skipped_invalid_schema == 1


def test_ignores_manifest_and_sidecar_files(tmp_path: Path):
    """``manifest.json`` and ``*.meta.json`` should not be mistaken for
    netlist payloads."""

    ref_dir = tmp_path / "divider"
    ref_dir.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"profile": "real"}), encoding="utf-8")
    (ref_dir / "ok.json").write_text(json.dumps(_minimal_netlist()), encoding="utf-8")
    (ref_dir / "ok.meta.json").write_text(json.dumps(_minimal_meta()), encoding="utf-8")

    samples, stats = load_real_samples(tmp_path)
    assert len(samples) == 1
    assert stats.n_skipped_no_meta == 0


def test_raises_on_missing_real_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="real_dir not found"):
        load_real_samples(tmp_path / "does_not_exist")


def test_allowed_outcomes_matches_evaluator_contract():
    """Real-loader's allowed outcomes must stay in lock-step with the
    synthetic evaluator's per-sample handling (see
    ``app/domain/gnn/evaluator.py:_evaluate_sample``)."""

    assert ALLOWED_OUTCOMES == frozenset({
        "positive", "wrong_observed", "missing_required",
    })
