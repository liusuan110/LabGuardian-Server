"""P0.8 收尾 · LabelManifest + assert_manifest_healthy + coverage_check wrapper.

Three pre-P1 hardening concerns:
- coverage gap must raise CoverageError (not silently produce broken labels)
- build_seal_samples_with_coverage_check wraps build+assert atomically
- LabelManifest tracks cross-sample distribution + periodic checkpoints
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.gnn import (
    LabelBuildResult,
    LabelSource,
    TaskType,
    build_from_logical_reference,
    build_seal_samples,
    identity_alignment,
)
from app.domain.gnn.label_builder import (
    CoverageError,
    build_seal_samples_with_coverage_check,
)
from app.domain.gnn.label_manifest import LabelManifest, assert_manifest_healthy
from app.domain.gnn.port_graph import build_hetero_circuit_graph

from .conftest import hcg_to_cur_nx

FIXTURE_RC = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_rc_v1.json"
)


def _build(perturbations=None):
    ref = build_from_logical_reference(json.loads(FIXTURE_RC.read_text()))
    cur_g = hcg_to_cur_nx(ref, perturbations=perturbations)
    cur = build_hetero_circuit_graph(cur_g, side="cur")
    return ref, cur, identity_alignment(ref, cur)


# ---------------------------------------------------------------------------
# CoverageError + wrapper
# ---------------------------------------------------------------------------


def test_coverage_check_passes_on_healthy_result() -> None:
    ref, cur, align = _build(
        perturbations=[
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ]
    )
    # Should not raise
    result = build_seal_samples_with_coverage_check(ref, cur, align)
    assert isinstance(result, LabelBuildResult)


def test_coverage_error_is_assertion_error_subclass() -> None:
    """pytest.raises(AssertionError) should still catch CoverageError to avoid
    breaking historical test patterns."""

    assert issubclass(CoverageError, AssertionError)


def test_coverage_error_carries_missing_pairs() -> None:
    """Synthesize a broken LabelBuildResult by stripping all samples and
    verify CoverageError lists the cur edges that lost coverage."""

    from dataclasses import replace

    from app.domain.gnn.label_builder import (
        LabelStats,
        assert_observed_edges_covered,
    )

    ref, cur, align = _build()
    # Build a valid result then nuke its samples to simulate a coverage gap.
    real = build_seal_samples(ref, cur, align)
    broken = replace(
        real,
        samples=(),
        stats=LabelStats(
            total_samples=0, n_positives=0, n_negatives=0, pos_neg_ratio=0.0,
            by_source={s.value: 0 for s in LabelSource},
            by_task_type={t.value: 0 for t in TaskType},
            n_groups=0, n_groups_without_positive=0,
            n_skipped_missing_component=0, n_skipped_optional_pin=0,
            n_skipped_forbidden_pin_no_violation=0, n_skipped_extract_error=0,
            n_unique_ports_covered=0, n_unique_nets_covered=0,
        ),
    )
    with pytest.raises(CoverageError) as exc:
        assert_observed_edges_covered(broken, cur, ref, align)
    assert exc.value.missing, "missing list should be populated"
    assert len(exc.value.missing) == len(cur.edges)


# ---------------------------------------------------------------------------
# LabelManifest core
# ---------------------------------------------------------------------------


def test_manifest_starts_empty() -> None:
    m = LabelManifest()
    assert m.n_processed == 0
    assert m.total_samples == 0
    assert m.failures == []
    assert m.by_source[LabelSource.REF_PRESENT.value] == 0


def test_manifest_add_accumulates_counts() -> None:
    ref, cur, align = _build()
    result = build_seal_samples(ref, cur, align)
    m = LabelManifest()
    m.add("sample_1", result)
    m.add("sample_2", result)
    assert m.n_processed == 2
    assert m.total_samples == 2 * result.stats.total_samples
    assert m.total_positives == 2 * result.stats.n_positives
    assert m.total_negatives == 2 * result.stats.n_negatives
    assert (
        m.by_source[LabelSource.REF_PRESENT.value]
        == 2 * result.stats.by_source[LabelSource.REF_PRESENT.value]
    )


def test_manifest_record_failure_does_not_increment_samples() -> None:
    m = LabelManifest()
    m.record_failure("bad_sample", "coverage gap")
    assert m.n_processed == 1
    assert m.n_skipped_failures == 1
    assert m.total_samples == 0
    assert m.failure_rate == 1.0
    assert m.failures == [("bad_sample", "coverage gap")]


def test_manifest_pos_neg_ratio_handles_zero_negatives() -> None:
    m = LabelManifest()
    # No negatives accumulated → ratio = total_positives / 1 (zero-safe)
    m.total_positives = 5
    m.total_negatives = 0
    assert m.pos_neg_ratio == 5.0


def test_manifest_checkpoint_fires_every_n() -> None:
    ref, cur, align = _build()
    result = build_seal_samples(ref, cur, align)
    m = LabelManifest()
    # Process 99 → no checkpoint at every=100
    for i in range(99):
        m.add(f"s_{i}", result)
    assert m.checkpoint(every=100) is None
    # Process #100 → checkpoint
    m.add("s_99", result)
    snap = m.checkpoint(every=100)
    assert snap is not None
    assert snap["n_processed"] == 100
    # Calling again at same count → already checkpointed, returns None
    assert m.checkpoint(every=100) is None
    # Process #101..#200, next checkpoint at 200
    for i in range(100, 200):
        m.add(f"s_{i}", result)
    snap2 = m.checkpoint(every=100)
    assert snap2 is not None
    assert snap2["n_processed"] == 200


def test_manifest_checkpoint_includes_required_keys() -> None:
    ref, cur, align = _build()
    result = build_seal_samples(ref, cur, align)
    m = LabelManifest()
    for i in range(10):
        m.add(f"s_{i}", result)
    snap = m.checkpoint(every=10)
    assert snap is not None
    required_keys = {
        "n_processed",
        "n_skipped_failures",
        "failure_rate",
        "total_samples",
        "total_positives",
        "total_negatives",
        "pos_neg_ratio",
        "avg_samples_per_build",
        "n_groups",
        "n_groups_without_positive",
        "n_skipped_missing_component",
        "n_skipped_optional_pin",
        "by_source",
        "by_task_type",
    }
    assert required_keys.issubset(snap.keys()), required_keys - snap.keys()


def test_manifest_summary_includes_failures_list() -> None:
    m = LabelManifest()
    m.record_failure("a", "reason a")
    m.record_failure("b", "reason b")
    summ = m.summary()
    assert "failures" in summ
    assert summ["failures"] == [
        {"sample_id": "a", "reason": "reason a"},
        {"sample_id": "b", "reason": "reason b"},
    ]


def test_manifest_to_json_writes_file(tmp_path: Path) -> None:
    ref, cur, align = _build()
    result = build_seal_samples(ref, cur, align)
    m = LabelManifest()
    m.add("s_1", result)
    m.record_failure("s_2", "coverage")
    out = tmp_path / "manifest.json"
    m.to_json(out)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["n_processed"] == 2
    assert payload["n_skipped_failures"] == 1
    assert payload["total_samples"] == result.stats.total_samples


def test_manifest_checkpoint_rejects_invalid_every() -> None:
    m = LabelManifest()
    with pytest.raises(ValueError, match="positive"):
        m.checkpoint(every=0)
    with pytest.raises(ValueError, match="positive"):
        m.checkpoint(every=-5)


# ---------------------------------------------------------------------------
# assert_manifest_healthy
# ---------------------------------------------------------------------------


def test_healthy_manifest_passes() -> None:
    ref, cur, align = _build(
        perturbations=[
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ]
    )
    result = build_seal_samples(ref, cur, align)
    m = LabelManifest()
    for i in range(10):
        m.add(f"s_{i}", result)
    # Should not raise — both required sources present, ratio in range, no failures
    assert_manifest_healthy(m)


def test_assert_manifest_fails_on_high_failure_rate() -> None:
    m = LabelManifest()
    m.record_failure("a", "x")
    m.record_failure("b", "y")
    # 100% failure rate vs default 5% cap
    with pytest.raises(ValueError, match="failure_rate"):
        assert_manifest_healthy(m)


def test_assert_manifest_fails_when_required_source_missing() -> None:
    """Healthy ratio but missing 'wrong_observed' should be flagged."""

    ref, cur, align = _build()  # perfect copy → no wrong_observed
    result = build_seal_samples(ref, cur, align)
    m = LabelManifest()
    m.add("s_1", result)
    with pytest.raises(ValueError, match="wrong_observed"):
        assert_manifest_healthy(m)


def test_assert_manifest_fails_on_extreme_pos_neg_ratio() -> None:
    m = LabelManifest()
    # Inject a wildly unbalanced distribution by hand
    m.n_processed = 1
    m.total_positives = 100
    m.total_negatives = 1  # ratio = 100 vs cap 3.0
    m.by_source = {s.value: 0 for s in LabelSource}
    m.by_source[LabelSource.REF_PRESENT.value] = 100
    m.by_source[LabelSource.WRONG_OBSERVED.value] = 1
    m.by_source[LabelSource.NEGATIVE_RANDOM.value] = 1
    with pytest.raises(ValueError, match="pos_neg_ratio"):
        assert_manifest_healthy(m)


# ---------------------------------------------------------------------------
# End-to-end micro pipeline (mimics P1 dataset_builder)
# ---------------------------------------------------------------------------


def test_dataset_builder_pattern_end_to_end(tmp_path: Path) -> None:
    """Mimics the P1 dataset_builder loop: build + coverage_check + manifest +
    write labels to disk. Verifies the documented usage works."""

    import json as _json

    from app.domain.gnn import serialize_label_build_result

    perturbations_list = [
        None,  # perfect copy
        [("drop_edge", "pin1", "VIN")],
        [
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ],
    ]
    m = LabelManifest()
    labels_dir = tmp_path / "labels" / "test_rc_v1"
    labels_dir.mkdir(parents=True)

    for i, pert in enumerate(perturbations_list):
        sample_id = f"test_rc_v1__sample_{i:03d}"
        try:
            ref, cur, align = _build(perturbations=pert)
            result = build_seal_samples_with_coverage_check(
                ref, cur, align, seed=hash(sample_id) & 0xFFFFFFFF
            )
            m.add(sample_id, result)
            (labels_dir / f"{sample_id}.json").write_text(
                _json.dumps(
                    serialize_label_build_result(
                        result, sample_id=sample_id, ref_id="test_rc_v1"
                    )
                )
            )
        except CoverageError as e:
            m.record_failure(sample_id, f"coverage: {e}")

    # All 3 samples succeeded (no coverage gaps expected for these perturbations)
    assert m.n_processed == 3
    assert m.n_skipped_failures == 0
    assert len(list(labels_dir.iterdir())) == 3

    # Write final manifest
    m.to_json(tmp_path / "manifest.json")
    final = json.loads((tmp_path / "manifest.json").read_text())
    assert final["n_processed"] == 3
    assert final["total_samples"] > 0
