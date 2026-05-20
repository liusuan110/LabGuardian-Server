"""P1 Phase A · end-to-end dataset_builder tests.

Verifies the full pipeline that P1 will run at scale:
  refs + perturbation plan → labels JSON on disk + manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.gnn import (
    DatasetSpec,
    PerturbationPlan,
    RefSpec,
    deserialize_label_build_result,
    generate_dataset,
)
from app.domain.gnn.label_manifest import LabelManifest

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "references"


def _make_ref(name: str, ref_id: str, subtypes: dict | None = None) -> RefSpec:
    return RefSpec(
        ref_id=ref_id,
        payload_path=FIXTURES_DIR / name,
        subtype_by_source_id=subtypes or {},
    )


def _basic_plan(n_per_op: int = 2) -> PerturbationPlan:
    return PerturbationPlan(
        counts={
            "identity": n_per_op,
            "pin_swap_symmetric": n_per_op,
            "wrong_connection": n_per_op,
        }
    )


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------


def test_generate_dataset_writes_labels_and_manifest(tmp_path: Path) -> None:
    spec = DatasetSpec(
        refs=(
            _make_ref("test_rc_v1.json", "rc_lowpass"),
            _make_ref("test_voltage_divider_v1.json", "divider"),
        ),
        plan=_basic_plan(n_per_op=2),
        output_dir=tmp_path / "ds",
        # Disable health enforcement because tiny dataset with low diversity
        # may trip pos_neg_ratio bounds — that's fine for the unit test;
        # Phase C will use large enough counts.
        enforce_healthy=False,
    )
    manifest = generate_dataset(spec)

    assert isinstance(manifest, LabelManifest)
    # 2 refs × 3 perturbations × 2 samples = 12
    assert manifest.n_processed == 12
    assert manifest.n_skipped_failures == 0

    # Labels written per ref_id
    rc_labels = list((tmp_path / "ds" / "labels" / "rc_lowpass").glob("*.json"))
    div_labels = list((tmp_path / "ds" / "labels" / "divider").glob("*.json"))
    assert len(rc_labels) == 6  # 3 ops × 2 samples
    assert len(div_labels) == 6

    # Manifest file exists & valid
    manifest_payload = json.loads((tmp_path / "ds" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["n_processed"] == 12


def test_each_label_file_is_valid_payload(tmp_path: Path) -> None:
    spec = DatasetSpec(
        refs=(_make_ref("test_rc_v1.json", "rc_lowpass"),),
        plan=PerturbationPlan(counts={"identity": 2, "wrong_connection": 2}),
        output_dir=tmp_path / "ds",
        enforce_healthy=False,
    )
    generate_dataset(spec)
    label_files = sorted(
        (tmp_path / "ds" / "labels" / "rc_lowpass").glob("*.json")
    )
    assert len(label_files) == 4
    for fp in label_files:
        payload = json.loads(fp.read_text(encoding="utf-8"))
        # Round-trip back through deserialize → ensures full schema validity
        result = deserialize_label_build_result(payload)
        assert result.stats.total_samples > 0
        # cur_metadata carries perturbation chain + alignment dict
        assert "perturbation_chain" in payload["cur_metadata"]
        assert payload["cur_metadata"]["alignment"]["ref_to_cur_component"]
        # Sample id matches file stem
        assert payload["sample_id"] == fp.stem


def test_sample_ids_are_unique_and_deterministic(tmp_path: Path) -> None:
    spec = DatasetSpec(
        refs=(_make_ref("test_rc_v1.json", "rc_lowpass"),),
        plan=PerturbationPlan(counts={"identity": 3, "wrong_connection": 3}),
        output_dir=tmp_path / "ds",
        base_seed=42,
        enforce_healthy=False,
    )
    generate_dataset(spec)
    files1 = sorted(
        f.name for f in (tmp_path / "ds" / "labels" / "rc_lowpass").glob("*.json")
    )

    # Run again with same seed → same file set, identical contents
    spec2 = DatasetSpec(
        refs=(_make_ref("test_rc_v1.json", "rc_lowpass"),),
        plan=PerturbationPlan(counts={"identity": 3, "wrong_connection": 3}),
        output_dir=tmp_path / "ds2",
        base_seed=42,
        enforce_healthy=False,
    )
    generate_dataset(spec2)
    files2 = sorted(
        f.name for f in (tmp_path / "ds2" / "labels" / "rc_lowpass").glob("*.json")
    )
    assert files1 == files2

    # File-by-file content equivalence (deterministic seed)
    for fname in files1:
        p1 = json.loads(((tmp_path / "ds" / "labels" / "rc_lowpass") / fname).read_text(encoding="utf-8"))
        p2 = json.loads(((tmp_path / "ds2" / "labels" / "rc_lowpass") / fname).read_text(encoding="utf-8"))
        # stats must match exactly
        assert p1["stats"] == p2["stats"]


# ---------------------------------------------------------------------------
# Manifest reflects per-source distribution
# ---------------------------------------------------------------------------


def test_manifest_by_source_aggregates_across_samples(tmp_path: Path) -> None:
    spec = DatasetSpec(
        refs=(_make_ref("test_voltage_divider_v1.json", "divider"),),
        plan=PerturbationPlan(
            counts={"identity": 3, "wrong_connection": 3}
        ),
        output_dir=tmp_path / "ds",
        enforce_healthy=False,
    )
    manifest = generate_dataset(spec)
    # identity samples produce REF_PRESENT positives
    assert manifest.by_source["ref_present"] > 0
    # wrong_connection samples produce WRONG_OBSERVED negatives
    assert manifest.by_source["wrong_observed"] > 0
    # Total positives + negatives = total_samples
    assert (
        manifest.total_positives + manifest.total_negatives
        == manifest.total_samples
    )


# ---------------------------------------------------------------------------
# Health enforcement
# ---------------------------------------------------------------------------


def test_enforce_healthy_raises_when_required_source_missing(tmp_path: Path) -> None:
    """Plan only includes 'identity' → no wrong_observed → assert_manifest_healthy
    should reject because 'wrong_observed' is in default required_sources."""

    spec = DatasetSpec(
        refs=(_make_ref("test_rc_v1.json", "rc_lowpass"),),
        plan=PerturbationPlan(counts={"identity": 5}),
        output_dir=tmp_path / "ds",
        enforce_healthy=True,
    )
    with pytest.raises(ValueError, match="wrong_observed"):
        generate_dataset(spec)


def test_enforce_healthy_disabled_lets_unbalanced_pass(tmp_path: Path) -> None:
    spec = DatasetSpec(
        refs=(_make_ref("test_rc_v1.json", "rc_lowpass"),),
        plan=PerturbationPlan(counts={"identity": 2}),
        output_dir=tmp_path / "ds",
        enforce_healthy=False,
    )
    # Should not raise; manifest still gets written
    manifest = generate_dataset(spec)
    assert manifest.n_processed == 2
    assert (tmp_path / "ds" / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Failure handling — manifest tracks even when one perturbation fails
# ---------------------------------------------------------------------------


def test_bad_ref_path_fails_fast_with_dataset_spec_error(tmp_path: Path) -> None:
    """P3 fix: a bad ref payload path must be caught at spec validation
    (before any sample work or partial output dir creation), surfacing a
    :class:`DatasetSpecError` rather than the bare ``FileNotFoundError``
    that used to leak from the first sample's load."""

    from app.domain.gnn.dataset_builder import DatasetSpecError

    bad_ref = RefSpec(
        ref_id="bogus",
        payload_path=tmp_path / "does_not_exist.json",
    )
    good_ref = _make_ref("test_rc_v1.json", "rc_lowpass")
    spec = DatasetSpec(
        refs=(bad_ref, good_ref),
        plan=PerturbationPlan(counts={"identity": 1}),
        output_dir=tmp_path / "ds",
        enforce_healthy=False,
    )
    with pytest.raises(DatasetSpecError, match="payload not found"):
        generate_dataset(spec)
    # No partial output: validation runs before mkdir
    assert not (tmp_path / "ds").exists()


# ---------------------------------------------------------------------------
# DatasetSpec helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# P2 audit regression: RefSpec.subtype_by_source_id applied to ref build
# ---------------------------------------------------------------------------


def test_ref_spec_subtype_override_applied_to_ref_hcg(tmp_path: Path) -> None:
    """P2 audit regression: when the fixture payload lacks ``subtype``, the
    ``RefSpec.subtype_by_source_id`` override must still apply at REF build
    time (not just cur build inside perturbation). Otherwise UA741 pin 8
    would be REQUIRED on ref side but FORBIDDEN on cur side → label_builder
    would surface bogus REF_PRESENT for pin 8 → coverage chaos.

    Strategy: copy the opamp fixture, strip the ``subtype`` field, run
    generate_dataset with the override, and verify a sample completes
    successfully with the correct ref-side pin policy semantics.
    """

    import json as _json

    opamp_payload = _json.loads(
        (FIXTURES_DIR / "test_opamp_buffer_v1.json").read_text(encoding="utf-8")
    )
    # Strip the subtype so the fixture is "naked" (no IC spec hints)
    for c in opamp_payload["components"]:
        c.pop("subtype", None)
    stripped = tmp_path / "opamp_no_subtype.json"
    stripped.write_text(_json.dumps(opamp_payload), encoding="utf-8")

    # WITHOUT override → ref builder treats U1 as a plain IC, no pin specs
    from app.domain.gnn import build_from_logical_reference

    ref_no_override = build_from_logical_reference(_json.loads(stripped.read_text(encoding="utf-8")))
    # All ports would be plain REQUIRED (no FORBIDDEN/OPTIONAL discrimination)
    forbidden_no_override = [
        p for p in ref_no_override.ports.values()
        if p.connection_policy == "forbidden"
    ]
    assert not forbidden_no_override, (
        "without subtype override, IC pin 8 should NOT be marked FORBIDDEN"
    )

    # WITH override → ref pin 8 = FORBIDDEN, pin 1/5 = OPTIONAL
    ref_with_override = build_from_logical_reference(
        _json.loads(stripped.read_text(encoding="utf-8")),
        extra_subtypes_by_source_id={"U1": "UA741"},
    )
    forbidden_with_override = [
        p for p in ref_with_override.ports.values()
        if p.connection_policy == "forbidden"
    ]
    assert len(forbidden_with_override) == 1, (
        f"with subtype override, expected 1 FORBIDDEN pin (UA741 pin 8), "
        f"got {len(forbidden_with_override)}"
    )

    # End-to-end: dataset_builder must honor the override on the ref side
    spec = DatasetSpec(
        refs=(
            RefSpec(
                ref_id="opamp_no_subtype",
                payload_path=stripped,
                subtype_by_source_id={"U1": "UA741"},
            ),
        ),
        plan=PerturbationPlan(counts={"identity": 1}),
        output_dir=tmp_path / "ds",
        enforce_healthy=False,
    )
    manifest = generate_dataset(spec)
    assert manifest.n_processed == 1
    assert manifest.n_skipped_failures == 0


# ---------------------------------------------------------------------------
# P3 audit regression: validate_dataset_spec catches all config errors upfront
# ---------------------------------------------------------------------------


def test_validate_rejects_unknown_perturbation_name(tmp_path: Path) -> None:
    from app.domain.gnn.dataset_builder import DatasetSpecError

    spec = DatasetSpec(
        refs=(_make_ref("test_rc_v1.json", "rc_lowpass"),),
        plan=PerturbationPlan(counts={"identity": 1, "rotate_resistor_clockwise": 1}),
        output_dir=tmp_path / "ds",
    )
    with pytest.raises(DatasetSpecError, match="unknown perturbation"):
        generate_dataset(spec)
    assert not (tmp_path / "ds").exists()


def test_validate_rejects_duplicate_ref_id(tmp_path: Path) -> None:
    from app.domain.gnn.dataset_builder import DatasetSpecError

    spec = DatasetSpec(
        refs=(
            _make_ref("test_rc_v1.json", "same"),
            _make_ref("test_voltage_divider_v1.json", "same"),
        ),
        plan=PerturbationPlan(counts={"identity": 1}),
        output_dir=tmp_path / "ds",
    )
    with pytest.raises(DatasetSpecError, match="duplicate ref_id"):
        generate_dataset(spec)


def test_validate_rejects_empty_refs_and_empty_plan(tmp_path: Path) -> None:
    from app.domain.gnn.dataset_builder import DatasetSpecError

    spec_empty_refs = DatasetSpec(
        refs=(),
        plan=PerturbationPlan(counts={"identity": 1}),
        output_dir=tmp_path / "ds1",
    )
    with pytest.raises(DatasetSpecError, match="refs is empty"):
        generate_dataset(spec_empty_refs)

    spec_empty_plan = DatasetSpec(
        refs=(_make_ref("test_rc_v1.json", "rc_lowpass"),),
        plan=PerturbationPlan(counts={}),
        output_dir=tmp_path / "ds2",
    )
    with pytest.raises(DatasetSpecError, match="counts is empty"):
        generate_dataset(spec_empty_plan)


def test_validate_rejects_negative_count(tmp_path: Path) -> None:
    from app.domain.gnn.dataset_builder import DatasetSpecError

    spec = DatasetSpec(
        refs=(_make_ref("test_rc_v1.json", "rc_lowpass"),),
        plan=PerturbationPlan(counts={"identity": -3}),
        output_dir=tmp_path / "ds",
    )
    with pytest.raises(DatasetSpecError, match="negative count"):
        generate_dataset(spec)


def test_validate_consolidates_multiple_issues(tmp_path: Path) -> None:
    """Validation collects ALL issues in one error message — failing fast
    on the first is unfriendly when configuring large specs."""

    from app.domain.gnn.dataset_builder import DatasetSpecError

    spec = DatasetSpec(
        refs=(
            RefSpec(ref_id="dup", payload_path=tmp_path / "nope1.json"),
            RefSpec(ref_id="dup", payload_path=tmp_path / "nope2.json"),
        ),
        plan=PerturbationPlan(counts={"made_up_op": 1}),
        output_dir=tmp_path / "ds",
    )
    with pytest.raises(DatasetSpecError) as ei:
        generate_dataset(spec)
    msg = str(ei.value)
    # All three issues should appear in the consolidated message
    assert "payload not found" in msg
    assert "duplicate ref_id" in msg
    assert "unknown perturbation" in msg


def test_total_samples_calculation() -> None:
    plan = PerturbationPlan(counts={"identity": 5, "wrong_connection": 10})
    spec = DatasetSpec(
        refs=(
            _make_ref("test_rc_v1.json", "a"),
            _make_ref("test_rc_v1.json", "b"),
            _make_ref("test_rc_v1.json", "c"),
        ),
        plan=plan,
        output_dir=Path("/tmp/unused"),
    )
    assert plan.total_per_ref() == 15
    assert spec.total_samples() == 45


# ---------------------------------------------------------------------------
# Phase B integration — all 12 operators run end-to-end through dataset_builder
# ---------------------------------------------------------------------------


def test_phase_b_operators_complete_through_pipeline(tmp_path: Path) -> None:
    """Smoke-test: spec containing every Phase B perturbation produces
    well-formed label JSON for every sample, no CoverageError leaks."""

    plan = PerturbationPlan(
        counts={
            # Phase A (sanity baseline)
            "identity": 1,
            # Phase B — 1 sample per op
            "missing_component": 1,
            "extra_component": 1,
            "floating_net": 1,
            "short_circuit": 1,
            "power_swapped": 1,
            "input_output_swapped": 1,
            "extra_wire_bridge": 1,
            "chained": 1,
        }
    )
    # UA741 buffer has power+ground+input+output roles — best fixture for
    # exercising power_swapped + input_output_swapped without fallback.
    spec = DatasetSpec(
        refs=(
            _make_ref("test_opamp_buffer_v1.json", "opamp"),
            _make_ref("test_voltage_divider_v1.json", "divider"),
        ),
        plan=plan,
        output_dir=tmp_path / "ds",
        enforce_healthy=False,
    )
    manifest = generate_dataset(spec)
    # 2 refs × 9 ops × 1 sample = 18
    assert manifest.n_processed == 18
    assert manifest.n_skipped_failures == 0
    # All label files are valid JSON
    labels_root = tmp_path / "ds" / "labels"
    assert {d.name for d in labels_root.iterdir()} == {"opamp", "divider"}
    for ref_dir in labels_root.iterdir():
        for label_file in ref_dir.glob("*.json"):
            payload = json.loads(label_file.read_text(encoding="utf-8"))
            # Phase B chain entries should appear in some samples
            assert "perturbation_chain" in payload["cur_metadata"]


# ---------------------------------------------------------------------------
# Phase C: resume support
# ---------------------------------------------------------------------------


def test_resume_replays_existing_labels_without_regen(tmp_path: Path) -> None:
    """First run writes labels; second run with ``resume=True`` re-uses
    them and produces an identical manifest."""

    spec = DatasetSpec(
        refs=(
            _make_ref("test_rc_v1.json", "rc"),
            _make_ref("test_voltage_divider_v1.json", "div"),
        ),
        plan=PerturbationPlan(counts={"identity": 2, "wrong_connection": 2}),
        output_dir=tmp_path / "ds",
        base_seed=42,
        enforce_healthy=False,
    )
    m1 = generate_dataset(spec)
    # Snapshot mtimes — resume must not rewrite the files
    label_files = sorted(
        (tmp_path / "ds" / "labels").rglob("*.json")
    )
    mtimes_before = {p: p.stat().st_mtime_ns for p in label_files}

    m2 = generate_dataset(spec, resume=True)
    assert m2.n_processed == m1.n_processed
    assert m2.total_samples == m1.total_samples
    assert m2.total_positives == m1.total_positives
    assert m2.total_negatives == m1.total_negatives
    assert m2.by_source == m1.by_source

    mtimes_after = {p: p.stat().st_mtime_ns for p in label_files}
    assert mtimes_after == mtimes_before, (
        "resume must not rewrite existing label files"
    )


def test_resume_regenerates_missing_samples(tmp_path: Path) -> None:
    """If some sample files are deleted, resume only re-generates those."""

    spec = DatasetSpec(
        refs=(_make_ref("test_rc_v1.json", "rc"),),
        plan=PerturbationPlan(counts={"identity": 4}),
        output_dir=tmp_path / "ds",
        base_seed=7,
        enforce_healthy=False,
    )
    m1 = generate_dataset(spec)
    assert m1.n_processed == 4

    # Delete 2 of 4
    rc_dir = tmp_path / "ds" / "labels" / "rc"
    deleted = sorted(rc_dir.glob("*.json"))[:2]
    snap_mtimes = {p: p.stat().st_mtime_ns for p in rc_dir.glob("*.json")}
    for p in deleted:
        p.unlink()
    assert len(list(rc_dir.glob("*.json"))) == 2

    m2 = generate_dataset(spec, resume=True)
    # 4 processed again (2 resumed + 2 regenerated)
    assert m2.n_processed == 4
    # All 4 files exist now
    assert len(list(rc_dir.glob("*.json"))) == 4
    # The 2 NOT deleted should still have original mtime
    for p in list(rc_dir.glob("*.json")):
        if p in snap_mtimes and p not in deleted:
            assert p.stat().st_mtime_ns == snap_mtimes[p]


def test_resume_with_corrupted_label_regenerates(tmp_path: Path) -> None:
    """Corrupted (non-JSON) label files must be regenerated, not silently dropped."""

    spec = DatasetSpec(
        refs=(_make_ref("test_rc_v1.json", "rc"),),
        plan=PerturbationPlan(counts={"identity": 2}),
        output_dir=tmp_path / "ds",
        enforce_healthy=False,
    )
    generate_dataset(spec)
    rc_dir = tmp_path / "ds" / "labels" / "rc"
    label_files = sorted(rc_dir.glob("*.json"))
    # Corrupt the first
    label_files[0].write_text("{not json}", encoding="utf-8")

    m2 = generate_dataset(spec, resume=True)
    assert m2.n_processed == 2
    assert m2.n_skipped_failures == 0
    # Corrupted file must now be valid JSON
    payload = json.loads(label_files[0].read_text(encoding="utf-8"))
    assert "stats" in payload


def test_resume_false_overwrites_labels(tmp_path: Path) -> None:
    """Without resume, generate_dataset truncates and overwrites."""

    spec = DatasetSpec(
        refs=(_make_ref("test_rc_v1.json", "rc"),),
        plan=PerturbationPlan(counts={"identity": 2}),
        output_dir=tmp_path / "ds",
        base_seed=0,
        enforce_healthy=False,
    )
    generate_dataset(spec)
    rc_dir = tmp_path / "ds" / "labels" / "rc"
    mtimes_before = {p: p.stat().st_mtime_ns for p in rc_dir.glob("*.json")}
    # Wait a tick then regen without resume — files should be rewritten
    import time

    time.sleep(0.01)
    generate_dataset(spec, resume=False)
    mtimes_after = {p: p.stat().st_mtime_ns for p in rc_dir.glob("*.json")}
    assert mtimes_after.keys() == mtimes_before.keys()
    assert any(
        mtimes_after[p] > mtimes_before[p] for p in mtimes_before
    ), "resume=False should rewrite files"


# ---------------------------------------------------------------------------
# Phase C: parallel execution (workers > 1)
# ---------------------------------------------------------------------------


def test_workers_gt1_produces_same_manifest_as_serial(tmp_path: Path) -> None:
    """Parity: the parallel path must produce identical aggregate stats to
    serial, given the same spec/seed. (Sample seeds are deterministic per
    sample_id, so the order in which workers complete doesn't matter.)"""

    plan = PerturbationPlan(
        counts={
            "identity": 3,
            "wrong_connection": 3,
            "extra_component": 2,
            "chained": 2,
        }
    )
    refs = (
        _make_ref("test_rc_v1.json", "rc"),
        _make_ref("test_voltage_divider_v1.json", "div"),
        _make_ref(
            "test_opamp_buffer_v1.json", "opamp", {"U1": "UA741"}
        ),
    )
    spec_serial = DatasetSpec(
        refs=refs,
        plan=plan,
        output_dir=tmp_path / "serial",
        base_seed=7,
        enforce_healthy=False,
    )
    spec_parallel = DatasetSpec(
        refs=refs,
        plan=plan,
        output_dir=tmp_path / "parallel",
        base_seed=7,
        enforce_healthy=False,
    )
    m_serial = generate_dataset(spec_serial, workers=1)
    m_parallel = generate_dataset(spec_parallel, workers=3)

    assert m_serial.n_processed == m_parallel.n_processed
    assert m_serial.total_samples == m_parallel.total_samples
    assert m_serial.total_positives == m_parallel.total_positives
    assert m_serial.total_negatives == m_parallel.total_negatives
    assert m_serial.n_groups == m_parallel.n_groups
    assert m_serial.by_source == m_parallel.by_source
    assert m_serial.by_task_type == m_parallel.by_task_type
    # All label files written (same set of sample_ids on disk)
    serial_files = sorted(
        f.name for f in (tmp_path / "serial" / "labels").rglob("*.json")
    )
    parallel_files = sorted(
        f.name for f in (tmp_path / "parallel" / "labels").rglob("*.json")
    )
    assert serial_files == parallel_files


def test_workers_gt1_writes_identical_label_payloads_as_serial(
    tmp_path: Path,
) -> None:
    """Per-sample seeds are deterministic, so each individual label JSON
    file should be byte-identical between the serial and parallel runs."""

    plan = PerturbationPlan(counts={"identity": 2, "wrong_connection": 2})
    spec_kwargs = dict(
        refs=(_make_ref("test_rc_v1.json", "rc"),),
        plan=plan,
        base_seed=42,
        enforce_healthy=False,
    )
    serial = DatasetSpec(output_dir=tmp_path / "s", **spec_kwargs)  # type: ignore[arg-type]
    parallel = DatasetSpec(output_dir=tmp_path / "p", **spec_kwargs)  # type: ignore[arg-type]
    generate_dataset(serial, workers=1)
    generate_dataset(parallel, workers=2)

    s_dir = tmp_path / "s" / "labels" / "rc"
    p_dir = tmp_path / "p" / "labels" / "rc"
    for fname in sorted(f.name for f in s_dir.glob("*.json")):
        s_payload = json.loads((s_dir / fname).read_text(encoding="utf-8"))
        p_payload = json.loads((p_dir / fname).read_text(encoding="utf-8"))
        assert s_payload == p_payload, (
            f"label payload mismatch for {fname} between serial and parallel"
        )


def test_workers_and_resume_compose_correctly(tmp_path: Path) -> None:
    """First do a serial generation, delete half the labels, then do a
    parallel run with ``resume=True``. Result must equal a fresh serial
    generation."""

    plan = PerturbationPlan(counts={"identity": 4, "wrong_connection": 4})
    spec = DatasetSpec(
        refs=(_make_ref("test_voltage_divider_v1.json", "div"),),
        plan=plan,
        output_dir=tmp_path / "ds",
        base_seed=99,
        enforce_healthy=False,
    )
    generate_dataset(spec, workers=1)
    # Delete half
    rc_dir = tmp_path / "ds" / "labels" / "div"
    all_files = sorted(rc_dir.glob("*.json"))
    for f in all_files[: len(all_files) // 2]:
        f.unlink()

    m_resumed = generate_dataset(spec, workers=2, resume=True)
    assert m_resumed.n_processed == len(all_files)
    assert m_resumed.n_skipped_failures == 0
    assert len(list(rc_dir.glob("*.json"))) == len(all_files)


def test_workers_propagate_coverage_failures_to_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a worker raises CoverageError, the manifest must record it
    via record_failure rather than crashing the pool. We force a coverage
    error by monkey-patching the coverage check to always raise."""

    from app.domain.gnn import label_builder

    def _always_raise(*args, **kwargs):  # noqa: ANN
        raise label_builder.CoverageError("synthetic coverage failure for test")

    # Patch on the module the worker imports from
    monkeypatch.setattr(
        "app.domain.gnn.dataset_builder.build_seal_samples_with_coverage_check",
        _always_raise,
    )
    spec = DatasetSpec(
        refs=(_make_ref("test_rc_v1.json", "rc"),),
        plan=PerturbationPlan(counts={"identity": 2}),
        output_dir=tmp_path / "ds",
        enforce_healthy=False,
    )
    # Use workers=1 so the monkeypatch (in-process only) takes effect.
    # ProcessPoolExecutor would spawn fresh processes without the patch.
    m = generate_dataset(spec, workers=1)
    assert m.n_processed == 2
    assert m.n_skipped_failures == 2
    assert all("coverage:" in r for _, r in m.failures)
