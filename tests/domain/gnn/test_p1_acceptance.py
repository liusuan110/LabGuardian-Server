"""P1 acceptance test — runs the production DEFAULT_CONFIG against a
**scaled-down** sample count so CI stays fast (~3 s) while exercising
every code path that the real 4 × 600 = 2400 production run exercises:

- every Phase A + Phase B perturbation operator at least once
- every required LabelSource (REF_PRESENT, WRONG_OBSERVED, REF_SYMMETRIC_SWAP,
  REF_ABSENT_REQUIRED, NEGATIVE_RANDOM, FORBIDDEN_NEGATIVE)
- parallel execution (workers > 1)
- splits with ref-disjoint test set (opamp_buffer held out)
- manifest health gate passing

Production-scale run (4 × 600 = 2400, ~0.8 s with workers=4) is reproducible
via the CLI:

    python -m scripts.gnn_generate_dataset \\
        --output-dir datasets/circuit_compare \\
        --base-seed 0 --workers 4 --progress

This test asserts the same shape on a tiny budget.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from app.domain.gnn import LabelSource
from scripts.gnn_generate_dataset import DEFAULT_CONFIG, main

# Counts scaled down (~factor 30) to keep test under ~3s but still cover
# every operator with ≥ 1 sample and produce ≥ 1 of every required
# LabelSource (FORBIDDEN_NEGATIVE only fires for the opamp_buffer ref).
ACCEPTANCE_SCALED_PLAN = {
    "identity": 3,
    "pin_swap_symmetric": 2,
    "wrong_connection": 3,
    "pin_reversed": 2,
    "missing_component": 2,
    "extra_component": 2,
    "floating_net": 2,
    "short_circuit": 2,
    "power_swapped": 1,
    "input_output_swapped": 1,
    "extra_wire_bridge": 2,
    "chained": 2,
    # Phase C · Stage 1 — same-net wire positive class (real-student form).
    "insert_same_net_wire": 2,
}


def _scaled_config(out_dir: Path, *, workers: int = 1) -> Path:
    """Clone DEFAULT_CONFIG, swap in the scaled plan, write to disk."""

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["plan"] = {"counts": ACCEPTANCE_SCALED_PLAN}
    cfg["checkpoint_every"] = 100
    # Keep enforce_healthy at the DEFAULT_CONFIG value (True) so the test
    # actually exercises assert_manifest_healthy on the scaled distribution.
    path = out_dir / "scaled.json"
    path.write_text(json.dumps(cfg))
    return path


def test_p1_acceptance_scaled_runs_end_to_end_serial(tmp_path: Path) -> None:
    """Smaller serial run — proves DEFAULT_CONFIG + scaled plan is healthy."""

    cfg = _scaled_config(tmp_path)
    out = tmp_path / "ds"
    rc = main(
        ["--output-dir", str(out), "--config", str(cfg), "--base-seed", "0"]
    )
    assert rc == 0, "scaled P1 run failed; see captured stdout"

    manifest = json.loads((out / "manifest.json").read_text())
    expected_total = sum(ACCEPTANCE_SCALED_PLAN.values()) * len(DEFAULT_CONFIG["refs"])
    assert manifest["n_processed"] == expected_total
    assert manifest["n_skipped_failures"] == 0
    assert manifest["failure_rate"] == 0.0

    # Every required source must have actually fired (otherwise the gate
    # was passed accidentally)
    required = {
        LabelSource.REF_PRESENT.value,
        LabelSource.REF_SYMMETRIC_SWAP.value,
        LabelSource.REF_ABSENT_REQUIRED.value,
        LabelSource.WRONG_OBSERVED.value,
        LabelSource.FORBIDDEN_NEGATIVE.value,
        LabelSource.NEGATIVE_RANDOM.value,
        # Phase C · Stage 1 gate: 没有这一 source 训练 → 模型仍学不到
        # "wire-两端同 net 是 positive" 这一模式。
        LabelSource.WIRE_SAME_NET_POSITIVE.value,
    }
    for src in required:
        assert manifest["by_source"][src] > 0, (
            f"required LabelSource {src!r} never fired — perturbation mix "
            f"would not exercise that path at production scale"
        )


def test_p1_acceptance_scaled_runs_end_to_end_parallel(tmp_path: Path) -> None:
    """Same scaled run with workers=3 — asserts the parallel path produces
    a manifest with identical aggregate stats to the serial path (modulo
    timing fields). Per the cross-process determinism contract."""

    cfg = _scaled_config(tmp_path)
    out_serial = tmp_path / "serial"
    out_parallel = tmp_path / "parallel"

    main(["--output-dir", str(out_serial), "--config", str(cfg),
          "--base-seed", "0", "--workers", "1"])
    main(["--output-dir", str(out_parallel), "--config", str(cfg),
          "--base-seed", "0", "--workers", "3"])

    m_serial = json.loads((out_serial / "manifest.json").read_text())
    m_parallel = json.loads((out_parallel / "manifest.json").read_text())

    # Strip non-deterministic / ratio fields then compare
    for k in (
        "n_processed", "n_skipped_failures", "total_samples",
        "total_positives", "total_negatives", "n_groups",
        "n_groups_without_positive",
        "n_skipped_missing_component", "n_skipped_optional_pin",
        "n_skipped_forbidden_pin_no_violation", "n_skipped_extract_error",
    ):
        assert m_serial[k] == m_parallel[k], f"{k} differs serial vs parallel"
    assert m_serial["by_source"] == m_parallel["by_source"]
    assert m_serial["by_task_type"] == m_parallel["by_task_type"]


def test_p1_acceptance_splits_held_out_topology(tmp_path: Path) -> None:
    """Splits must hold the opamp_buffer ref entirely out of train+val
    (plan §五 generalisation constraint)."""

    cfg = _scaled_config(tmp_path)
    out = tmp_path / "ds"
    main(["--output-dir", str(out), "--config", str(cfg), "--base-seed", "0"])

    train = json.loads((out / "splits" / "train.json").read_text())
    val = json.loads((out / "splits" / "val.json").read_text())
    test = json.loads((out / "splits" / "test.json").read_text())

    assert all(s.startswith("opamp_buffer/") for s in test)
    assert all(not s.startswith("opamp_buffer/") for s in train)
    assert all(not s.startswith("opamp_buffer/") for s in val)
    # Every label file referenced by every split entry exists on disk
    labels = out / "labels"
    for entry in (*train, *val, *test):
        ref_id, sid = entry.split("/", 1)
        assert (labels / ref_id / f"{sid}.json").is_file(), (
            f"split entry {entry} missing label file"
        )
