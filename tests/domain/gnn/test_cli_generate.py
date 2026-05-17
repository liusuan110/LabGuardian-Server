"""P1 Phase C · CLI smoke tests for scripts.gnn_generate_dataset.

Drives the entrypoint with a tiny config and checks the on-disk
artifacts (labels + manifest + splits) match the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.gnn_generate_dataset import (
    DEFAULT_CONFIG,
    build_dataset_spec,
    build_split_spec,
    main,
)

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "references"


def _write_tiny_config(tmp_path: Path) -> Path:
    cfg = {
        "refs": [
            {
                "ref_id": "rc",
                "payload_path": str(FIXTURES_DIR / "test_rc_v1.json"),
            },
            {
                "ref_id": "div",
                "payload_path": str(FIXTURES_DIR / "test_voltage_divider_v1.json"),
            },
            {
                "ref_id": "opamp",
                "payload_path": str(FIXTURES_DIR / "test_opamp_buffer_v1.json"),
            },
        ],
        "plan": {
            "counts": {
                "identity": 1,
                "wrong_connection": 1,
            }
        },
        "test_ref_ids": ["opamp"],
        "val_fraction": 0.25,
        "enforce_healthy": False,
        "checkpoint_every": 100,
        "subtypes_by_ref_id": {"opamp": {"U1": "UA741"}},
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def test_cli_default_config_resolves_to_real_fixtures() -> None:
    """The built-in MVP config must point at fixtures that actually exist
    on disk."""

    for r in DEFAULT_CONFIG["refs"]:
        assert Path(r["payload_path"]).is_file(), (
            f"DEFAULT_CONFIG ref {r['ref_id']} points at missing fixture: "
            f"{r['payload_path']}"
        )


def test_build_dataset_spec_propagates_kwargs(tmp_path: Path) -> None:
    cfg = {
        "refs": [
            {"ref_id": "rc", "payload_path": str(FIXTURES_DIR / "test_rc_v1.json")},
        ],
        "plan": {"counts": {"identity": 3}},
        "negatives_per_positive": 2.5,
        "forbidden_negative_samples": 7,
        "missing_edge_group_size": 4,
        "include_optional": True,
        "num_hops": 3,
        "checkpoint_every": 33,
        "enforce_healthy": False,
        "subtypes_by_ref_id": {},
    }
    spec = build_dataset_spec(cfg, tmp_path / "out", base_seed=11)
    assert spec.base_seed == 11
    assert spec.negatives_per_positive == 2.5
    assert spec.forbidden_negative_samples == 7
    assert spec.missing_edge_group_size == 4
    assert spec.include_optional is True
    assert spec.num_hops == 3
    assert spec.checkpoint_every == 33
    assert spec.enforce_healthy is False
    assert spec.refs[0].ref_id == "rc"


def test_build_split_spec_picks_up_seed_from_main_arg() -> None:
    cfg = {"test_ref_ids": ["opamp"], "val_fraction": 0.2}
    s = build_split_spec(cfg, base_seed=99)
    assert s.test_ref_ids == ("opamp",)
    assert s.val_fraction == 0.2
    assert s.seed == 99


def test_cli_end_to_end_creates_labels_manifest_splits(tmp_path: Path) -> None:
    cfg_path = _write_tiny_config(tmp_path)
    out = tmp_path / "ds"
    rc = main(
        [
            "--output-dir",
            str(out),
            "--config",
            str(cfg_path),
        ]
    )
    assert rc == 0
    # labels: 3 refs × 2 ops × 1 sample = 6
    label_files = list((out / "labels").rglob("*.json"))
    assert len(label_files) == 6
    # manifest
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_processed"] == 6
    # splits: opamp = test, rc+div = train/val
    train = json.loads((out / "splits" / "train.json").read_text())
    val = json.loads((out / "splits" / "val.json").read_text())
    test = json.loads((out / "splits" / "test.json").read_text())
    assert all(s.startswith("opamp/") for s in test)
    assert all(not s.startswith("opamp/") for s in train)
    assert all(not s.startswith("opamp/") for s in val)
    assert len(test) + len(train) + len(val) == 6


def test_cli_resume_idempotent(tmp_path: Path) -> None:
    cfg_path = _write_tiny_config(tmp_path)
    out = tmp_path / "ds"
    assert main(["--output-dir", str(out), "--config", str(cfg_path)]) == 0
    rc2 = main(
        ["--output-dir", str(out), "--config", str(cfg_path), "--resume"]
    )
    assert rc2 == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_processed"] == 6


def test_cli_returns_nonzero_on_bad_config(tmp_path: Path) -> None:
    bad_cfg = tmp_path / "bad.json"
    bad_cfg.write_text(
        json.dumps(
            {
                "refs": [
                    {
                        "ref_id": "missing",
                        "payload_path": str(tmp_path / "does_not_exist.json"),
                    }
                ],
                "plan": {"counts": {"identity": 1}},
                "test_ref_ids": [],
                "enforce_healthy": False,
            }
        )
    )
    rc = main(
        ["--output-dir", str(tmp_path / "ds"), "--config", str(bad_cfg)]
    )
    assert rc == 2  # spec validation failure


def test_cli_skip_splits_omits_splits_dir(tmp_path: Path) -> None:
    cfg_path = _write_tiny_config(tmp_path)
    out = tmp_path / "ds"
    main(
        [
            "--output-dir",
            str(out),
            "--config",
            str(cfg_path),
            "--skip-splits",
        ]
    )
    assert not (out / "splits").exists()
    # labels + manifest still written
    assert (out / "manifest.json").is_file()


def test_cli_help_does_not_crash() -> None:
    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
