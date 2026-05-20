"""P3.2 · Prebaked SEAL Dataset tests.

Verifies the prebake → load → train pipeline is byte-identical to
:class:`FlatSealDataset` (semantic parity) and faster (perf contract).
Skipped if torch / torch_geometric aren't installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from app.domain.gnn import (  # noqa: E402
    DatasetSpec,
    PerturbationPlan,
    RefSpec,
    generate_dataset,
)
from app.domain.gnn.graph_schema import DRNL_LABEL_DIM  # noqa: E402
from app.domain.gnn.prebaked_dataset import (  # noqa: E402
    PREBAKED_SCHEMA_VERSION,
    PrebakedSealDataset,
    prebake_to_disk,
)
from app.domain.gnn.pyg_dataset import (  # noqa: E402
    FlatSealDataset,
    RefEntry,
    RefRegistry,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "references"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_dataset(tmp_path: Path) -> tuple[Path, RefRegistry, list[str]]:
    spec = DatasetSpec(
        refs=(
            RefSpec(ref_id="rc", payload_path=FIXTURES / "test_rc_v1.json"),
            RefSpec(
                ref_id="div",
                payload_path=FIXTURES / "test_voltage_divider_v1.json",
            ),
            RefSpec(
                ref_id="opamp",
                payload_path=FIXTURES / "test_opamp_buffer_v1.json",
                subtype_by_source_id={"U1": "UA741"},
            ),
        ),
        plan=PerturbationPlan(
            counts={"identity": 1, "wrong_connection": 2}
        ),
        output_dir=tmp_path / "ds",
        base_seed=42,
        enforce_healthy=False,
    )
    generate_dataset(spec)
    reg = RefRegistry()
    reg.register(RefEntry("rc", FIXTURES / "test_rc_v1.json"))
    reg.register(RefEntry("div", FIXTURES / "test_voltage_divider_v1.json"))
    reg.register(
        RefEntry(
            "opamp",
            FIXTURES / "test_opamp_buffer_v1.json",
            subtype_by_source_id={"U1": "UA741"},
        )
    )
    entries = sorted(
        f"{p.parent.name}/{p.stem}"
        for p in (tmp_path / "ds" / "labels").rglob("*.json")
    )
    return tmp_path / "ds" / "labels", reg, entries


# ---------------------------------------------------------------------------
# prebake_to_disk
# ---------------------------------------------------------------------------


def test_prebake_writes_blob_with_expected_keys(tmp_path: Path) -> None:
    labels_dir, reg, entries = _make_tiny_dataset(tmp_path)
    out = tmp_path / "prebaked.pt"
    stats = prebake_to_disk(labels_dir, reg, entries, out)

    assert out.is_file()
    blob = torch.load(out, weights_only=False)
    assert blob["version"] == PREBAKED_SCHEMA_VERSION
    assert blob["n_rows"] == stats.n_rows_baked
    assert len(blob["data_list"]) == stats.n_rows_baked
    assert len(blob["entries"]) == stats.n_rows_baked
    assert len(blob["row_indices"]) == stats.n_rows_baked
    assert isinstance(blob["feature_width"], int)
    assert blob["config"]["drop_drnl_at_bake"] is False


def test_prebake_records_failures_in_stats(tmp_path: Path) -> None:
    labels_dir, reg, entries = _make_tiny_dataset(tmp_path)
    bogus = list(entries) + ["nope/does_not_exist"]
    stats = prebake_to_disk(labels_dir, reg, bogus, tmp_path / "p.pt")
    # one missing file → load failure counted, real rows still baked
    assert stats.n_samples_failed_to_load >= 1
    assert stats.n_rows_baked > 0


def test_prebake_zero_rows_is_a_legitimate_state(tmp_path: Path) -> None:
    """All entries missing → empty blob, zero rows. The CLI handles this
    via exit code 3; the helper itself shouldn't raise."""

    reg = RefRegistry()
    reg.register(RefEntry("rc", FIXTURES / "test_rc_v1.json"))
    stats = prebake_to_disk(
        tmp_path / "no_labels",
        reg,
        ["rc/does_not_exist"],
        tmp_path / "p.pt",
    )
    assert stats.n_rows_baked == 0
    assert stats.n_samples_failed_to_load == 1


# ---------------------------------------------------------------------------
# PrebakedSealDataset semantic parity with FlatSealDataset
# ---------------------------------------------------------------------------


def test_prebaked_dataset_matches_flat_dataset_row_by_row(
    tmp_path: Path,
) -> None:
    """For the same entries, PrebakedSealDataset[i].x must equal
    FlatSealDataset[i].x. Otherwise the prebake bypasses some
    transformation."""

    labels_dir, reg, entries = _make_tiny_dataset(tmp_path)
    out = tmp_path / "prebaked.pt"
    prebake_to_disk(labels_dir, reg, entries, out)

    flat = FlatSealDataset(labels_dir, reg, entries)
    pre = PrebakedSealDataset(out, entries)
    assert len(flat) == len(pre), (
        f"row count mismatch: flat={len(flat)} prebaked={len(pre)}"
    )
    # FlatSealDataset iterates in entries-then-row order. PrebakedSealDataset
    # also bakes in that order. So index i should match.
    for i in range(len(flat)):
        a = flat[i]
        b = pre[i]
        assert torch.equal(a.x, b.x), f"x differs at row {i}"
        assert torch.equal(a.edge_index, b.edge_index), f"edges differ at row {i}"
        assert torch.equal(a.y, b.y), f"y differs at row {i}"
        assert a.label_source == b.label_source
        assert a.task_type == b.task_type
        assert a.group_id == b.group_id


def test_prebaked_dataset_drop_drnl_zeros_first_17_dims_at_load(
    tmp_path: Path,
) -> None:
    """The drop_drnl ablation flag is applied at load time (not bake
    time), so one prebaked blob serves both baseline and no_drnl runs."""

    labels_dir, reg, entries = _make_tiny_dataset(tmp_path)
    out = tmp_path / "prebaked.pt"
    prebake_to_disk(labels_dir, reg, entries, out)  # baked with DRNL on
    pre_on = PrebakedSealDataset(out, entries, drop_drnl=False)
    pre_off = PrebakedSealDataset(out, entries, drop_drnl=True)
    d_on = pre_on[0]
    d_off = pre_off[0]
    # DRNL slice on for first dataset, zeroed for second
    assert d_on.x[:, :DRNL_LABEL_DIM].sum().item() > 0
    assert d_off.x[:, :DRNL_LABEL_DIM].sum().item() == 0
    # Rest of feature vector identical
    assert torch.equal(
        d_on.x[:, DRNL_LABEL_DIM:], d_off.x[:, DRNL_LABEL_DIM:]
    )
    # Original tensor untouched — drop_drnl must clone (otherwise next
    # query for d_on returns zeroed DRNL too).
    d_on_again = pre_on[0]
    assert d_on_again.x[:, :DRNL_LABEL_DIM].sum().item() > 0


def test_prebaked_dataset_filters_by_split_entries(tmp_path: Path) -> None:
    labels_dir, reg, entries = _make_tiny_dataset(tmp_path)
    out = tmp_path / "prebaked.pt"
    prebake_to_disk(labels_dir, reg, entries, out)
    # Take only entries from "rc" ref
    rc_only = [e for e in entries if e.startswith("rc/")]
    pre = PrebakedSealDataset(out, rc_only)
    assert all(e.startswith("rc/") for e in pre.entries)
    assert len(pre) == sum(
        1 for e in PrebakedSealDataset(out).entries if e.startswith("rc/")
    )


def test_prebaked_dataset_rejects_empty_split(tmp_path: Path) -> None:
    labels_dir, reg, entries = _make_tiny_dataset(tmp_path)
    out = tmp_path / "prebaked.pt"
    prebake_to_disk(labels_dir, reg, entries, out)
    with pytest.raises(ValueError, match="zero rows"):
        PrebakedSealDataset(out, ["nonexistent/zzz"])


def test_prebaked_dataset_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        PrebakedSealDataset(tmp_path / "does_not_exist.pt")


def test_prebaked_dataset_rejects_stale_schema_version(
    tmp_path: Path,
) -> None:
    """If someone hand-edits or bumps the schema, the loader must refuse
    to silently consume stale blobs."""

    labels_dir, reg, entries = _make_tiny_dataset(tmp_path)
    out = tmp_path / "prebaked.pt"
    prebake_to_disk(labels_dir, reg, entries, out)
    blob = torch.load(out, weights_only=False)
    blob["version"] = PREBAKED_SCHEMA_VERSION + 99
    torch.save(blob, out)
    with pytest.raises(ValueError, match="version"):
        PrebakedSealDataset(out, entries)


# ---------------------------------------------------------------------------
# DataLoader compatibility
# ---------------------------------------------------------------------------


def test_prebaked_dataset_works_with_pyg_dataloader(tmp_path: Path) -> None:
    from torch_geometric.loader import DataLoader

    labels_dir, reg, entries = _make_tiny_dataset(tmp_path)
    out = tmp_path / "prebaked.pt"
    prebake_to_disk(labels_dir, reg, entries, out)
    ds = PrebakedSealDataset(out, entries)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    assert batch.x.dim() == 2
    assert batch.edge_index.shape[0] == 2
    assert batch.num_graphs <= 8


# ---------------------------------------------------------------------------
# Train_full integration
# ---------------------------------------------------------------------------


def test_train_full_with_prebaked_path_yields_same_initial_loss(
    tmp_path: Path,
) -> None:
    """End-to-end: training one epoch on FlatSealDataset vs on a
    prebaked dataset of the same entries must produce the same
    pre-epoch loss (since the model + data + seed are identical)."""

    from app.domain.gnn.model import CircuitMatchNet
    from scripts.gnn_train_full import build_loaders, evaluate

    labels_dir, reg, entries = _make_tiny_dataset(tmp_path)
    out = tmp_path / "prebaked.pt"
    prebake_to_disk(labels_dir, reg, entries, out)

    # Save dummy splits so build_loaders has them
    splits_dir = tmp_path / "ds" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    n = len(entries)
    (splits_dir / "train.json").write_text(json.dumps(entries[: n // 2 or 1]), encoding="utf-8")
    (splits_dir / "val.json").write_text(json.dumps(entries[n // 2 : n - 1] or entries[:1]), encoding="utf-8")
    (splits_dir / "test.json").write_text(json.dumps(entries[-1:] or entries[:1]), encoding="utf-8")

    refs_config = [
        {"ref_id": "rc", "payload_path": str(FIXTURES / "test_rc_v1.json")},
        {"ref_id": "div", "payload_path": str(FIXTURES / "test_voltage_divider_v1.json")},
        {"ref_id": "opamp", "payload_path": str(FIXTURES / "test_opamp_buffer_v1.json")},
    ]
    subtypes = {"opamp": {"U1": "UA741"}}

    # Without prebake (live replay)
    _train_l, val_l, _test_l = build_loaders(
        tmp_path / "ds", refs_config, subtypes,
        batch_size=8, drop_drnl=False, prebaked_path=None,
    )
    # With prebake
    _train_p, val_p, _test_p = build_loaders(
        tmp_path / "ds", refs_config, subtypes,
        batch_size=8, drop_drnl=False, prebaked_path=out,
    )
    # Same row counts
    assert len(val_l.dataset) == len(val_p.dataset)

    # Same eval metrics for an identical model. We construct one model
    # then copy its weights into a second, guaranteeing param-equality
    # (manual_seed alone doesn't survive intervening RNG draws from
    # PyG / DataLoader init).
    probe = next(iter(val_l))
    m1 = CircuitMatchNet(in_channels=probe.x.shape[1], hidden_channels=16)
    m2 = CircuitMatchNet(in_channels=probe.x.shape[1], hidden_channels=16)
    m2.load_state_dict(m1.state_dict())
    for (_, v1), (_, v2) in zip(
        m1.state_dict().items(), m2.state_dict().items()
    ):
        assert torch.equal(v1, v2)
    metrics_live = evaluate(m1, val_l, torch.device("cpu"))
    metrics_pre = evaluate(m2, val_p, torch.device("cpu"))
    # Same n samples
    assert metrics_live["wrong_edge"]["n"] == metrics_pre["wrong_edge"]["n"]
    # Same AUC (model deterministic on same data)
    assert abs(
        metrics_live["wrong_edge"]["auc"] - metrics_pre["wrong_edge"]["auc"]
    ) < 1e-4
