"""P2 · PyG FlatSealDataset tests.

End-to-end: P1 dataset_builder writes labels/<ref>/<sample>.json →
FlatSealDataset yields one PyG Data per SealSample row → DataLoader
batches → ready for SEAL DGCNN head.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.loader import DataLoader  # noqa: E402

from app.domain.gnn import (  # noqa: E402
    DatasetSpec,
    PerturbationPlan,
    RefSpec,
    generate_dataset,
)
from app.domain.gnn.pyg_dataset import (  # noqa: E402
    FlatSealDataset,
    RefEntry,
    RefRegistry,
    reconstruct_cur_hcg,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "references"


def _generate_small_dataset(tmp_path: Path) -> tuple[Path, RefRegistry, list[str]]:
    """Run a tiny dataset_builder pass and build a matching RefRegistry."""

    spec = DatasetSpec(
        refs=(
            RefSpec(
                ref_id="rc", payload_path=FIXTURES / "test_rc_v1.json"
            ),
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
            counts={"identity": 2, "wrong_connection": 2, "extra_component": 1}
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

    # Build entries from all labels (deterministic)
    labels_dir = tmp_path / "ds" / "labels"
    entries = sorted(
        f"{p.parent.name}/{p.stem}"
        for p in labels_dir.rglob("*.json")
    )
    return labels_dir, reg, entries


# ---------------------------------------------------------------------------
# RefRegistry
# ---------------------------------------------------------------------------


def test_ref_registry_caches_ref_hcg() -> None:
    reg = RefRegistry()
    reg.register(RefEntry("rc", FIXTURES / "test_rc_v1.json"))
    a = reg.ref_hcg("rc")
    b = reg.ref_hcg("rc")
    assert a is b, "ref_hcg cache must return the same object instance"


def test_ref_registry_populate_helper_handles_subtype_dict() -> None:
    reg = RefRegistry()
    reg.populate(
        [
            {"ref_id": "opamp", "payload_path": str(FIXTURES / "test_opamp_buffer_v1.json")},
        ],
        {"opamp": {"U1": "UA741"}},
    )
    assert reg.entries["opamp"].subtype_by_source_id == {"U1": "UA741"}
    hcg = reg.ref_hcg("opamp")
    # UA741 buffer should fully materialise 8 ports (incl FORBIDDEN/OPTIONAL)
    assert len(hcg.ports) == 8


# ---------------------------------------------------------------------------
# reconstruct_cur_hcg — replays the perturbation deterministically
# ---------------------------------------------------------------------------


def test_reconstruct_cur_hcg_matches_dataset_builder_output(
    tmp_path: Path,
) -> None:
    labels_dir, reg, entries = _generate_small_dataset(tmp_path)
    # Pick the first entry, replay, and verify the cur_hcg shape matches
    # what dataset_builder saw (we infer from label payload's subgraph
    # references: every subgraph anchor port must exist in cur_hcg).
    entry = entries[0]
    ref_id, sample_id = entry.split("/", 1)
    payload = json.loads(
        (labels_dir / ref_id / f"{sample_id}.json").read_text(encoding="utf-8")
    )
    cur_hcg = reconstruct_cur_hcg(
        reg.ref_hcg(ref_id),
        payload["cur_metadata"],
        subtype_by_source_id=(
            reg.entries[ref_id].subtype_by_source_id or None
        ),
    )
    # Every anchor port mentioned in any subgraph must exist in cur_hcg.ports
    for s in payload["samples"]:
        port_id = s["candidate_edge"][0]
        assert port_id in cur_hcg.ports, (
            f"replayed cur_hcg is missing port {port_id} from sample "
            f"{s.get('label_source')}"
        )


# ---------------------------------------------------------------------------
# FlatSealDataset
# ---------------------------------------------------------------------------


def test_flat_seal_dataset_yields_one_data_per_row(tmp_path: Path) -> None:
    labels_dir, reg, entries = _generate_small_dataset(tmp_path)
    ds = FlatSealDataset(labels_dir, reg, entries)
    assert len(ds) > 0
    # Total rows = sum of len(payload['samples']) across all entries
    expected_total = 0
    for entry in entries:
        ref_id, sid = entry.split("/", 1)
        payload = json.loads(
            (labels_dir / ref_id / f"{sid}.json").read_text(encoding="utf-8")
        )
        expected_total += len(payload["samples"])
    assert len(ds) == expected_total


def test_flat_seal_dataset_data_has_expected_attrs(tmp_path: Path) -> None:
    labels_dir, reg, entries = _generate_small_dataset(tmp_path)
    ds = FlatSealDataset(labels_dir, reg, entries)
    d = ds[0]
    # PyG required attrs
    assert d.x.dim() == 2
    assert d.edge_index.dim() == 2
    assert d.edge_index.shape[0] == 2
    # Our enrichments
    assert hasattr(d, "y")
    assert d.y.numel() == 1
    assert d.label_source in {
        "ref_present", "ref_symmetric_swap", "ref_absent_required",
        "wrong_observed", "forbidden_violated", "forbidden_negative",
        "negative_random", "negative_hard",
    }
    assert d.task_type in {"wrong_edge", "missing_edge"}
    assert d.ref_id in {e.split("/", 1)[0] for e in entries}
    assert isinstance(d.sample_id, str)


def test_flat_seal_dataset_is_compatible_with_pyg_dataloader(
    tmp_path: Path,
) -> None:
    labels_dir, reg, entries = _generate_small_dataset(tmp_path)
    ds = FlatSealDataset(labels_dir, reg, entries)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    n_batches = 0
    n_rows = 0
    for batch in loader:
        n_batches += 1
        n_rows += batch.num_graphs
        # x is concatenated, batch vector aligns rows to subgraph ids
        assert batch.x.shape[0] == batch.batch.shape[0]
        assert batch.batch.max().item() < batch.num_graphs
    assert n_rows == len(ds)
    assert n_batches > 0


def test_flat_seal_dataset_missing_label_file_raises(
    tmp_path: Path,
) -> None:
    labels_dir, reg, entries = _generate_small_dataset(tmp_path)
    bogus = entries + ["nope/does_not_exist"]
    with pytest.raises(FileNotFoundError):
        FlatSealDataset(labels_dir, reg, bogus)


def test_flat_seal_dataset_is_deterministic(tmp_path: Path) -> None:
    """Row order is determined by ``split_entries`` order (the
    ``__init__`` index iterates that list)."""

    labels_dir, reg, entries = _generate_small_dataset(tmp_path)
    ds1 = FlatSealDataset(labels_dir, reg, entries)
    ds2 = FlatSealDataset(labels_dir, reg, entries)
    assert len(ds1) == len(ds2)
    # First row across both must be the same SealSample
    d1 = ds1[0]
    d2 = ds2[0]
    assert d1.ref_id == d2.ref_id
    assert d1.sample_id == d2.sample_id
    assert d1.row_idx == d2.row_idx
    assert torch.equal(d1.x, d2.x)
    assert torch.equal(d1.edge_index, d2.edge_index)


def test_flat_seal_dataset_replays_with_correct_subtypes(
    tmp_path: Path,
) -> None:
    """For UA741, cur_hcg must include FORBIDDEN/OPTIONAL pin
    materialisation. Verify by picking an *identity* opamp sample (no
    component injection) and confirming exactly 8 ports were rebuilt.
    Other ops (extra_component / extra_wire_bridge / chained) can add
    parasitic components and bump the count above 8 — see
    test_flat_seal_dataset_replays_extra_component for that case."""

    labels_dir, reg, _entries = _generate_small_dataset(tmp_path)
    # Find an opamp identity sample specifically
    opamp_identity = labels_dir / "opamp" / "opamp__identity_0000.json"
    assert opamp_identity.is_file(), "expected identity opamp sample to exist"
    payload = json.loads(opamp_identity.read_text(encoding="utf-8"))
    cur_hcg = reconstruct_cur_hcg(
        reg.ref_hcg("opamp"),
        payload["cur_metadata"],
        subtype_by_source_id=reg.entries["opamp"].subtype_by_source_id,
    )
    assert len(cur_hcg.ports) == 8, (
        f"opamp identity cur should have 8 ports (FORBIDDEN + OPTIONAL + "
        f"REQUIRED), got {len(cur_hcg.ports)} — subtype override likely "
        f"didn't reach reconstruct_cur_hcg"
    )
    # And FORBIDDEN pin 8 must exist as a floating port
    forbidden = [
        p for p in cur_hcg.ports.values()
        if p.connection_policy == "forbidden"
    ]
    assert len(forbidden) == 1
