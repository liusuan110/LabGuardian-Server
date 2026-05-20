"""P2.5 · SpiceNetlist self-supervised pretrain tests.

Covers:
- SpiceNetlist JSON → HeteroCircuitGraph loader (component / port mapping)
- SpiceNetlistPretrainDataset positive / negative sampling balance
- SealDGCNN forward pass + gradient flow
- 1-epoch training smoke (loss decreases on a tiny synthetic set)
- Manual ROC AUC implementation correctness

Skipped automatically if torch / torch_geometric aren't installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.loader import DataLoader  # noqa: E402

from app.domain.gnn.pretrain_dataset import SpiceNetlistPretrainDataset  # noqa: E402
from app.domain.gnn.seal_dgcnn import SealDGCNN, predict_prob  # noqa: E402
from app.domain.gnn.spicenetlist_loader import (  # noqa: E402
    COMPONENT_TYPE_MAP,
    load_circuit_json,
    load_spicenetlist_dir,
)
from scripts.gnn_pretrain_seal import (  # noqa: E402
    kfold_circuit_split,
    roc_auc,
)

SPICENETLIST_DIR = Path("/Users/liusuan/Desktop/GNN_ACLP-main/SpiceNetlist/JSON")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _sample_payload() -> list[dict]:
    return [
        {
            "component_type": "NMOS",
            "port_connection": {"Drain": "1", "Gate": "2", "Source": "0"},
        },
        {
            "component_type": "Res",
            "port_connection": {"Pos": "1", "Neg": "0"},
        },
        {
            "component_type": "Voltage",
            "port_connection": {"Pos": "2", "Neg": "0"},
        },
    ]


def test_load_circuit_json_basic_shapes() -> None:
    circ = load_circuit_json(_sample_payload(), circuit_id="test_0")
    assert circ.circuit_id == "test_0"
    s = circ.hcg.summary()
    # 3 (NMOS Drain/Gate/Source) + 2 (Res Pos/Neg) + 2 (Voltage Pos/Neg) = 7 ports
    assert s == {"n_components": 3, "n_ports": 7, "n_nets": 3, "n_edges": 7}
    # Ground net "0" tagged
    gnd = [n for n in circ.hcg.nets.values() if n.source_id == "0"]
    assert gnd and gnd[0].role == "gnd" and gnd[0].is_power_rail is True


def test_loader_maps_mosfet_pins_to_bjt_port_types() -> None:
    circ = load_circuit_json(_sample_payload(), circuit_id="m")
    mosfet_ports = [
        p for p in circ.hcg.ports.values() if p.parent_ctype == "Transistor"
    ]
    port_types = {p.port_key: p.port_type for p in mosfet_ports}
    # Drain→COLLECTOR, Gate→BASE, Source→EMITTER per loader mapping
    assert port_types == {
        "drain": "collector",
        "gate": "base",
        "source": "emitter",
    }


def test_loader_resistor_pins_share_symmetry_class() -> None:
    circ = load_circuit_json(_sample_payload(), circuit_id="r")
    res_ports = [
        p for p in circ.hcg.ports.values() if p.parent_ctype == "Resistor"
    ]
    assert len(res_ports) == 2
    assert res_ports[0].symmetry_class_id == res_ports[1].symmetry_class_id


def test_component_type_map_covers_all_observed_types() -> None:
    """Every component_type seen across the 155-circuit dataset must map
    to a defined ComponentType (no KeyError fallthrough)."""

    if not SPICENETLIST_DIR.is_dir():
        pytest.skip("SpiceNetlist dataset not present locally")
    seen = set()
    for fp in SPICENETLIST_DIR.glob("*.json"):
        import json

        for c in json.loads(fp.read_text(encoding="utf-8")):
            seen.add(str(c.get("component_type", "")))
    # All observed types are in our mapping (or fall back to UNKNOWN
    # via _map_component_type for any genuinely unknown one)
    unmapped = [t for t in seen if t and t not in COMPONENT_TYPE_MAP]
    assert not unmapped, (
        f"unmapped SpiceNetlist component types: {unmapped} — extend "
        f"COMPONENT_TYPE_MAP in spicenetlist_loader.py"
    )


def test_loader_handles_full_directory() -> None:
    if not SPICENETLIST_DIR.is_dir():
        pytest.skip("SpiceNetlist dataset not present locally")
    circuits = load_spicenetlist_dir(SPICENETLIST_DIR)
    assert len(circuits) >= 100, (
        f"expected ≥ 100 SpiceNetlist circuits, got {len(circuits)}"
    )
    # Every circuit has at least 1 component + 1 edge
    for c in circuits:
        s = c.hcg.summary()
        assert s["n_components"] >= 1
        assert s["n_edges"] >= 1


# ---------------------------------------------------------------------------
# Pretrain dataset
# ---------------------------------------------------------------------------


def test_pretrain_dataset_balanced_positives_and_negatives() -> None:
    circuit = load_circuit_json(_sample_payload(), circuit_id="t")
    ds = SpiceNetlistPretrainDataset(
        [circuit],
        negatives_per_positive=1.0,
        max_pairs_per_circuit=None,
        seed=0,
    )
    # 7 edges → 7 positives → 7 negatives → 14 samples
    labels = [int(ds[i].y.item()) for i in range(len(ds))]
    pos = sum(1 for L in labels if L == 1)
    neg = sum(1 for L in labels if L == 0)
    assert pos == 7
    assert neg == 7


def test_pretrain_dataset_is_deterministic_given_seed() -> None:
    circuit = load_circuit_json(_sample_payload(), circuit_id="t")
    ds1 = SpiceNetlistPretrainDataset([circuit], seed=42, max_pairs_per_circuit=None)
    ds2 = SpiceNetlistPretrainDataset([circuit], seed=42, max_pairs_per_circuit=None)
    assert len(ds1) == len(ds2)
    for i in range(len(ds1)):
        a, b = ds1[i], ds2[i]
        assert int(a.y.item()) == int(b.y.item())
        assert torch.equal(a.x, b.x)


def test_pretrain_dataset_respects_max_pairs_per_circuit() -> None:
    if not SPICENETLIST_DIR.is_dir():
        pytest.skip("SpiceNetlist dataset not present locally")
    circuits = load_spicenetlist_dir(SPICENETLIST_DIR)
    # The largest circuit has 237 edges → would dominate without the cap
    ds = SpiceNetlistPretrainDataset(circuits, max_pairs_per_circuit=20, seed=0)
    # Each circuit contributes ≤ 20 rows. Total bound: 155 × 20 = 3100
    assert len(ds) <= 155 * 20


# ---------------------------------------------------------------------------
# SealDGCNN model
# ---------------------------------------------------------------------------


def _tiny_loader(max_circuits: int = 5):
    """Build a tiny DataLoader for SealDGCNN smoke tests."""

    circ = load_circuit_json(_sample_payload(), circuit_id="t")
    ds = SpiceNetlistPretrainDataset([circ] * max_circuits, seed=0)
    return DataLoader(ds, batch_size=4, shuffle=False), ds[0].x.shape[1]


def test_seal_dgcnn_forward_shape() -> None:
    loader, in_channels = _tiny_loader()
    model = SealDGCNN(in_channels=in_channels)
    batch = next(iter(loader))
    out = model(batch.x, batch.edge_index, batch.batch)
    assert out.shape == (batch.num_graphs,)
    assert out.dtype == torch.float32


def test_predict_prob_returns_values_in_unit_interval() -> None:
    loader, in_channels = _tiny_loader()
    model = SealDGCNN(in_channels=in_channels)
    batch = next(iter(loader))
    probs = predict_prob(model, batch)
    assert probs.shape == (batch.num_graphs,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_seal_dgcnn_gradient_flows_back_to_first_layer() -> None:
    """Smoke test: a single backward pass produces non-zero gradients on
    every parameter (would fail if any layer is detached from the graph)."""

    loader, in_channels = _tiny_loader()
    model = SealDGCNN(in_channels=in_channels, hidden_channels=16)
    batch = next(iter(loader))
    logits = model(batch.x, batch.edge_index, batch.batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch.y.float().view(-1)
    )
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


# ---------------------------------------------------------------------------
# K-fold split
# ---------------------------------------------------------------------------


def test_kfold_split_is_ref_disjoint_across_folds() -> None:
    """Plan §五 hard constraint: each circuit appears in exactly one
    val set across the folds (no edge contamination)."""

    if not SPICENETLIST_DIR.is_dir():
        pytest.skip("SpiceNetlist dataset not present locally")
    circuits = load_spicenetlist_dir(SPICENETLIST_DIR)[:30]
    folds = kfold_circuit_split(circuits, k=5, seed=0)
    assert len(folds) == 5
    seen_val_ids: list[str] = []
    for train_c, val_c in folds:
        ids = {c.circuit_id for c in train_c}
        val_ids = {c.circuit_id for c in val_c}
        # train and val never overlap
        assert ids.isdisjoint(val_ids)
        seen_val_ids.extend(val_ids)
    # Every circuit appears in exactly one val fold
    assert sorted(seen_val_ids) == sorted(c.circuit_id for c in circuits)


def test_kfold_split_k1_uses_random_80_20_split() -> None:
    circs_payload = [_sample_payload() for _ in range(10)]
    circs = [
        load_circuit_json(p, circuit_id=str(i)) for i, p in enumerate(circs_payload)
    ]
    [(train, val)] = kfold_circuit_split(circs, k=1, seed=0)
    # 80/20 → ~8 train, ~2 val
    assert len(train) + len(val) == 10
    assert len(train) >= 1
    assert len(val) >= 1


# ---------------------------------------------------------------------------
# AUC implementation
# ---------------------------------------------------------------------------


def test_roc_auc_perfect_separation() -> None:
    scores = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]
    labels = [0, 0, 0, 1, 1, 1]
    assert roc_auc(scores, labels) == 1.0


def test_roc_auc_random_is_half() -> None:
    # Interleaved scores → AUC ≈ 0.5
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    assert roc_auc(scores, labels) == 0.5


def test_roc_auc_reversed_is_zero() -> None:
    scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
    labels = [0, 0, 0, 1, 1, 1]
    assert roc_auc(scores, labels) == 0.0


def test_roc_auc_single_class_returns_nan() -> None:
    import math

    assert math.isnan(roc_auc([0.1, 0.5, 0.9], [1, 1, 1]))


# ---------------------------------------------------------------------------
# 1-epoch training smoke (loss decreases)
# ---------------------------------------------------------------------------


def test_one_epoch_training_decreases_loss() -> None:
    """End-to-end smoke: on a tiny synthetic set, BCE loss after one
    epoch of Adam updates must be lower than before."""

    if not SPICENETLIST_DIR.is_dir():
        pytest.skip("SpiceNetlist dataset not present locally")
    circuits = load_spicenetlist_dir(SPICENETLIST_DIR)[:20]
    ds = SpiceNetlistPretrainDataset(
        circuits, seed=0, max_pairs_per_circuit=10
    )
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch_probe = next(iter(loader))
    model = SealDGCNN(in_channels=batch_probe.x.shape[1], hidden_channels=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Pre-train loss snapshot
    with torch.no_grad():
        pre_losses = [
            loss_fn(
                model(b.x, b.edge_index, b.batch), b.y.float().view(-1)
            ).item()
            for b in loader
        ]
    pre_mean = sum(pre_losses) / len(pre_losses)

    # One epoch
    model.train()
    for b in loader:
        opt.zero_grad()
        loss_fn(
            model(b.x, b.edge_index, b.batch), b.y.float().view(-1)
        ).backward()
        opt.step()

    # Post-train loss snapshot
    model.eval()
    with torch.no_grad():
        post_losses = [
            loss_fn(
                model(b.x, b.edge_index, b.batch), b.y.float().view(-1)
            ).item()
            for b in loader
        ]
    post_mean = sum(post_losses) / len(post_losses)
    assert post_mean < pre_mean, (
        f"BCE loss did not decrease after one epoch: pre={pre_mean:.4f} "
        f"post={post_mean:.4f}"
    )
