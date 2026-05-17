"""P3 · CircuitMatchNet + training scaffolding tests.

Covers:
- Model forward pass + dict shape
- Backbone load from a P2.5 checkpoint
- Checkpoint save / load round-trip
- F1 / top-k metrics correctness
- 1-epoch training smoke on a tiny subset of P1 dataset
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.loader import DataLoader  # noqa: E402

from app.domain.gnn import DatasetSpec, PerturbationPlan, RefSpec, generate_dataset  # noqa: E402
from app.domain.gnn.model import CircuitMatchNet  # noqa: E402
from app.domain.gnn.pyg_dataset import FlatSealDataset, RefEntry, RefRegistry  # noqa: E402
from app.domain.gnn.seal_dgcnn import SealDGCNN  # noqa: E402
from scripts.gnn_train_full import (  # noqa: E402
    evaluate,
    f1_at_threshold,
    top_k_accuracy,
    train_one_epoch,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "references"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_p1_dataset(tmp_path: Path) -> tuple[Path, RefRegistry, list[str]]:
    spec = DatasetSpec(
        refs=(
            RefSpec(ref_id="rc", payload_path=FIXTURES / "test_rc_v1.json"),
            RefSpec(ref_id="div", payload_path=FIXTURES / "test_voltage_divider_v1.json"),
            RefSpec(
                ref_id="opamp",
                payload_path=FIXTURES / "test_opamp_buffer_v1.json",
                subtype_by_source_id={"U1": "UA741"},
            ),
        ),
        plan=PerturbationPlan(counts={
            "identity": 1, "wrong_connection": 2, "pin_swap_symmetric": 1
        }),
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
# Model
# ---------------------------------------------------------------------------


def test_circuit_match_net_forward_returns_dict_with_seal_logits() -> None:
    m = CircuitMatchNet(in_channels=68)
    x = torch.randn(20, 68)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    b = torch.zeros(20, dtype=torch.long)
    out = m(x, ei, b)
    assert isinstance(out, dict)
    assert "seal_logits" in out
    assert out["seal_logits"].shape == (1,)


def test_circuit_match_net_save_and_load_round_trip(tmp_path: Path) -> None:
    m = CircuitMatchNet(in_channels=68, hidden_channels=16)
    path = tmp_path / "ckpt.pt"
    m.save(path, extra={"foo": "bar"})
    loaded = CircuitMatchNet.load(path)
    # Configs identical
    assert loaded.config == m.config
    # Weights identical (compare first GCN layer)
    p1 = next(m.seal_head.gcns[0].parameters())
    p2 = next(loaded.seal_head.gcns[0].parameters())
    assert torch.equal(p1, p2)


def test_circuit_match_net_loads_p2_5_backbone(tmp_path: Path) -> None:
    """Simulate a P2.5 checkpoint and verify weight transfer."""

    # Build a SealDGCNN, save it in the P2.5 format
    src = SealDGCNN(in_channels=68, hidden_channels=32, sort_k=30)
    ckpt_path = tmp_path / "fake_backbone.pt"
    torch.save({
        "state_dict": src.state_dict(),
        "hidden": 32,
        "sort_k": 30,
        "in_channels": 68,
        "best_val_auc": 0.95,
        "fold": 0,
    }, ckpt_path)
    # Load into CircuitMatchNet
    m = CircuitMatchNet.from_pretrained_backbone(ckpt_path, strict=True)
    # Weights of m.seal_head must equal src
    for (k1, v1), (k2, v2) in zip(
        m.seal_head.state_dict().items(), src.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(v1, v2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_f1_at_threshold_perfect_separation() -> None:
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    p, r, f = f1_at_threshold(scores, labels, threshold=0.5)
    assert p == 1.0
    assert r == 1.0
    assert f == 1.0


def test_f1_at_threshold_zero_recall_returns_zero_f1() -> None:
    """All predictions below threshold → 0 recall → 0 F1 (not NaN)."""
    scores = [0.1, 0.2, 0.1]
    labels = [1, 1, 0]
    p, r, f = f1_at_threshold(scores, labels, threshold=0.5)
    assert r == 0.0
    assert f == 0.0


def test_top_k_accuracy_perfect_when_label_1_is_top_score() -> None:
    groups = {
        "g0": [(0.1, 0), (0.9, 1), (0.2, 0)],  # 0.9 is highest → top-1 hit
        "g1": [(0.5, 0), (0.6, 1), (0.7, 0)],  # 0.6 is 2nd → top-1 miss, top-2 hit
    }
    assert top_k_accuracy(groups, k=1) == 0.5
    assert top_k_accuracy(groups, k=2) == 1.0
    assert top_k_accuracy(groups, k=3) == 1.0


def test_top_k_accuracy_skips_groups_without_positive() -> None:
    groups = {
        "g0": [(0.9, 1), (0.1, 0)],  # top-1 hit
        "g1": [(0.5, 0), (0.7, 0)],  # no positive → skipped
    }
    assert top_k_accuracy(groups, k=1) == 1.0  # 1/1 valid groups hit


# ---------------------------------------------------------------------------
# End-to-end train + eval smoke
# ---------------------------------------------------------------------------


def test_train_one_epoch_reduces_loss_on_tiny_dataset(tmp_path: Path) -> None:
    labels_dir, reg, entries = _make_tiny_p1_dataset(tmp_path)
    ds = FlatSealDataset(labels_dir, reg, entries)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    probe = next(iter(loader))
    model = CircuitMatchNet(
        in_channels=probe.x.shape[1], hidden_channels=16
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    device = torch.device("cpu")

    # Snapshot pre-train loss
    pre_loss = train_one_epoch(model, loader, opt, device)
    # One more epoch — loss should drop
    post_loss = train_one_epoch(model, loader, opt, device)
    assert post_loss < pre_loss, (
        f"loss did not decrease: pre={pre_loss:.4f} post={post_loss:.4f}"
    )


def test_evaluate_splits_metrics_by_task_type(tmp_path: Path) -> None:
    labels_dir, reg, entries = _make_tiny_p1_dataset(tmp_path)
    ds = FlatSealDataset(labels_dir, reg, entries)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    probe = next(iter(loader))
    model = CircuitMatchNet(
        in_channels=probe.x.shape[1], hidden_channels=16
    )
    metrics = evaluate(model, loader, torch.device("cpu"))
    assert "mean_loss" in metrics
    # Both task types should be present given the perturbation mix
    assert "wrong_edge" in metrics
    we = metrics["wrong_edge"]
    assert {"n", "auc", "precision", "recall", "f1", "accuracy"}.issubset(we)
    assert 0.0 <= we["f1"] <= 1.0
    if "missing_edge" in metrics:
        me = metrics["missing_edge"]
        assert {"n_groups", "top1", "top3", "top5"}.issubset(me)
        assert 0.0 <= me["top3"] <= 1.0


def test_train_full_cli_exits_3_when_any_gate_fails(tmp_path: Path) -> None:
    """**Regression** for the original gate bug: the script used
    ``if best_f1 < min_f1 AND best_top3 < min_top3`` so a run with one
    gate failing would still exit 0. The fix uses ``OR`` semantics —
    BOTH gates must pass per plan §九.

    Strategy: invoke `main(...)` on the real (tiny) P1 dataset with
    intentionally impossible thresholds (e.g. min_f1=0.999) and check
    that the exit code is 3 even if only that one gate is missed.
    """

    from scripts.gnn_train_full import main as train_main

    # Build tiny dataset on disk via the same fixture helper, then point
    # the CLI at it.
    _make_tiny_p1_dataset(tmp_path)

    # Refs config that matches the tiny dataset
    cfg = {
        "refs": [
            {"ref_id": "rc", "payload_path": str(FIXTURES / "test_rc_v1.json")},
            {"ref_id": "div", "payload_path": str(FIXTURES / "test_voltage_divider_v1.json")},
            {
                "ref_id": "opamp",
                "payload_path": str(FIXTURES / "test_opamp_buffer_v1.json"),
            },
        ],
        "subtypes_by_ref_id": {"opamp": {"U1": "UA741"}},
    }
    refs_config_path = tmp_path / "refs.json"
    refs_config_path.write_text(json.dumps(cfg))

    # Build a splits/ dir so FlatSealDataset finds them. The tiny dataset
    # generated by _make_tiny_p1_dataset doesn't write splits, so synthesise:
    labels_dir = tmp_path / "ds" / "labels"
    splits_dir = tmp_path / "ds" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    all_entries = sorted(
        f"{p.parent.name}/{p.stem}" for p in labels_dir.rglob("*.json")
    )
    # Put a few in train + val + test so loaders are non-empty
    n = len(all_entries)
    (splits_dir / "train.json").write_text(json.dumps(all_entries[: n // 2 or 1]))
    (splits_dir / "val.json").write_text(json.dumps(all_entries[n // 2 : n - 1] or all_entries[:1]))
    (splits_dir / "test.json").write_text(json.dumps(all_entries[-1:] or all_entries[:1]))

    out = tmp_path / "p3_out"
    # Strategy: use UNREACHABLE thresholds (> 1.0) so the model can never
    # accidentally pass a "failing" gate — top1/top3/F1 are all in [0, 1].
    # That isolates the gate-logic from training-loop variance.
    UNREACHABLE = 1.5  # > 1.0 max possible, so any value < UNREACHABLE
    REACHABLE = 0.0    # any value ≥ 0, always passes

    # Case 1: BOTH gates fail (would exit 3 under either AND or OR)
    rc_both = train_main([
        "--dataset-dir", str(tmp_path / "ds"),
        "--refs-config", str(refs_config_path),
        "--output-dir", str(out / "both"),
        "--epochs", "1",
        "--batch-size", "16",
        "--min-f1", str(UNREACHABLE),
        "--min-top3", str(UNREACHABLE),
    ])
    assert rc_both == 3, "both-gates-fail must exit 3"

    # Case 2: ONLY f1 fails (top3 gate set to always pass) → OR must
    # still exit 3. The original `and` bug would have returned 0 here.
    rc_only_f1 = train_main([
        "--dataset-dir", str(tmp_path / "ds"),
        "--refs-config", str(refs_config_path),
        "--output-dir", str(out / "only_f1"),
        "--epochs", "1",
        "--batch-size", "16",
        "--min-f1", str(UNREACHABLE),
        "--min-top3", str(REACHABLE),
    ])
    assert rc_only_f1 == 3, (
        "single-gate-fail (F1 only) must exit 3 — regression: original "
        "AND-bug let this slip as exit 0"
    )

    # Case 3: ONLY top3 fails (f1 gate set to always pass) → OR must
    # still exit 3.
    rc_only_top3 = train_main([
        "--dataset-dir", str(tmp_path / "ds"),
        "--refs-config", str(refs_config_path),
        "--output-dir", str(out / "only_top3"),
        "--epochs", "1",
        "--batch-size", "16",
        "--min-f1", str(REACHABLE),
        "--min-top3", str(UNREACHABLE),
    ])
    assert rc_only_top3 == 3, (
        "single-gate-fail (top3 only) must exit 3 — regression: original "
        "AND-bug let this slip as exit 0"
    )

    # Case 4: BOTH gates always pass → exit 0
    rc_pass = train_main([
        "--dataset-dir", str(tmp_path / "ds"),
        "--refs-config", str(refs_config_path),
        "--output-dir", str(out / "pass"),
        "--epochs", "1",
        "--batch-size", "16",
        "--min-f1", str(REACHABLE),
        "--min-top3", str(REACHABLE),
    ])
    assert rc_pass == 0


def test_backbone_load_then_train_smoke(tmp_path: Path) -> None:
    """End-to-end: save fake backbone, load into CircuitMatchNet, run
    1 epoch on tiny P1 dataset, verify no crash + loss is finite."""

    # 1) Synthesize a fake backbone ckpt
    src = SealDGCNN(in_channels=68, hidden_channels=32, sort_k=30)
    backbone_path = tmp_path / "backbone.pt"
    torch.save({
        "state_dict": src.state_dict(),
        "hidden": 32, "sort_k": 30, "in_channels": 68,
        "best_val_auc": 0.99, "fold": 0,
    }, backbone_path)

    # 2) Tiny P1 dataset
    labels_dir, reg, entries = _make_tiny_p1_dataset(tmp_path)
    ds = FlatSealDataset(labels_dir, reg, entries)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    # 3) Load backbone
    probe = next(iter(loader))
    model = CircuitMatchNet.from_pretrained_backbone(
        backbone_path, strict=False, override_in_channels=probe.x.shape[1]
    )

    # 4) One epoch + eval
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")
    loss = train_one_epoch(model, loader, opt, device)
    assert torch.isfinite(torch.tensor(loss))
    metrics = evaluate(model, loader, device)
    assert "mean_loss" in metrics
    assert metrics["mean_loss"] > 0
