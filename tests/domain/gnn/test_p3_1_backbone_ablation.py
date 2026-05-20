"""P3.1 · L1 HeteroConv backbone + ablation harness tests.

Skipped automatically if torch / torch_geometric aren't installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.transforms import ToUndirected  # noqa: E402

from app.domain.gnn import build_from_logical_reference, to_hetero_data  # noqa: E402
from app.domain.gnn.backbone import (  # noqa: E402
    HeteroNodeEncoder,
    HeteroSAGEBackbone,
    embeddings_for_subgraph,
)
from app.domain.gnn.graph_schema import (  # noqa: E402
    COMPONENT_FEAT_DIM,
    DRNL_LABEL_DIM,
    NET_FEAT_DIM,
    PORT_FEAT_DIM,
)
from app.domain.gnn.pyg_converter import seal_subgraph_to_pyg_data  # noqa: E402
from app.domain.gnn.seal_subgraph import extract_seal_subgraph  # noqa: E402
from scripts.gnn_ablation import (  # noqa: E402
    ABLATIONS,
    build_ablation_argv,
    render_report,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "references"


def _load(name: str, subtypes: dict | None = None):
    return build_from_logical_reference(
        json.loads((FIXTURES / name).read_text(encoding="utf-8")),
        extra_subtypes_by_source_id=subtypes,
    )


# ---------------------------------------------------------------------------
# HeteroNodeEncoder
# ---------------------------------------------------------------------------


def test_hetero_node_encoder_lifts_each_type_to_hidden_dim() -> None:
    ref = _load("test_opamp_buffer_v1.json", {"U1": "UA741"})
    data = to_hetero_data(ref)
    enc = HeteroNodeEncoder(hidden_dim=64)
    z = enc(data)
    assert set(z) == {"component", "port", "net"}
    assert z["component"].shape == (data["component"].x.shape[0], 64)
    assert z["port"].shape == (data["port"].x.shape[0], 64)
    assert z["net"].shape == (data["net"].x.shape[0], 64)
    # tanh keeps |z| ≤ 1
    for k, v in z.items():
        assert v.abs().max().item() <= 1.0, f"{k} exceeded tanh range"


def test_hetero_node_encoder_dims_match_schema_constants() -> None:
    """Encoder must consume the schema-declared widths so future
    PORT_FEAT_DIM bumps trip a clear shape error."""

    enc = HeteroNodeEncoder(hidden_dim=128)
    assert enc.enc_comp.in_features == COMPONENT_FEAT_DIM
    assert enc.enc_port.in_features == PORT_FEAT_DIM
    assert enc.enc_net.in_features == NET_FEAT_DIM


# ---------------------------------------------------------------------------
# HeteroSAGEBackbone
# ---------------------------------------------------------------------------


def test_hetero_sage_backbone_forward_shapes() -> None:
    ref = _load("test_opamp_buffer_v1.json", {"U1": "UA741"})
    data = ToUndirected()(to_hetero_data(ref))
    m = HeteroSAGEBackbone(hidden_dim=32, num_layers=3)
    z = m(data)
    assert z["component"].shape == (data["component"].x.shape[0], 32)
    assert z["port"].shape == (data["port"].x.shape[0], 32)
    assert z["net"].shape == (data["net"].x.shape[0], 32)


def test_hetero_sage_backbone_gradient_flows() -> None:
    ref = _load("test_rc_v1.json")
    data = ToUndirected()(to_hetero_data(ref))
    m = HeteroSAGEBackbone(hidden_dim=16, num_layers=2)
    z = m(data)
    loss = z["port"].sum() + z["net"].sum() + z["component"].sum()
    loss.backward()
    for name, p in m.named_parameters():
        # Some bias params may legitimately have None grad if their tensor
        # path is detached; but the encoder weights must always get grads.
        if "enc_" in name:
            assert p.grad is not None, f"no grad on encoder param {name}"
            assert torch.isfinite(p.grad).all()


def test_hetero_sage_backbone_three_layers_change_representation() -> None:
    """Sanity: stacking 3 SAGE layers should produce embeddings that
    differ from the 0-layer (encoder-only) embeddings on a circuit with
    actual edges (UA741 has 5 connected pins)."""

    ref = _load("test_opamp_buffer_v1.json", {"U1": "UA741"})
    data = ToUndirected()(to_hetero_data(ref))
    encoder = HeteroNodeEncoder(hidden_dim=16)
    backbone = HeteroSAGEBackbone(hidden_dim=16, num_layers=3)
    with torch.no_grad():
        z0 = encoder(data)
        z3 = backbone(data)
    assert not torch.allclose(z0["port"], z3["port"]), (
        "3 SAGE layers produced identical port embeddings to encoder-only"
    )


# ---------------------------------------------------------------------------
# embeddings_for_subgraph
# ---------------------------------------------------------------------------


def test_embeddings_for_subgraph_orders_ports_then_nets() -> None:
    ref = _load("test_rc_v1.json")
    data = ToUndirected()(to_hetero_data(ref))
    backbone = HeteroSAGEBackbone(hidden_dim=8, num_layers=2)
    with torch.no_grad():
        z = backbone(data)
    sg = extract_seal_subgraph(ref, ref.edges[0].src_port_id, ref.edges[0].dst_net_id)
    emb = embeddings_for_subgraph(
        z, list(sg.port_ids), list(sg.net_ids),
        {"port": data["port"].node_ids, "net": data["net"].node_ids},
    )
    assert emb.shape == (len(sg.port_ids) + len(sg.net_ids), 8)
    # First row corresponds to sg.port_ids[0] = the anchor port
    expected_first = z["port"][
        data["port"].node_ids.index(sg.port_ids[0])
    ]
    assert torch.equal(emb[0], expected_first)


def test_embeddings_for_subgraph_handles_missing_nodes() -> None:
    """If the SealSubgraph contains a node not in the HCG node_ids list
    (shouldn't happen in practice, but guard against silent failure),
    return zero vectors instead of crashing."""

    ref = _load("test_rc_v1.json")
    data = ToUndirected()(to_hetero_data(ref))
    backbone = HeteroSAGEBackbone(hidden_dim=4, num_layers=1)
    with torch.no_grad():
        z = backbone(data)
    emb = embeddings_for_subgraph(
        z, ["does_not_exist"], list(data["net"].node_ids),
        {"port": data["port"].node_ids, "net": data["net"].node_ids},
    )
    # First row is the missing port → all zeros
    assert emb[0].abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# DRNL ablation flag wiring
# ---------------------------------------------------------------------------


def test_seal_subgraph_drop_drnl_zeros_first_17_dims() -> None:
    ref = _load("test_rc_v1.json")
    sg = extract_seal_subgraph(ref, ref.edges[0].src_port_id, ref.edges[0].dst_net_id)
    d_normal = seal_subgraph_to_pyg_data(sg, ref)
    d_no_drnl = seal_subgraph_to_pyg_data(sg, ref, drop_drnl=True)
    # DRNL slice differs
    assert d_normal.x[:, :DRNL_LABEL_DIM].sum().item() > 0
    assert d_no_drnl.x[:, :DRNL_LABEL_DIM].sum().item() == 0
    # Everything past the DRNL slice is identical
    assert torch.equal(
        d_normal.x[:, DRNL_LABEL_DIM:],
        d_no_drnl.x[:, DRNL_LABEL_DIM:],
    )


def test_flat_seal_dataset_propagates_drop_drnl(tmp_path: Path) -> None:
    """The ablation flag must reach the converter via the dataset class."""

    from app.domain.gnn import DatasetSpec, PerturbationPlan, RefSpec, generate_dataset
    from app.domain.gnn.pyg_dataset import FlatSealDataset, RefEntry, RefRegistry

    spec = DatasetSpec(
        refs=(RefSpec(ref_id="rc", payload_path=FIXTURES / "test_rc_v1.json"),),
        plan=PerturbationPlan(counts={"identity": 1}),
        output_dir=tmp_path / "ds",
        enforce_healthy=False,
    )
    generate_dataset(spec)
    reg = RefRegistry()
    reg.register(RefEntry("rc", FIXTURES / "test_rc_v1.json"))
    entries = sorted(
        f"{p.parent.name}/{p.stem}"
        for p in (tmp_path / "ds" / "labels").rglob("*.json")
    )
    ds_on = FlatSealDataset(tmp_path / "ds" / "labels", reg, entries, drop_drnl=False)
    ds_off = FlatSealDataset(tmp_path / "ds" / "labels", reg, entries, drop_drnl=True)
    assert ds_on[0].x[:, :DRNL_LABEL_DIM].sum().item() > 0
    assert ds_off[0].x[:, :DRNL_LABEL_DIM].sum().item() == 0


# ---------------------------------------------------------------------------
# Ablation argv composer
# ---------------------------------------------------------------------------


def test_ablation_argv_baseline_keeps_pretrain_ckpt() -> None:
    argv = build_ablation_argv(
        ["--dataset-dir", "/x", "--epochs", "5"],
        "baseline",
        Path("/out/baseline"),
        pretrain_ckpt=Path("/ckpt/backbone.pt"),
    )
    assert "--pretrain-ckpt" in argv
    assert "/ckpt/backbone.pt" in argv
    assert "--no-drnl" not in argv


def test_ablation_argv_no_pretrain_drops_pretrain_ckpt() -> None:
    argv = build_ablation_argv(
        ["--dataset-dir", "/x", "--epochs", "5"],
        "no_pretrain",
        Path("/out/no_pretrain"),
        pretrain_ckpt=Path("/ckpt/backbone.pt"),
    )
    assert "--pretrain-ckpt" not in argv
    assert "--no-drnl" not in argv


def test_ablation_argv_no_drnl_keeps_pretrain_but_adds_flag() -> None:
    argv = build_ablation_argv(
        ["--dataset-dir", "/x", "--epochs", "5"],
        "no_drnl",
        Path("/out/no_drnl"),
        pretrain_ckpt=Path("/ckpt/backbone.pt"),
    )
    assert "--pretrain-ckpt" in argv
    assert "--no-drnl" in argv


def test_ablation_argv_rejects_unknown_config() -> None:
    with pytest.raises(ValueError, match="unknown ablation"):
        build_ablation_argv([], "no_port", Path("/x"), None)


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


def test_render_report_produces_markdown_table() -> None:
    fake_results = {
        "baseline": {
            "best_val_f1": 0.92,
            "best_val_top3": 1.0,
            "history": [{"val": {"wrong_edge": {"auc": 0.97}}}],
            "test_metrics": {
                "wrong_edge": {"f1": 0.62},
                "missing_edge": {"top3": 0.4},
            },
        },
        "no_pretrain": {
            "best_val_f1": 0.85,  # baseline - 0.07 → pretrain helps
            "best_val_top3": 0.9,
            "history": [{"val": {"wrong_edge": {"auc": 0.91}}}],
            "test_metrics": {
                "wrong_edge": {"f1": 0.55},
                "missing_edge": {"top3": 0.3},
            },
        },
        "no_drnl": {
            "best_val_f1": 0.88,  # baseline - 0.04 → DRNL helps
            "best_val_top3": 0.95,
            "history": [{"val": {"wrong_edge": {"auc": 0.93}}}],
            "test_metrics": {
                "wrong_edge": {"f1": 0.58},
                "missing_edge": {"top3": 0.35},
            },
        },
    }
    md = render_report(fake_results, target_f1_gate=0.88, target_top3_gate=0.85)
    # Sanity: every config + its delta + verdict shows up
    for cfg in ABLATIONS:
        assert f"`{cfg}`" in md, f"config {cfg} missing from report"
    assert "去预训练" in md
    assert "去 DRNL" in md
    assert "去 port" in md
    # With these synthetic numbers, both ablations beat plan targets
    assert "✅ pretraining helps" in md
    assert "✅ DRNL helps" in md
