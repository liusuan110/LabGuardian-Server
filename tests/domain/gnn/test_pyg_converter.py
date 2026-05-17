"""P2 · PyG converter tests.

Verifies the HeteroCircuitGraph → PyG HeteroData encoding and the
SealSubgraph → PyG Data encoding match plan §三 + §三.6 dimensions.
Skipped automatically if torch / torch_geometric aren't installed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from app.domain.gnn import (  # noqa: E402
    COMPONENT_FEAT_DIM,
    NET_FEAT_DIM,
    PORT_FEAT_DIM,
    build_from_logical_reference,
    extract_seal_subgraph,
)
from app.domain.gnn.graph_schema import (  # noqa: E402
    DRNL_LABEL_DIM,
    PORT_NET_EDGE_FEAT_DIM,
)
from app.domain.gnn.pyg_converter import (  # noqa: E402
    encode_component_features,
    encode_net_features,
    encode_port_features,
    encode_port_net_edge_features,
    seal_subgraph_to_pyg_data,
    to_hetero_data,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "references"


def _load(name: str, subtypes: dict | None = None):
    return build_from_logical_reference(
        json.loads((FIXTURES / name).read_text()),
        extra_subtypes_by_source_id=subtypes,
    )


# ---------------------------------------------------------------------------
# Per-encoder shape contracts (plan §三)
# ---------------------------------------------------------------------------


def test_encode_component_features_shape_and_ids() -> None:
    hcg = _load("test_rc_v1.json")
    x, ids = encode_component_features(hcg)
    assert x.shape == (2, COMPONENT_FEAT_DIM)
    assert x.dtype == torch.float32
    assert ids == list(hcg.components)


def test_encode_port_features_shape_and_dim() -> None:
    hcg = _load("test_opamp_buffer_v1.json", {"U1": "UA741"})
    x, ids = encode_port_features(hcg)
    # UA741 buffer has all 8 ports materialised
    assert x.shape == (8, PORT_FEAT_DIM)
    assert x.dtype == torch.float32
    assert ids == list(hcg.ports)


def test_encode_net_features_shape_and_dim() -> None:
    hcg = _load("test_voltage_divider_v1.json")
    x, ids = encode_net_features(hcg)
    assert x.shape == (3, NET_FEAT_DIM)
    assert x.dtype == torch.float32


def test_encode_port_net_edge_features_shape() -> None:
    hcg = _load("test_rc_v1.json")
    edge_index, edge_attr = encode_port_net_edge_features(hcg)
    # 2 components × 2 pins each = 4 (port,connects,net) edges
    assert edge_index.shape == (2, 4)
    assert edge_attr.shape == (4, PORT_NET_EDGE_FEAT_DIM)
    # confidence column ≥ 0
    assert (edge_attr[:, 0] >= 0).all()


# ---------------------------------------------------------------------------
# to_hetero_data — structural completeness
# ---------------------------------------------------------------------------


def test_to_hetero_data_has_expected_node_and_edge_types() -> None:
    hcg = _load("test_opamp_buffer_v1.json", {"U1": "UA741"})
    data = to_hetero_data(hcg)
    assert set(data.node_types) == {"component", "port", "net"}
    assert ("component", "has_port", "port") in data.edge_types
    assert ("port", "connects", "net") in data.edge_types
    # node counts
    assert data["component"].x.shape[0] == 1   # U1
    assert data["port"].x.shape[0] == 8         # full UA741 materialisation
    assert data["net"].x.shape[0] == 4
    # has_port edges = port count
    assert data["component", "has_port", "port"].edge_index.shape[1] == 8
    # connects edges = the 5 connected pins (NC/OPTIONAL stay floating)
    assert data["port", "connects", "net"].edge_index.shape[1] == 5


def test_to_hetero_data_is_undirected_compatible() -> None:
    """PyG's ToUndirected transform must not crash on our HeteroData."""

    from torch_geometric.transforms import ToUndirected

    hcg = _load("test_rc_v1.json")
    data = to_hetero_data(hcg)
    aug = ToUndirected()(data)
    # Reverse edge types added by the transform
    assert ("port", "rev_has_port", "component") in aug.edge_types
    assert ("net", "rev_connects", "port") in aug.edge_types


# ---------------------------------------------------------------------------
# SEAL subgraph → PyG Data
# ---------------------------------------------------------------------------


def test_seal_subgraph_to_pyg_data_feature_layout() -> None:
    hcg = _load("test_voltage_divider_v1.json")
    # Pick the first observed edge as the anchor (R1.pin1 — VIN)
    edge = hcg.edges[0]
    sg = extract_seal_subgraph(
        hcg, edge.src_port_id, edge.dst_net_id, num_hops=2
    )
    data = seal_subgraph_to_pyg_data(
        sg, hcg, label=1, label_source="ref_present", task_type="wrong_edge"
    )
    n_nodes = len(sg.port_ids) + len(sg.net_ids)
    feat_width = DRNL_LABEL_DIM + max(PORT_FEAT_DIM, NET_FEAT_DIM) + 1
    assert data.x.shape == (n_nodes, feat_width)
    # DRNL one-hots cover dims [0, 17)
    assert (data.x[:, :DRNL_LABEL_DIM].sum(dim=1) == 1).all(), (
        "every node must have exactly one DRNL one-hot bit set"
    )
    # target_flag is last column — exactly 2 anchors must be set
    assert int(data.x[:, -1].sum().item()) == 2
    assert data.y.item() == 1.0
    assert data.label_source == "ref_present"
    assert data.task_type == "wrong_edge"


def test_seal_subgraph_to_pyg_data_anchor_indices() -> None:
    hcg = _load("test_rc_v1.json")
    edge = hcg.edges[0]
    sg = extract_seal_subgraph(hcg, edge.src_port_id, edge.dst_net_id)
    data = seal_subgraph_to_pyg_data(sg, hcg, label=0)
    # Anchors are at sg.port_ids[0] / sg.net_ids[0] by SEAL contract
    port_anchor_idx = int(data.target_port_idx.item())
    net_anchor_idx = int(data.target_net_idx.item())
    assert port_anchor_idx == 0
    assert net_anchor_idx == len(sg.port_ids)  # nets start after ports
    # Both anchors must have target_flag = 1
    assert data.x[port_anchor_idx, -1] == 1.0
    assert data.x[net_anchor_idx, -1] == 1.0


def test_seal_subgraph_to_pyg_data_edge_index_is_undirected() -> None:
    """Each (port, net) edge should appear in both directions."""

    hcg = _load("test_rc_v1.json")
    edge = hcg.edges[0]
    sg = extract_seal_subgraph(hcg, edge.src_port_id, edge.dst_net_id)
    data = seal_subgraph_to_pyg_data(sg, hcg)
    # 2 × number of subgraph edges (excluding the candidate edge)
    assert data.edge_index.shape[1] == 2 * len(sg.edges)


def test_seal_subgraph_carries_optional_labels_only_when_provided() -> None:
    hcg = _load("test_rc_v1.json")
    edge = hcg.edges[0]
    sg = extract_seal_subgraph(hcg, edge.src_port_id, edge.dst_net_id)
    data = seal_subgraph_to_pyg_data(sg, hcg)  # no label kwargs
    # PyG Data exposes ``y`` as a sentinel attr (= None when unset). For
    # collation safety the converter always sets the string fields to ""
    # so DataLoader can stack heterogeneous batches.
    assert getattr(data, "y", None) is None
    assert data.label_source == ""
    assert data.task_type == ""
    assert data.group_id == ""


# ---------------------------------------------------------------------------
# Determinism — same HCG, same tensors
# ---------------------------------------------------------------------------


def test_encoding_is_deterministic_same_hcg() -> None:
    hcg = _load("test_voltage_divider_v1.json")
    x1, _ = encode_port_features(hcg)
    x2, _ = encode_port_features(hcg)
    assert torch.equal(x1, x2)
