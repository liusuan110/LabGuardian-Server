"""Unit tests for the graph → tensor feature encoder."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import torch

from app.domain.dsl.loader import load_dsl_reference
from app.domain.logical_reference import logical_reference_to_graph
from app.domain.topology.features import (
    FEATURE_DIM,
    encode_graph,
    encode_node,
    encoded_to_pyg_data,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCES_DIR = REPO_ROOT / "knowledge" / "references"


def test_feature_dim_is_23() -> None:
    """v2 — added num_R_neighbors + num_C_neighbors (2 dims).
    If this drifts, model checkpoints become invalid."""
    assert FEATURE_DIM == 23


def test_encode_node_returns_correct_dim() -> None:
    vec = encode_node({"kind": "comp", "ctype": "Resistor"}, degree=2, max_degree=4)
    assert len(vec) == FEATURE_DIM


def test_encode_comp_node_sets_kind_and_ctype() -> None:
    vec = encode_node(
        {"kind": "comp", "ctype": "IC", "subtype": "UA741"},
        degree=5, max_degree=5,
    )
    # kind index for "comp" = 0; one-hot at index 0
    assert vec[0] == 1.0
    assert vec[1] == 0.0
    # ctype "IC" sits at index 3 of _COMP_TYPE_ORDER (0-based),
    # which lands at offset 2 + 3 = 5 in the vector.
    assert vec[5] == 1.0


def test_encode_net_node_sets_role() -> None:
    vec = encode_node({"kind": "net", "role": "ground"}, degree=1, max_degree=4)
    # kind "net" -> vec[1] = 1
    assert vec[1] == 1.0
    # net_role "ground" sits at offset 2 + 8 (comp_type) + 4 (subtype) + 3 = 17
    assert vec[17] == 1.0


def test_encode_graph_dimensions(tmp_path: Path) -> None:
    payload = load_dsl_reference(REFERENCES_DIR / "ua741_inverting_amp_gain10_v1.py")
    g = logical_reference_to_graph(payload)
    encoded = encode_graph(g)
    assert encoded.x.shape == (g.number_of_nodes(), FEATURE_DIM)
    # Each undirected edge contributes 2 entries in PyG edge_index.
    assert encoded.edge_index.shape == (2, g.number_of_edges() * 2)
    assert len(encoded.node_order) == g.number_of_nodes()


def test_encoded_to_pyg_data_with_label() -> None:
    payload = load_dsl_reference(REFERENCES_DIR / "ce_amp_fixed_bias_v1.py")
    g = logical_reference_to_graph(payload)
    encoded = encode_graph(g)
    data = encoded_to_pyg_data(encoded, label_index=1)
    assert data.x is encoded.x
    assert data.edge_index is encoded.edge_index
    assert data.y is not None
    assert int(data.y[0]) == 1


def test_encoded_to_pyg_data_without_label_omits_y() -> None:
    g = nx.Graph()
    g.add_node("n1", kind="comp", ctype="Resistor")
    encoded = encode_graph(g)
    data = encoded_to_pyg_data(encoded, label_index=None)
    assert data.y is None


def test_encoding_is_deterministic() -> None:
    """Same graph in → byte-identical tensors out. Critical for ONNX
    export reproducibility."""
    payload = load_dsl_reference(REFERENCES_DIR / "ua741_integrator_v1.py")
    g = logical_reference_to_graph(payload)
    e1 = encode_graph(g)
    e2 = encode_graph(g)
    assert torch.equal(e1.x, e2.x)
    assert torch.equal(e1.edge_index, e2.edge_index)
    assert e1.node_order == e2.node_order


def test_empty_graph_does_not_crash() -> None:
    g = nx.Graph()
    encoded = encode_graph(g)
    assert encoded.x.shape == (0, FEATURE_DIM)
    assert encoded.edge_index.shape == (2, 0)
    assert encoded.node_order == []


def test_normalized_subtype_falls_through_to_other() -> None:
    """Unknown subtype strings get bucketed to ``other`` (index 3)."""
    vec = encode_node(
        {"kind": "comp", "ctype": "IC", "subtype": "LM358"},
        degree=1, max_degree=1,
    )
    # subtype index for "other" = 3; offset = 2 (kind) + 8 (ctype) + 3 = 13
    assert vec[13] == 1.0
