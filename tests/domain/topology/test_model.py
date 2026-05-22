"""Smoke tests for the GraphSAGE topology classifier.

These tests verify shape/dtype contracts and parameter count constraints —
they do NOT train the model (training is a separate offline step).
"""

from __future__ import annotations

from pathlib import Path

import torch

from app.domain.dsl.loader import load_dsl_reference
from app.domain.logical_reference import logical_reference_to_graph
from app.domain.topology.features import FEATURE_DIM, encode_graph, encoded_to_pyg_data
from app.domain.topology.labels import TOPOLOGY_LABELS
from app.domain.topology.model import (
    DEFAULT_HIDDEN_DIM,
    NUM_CLASSES,
    TopologyClassifier,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCES_DIR = REPO_ROOT / "knowledge" / "references"


def test_num_classes_matches_label_count() -> None:
    assert NUM_CLASSES == len(TOPOLOGY_LABELS) == 7


def test_default_model_param_count_under_phase1_budget() -> None:
    """Phase 1 design goal: model < 100K params for edge friendliness."""
    model = TopologyClassifier()
    params = model.count_parameters()
    assert params < 100_000, (
        f"GraphSAGE expected < 100K params, got {params}. "
        f"Reduce DEFAULT_HIDDEN_DIM ({DEFAULT_HIDDEN_DIM}) or shrink MLP head."
    )


def test_forward_pass_returns_logits_with_correct_shape() -> None:
    """Single-graph batch forward."""
    payload = load_dsl_reference(REFERENCES_DIR / "ua741_inverting_amp_gain10_v1.py")
    g = logical_reference_to_graph(payload)
    encoded = encode_graph(g)
    data = encoded_to_pyg_data(encoded, label_index=3)

    model = TopologyClassifier()
    model.eval()
    batch = torch.zeros(data.x.shape[0], dtype=torch.long)  # all nodes in batch 0
    with torch.no_grad():
        logits = model(data.x, data.edge_index, batch)
    assert logits.shape == (1, NUM_CLASSES)
    assert logits.dtype == torch.float32


def test_predict_proba_returns_normalized_distribution() -> None:
    payload = load_dsl_reference(REFERENCES_DIR / "ce_amp_fixed_bias_v1.py")
    g = logical_reference_to_graph(payload)
    encoded = encode_graph(g)
    data = encoded_to_pyg_data(encoded)

    model = TopologyClassifier()
    batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    probs = model.predict_proba(data.x, data.edge_index, batch)
    assert probs.shape == (1, NUM_CLASSES)
    # Softmax outputs sum to 1 per graph.
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)


def test_forward_pass_with_batched_graphs() -> None:
    """Two graphs batched: batch tensor encodes per-node graph membership."""
    payload_a = load_dsl_reference(REFERENCES_DIR / "rc_first_order_v1.py")
    payload_b = load_dsl_reference(REFERENCES_DIR / "ua741_integrator_v1.py")
    g_a = logical_reference_to_graph(payload_a)
    g_b = logical_reference_to_graph(payload_b)

    enc_a = encode_graph(g_a)
    enc_b = encode_graph(g_b)

    x = torch.cat([enc_a.x, enc_b.x], dim=0)
    # Shift graph B's edge indices by graph A's node count.
    shift = enc_a.x.shape[0]
    edge_b_shifted = enc_b.edge_index + shift
    edge_index = torch.cat([enc_a.edge_index, edge_b_shifted], dim=1)
    batch = torch.cat([
        torch.zeros(enc_a.x.shape[0], dtype=torch.long),
        torch.ones(enc_b.x.shape[0], dtype=torch.long),
    ])

    model = TopologyClassifier()
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, batch)
    assert logits.shape == (2, NUM_CLASSES)


def test_model_accepts_custom_hidden_dim() -> None:
    model = TopologyClassifier(hidden_dim=32)
    assert model.hidden_dim == 32
    # Smaller model should have fewer params than the default 64-wide one.
    smaller = model.count_parameters()
    default = TopologyClassifier(hidden_dim=64).count_parameters()
    assert smaller < default


def test_model_in_dim_matches_feature_dim_default() -> None:
    model = TopologyClassifier()
    assert model.in_dim == FEATURE_DIM
