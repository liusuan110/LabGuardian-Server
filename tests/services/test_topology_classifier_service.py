"""Unit tests for ``app.services.topology_classifier_service``.

These tests stay below the FastAPI layer — they exercise the service
dataclasses + singleton model loader directly, so a failure here points
at a service bug rather than an HTTP wiring problem.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from app.domain.dsl.loader import load_dsl_reference
from app.domain.logical_reference import logical_reference_to_graph
from app.services.topology_classifier_service import (
    DEFAULT_CKPT_PATH,
    REASON_TINY_GRAPH,
    TopologySuggestion,
    reset_model_cache,
    suggest_from_graph,
    suggest_from_netlist_v2,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "real_student"
REFERENCES = REPO_ROOT / "knowledge" / "references"


@pytest.fixture(autouse=True)
def _reset():
    reset_model_cache()
    yield
    reset_model_cache()


def _skip_if_no_ckpt() -> None:
    if not DEFAULT_CKPT_PATH.exists():
        pytest.skip(
            "checkpoints/gnn_a_v1/best.pt missing — run train script first."
        )


# ---------------------------------------------------------------------------
# Disabled branches
# ---------------------------------------------------------------------------


def test_empty_graph_returns_tiny_graph_reason() -> None:
    g = nx.Graph()
    result = suggest_from_graph(g)
    assert isinstance(result, TopologySuggestion)
    assert result.enabled is False
    assert result.disabled_reason == REASON_TINY_GRAPH
    assert result.gnn_predictions == []
    assert result.consensus is None


def test_missing_ckpt_returns_disabled_reason(tmp_path: Path) -> None:
    """Pointing at a non-existent ckpt should give clean disabled output,
    not an exception."""
    bogus = tmp_path / "nonexistent.pt"
    # Use a real graph so we get past the tiny_graph check.
    ref = load_dsl_reference(REFERENCES / "ce_amp_fixed_bias_v1.py")
    g = logical_reference_to_graph(ref)
    result = suggest_from_graph(g, ckpt_path=bogus)
    assert result.enabled is False
    assert "checkpoint" in (result.disabled_reason or "")


def test_invalid_netlist_returns_invalid_branch() -> None:
    result = suggest_from_netlist_v2({"hello": "world"})
    assert result.enabled is False
    assert result.disabled_reason is not None


# ---------------------------------------------------------------------------
# Happy path — requires real ckpt
# ---------------------------------------------------------------------------


def test_suggest_runs_and_returns_predictions_for_lpf() -> None:
    _skip_if_no_ckpt()
    payload = json.loads(
        (FIXTURES / "opamp_inverting_lpf_correct_v1.json").read_text()
    )
    result = suggest_from_netlist_v2(payload)
    assert result.enabled is True
    assert len(result.gnn_predictions) == 7  # default top_k
    assert result.gnn_predictions[0].rank == 1
    # LPF correctly identified as lossy integrator.
    assert result.gnn_predictions[0].label == "integrator_ua741"


def test_inference_is_fast() -> None:
    """Latency budget: 50ms on CPU. Real measurements typically < 5ms."""
    _skip_if_no_ckpt()
    payload = json.loads((FIXTURES / "inverting_amp_correct_v1.json").read_text())
    result = suggest_from_netlist_v2(payload)
    assert result.enabled is True
    assert result.inference_ms < 50.0, (
        f"GNN-A inference unexpectedly slow: {result.inference_ms:.1f}ms"
    )


def test_consensus_high_for_canonical_diff_amp() -> None:
    _skip_if_no_ckpt()
    payload = json.loads((FIXTURES / "bjt_diff_amp_correct_v1.json").read_text())
    result = suggest_from_netlist_v2(payload)
    assert result.consensus is not None
    assert result.consensus.agreed is True
    assert result.consensus.confidence_band == "high"
    assert result.consensus.recommended_label == "differential_pair"


def test_predictions_are_sorted_by_confidence_desc() -> None:
    _skip_if_no_ckpt()
    payload = json.loads((FIXTURES / "opamp_summing_correct_v1.json").read_text())
    result = suggest_from_netlist_v2(payload)
    confs = [p.confidence for p in result.gnn_predictions]
    assert confs == sorted(confs, reverse=True)
    ranks = [p.rank for p in result.gnn_predictions]
    assert ranks == list(range(1, len(ranks) + 1))


def test_top_k_truncates_output() -> None:
    _skip_if_no_ckpt()
    payload = json.loads((FIXTURES / "inverting_amp_correct_v1.json").read_text())
    result = suggest_from_netlist_v2(payload, top_k=3)
    assert len(result.gnn_predictions) == 3


def test_to_dict_is_json_serializable() -> None:
    _skip_if_no_ckpt()
    payload = json.loads((FIXTURES / "bjt_diff_amp_correct_v1.json").read_text())
    result = suggest_from_netlist_v2(payload)
    serialized = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "differential_pair" in serialized
    assert "consensus" in serialized


def test_template_matches_align_with_gnn_for_canonical() -> None:
    """When GNN strongly identifies a topology, the symbolic template
    matcher should usually agree (this is the basis for 'high consensus').
    """
    _skip_if_no_ckpt()
    payload = json.loads((FIXTURES / "inverting_amp_correct_v1.json").read_text())
    result = suggest_from_netlist_v2(payload)
    assert result.template_matches, "template matcher returned nothing"
    top_template = result.template_matches[0]
    top_gnn = result.gnn_predictions[0]
    assert top_template["topology_label"] == top_gnn.label
