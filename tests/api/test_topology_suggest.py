"""End-to-end tests for ``POST /api/v1/topology/suggest`` + ``GET /model-info``.

These tests spin up the FastAPI app via ``TestClient`` and exercise the
full request → service → response chain. They require the trained ckpt
to be present at ``checkpoints/gnn_a_v1/best.pt`` — if missing, the
``enabled=False`` branch is tested instead (matches CI behavior on a
freshly cloned repo without a trained model).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.topology_classifier_service import (
    DEFAULT_CKPT_PATH,
    reset_model_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "real_student"


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Clear the cached model between tests so ckpt-missing branch can
    flip without leaking state."""
    reset_model_cache()
    yield
    reset_model_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def ckpt_present() -> bool:
    return DEFAULT_CKPT_PATH.exists()


# ---------------------------------------------------------------------------
# /model-info
# ---------------------------------------------------------------------------


def test_model_info_returns_label_set(client: TestClient) -> None:
    resp = client.get("/api/v1/topology/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "gnn_a_v2"
    assert body["num_classes"] == 7
    assert "rc_first_order" in body["labels"]
    assert "unknown" in body["labels"]


def test_model_info_reports_ckpt_presence(
    client: TestClient, ckpt_present: bool
) -> None:
    resp = client.get("/api/v1/topology/model-info")
    body = resp.json()
    assert body["ckpt_exists"] == ckpt_present
    assert body["available"] == ckpt_present


# ---------------------------------------------------------------------------
# /suggest — validation
# ---------------------------------------------------------------------------


def test_suggest_requires_payload(client: TestClient) -> None:
    """No netlist + no logical_reference → 400."""
    resp = client.post("/api/v1/topology/suggest", json={})
    assert resp.status_code == 400
    assert "netlist_v2" in resp.json()["detail"]


def test_suggest_rejects_invalid_logical_reference(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/topology/suggest",
        json={"logical_reference": {"garbage": "no required fields"}},
    )
    # Either 400 or 200 with enabled=False — both are acceptable
    # ("garbage in" should not 500).
    assert resp.status_code in (200, 400)


def test_suggest_invalid_top_k(client: TestClient) -> None:
    """top_k must be in [1, 7]."""
    resp = client.post(
        "/api/v1/topology/suggest",
        json={"netlist_v2": {"components": [], "nets": []}, "top_k": 99},
    )
    assert resp.status_code == 422  # pydantic validation


# ---------------------------------------------------------------------------
# /suggest — disabled branches
# ---------------------------------------------------------------------------


def test_suggest_returns_tiny_graph_for_empty_netlist(client: TestClient) -> None:
    """Empty netlist → graph has 0 nodes → tiny_graph branch."""
    resp = client.post(
        "/api/v1/topology/suggest",
        json={"netlist_v2": {"components": [], "nets": []}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["disabled_reason"] == "tiny_graph"


# ---------------------------------------------------------------------------
# /suggest — happy path (requires ckpt)
# ---------------------------------------------------------------------------


def _require_ckpt(ckpt_present: bool) -> None:
    if not ckpt_present:
        pytest.skip(
            "checkpoints/gnn_a_v1/best.pt not present — "
            "run scripts/cadx/train_topology_classifier.py first."
        )


@pytest.mark.parametrize(
    "fixture_basename,expected_label",
    [
        ("inverting_amp_correct_v1",       "inverting_amp_ua741"),
        ("bjt_diff_amp_correct_v1",        "differential_pair"),
        ("opamp_inverting_lpf_correct_v1", "integrator_ua741"),  # LPF == lossy integrator
        ("opamp_summing_correct_v1",       "summing_amp_ua741"),
    ],
)
def test_suggest_correctly_identifies_topology(
    client: TestClient,
    ckpt_present: bool,
    fixture_basename: str,
    expected_label: str,
) -> None:
    _require_ckpt(ckpt_present)

    payload = json.loads((FIXTURES / f"{fixture_basename}.json").read_text())
    resp = client.post(
        "/api/v1/topology/suggest",
        json={"netlist_v2": payload, "top_k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["disabled_reason"] is None
    assert len(body["gnn_predictions"]) == 3
    top1 = body["gnn_predictions"][0]
    assert top1["label"] == expected_label, (
        f"{fixture_basename}: expected top-1 {expected_label}, got {top1['label']} "
        f"(conf={top1['confidence']:.3f})"
    )
    assert top1["confidence"] > 0.7
    assert body["consensus"]["recommended_label"] == expected_label


def test_suggest_response_schema_complete(
    client: TestClient, ckpt_present: bool
) -> None:
    """Every Pydantic field must be populated in a happy-path response."""
    _require_ckpt(ckpt_present)

    payload = json.loads((FIXTURES / "inverting_amp_correct_v1.json").read_text())
    resp = client.post(
        "/api/v1/topology/suggest",
        json={"netlist_v2": payload},
    )
    body = resp.json()
    # Top-level
    assert body["enabled"] is True
    assert body["model_version"] == "gnn_a_v2"
    assert body["inference_ms"] > 0
    # Graph stats
    gs = body["graph_stats"]
    assert gs["num_nodes"] > 0
    assert gs["num_comp_nodes"] > 0
    assert gs["num_net_nodes"] > 0
    # Predictions are ranked descending
    confs = [p["confidence"] for p in body["gnn_predictions"]]
    assert confs == sorted(confs, reverse=True)
    # Each prediction has display name
    for p in body["gnn_predictions"]:
        assert p["display_name_zh"]
        assert p["display_name_en"]
        assert "rank" in p
    # Consensus
    c = body["consensus"]
    assert c["confidence_band"] in {"high", "medium", "low", "disagreement"}
    assert "recommended_label" in c


def test_suggest_with_logical_reference_branch(
    client: TestClient, ckpt_present: bool
) -> None:
    """Verify the logical_reference fallback path works (used by debug tools)."""
    _require_ckpt(ckpt_present)

    from app.domain.dsl.loader import load_dsl_reference

    ref = load_dsl_reference(
        REPO_ROOT / "knowledge" / "references" / "ce_amp_fixed_bias_v1.py"
    )
    resp = client.post(
        "/api/v1/topology/suggest",
        json={"logical_reference": ref},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["gnn_predictions"][0]["label"] == "common_emitter"


def test_suggest_high_consensus_for_canonical_circuit(
    client: TestClient, ckpt_present: bool
) -> None:
    """When GNN agrees with template matcher, consensus.confidence_band
    must be 'high' — this is what unlocks the auto-select UX."""
    _require_ckpt(ckpt_present)

    payload = json.loads((FIXTURES / "bjt_diff_amp_correct_v1.json").read_text())
    resp = client.post(
        "/api/v1/topology/suggest",
        json={"netlist_v2": payload},
    )
    body = resp.json()
    assert body["consensus"]["agreed"] is True
    assert body["consensus"]["confidence_band"] == "high"
