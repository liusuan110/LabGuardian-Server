"""Tests for POST /api/v1/pipeline/compare-netlist"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _reference_payload() -> dict:
    return {
        "format": "logical_reference_v1",
        "reference_id": "test_rc_v1",
        "name": "Test RC",
        "components": [
            {
                "ref_id": "R1",
                "type": "Resistor",
                "pins": [{"pin": "pin1", "net": "VCC"}, {"pin": "pin2", "net": "GND"}],
            },
        ],
        "nets": [
            {"net": "VCC", "role": "power"},
            {"net": "GND", "role": "ground"},
        ],
    }


def _netlist_v2_match() -> dict:
    return {
        "components": [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {"pin_id": 1, "pin_name": "pin1", "electrical_net_id": "NET_001"},
                    {"pin_id": 2, "pin_name": "pin2", "electrical_net_id": "NET_002"},
                ],
            },
        ],
        "nets": [
            {"electrical_net_id": "NET_001", "role": "power", "role_label": "VCC", "member_hole_ids": ["A1"]},
            {"electrical_net_id": "NET_002", "role": "ground", "role_label": "GND", "member_hole_ids": ["A3"]},
        ],
    }


def test_compare_netlist_full_match(client: TestClient) -> None:
    resp = client.post("/api/v1/pipeline/compare-netlist", json={
        "reference_circuit": _reference_payload(),
        "current_netlist_v2": _netlist_v2_match(),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is True
    assert data["similarity"] == 1.0
    assert data["comparison_report"]["summary"]["comparison_mode"] == "logical_graph"
    assert data["comparison_report"]["summary"]["strict_functional_pin_roles"] is True
    assert data["comparison_report"]["summary"]["equivalence_rule"] == "logical_topology_with_port_semantics"


def test_compare_netlist_missing_component(client: TestClient) -> None:
    netlist = {
        "components": [],
        "nets": [
            {"electrical_net_id": "NET_001", "role": "power", "member_hole_ids": ["A1"]},
        ],
    }
    resp = client.post("/api/v1/pipeline/compare-netlist", json={
        "reference_circuit": _reference_payload(),
        "current_netlist_v2": netlist,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is False
    codes = {item["error_code"] for item in data["comparison_report"]["items"]}
    assert "COMPONENT_MISSING" in codes


def test_compare_netlist_requires_reference(client: TestClient) -> None:
    resp = client.post("/api/v1/pipeline/compare-netlist", json={
        "current_netlist_v2": _netlist_v2_match(),
    })
    assert resp.status_code == 400


def test_compare_netlist_requires_logical_reference_v1(client: TestClient) -> None:
    resp = client.post("/api/v1/pipeline/compare-netlist", json={
        "reference_circuit": {"format": "legacy_v4"},
        "current_netlist_v2": _netlist_v2_match(),
    })
    assert resp.status_code == 400
