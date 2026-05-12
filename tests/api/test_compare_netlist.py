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


def _rc_reference_with_vlp() -> dict:
    return {
        "format": "logical_reference_v1",
        "reference_id": "rc_vlp_v1",
        "name": "RC VLP",
        "components": [
            {
                "ref_id": "R1",
                "type": "Resistor",
                "pins": [{"pin": "pin1", "net": "VIN"}, {"pin": "pin2", "net": "VLP"}],
            },
            {
                "ref_id": "C1",
                "type": "CapacitorCeramic",
                "pins": [{"pin": "pin1", "net": "VLP"}, {"pin": "pin2", "net": "GND"}],
            },
        ],
        "nets": [
            {"net": "VIN", "role": "input"},
            {"net": "VLP", "role": "signal"},
            {"net": "GND", "role": "ground"},
        ],
    }


def _rc_netlist_unlabeled() -> dict:
    return {
        "components": [
            {
                "component_id": "R2",
                "component_type": "Resistor",
                "pins": [
                    {"pin_id": 1, "pin_name": "pin1", "electrical_net_id": "NET_001"},
                    {"pin_id": 2, "pin_name": "pin2", "electrical_net_id": "NET_002"},
                ],
            },
            {
                "component_id": "C2",
                "component_type": "CapacitorCeramic",
                "pins": [
                    {"pin_id": 1, "pin_name": "pin1", "electrical_net_id": "NET_002"},
                    {"pin_id": 2, "pin_name": "pin2", "electrical_net_id": "NET_003"},
                ],
            },
        ],
        "nets": [
            {"electrical_net_id": "NET_001", "member_hole_ids": ["A1"]},
            {"electrical_net_id": "NET_002", "member_hole_ids": ["A3", "B3"]},
            {"electrical_net_id": "NET_003", "member_hole_ids": ["B5"]},
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
    assert data["comparison_report"]["summary"]["report_layers"]["reference_compare"]["included"] is True


def test_compare_netlist_infers_canonical_vlp(client: TestClient) -> None:
    resp = client.post("/api/v1/pipeline/compare-netlist", json={
        "reference_circuit": _rc_reference_with_vlp(),
        "current_netlist_v2": _rc_netlist_unlabeled(),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is True
    inferred = data["comparison_report"]["summary"]["net_normalization"]["inferred_aliases"]
    assert any(item["source_id"] == "NET_002" and item["canonical_name"] == "VLP" for item in inferred)
    logical_nets = data["comparison_report"]["summary"]["net_normalization"]["logical_nets"]
    assert any(net["source_id"] == "NET_002" and net["canonical_name"] == "VLP" for net in logical_nets)


def test_compare_netlist_accepts_minimal_port_annotations(client: TestClient) -> None:
    resp = client.post("/api/v1/pipeline/compare-netlist", json={
        "reference_circuit": _rc_reference_with_vlp(),
        "current_netlist_v2": _rc_netlist_unlabeled(),
        "port_annotations": [
            {
                "role": "input",
                "target": {"component_id": "R2", "pin_name": "pin1"},
            }
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is True
    applied = data["comparison_report"]["summary"]["port_annotations_applied"]
    assert applied == [
        {
            "role": "input",
            "role_label": "",
            "electrical_net_id": "NET_001",
            "source": "port_annotation",
            "resolved_by": "component_pin",
            "component_id": "R2",
            "pin_name": "pin1",
        }
    ]


def test_compare_netlist_manual_alias_overrides_inference(client: TestClient) -> None:
    resp = client.post("/api/v1/pipeline/compare-netlist", json={
        "reference_circuit": _rc_reference_with_vlp(),
        "current_netlist_v2": _rc_netlist_unlabeled(),
        "net_alias_assignments": [
            {"electrical_net_id": "NET_002", "canonical_name": "MID"}
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    logical_nets = data["comparison_report"]["summary"]["net_normalization"]["logical_nets"]
    assert any(net["source_id"] == "NET_002" and net["canonical_name"] == "MID" for net in logical_nets)


def test_compare_netlist_manual_merge_repairs_split_vlp(client: TestClient) -> None:
    netlist = _rc_netlist_unlabeled()
    netlist["components"][1]["pins"][0]["electrical_net_id"] = "NET_004"
    netlist["nets"].append({"electrical_net_id": "NET_004", "member_hole_ids": ["B3"]})
    resp = client.post("/api/v1/pipeline/compare-netlist", json={
        "reference_circuit": _rc_reference_with_vlp(),
        "current_netlist_v2": netlist,
        "net_merge_assignments": [
            {"source_net_ids": ["NET_002", "NET_004"], "target_canonical_name": "VLP"}
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is True
    merges = data["comparison_report"]["summary"]["net_normalization"]["applied_merges"]
    assert merges and merges[0]["merged_source_ids"] == ["NET_002", "NET_004"]


def test_compare_netlist_reference_nc_pin_may_be_unconnected(client: TestClient) -> None:
    reference = {
        "format": "logical_reference_v1",
        "reference_id": "nc_v1",
        "components": [
            {
                "ref_id": "J1",
                "type": "Header",
                "pins": [{"pin": "pin1", "net": "VIN"}, {"pin": "pin2", "nc": True}],
            }
        ],
        "nets": [{"net": "VIN", "role": "input"}],
    }
    netlist = {
        "components": [
            {
                "component_id": "J2",
                "component_type": "Header",
                "pins": [{"pin_id": 1, "pin_name": "pin1", "electrical_net_id": "NET_001"}],
            }
        ],
        "nets": [{"electrical_net_id": "NET_001"}],
    }
    resp = client.post("/api/v1/pipeline/compare-netlist", json={
        "reference_circuit": reference,
        "current_netlist_v2": netlist,
    })
    assert resp.status_code == 200
    assert resp.json()["is_correct"] is True


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
        "reference_circuit": {"meta": {"format": "labguardian_ref_v4"}, "netlist_v2": {}},
        "current_netlist_v2": _netlist_v2_match(),
    })
    assert resp.status_code == 400
    assert "不支持旧参考电路格式" in resp.json()["detail"]
