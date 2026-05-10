from __future__ import annotations

from app.pipeline.stages.s4_validate import run_validate


def _reference() -> dict:
    return {
        "format": "logical_reference_v1",
        "reference_id": "rc_first_order_v1",
        "name": "一阶 RC 电路",
        "components": [
            {
                "ref_id": "R1",
                "type": "Resistor",
                "pins": [{"pin": "pin1", "net": "VIN"}, {"pin": "pin2", "net": "VC"}],
            },
            {
                "ref_id": "C1",
                "type": "CapacitorCeramic",
                "pins": [{"pin": "pin1", "net": "VC"}, {"pin": "pin2", "net": "GND"}],
            },
        ],
        "nets": [
            {"net": "VIN", "role": "input"},
            {"net": "VC", "role": "signal"},
            {"net": "GND", "role": "ground"},
        ],
    }


def _components_match() -> list[dict]:
    return [
        {
            "component_id": "R2",
            "component_type": "Resistor",
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1", "electrical_node_id": "ROW_1_L"},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3", "electrical_node_id": "ROW_3_L"},
            ],
        },
        {
            "component_id": "C2",
            "component_type": "CapacitorCeramic",
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "B3", "electrical_node_id": "ROW_3_L"},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "PWR_MINUS", "electrical_node_id": "PWR_MINUS"},
            ],
        },
    ]


def _components_missing_cap() -> list[dict]:
    return [
        {
            "component_id": "R2",
            "component_type": "Resistor",
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1", "electrical_node_id": "ROW_1_L"},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3", "electrical_node_id": "ROW_3_L"},
            ],
        },
    ]


def test_s4_detailed_missing_component() -> None:
    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=_reference(),
        components=_components_missing_cap(),
    )
    report = result["comparison_report"]
    assert report["summary"]["comparison_mode"] == "logical_graph"
    assert report["summary"]["reference_id"] == "rc_first_order_v1"
    assert report["summary"]["reference_name"] == "一阶 RC 电路"

    items = report["items"]
    missing = [i for i in items if i["error_code"] == "COMPONENT_MISSING"]
    assert len(missing) == 1
    assert missing[0]["expected"]["ref_id"] == "C1"
    assert missing[0]["component_ref"]["ref_id"] == "C1"
    assert missing[0]["title"] == "缺元件"

    incomplete = [i for i in items if i["error_code"] == "INCOMPLETE_CIRCUIT"]
    assert len(incomplete) == 1
    assert incomplete[0]["title"] == "电路未完成"


def test_s4_detailed_full_match() -> None:
    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=_reference(),
        components=_components_match(),
    )
    assert result["is_correct"] is True
    report = result["comparison_report"]
    assert report["summary"]["logic_correct"] is True
    assert report["summary"]["reference_id"] == "rc_first_order_v1"
    assert report["items"] == []
