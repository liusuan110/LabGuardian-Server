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
        "compare_options": {
            "ignore_hole_id": True,
            "ignore_component_id": True,
            "ignore_polarity": True,
        },
    }


def _components() -> list[dict]:
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
                {
                    "pin_id": 2,
                    "pin_name": "pin2",
                    "hole_id": "PWR_MINUS",
                    "electrical_node_id": "PWR_MINUS",
                },
            ],
        },
    ]


def _find_net_by_hole(netlist_v2: dict, hole_id: str) -> dict | None:
    for net in netlist_v2.get("nets", []):
        if hole_id in net.get("member_hole_ids", []):
            return net
    return None


def test_s4_logical_reference_v1_full_match() -> None:
    from app.pipeline.stages.s3_topology import run_topology

    s3 = run_topology(components=_components())
    netlist_v2 = dict(s3.get("netlist_v2") or {})
    # 手动标注与参考一致的角色，使逻辑图完全匹配
    # 通过 hole_id 定位 net，避免依赖 NET_xxx 编号顺序
    vin_net = _find_net_by_hole(netlist_v2, "A1")
    assert vin_net is not None
    vin_net["role"] = "input"

    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=_reference(),
        components=_components(),
        current_netlist_v2=netlist_v2,
    )

    assert result["is_correct"] is True
    assert result["similarity"] == 1.0
    assert result["comparison_report"]["summary"]["comparison_mode"] == "logical_graph"
    assert result["comparison_report"]["hole_errors"] == []
    assert result["comparison_report"]["polarity_errors"] == []


def test_s4_logical_reference_v1_missing_component() -> None:
    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=_reference(),
        components=_components()[:1],
    )

    codes = {item["error_code"] for item in result["comparison_report"]["items"]}
    assert result["is_correct"] is False
    assert "COMPONENT_MISSING" in codes


def test_s4_keeps_labguardian_ref_v4_branch() -> None:
    reference = {
        "netlist_v2": {
            "components": [
                {
                    "component_id": "R1",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1", "electrical_node_id": "ROW_1_L"},
                        {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3", "electrical_node_id": "ROW_3_L"},
                    ],
                }
            ],
            "nets": [],
        }
    }

    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=reference,
        components=[
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1", "electrical_node_id": "ROW_1_L"},
                    {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3", "electrical_node_id": "ROW_3_L"},
                ],
            }
        ],
    )

    assert result["comparison_report"]["summary"].get("comparison_mode") != "logical_graph"
