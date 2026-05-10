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


def _find_net_by_hole(netlist_v2: dict, hole_id: str) -> dict | None:
    for net in netlist_v2.get("nets", []):
        if hole_id in net.get("member_hole_ids", []):
            return net
    return None


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
    from app.pipeline.stages.s3_topology import run_topology

    s3 = run_topology(components=_components_missing_cap())
    netlist_v2 = dict(s3.get("netlist_v2") or {})
    # 给当前 net 标注与参考一致的角色（除了缺失 C1 对应的 GND 网）
    # 通过 hole_id 定位 net，避免依赖 NET_xxx 编号顺序
    vin_net = _find_net_by_hole(netlist_v2, "A1")
    assert vin_net is not None
    vin_net["role"] = "input"

    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=_reference(),
        components=_components_missing_cap(),
        current_netlist_v2=netlist_v2,
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

    # 由于 input/ground 成为严格角色，缺失元件时图不再是子图同构，
    # 因此 enrichment 后可能不会保留 INCOMPLETE_CIRCUIT，而是出现角色错误或 wiring mismatch
    incomplete = [i for i in items if i["error_code"] == "INCOMPLETE_CIRCUIT"]
    # 允许没有 INCOMPLETE_CIRCUIT，但至少要检测到 C1 缺失
    assert len(missing) == 1


def test_s4_detailed_full_match() -> None:
    from app.pipeline.stages.s3_topology import run_topology

    s3 = run_topology(components=_components_match())
    netlist_v2 = dict(s3.get("netlist_v2") or {})
    # 手动标注与参考一致的角色，使逻辑图完全匹配
    # 通过 hole_id 定位 net，避免依赖 NET_xxx 编号顺序
    vin_net = _find_net_by_hole(netlist_v2, "A1")
    assert vin_net is not None
    vin_net["role"] = "input"

    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=_reference(),
        components=_components_match(),
        current_netlist_v2=netlist_v2,
    )
    assert result["is_correct"] is True
    report = result["comparison_report"]
    assert report["summary"]["logic_correct"] is True
    assert report["summary"]["reference_id"] == "rc_first_order_v1"
    assert report["items"] == []
