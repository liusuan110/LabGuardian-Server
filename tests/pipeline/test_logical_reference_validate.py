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
    from app.pipeline.stages.s3_topology import run_topology

    components = _components()[:1]
    s3 = run_topology(components=components)
    netlist_v2 = dict(s3.get("netlist_v2") or {})

    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=_reference(),
        components=components,
        current_netlist_v2=netlist_v2,
    )

    codes = {item["error_code"] for item in result["comparison_report"]["items"]}
    assert result["is_correct"] is False
    assert "COMPONENT_MISSING" in codes


def test_s4_unsupported_reference_format() -> None:
    """非 logical_reference_v1 格式应返回 UNSUPPORTED_REFERENCE_FORMAT。"""
    reference = {
        "format": "labguardian_ref_v4",
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

    codes = {item["error_code"] for item in result["comparison_report"]["items"]}
    assert "UNSUPPORTED_REFERENCE_FORMAT" in codes
    assert result["is_correct"] is False
    assert result["comparison_report"]["summary"]["comparison_mode"] == "logical_graph"


def test_s4_reference_not_set() -> None:
    """reference_circuit 为空时应返回 REFERENCE_NOT_SET。"""
    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=None,
        components=[],
    )

    codes = {item["error_code"] for item in result["comparison_report"]["items"]}
    assert "REFERENCE_NOT_SET" in codes
    assert result["is_correct"] is False
    assert result["comparison_report"]["summary"]["comparison_mode"] == "logical_graph"


def _reference_led() -> dict:
    return {
        "format": "logical_reference_v1",
        "reference_id": "led_basic_v1",
        "name": "LED 基础电路",
        "components": [
            {
                "ref_id": "R1",
                "type": "Resistor",
                "pins": [{"pin": "pin1", "net": "VCC"}, {"pin": "pin2", "net": "LED_NET"}],
            },
            {
                "ref_id": "D1",
                "type": "LED",
                "pins": [{"pin": "anode", "net": "LED_NET"}, {"pin": "cathode", "net": "GND"}],
            },
        ],
        "nets": [
            {"net": "VCC", "role": "power"},
            {"net": "LED_NET", "role": "signal"},
            {"net": "GND", "role": "ground"},
        ],
    }


def _components_led_match() -> list[dict]:
    return [
        {
            "component_id": "R1",
            "component_type": "Resistor",
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1", "electrical_node_id": "ROW_1_L"},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3", "electrical_node_id": "ROW_3_L"},
            ],
        },
        {
            "component_id": "D1",
            "component_type": "LED",
            "polarity": "forward",
            "pins": [
                {"pin_id": 1, "pin_name": "anode", "hole_id": "B3", "electrical_node_id": "ROW_3_L"},
                {"pin_id": 2, "pin_name": "cathode", "hole_id": "B5", "electrical_node_id": "ROW_5_L"},
            ],
        },
    ]


def _find_net_by_hole(netlist_v2: dict, hole_id: str) -> dict | None:
    for net in netlist_v2.get("nets", []):
        if hole_id in net.get("member_hole_ids", []):
            return net
    return None


def test_resistor_pin_swap_still_correct() -> None:
    from app.pipeline.stages.s3_topology import run_topology

    ref = {
        "format": "logical_reference_v1",
        "reference_id": "resistor_swap_v1",
        "name": "电阻交换引脚",
        "components": [
            {
                "ref_id": "R1",
                "type": "Resistor",
                "pins": [{"pin": "pin1", "net": "NET_A"}, {"pin": "pin2", "net": "NET_B"}],
            },
        ],
        "nets": [
            {"net": "NET_A", "role": "signal"},
            {"net": "NET_B", "role": "signal"},
        ],
    }
    components = [
        {
            "component_id": "R1",
            "component_type": "Resistor",
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1", "electrical_node_id": "ROW_1_L"},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3", "electrical_node_id": "ROW_3_L"},
            ],
        },
    ]
    s3 = run_topology(components=components)
    netlist_v2 = dict(s3.get("netlist_v2") or {})

    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=ref,
        components=components,
        current_netlist_v2=netlist_v2,
    )
    assert result["is_correct"] is True
    assert result["similarity"] == 1.0


def test_led_polarity_ignored() -> None:
    from app.pipeline.stages.s3_topology import run_topology

    components = _components_led_match()
    # 反转 polarity 字段，但引脚连接不变
    components[1]["polarity"] = "reversed"

    s3 = run_topology(components=components)
    netlist_v2 = dict(s3.get("netlist_v2") or {})
    vin_net = _find_net_by_hole(netlist_v2, "A1")
    gnd_net = _find_net_by_hole(netlist_v2, "B5")
    if vin_net:
        vin_net["role"] = "power"
    if gnd_net:
        gnd_net["role"] = "ground"

    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=_reference_led(),
        components=components,
        current_netlist_v2=netlist_v2,
    )
    assert result["is_correct"] is True
    assert result["comparison_report"]["summary"].get("ignore_polarity") is True
    assert result["comparison_report"]["polarity_errors"] == []


def test_open_circuit_detected() -> None:
    from app.pipeline.stages.s3_topology import run_topology

    ref = {
        "format": "logical_reference_v1",
        "reference_id": "series_v1",
        "name": "串联电路",
        "components": [
            {
                "ref_id": "R1",
                "type": "Resistor",
                "pins": [{"pin": "pin1", "net": "VCC"}, {"pin": "pin2", "net": "MID"}],
            },
            {
                "ref_id": "R2",
                "type": "Resistor",
                "pins": [{"pin": "pin1", "net": "MID"}, {"pin": "pin2", "net": "GND"}],
            },
        ],
        "nets": [
            {"net": "VCC", "role": "power"},
            {"net": "MID", "role": "signal"},
            {"net": "GND", "role": "ground"},
        ],
    }
    # 故意把 R1.pin2 和 R2.pin1 放到面包板同一行的左右两侧（不通），形成开路
    components = [
        {
            "component_id": "R1",
            "component_type": "Resistor",
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1", "electrical_node_id": "ROW_1_L"},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3", "electrical_node_id": "ROW_3_L"},
            ],
        },
        {
            "component_id": "R2",
            "component_type": "Resistor",
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "E3", "electrical_node_id": "ROW_3_R"},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "A5", "electrical_node_id": "ROW_5_L"},
            ],
        },
    ]
    s3 = run_topology(components=components)
    netlist_v2 = dict(s3.get("netlist_v2") or {})
    vcc_net = _find_net_by_hole(netlist_v2, "A1")
    gnd_net = _find_net_by_hole(netlist_v2, "A5")
    if vcc_net:
        vcc_net["role"] = "power"
    if gnd_net:
        gnd_net["role"] = "ground"

    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=ref,
        components=components,
        current_netlist_v2=netlist_v2,
    )
    assert result["is_correct"] is False
    codes = {item["error_code"] for item in result["comparison_report"]["items"]}
    assert "OPEN_CIRCUIT" in codes or "COMPONENT_MISSING" in codes or "WRONG_CONNECTION" in codes


def test_extra_connection_detected() -> None:
    from app.pipeline.stages.s3_topology import run_topology

    ref = {
        "format": "logical_reference_v1",
        "reference_id": "single_resistor_v1",
        "name": "单电阻",
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
    # 额外加一个电阻 R2
    components = [
        {
            "component_id": "R1",
            "component_type": "Resistor",
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1", "electrical_node_id": "ROW_1_L"},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3", "electrical_node_id": "ROW_3_L"},
            ],
        },
        {
            "component_id": "R2",
            "component_type": "Resistor",
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "B1", "electrical_node_id": "ROW_1_L"},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "B3", "electrical_node_id": "ROW_3_L"},
            ],
        },
    ]
    s3 = run_topology(components=components)
    netlist_v2 = dict(s3.get("netlist_v2") or {})
    vcc_net = _find_net_by_hole(netlist_v2, "A1")
    gnd_net = _find_net_by_hole(netlist_v2, "A3")
    if vcc_net:
        vcc_net["role"] = "power"
    if gnd_net:
        gnd_net["role"] = "ground"

    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=ref,
        components=components,
        current_netlist_v2=netlist_v2,
    )
    assert result["is_correct"] is False
    codes = {item["error_code"] for item in result["comparison_report"]["items"]}
    assert "COMPONENT_EXTRA" in codes


def test_component_missing() -> None:
    from app.pipeline.stages.s3_topology import run_topology

    components = _components_led_match()[:1]  # 只有 R1，缺少 D1
    s3 = run_topology(components=components)
    netlist_v2 = dict(s3.get("netlist_v2") or {})
    vcc_net = _find_net_by_hole(netlist_v2, "A1")
    if vcc_net:
        vcc_net["role"] = "power"

    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=_reference_led(),
        components=components,
        current_netlist_v2=netlist_v2,
    )
    assert result["is_correct"] is False
    codes = {item["error_code"] for item in result["comparison_report"]["items"]}
    assert "COMPONENT_MISSING" in codes


def test_component_extra() -> None:
    from app.pipeline.stages.s3_topology import run_topology

    components = _components_led_match()
    # 额外添加一个电容
    components.append({
        "component_id": "C1",
        "component_type": "CapacitorCeramic",
        "pins": [
            {"pin_id": 1, "pin_name": "pin1", "hole_id": "C1", "electrical_node_id": "ROW_1_L"},
            {"pin_id": 2, "pin_name": "pin2", "hole_id": "C3", "electrical_node_id": "ROW_3_L"},
        ],
    })

    s3 = run_topology(components=components)
    netlist_v2 = dict(s3.get("netlist_v2") or {})
    vcc_net = _find_net_by_hole(netlist_v2, "A1")
    gnd_net = _find_net_by_hole(netlist_v2, "B5")
    if vcc_net:
        vcc_net["role"] = "power"
    if gnd_net:
        gnd_net["role"] = "ground"

    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=_reference_led(),
        components=components,
        current_netlist_v2=netlist_v2,
    )
    assert result["is_correct"] is False
    codes = {item["error_code"] for item in result["comparison_report"]["items"]}
    assert "COMPONENT_EXTRA" in codes


def test_vcc_gnd_role_error() -> None:
    from app.pipeline.stages.s3_topology import run_topology

    ref = {
        "format": "logical_reference_v1",
        "reference_id": "led_power_gnd_v1",
        "name": "LED 电源地测试",
        "components": [
            {
                "ref_id": "D1",
                "type": "LED",
                "pins": [{"pin": "anode", "net": "VCC"}, {"pin": "cathode", "net": "GND"}],
            },
        ],
        "nets": [
            {"net": "VCC", "role": "power"},
            {"net": "GND", "role": "ground"},
        ],
    }
    components = [
        {
            "component_id": "D1",
            "component_type": "LED",
            "pins": [
                {"pin_id": 1, "pin_name": "anode", "hole_id": "A1", "electrical_node_id": "ROW_1_L"},
                {"pin_id": 2, "pin_name": "cathode", "hole_id": "A3", "electrical_node_id": "ROW_3_L"},
            ],
        },
    ]
    s3 = run_topology(components=components)
    netlist_v2 = dict(s3.get("netlist_v2") or {})
    # 故意把 VCC 和 GND 的角色标反
    for net in netlist_v2.get("nets", []):
        if "A1" in net.get("member_hole_ids", []):
            net["role"] = "ground"
            net["power_role"] = "GND"
        elif "A3" in net.get("member_hole_ids", []):
            net["role"] = "power"
            net["power_role"] = "VCC"

    result = run_validate(
        topology_graph={"nodes": [], "links": []},
        reference_circuit=ref,
        components=components,
        current_netlist_v2=netlist_v2,
    )
    assert result["is_correct"] is False
    items = result["comparison_report"]["items"]
    role_errors = [i for i in items if i["error_code"] in {
        "POWER_NODE_MISMATCH", "GROUND_NODE_MISMATCH", "ROLE_MISMATCH"
    }]
    assert len(role_errors) >= 1
