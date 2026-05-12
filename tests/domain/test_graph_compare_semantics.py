"""
语义级逻辑图比较测试 — 覆盖 role_label、功能引脚、对称等价、短路检测。
"""

from __future__ import annotations

import networkx as nx
import pytest

from app.domain.graph_compare import compare_logical_graphs
from app.domain.logical_reference import current_netlist_v2_to_graph, logical_reference_to_graph


class TestSameLogicDifferentHoles:
    """相同逻辑，不同 hole_id，结果应为 true。"""

    def test_same_logic_different_holes(self) -> None:
        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {
                    "ref_id": "R1",
                    "type": "Resistor",
                    "pins": [
                        {"pin": "pin1", "net": "VIN"},
                        {"pin": "pin2", "net": "GND"},
                    ],
                },
            ],
            "nets": [
                {"net": "VIN", "role": "input"},
                {"net": "GND", "role": "ground"},
            ],
        }
        cur_netlist = {
            "components": [
                {
                    "component_id": "R2",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "N_0", "hole_id": "X1"},
                        {"pin_name": "pin2", "electrical_net_id": "N_1", "hole_id": "Y99"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "N_0", "role": "input"},
                {"electrical_net_id": "N_1", "role": "ground", "role_label": "GND"},
            ],
        }
        ref = logical_reference_to_graph(ref_payload)
        cur = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist)
        assert result["logic_correct"] is True
        assert result["details"]["match_type"] == "full_isomorphism"


class TestRoleLabelMatch:
    """UI1/UI2/UO1/UO2/VCC/VEE/GND 正确标注，结果应为 true。"""

    def test_role_label_match(self) -> None:
        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {
                    "ref_id": "U1",
                    "type": "IC",
                    "pins": [
                        {"pin": "IN1", "net": "UI1"},
                        {"pin": "IN2", "net": "UI2"},
                        {"pin": "OUT1", "net": "UO1"},
                        {"pin": "OUT2", "net": "UO2"},
                        {"pin": "VCC", "net": "VCC"},
                        {"pin": "VEE", "net": "VEE"},
                        {"pin": "GND", "net": "GND"},
                    ],
                },
            ],
            "nets": [
                {"net": "UI1", "role": "input", "role_label": "UI1"},
                {"net": "UI2", "role": "input", "role_label": "UI2"},
                {"net": "UO1", "role": "output", "role_label": "UO1"},
                {"net": "UO2", "role": "output", "role_label": "UO2"},
                {"net": "VCC", "role": "power", "role_label": "VCC"},
                {"net": "VEE", "role": "power", "role_label": "VEE"},
                {"net": "GND", "role": "ground", "role_label": "GND"},
            ],
        }
        cur_netlist = {
            "components": [
                {
                    "component_id": "U2",
                    "component_type": "IC",
                    "pins": [
                        {"pin_name": "IN1", "electrical_net_id": "N0"},
                        {"pin_name": "IN2", "electrical_net_id": "N1"},
                        {"pin_name": "OUT1", "electrical_net_id": "N2"},
                        {"pin_name": "OUT2", "electrical_net_id": "N3"},
                        {"pin_name": "VCC", "electrical_net_id": "N4"},
                        {"pin_name": "VEE", "electrical_net_id": "N5"},
                        {"pin_name": "GND", "electrical_net_id": "N6"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "N0", "role": "input", "role_label": "UI1"},
                {"electrical_net_id": "N1", "role": "input", "role_label": "UI2"},
                {"electrical_net_id": "N2", "role": "output", "role_label": "UO1"},
                {"electrical_net_id": "N3", "role": "output", "role_label": "UO2"},
                {"electrical_net_id": "N4", "role": "power", "role_label": "VCC"},
                {"electrical_net_id": "N5", "role": "power", "role_label": "VEE"},
                {"electrical_net_id": "N6", "role": "ground", "role_label": "GND"},
            ],
        }
        ref = logical_reference_to_graph(ref_payload)
        cur = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist)
        assert result["logic_correct"] is True
        assert result["similarity"] == 1.0


class TestRoleLabelMismatch:
    """UI1 和 UI2 互换，没有 symmetry_groups，结果应为 false。"""

    def test_role_label_mismatch(self) -> None:
        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {
                    "ref_id": "U1",
                    "type": "IC",
                    "pins": [
                        {"pin": "IN1", "net": "UI1"},
                        {"pin": "IN2", "net": "UI2"},
                    ],
                },
            ],
            "nets": [
                {"net": "UI1", "role": "input", "role_label": "UI1"},
                {"net": "UI2", "role": "input", "role_label": "UI2"},
            ],
        }
        cur_netlist = {
            "components": [
                {
                    "component_id": "U2",
                    "component_type": "IC",
                    "pins": [
                        {"pin_name": "IN1", "electrical_net_id": "N0"},
                        {"pin_name": "IN2", "electrical_net_id": "N1"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "N0", "role": "input", "role_label": "UI2"},
                {"electrical_net_id": "N1", "role": "input", "role_label": "UI1"},
            ],
        }
        ref = logical_reference_to_graph(ref_payload)
        cur = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist)
        assert result["logic_correct"] is False
        items = result["report"]["items"]
        assert any(
            i["error_code"] in {"ROLE_LABEL_MISMATCH", "INPUT_NODE_MISMATCH", "WRONG_CONNECTION"}
            for i in items
        )


class TestVccVeeMismatch:
    """VCC 和 VEE 搞反，结果应为 false。"""

    def test_vcc_vee_mismatch(self) -> None:
        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {
                    "ref_id": "U1",
                    "type": "IC",
                    "pins": [
                        {"pin": "VCC", "net": "VCC"},
                        {"pin": "VEE", "net": "VEE"},
                    ],
                },
            ],
            "nets": [
                {"net": "VCC", "role": "power", "role_label": "VCC"},
                {"net": "VEE", "role": "power", "role_label": "VEE"},
            ],
        }
        cur_netlist = {
            "components": [
                {
                    "component_id": "U2",
                    "component_type": "IC",
                    "pins": [
                        {"pin_name": "VCC", "electrical_net_id": "N0"},
                        {"pin_name": "VEE", "electrical_net_id": "N1"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "N0", "role": "power", "role_label": "VEE"},
                {"electrical_net_id": "N1", "role": "power", "role_label": "VCC"},
            ],
        }
        ref = logical_reference_to_graph(ref_payload)
        cur = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist)
        assert result["logic_correct"] is False
        items = result["report"]["items"]
        assert any(
            i["error_code"] in {"ROLE_LABEL_MISMATCH", "POWER_NODE_MISMATCH", "WRONG_CONNECTION"}
            for i in items
        )


class TestTransistorPinMismatch:
    """三极管 collector/emitter 接反，结果应为 false，包含 PIN_ROLE_MISMATCH 或 WRONG_CONNECTION。"""

    def test_transistor_pin_mismatch(self) -> None:
        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {
                    "ref_id": "Q1",
                    "type": "Transistor",
                    "pins": [
                        {"pin": "E", "net": "GND"},
                        {"pin": "B", "net": "BASE"},
                        {"pin": "C", "net": "VCC"},
                    ],
                },
            ],
            "nets": [
                {"net": "GND", "role": "ground", "role_label": "GND"},
                {"net": "BASE", "role": "signal"},
                {"net": "VCC", "role": "power", "role_label": "VCC"},
            ],
        }
        # C/E 接反
        cur_netlist = {
            "components": [
                {
                    "component_id": "Q2",
                    "component_type": "Transistor",
                    "pins": [
                        {"pin_name": "E", "electrical_net_id": "N0", "polarity_role": "E"},
                        {"pin_name": "B", "electrical_net_id": "N1", "polarity_role": "B"},
                        {"pin_name": "C", "electrical_net_id": "N2", "polarity_role": "C"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "N0", "role": "power", "role_label": "VCC"},
                {"electrical_net_id": "N1", "role": "signal"},
                {"electrical_net_id": "N2", "role": "ground", "role_label": "GND"},
            ],
        }
        ref = logical_reference_to_graph(ref_payload)
        cur = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist)
        assert result["logic_correct"] is False
        items = result["report"]["items"]
        assert any(
            i["error_code"] in {"PIN_ROLE_MISMATCH", "WRONG_CONNECTION"}
            for i in items
        )


class TestEquivalentWithExtra:
    """参考逻辑存在，额外元件不影响关键网络，结果 match_type=equivalent_with_extra，warning。"""

    def test_equivalent_with_extra(self) -> None:
        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {
                    "ref_id": "R1",
                    "type": "Resistor",
                    "pins": [
                        {"pin": "pin1", "net": "VCC"},
                        {"pin": "pin2", "net": "GND"},
                    ],
                },
            ],
            "nets": [
                {"net": "VCC", "role": "power", "role_label": "VCC"},
                {"net": "GND", "role": "ground", "role_label": "GND"},
            ],
        }
        # 多了一个电阻 R2 接在独立的信号网络上，不影响 VCC/GND
        cur_netlist = {
            "components": [
                {
                    "component_id": "R1",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "N0"},
                        {"pin_name": "pin2", "electrical_net_id": "N1"},
                    ],
                },
                {
                    "component_id": "R2",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "N2"},
                        {"pin_name": "pin2", "electrical_net_id": "N3"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "N0", "role": "power", "role_label": "VCC"},
                {"electrical_net_id": "N1", "role": "ground", "role_label": "GND"},
                {"electrical_net_id": "N2", "role": "signal"},
                {"electrical_net_id": "N3", "role": "signal"},
            ],
        }
        ref = logical_reference_to_graph(ref_payload)
        cur = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist)
        assert result["logic_correct"] is True
        assert result["details"]["match_type"] == "equivalent_with_extra"
        items = result["report"]["items"]
        assert any(i["severity"] == "warning" for i in items)


class TestShortCircuit:
    """UO1/UO2 或 VCC/GND 被合并，结果应为 false，包含 SHORT_CIRCUIT。"""

    def test_short_circuit_uo1_uo2(self) -> None:
        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {
                    "ref_id": "U1",
                    "type": "IC",
                    "pins": [
                        {"pin": "OUT1", "net": "UO1"},
                        {"pin": "OUT2", "net": "UO2"},
                    ],
                },
            ],
            "nets": [
                {"net": "UO1", "role": "output", "role_label": "UO1"},
                {"net": "UO2", "role": "output", "role_label": "UO2"},
            ],
        }
        # 把 UO1/UO2 短接到同一网络
        cur_netlist = {
            "components": [
                {
                    "component_id": "U2",
                    "component_type": "IC",
                    "pins": [
                        {"pin_name": "OUT1", "electrical_net_id": "N0"},
                        {"pin_name": "OUT2", "electrical_net_id": "N0"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "N0", "role": "output"},
            ],
        }
        ref = logical_reference_to_graph(ref_payload)
        cur = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist)
        assert result["logic_correct"] is False
        items = result["report"]["items"]
        assert any(i["error_code"] == "SHORT_CIRCUIT" for i in items)

    def test_short_circuit_vcc_gnd(self) -> None:
        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {
                    "ref_id": "R1",
                    "type": "Resistor",
                    "pins": [
                        {"pin": "pin1", "net": "VCC"},
                        {"pin": "pin2", "net": "GND"},
                    ],
                },
            ],
            "nets": [
                {"net": "VCC", "role": "power", "role_label": "VCC"},
                {"net": "GND", "role": "ground", "role_label": "GND"},
            ],
        }
        # VCC 和 GND 被短接
        cur_netlist = {
            "components": [
                {
                    "component_id": "R1",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "N0"},
                        {"pin_name": "pin2", "electrical_net_id": "N0"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "N0", "role": "power", "role_label": "VCC"},
            ],
        }
        ref = logical_reference_to_graph(ref_payload)
        cur = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist)
        assert result["logic_correct"] is False
        items = result["report"]["items"]
        assert any(i["error_code"] == "SHORT_CIRCUIT" for i in items)


class TestSymmetryAllowed:
    """UI1/UI2、UO1/UO2 互换，reference 声明 symmetry_groups swap_allowed，结果应为 true。"""

    def test_symmetry_allowed(self) -> None:
        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {
                    "ref_id": "U1",
                    "type": "IC",
                    "pins": [
                        {"pin": "IN1", "net": "UI1"},
                        {"pin": "IN2", "net": "UI2"},
                        {"pin": "OUT1", "net": "UO1"},
                        {"pin": "OUT2", "net": "UO2"},
                    ],
                },
            ],
            "nets": [
                {"net": "UI1", "role": "input", "role_label": "UI1"},
                {"net": "UI2", "role": "input", "role_label": "UI2"},
                {"net": "UO1", "role": "output", "role_label": "UO1"},
                {"net": "UO2", "role": "output", "role_label": "UO2"},
            ],
            "symmetry_groups": [
                {
                    "name": "diff_pair",
                    "nets": [["UI1", "UI2"], ["UO1", "UO2"]],
                    "mode": "swap_allowed",
                }
            ],
        }
        # UI1/UI2 互换，UO1/UO2 互换
        cur_netlist = {
            "components": [
                {
                    "component_id": "U2",
                    "component_type": "IC",
                    "pins": [
                        {"pin_name": "IN1", "electrical_net_id": "N0"},
                        {"pin_name": "IN2", "electrical_net_id": "N1"},
                        {"pin_name": "OUT1", "electrical_net_id": "N2"},
                        {"pin_name": "OUT2", "electrical_net_id": "N3"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "N0", "role": "input", "role_label": "UI2"},
                {"electrical_net_id": "N1", "role": "input", "role_label": "UI1"},
                {"electrical_net_id": "N2", "role": "output", "role_label": "UO2"},
                {"electrical_net_id": "N3", "role": "output", "role_label": "UO1"},
            ],
        }
        ref = logical_reference_to_graph(ref_payload)
        cur = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist)
        assert result["logic_correct"] is True
        assert result["details"]["match_type"] == "equivalent_with_allowed_symmetry"


class TestUnsupportedReferenceFormat:
    """输入 format="labguardian_ref_v4"，返回 UNSUPPORTED_REFERENCE_FORMAT。"""

    def test_unsupported_reference_format(self) -> None:
        from app.pipeline.stages.s4_validate import run_validate

        reference = {
            "format": "labguardian_ref_v4",
            "netlist_v2": {
                "components": [],
                "nets": [],
            },
        }
        result = run_validate(
            topology_graph={"nodes": [], "links": []},
            reference_circuit=reference,
            components=[],
        )
        codes = {item["error_code"] for item in result["comparison_report"]["items"]}
        assert "UNSUPPORTED_REFERENCE_FORMAT" in codes
        assert result["is_correct"] is False
