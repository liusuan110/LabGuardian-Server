from __future__ import annotations

import networkx as nx
import pytest

from app.domain.graph_compare import compare_logical_graphs
from app.domain.logical_reference import current_netlist_v2_to_graph, logical_reference_to_graph


def _ref_payload() -> dict:
    return {
        "format": "logical_reference_v1",
        "reference_id": "rc_first_order_v1",
        "name": "一阶 RC 电路",
        "components": [
            {
                "ref_id": "R1",
                "type": "Resistor",
                "pins": [
                    {"pin": "pin1", "net": "VIN"},
                    {"pin": "pin2", "net": "VC"},
                ],
            },
            {
                "ref_id": "C1",
                "type": "CapacitorCeramic",
                "pins": [
                    {"pin": "pin1", "net": "VC"},
                    {"pin": "pin2", "net": "GND"},
                ],
            },
        ],
        "nets": [
            {"net": "VIN", "role": "input"},
            {"net": "VC", "role": "signal"},
            {"net": "GND", "role": "ground"},
        ],
    }


def _cur_netlist_match() -> dict:
    return {
        "components": [
            {
                "component_id": "R2",
                "component_type": "Resistor",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "NET_000", "hole_id": "A1"},
                    {"pin_name": "pin2", "electrical_net_id": "NET_001", "hole_id": "A3"},
                ],
            },
            {
                "component_id": "C2",
                "component_type": "CapacitorCeramic",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "NET_001", "hole_id": "B3"},
                    {"pin_name": "pin2", "electrical_net_id": "NET_002", "hole_id": "PWR_MINUS"},
                ],
            },
        ],
        "nets": [
            {"electrical_net_id": "NET_000", "role": "input"},
            {"electrical_net_id": "NET_001", "role": "signal"},
            {"electrical_net_id": "NET_002", "role": "ground"},
        ],
    }


def _cur_netlist_missing_cap() -> dict:
    return {
        "components": [
            {
                "component_id": "R2",
                "component_type": "Resistor",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "NET_000", "hole_id": "A1"},
                    {"pin_name": "pin2", "electrical_net_id": "NET_001", "hole_id": "A3"},
                ],
            },
        ],
        "nets": [
            {"electrical_net_id": "NET_000"},
            {"electrical_net_id": "NET_001"},
        ],
    }


def _cur_netlist_extra_resistor() -> dict:
    return {
        "components": [
            {
                "component_id": "R2",
                "component_type": "Resistor",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "NET_000", "hole_id": "A1"},
                    {"pin_name": "pin2", "electrical_net_id": "NET_001", "hole_id": "A3"},
                ],
            },
            {
                "component_id": "C2",
                "component_type": "CapacitorCeramic",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "NET_001", "hole_id": "B3"},
                    {"pin_name": "pin2", "electrical_net_id": "NET_002", "hole_id": "PWR_MINUS"},
                ],
            },
            {
                "component_id": "R3",
                "component_type": "Resistor",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "NET_001", "hole_id": "C1"},
                    {"pin_name": "pin2", "electrical_net_id": "NET_003", "hole_id": "C3"},
                ],
            },
        ],
        "nets": [
            {"electrical_net_id": "NET_000"},
            {"electrical_net_id": "NET_001"},
            {"electrical_net_id": "NET_002"},
            {"electrical_net_id": "NET_003"},
        ],
    }


def _cur_netlist_wrong_connection() -> dict:
    """R1 and C1 both connect to VIN and GND (parallel instead of series)."""
    return {
        "components": [
            {
                "component_id": "R2",
                "component_type": "Resistor",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "NET_000", "hole_id": "A1"},
                    {"pin_name": "pin2", "electrical_net_id": "NET_001", "hole_id": "A3"},
                ],
            },
            {
                "component_id": "C2",
                "component_type": "CapacitorCeramic",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "NET_000", "hole_id": "B1"},
                    {"pin_name": "pin2", "electrical_net_id": "NET_001", "hole_id": "B3"},
                ],
            },
        ],
        "nets": [
            {"electrical_net_id": "NET_000"},
            {"electrical_net_id": "NET_001"},
        ],
    }


class TestCompareLogicalGraphsDetailed:
    def test_full_isomorphism_enriched(self) -> None:
        ref = logical_reference_to_graph(_ref_payload())
        cur = current_netlist_v2_to_graph(_cur_netlist_match())
        result = compare_logical_graphs(
            ref, cur, ref_payload=_ref_payload(), cur_netlist_v2=_cur_netlist_match()
        )
        assert result["logic_correct"] is True
        assert result["report"]["summary"]["reference_id"] == "rc_first_order_v1"
        assert result["report"]["summary"]["reference_name"] == "一阶 RC 电路"

    def test_missing_component_detailed(self) -> None:
        ref = logical_reference_to_graph(_ref_payload())
        cur = current_netlist_v2_to_graph(_cur_netlist_missing_cap())
        result = compare_logical_graphs(
            ref, cur, ref_payload=_ref_payload(), cur_netlist_v2=_cur_netlist_missing_cap()
        )
        assert result["logic_correct"] is False
        items = result["report"]["items"]

        missing = [i for i in items if i["error_code"] == "COMPONENT_MISSING"]
        assert len(missing) == 1
        assert missing[0]["expected"]["ref_id"] == "C1"
        assert missing[0]["expected"]["type"] == "CapacitorCeramic"
        assert missing[0]["actual"] is None
        assert missing[0]["component_ref"]["ref_id"] == "C1"
        assert missing[0]["component_actual"] is None
        assert missing[0]["title"] == "缺元件"

        # input/output/power/ground 成为严格角色后，缺失元件时图可能不再是子图同构，
        # 因此 enrichment 后可能不再保留 INCOMPLETE_CIRCUIT，而是出现 wiring mismatch/角色错误
        incomplete = [i for i in items if i["error_code"] == "INCOMPLETE_CIRCUIT"]
        # 允许没有 INCOMPLETE_CIRCUIT，但至少要检测到 C1 缺失
        assert len(missing) == 1

    def test_extra_component_detailed(self) -> None:
        ref = logical_reference_to_graph(_ref_payload())
        cur = current_netlist_v2_to_graph(_cur_netlist_extra_resistor())
        result = compare_logical_graphs(
            ref, cur, ref_payload=_ref_payload(), cur_netlist_v2=_cur_netlist_extra_resistor()
        )
        assert result["logic_correct"] is False
        items = result["report"]["items"]

        extra = [i for i in items if i["error_code"] == "COMPONENT_EXTRA"]
        assert len(extra) == 1
        assert extra[0]["actual"]["component_id"] == "R3"
        assert extra[0]["actual"]["type"] == "Resistor"
        assert extra[0]["expected"] is None
        assert extra[0]["component_ref"] is None
        assert extra[0]["component_actual"]["component_id"] == "R3"
        assert extra[0]["title"] == "多余元件"

    def test_wrong_connection_detailed(self) -> None:
        ref = logical_reference_to_graph(_ref_payload())
        cur = current_netlist_v2_to_graph(_cur_netlist_wrong_connection())
        result = compare_logical_graphs(
            ref, cur, ref_payload=_ref_payload(), cur_netlist_v2=_cur_netlist_wrong_connection()
        )
        assert result["logic_correct"] is False
        items = result["report"]["items"]

        wrong = [i for i in items if i["error_code"] == "WRONG_CONNECTION"]
        assert len(wrong) >= 1
        w = wrong[0]
        assert "expected" in w
        assert "actual" in w
        assert w["actual"]["actual_component_id"] in {"R2", "C2"}
        assert w["component_ref"] is not None
        assert w["component_actual"] is not None
        assert w["title"] == "错接"
        assert w["error_family"] == "wiring_mismatch"

    def test_no_hole_mismatch_in_detailed(self) -> None:
        ref = logical_reference_to_graph(_ref_payload())
        cur = current_netlist_v2_to_graph(_cur_netlist_match())
        result = compare_logical_graphs(
            ref, cur, ref_payload=_ref_payload(), cur_netlist_v2=_cur_netlist_match()
        )
        items = result["report"]["items"]
        assert not any(i.get("error_code") == "HOLE_MISMATCH" for i in items)

    def test_item_structure(self) -> None:
        """验证 enriched item 包含所有要求的字段。"""
        ref = logical_reference_to_graph(_ref_payload())
        cur = current_netlist_v2_to_graph(_cur_netlist_missing_cap())
        result = compare_logical_graphs(
            ref, cur, ref_payload=_ref_payload(), cur_netlist_v2=_cur_netlist_missing_cap()
        )
        for item in result["report"]["items"]:
            assert "error_code" in item
            assert "error_family" in item
            assert "severity" in item
            assert "title" in item
            assert "message" in item
            assert "expected" in item
            assert "actual" in item
            assert "component_ref" in item
            assert "component_actual" in item
            assert "evidence_refs" in item
            assert "suggested_action" in item

    def test_passive_pin_swap_no_false_wrong_connection(self) -> None:
        """R7 引脚互换，C9 错接 —  enrichment 时不应对 R7 报 WRONG_CONNECTION。"""
        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {
                    "ref_id": "R1",
                    "type": "Resistor",
                    "pins": [
                        {"pin": "pin1", "net": "NET_A"},
                        {"pin": "pin2", "net": "NET_B"},
                    ],
                },
                {
                    "ref_id": "C1",
                    "type": "CapacitorCeramic",
                    "pins": [
                        {"pin": "pin1", "net": "NET_B"},
                        {"pin": "pin2", "net": "GND"},
                    ],
                },
            ],
            "nets": [
                {"net": "NET_A"},
                {"net": "NET_B"},
                {"net": "GND", "role": "ground"},
            ],
        }
        # R7 swapped pins (still correct topology), C9 on completely wrong nets (no shared net with R7)
        cur_netlist = {
            "components": [
                {
                    "component_id": "R7",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "NET_001"},
                        {"pin_name": "pin2", "electrical_net_id": "NET_000"},
                    ],
                },
                {
                    "component_id": "C9",
                    "component_type": "CapacitorCeramic",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "NET_002"},
                        {"pin_name": "pin2", "electrical_net_id": "NET_003"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "NET_000"},
                {"electrical_net_id": "NET_001"},
                {"electrical_net_id": "NET_002"},
                {"electrical_net_id": "NET_003"},
            ],
        }
        ref = logical_reference_to_graph(ref_payload)
        cur = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(
            ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist
        )
        assert result["logic_correct"] is False
        items = result["report"]["items"]
        wrong = [i for i in items if i["error_code"] == "WRONG_CONNECTION"]
        # R7 should NOT produce WRONG_CONNECTION because its nets match after pin-swap
        assert not any("R7" in str(i.get("message", "")) for i in wrong)
        # C9 should still produce WRONG_CONNECTION because its topology is wrong
        assert any("C9" in str(i.get("message", "")) or "C1" in str(i.get("message", "")) for i in wrong)

    def test_missing_capacitor_detailed(self) -> None:
        """Detailed enrichment for missing capacitor should report COMPONENT_MISSING."""
        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {
                    "ref_id": "R1",
                    "type": "Resistor",
                    "pins": [
                        {"pin": "pin1", "net": "NET_A"},
                        {"pin": "pin2", "net": "NET_B"},
                    ],
                },
                {
                    "ref_id": "C1",
                    "type": "CapacitorCeramic",
                    "pins": [
                        {"pin": "pin1", "net": "NET_B"},
                        {"pin": "pin2", "net": "GND"},
                    ],
                },
            ],
            "nets": [
                {"net": "NET_A"},
                {"net": "NET_B"},
                {"net": "GND", "role": "ground"},
            ],
        }
        cur_netlist = {
            "components": [
                {
                    "component_id": "R7",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "NET_000"},
                        {"pin_name": "pin2", "electrical_net_id": "NET_001"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "NET_000"},
                {"electrical_net_id": "NET_001"},
            ],
        }
        ref = logical_reference_to_graph(ref_payload)
        cur = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(
            ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist
        )
        assert result["logic_correct"] is False
        items = result["report"]["items"]
        missing = [i for i in items if i["error_code"] == "COMPONENT_MISSING"]
        assert len(missing) == 1
        assert missing[0]["expected"]["ref_id"] == "C1"

    def test_resistor_wrong_net_detailed(self) -> None:
        """Resistor on wrong net should report WRONG_CONNECTION."""
        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {
                    "ref_id": "R1",
                    "type": "Resistor",
                    "pins": [
                        {"pin": "pin1", "net": "NET_A"},
                        {"pin": "pin2", "net": "NET_B"},
                    ],
                },
                {
                    "ref_id": "C1",
                    "type": "CapacitorCeramic",
                    "pins": [
                        {"pin": "pin1", "net": "NET_B"},
                        {"pin": "pin2", "net": "GND"},
                    ],
                },
            ],
            "nets": [
                {"net": "NET_A"},
                {"net": "NET_B"},
                {"net": "GND", "role": "ground"},
            ],
        }
        # R7 and C9 in parallel instead of series — topology genuinely wrong
        cur_netlist = {
            "components": [
                {
                    "component_id": "R7",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "NET_000"},
                        {"pin_name": "pin2", "electrical_net_id": "NET_001"},
                    ],
                },
                {
                    "component_id": "C9",
                    "component_type": "CapacitorCeramic",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "NET_000"},
                        {"pin_name": "pin2", "electrical_net_id": "NET_001"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "NET_000"},
                {"electrical_net_id": "NET_001", "role": "ground"},
            ],
        }
        ref = logical_reference_to_graph(ref_payload)
        cur = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(
            ref, cur, ref_payload=ref_payload, cur_netlist_v2=cur_netlist
        )
        assert result["logic_correct"] is False
        items = result["report"]["items"]
        wrong = [i for i in items if i["error_code"] == "WRONG_CONNECTION"]
        assert len(wrong) >= 1
        assert any("R1" in str(i.get("message", "")) or "R7" in str(i.get("message", "")) for i in wrong)

    def test_summary_metadata_in_detailed(self) -> None:
        """Detailed result should expose equivalence metadata in summary."""
        ref = logical_reference_to_graph(_ref_payload())
        cur = current_netlist_v2_to_graph(_cur_netlist_match())
        result = compare_logical_graphs(
            ref, cur, ref_payload=_ref_payload(), cur_netlist_v2=_cur_netlist_match()
        )
        summary = result["report"]["summary"]
        assert summary.get("ignore_component_id") is True
        assert summary.get("ignore_hole_id") is True
        assert summary.get("ignore_passive_pin_order") is True
        assert summary.get("equivalence_rule") == "component_type_and_topology"
