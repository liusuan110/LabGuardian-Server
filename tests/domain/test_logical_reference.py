from __future__ import annotations

import pytest

from app.domain.logical_reference import (
    current_netlist_v2_to_graph,
    logical_reference_to_graph,
    normalize_component_type,
    normalize_net_role,
    validate_logical_reference,
)


class TestNormalizeComponentType:
    def test_resistor(self) -> None:
        assert normalize_component_type("resistor") == "Resistor"
        assert normalize_component_type("Resistor") == "Resistor"
        assert normalize_component_type("RESISTOR") == "Resistor"

    def test_led(self) -> None:
        assert normalize_component_type("led") == "LED"
        assert normalize_component_type("LED") == "LED"

    def test_capacitor(self) -> None:
        assert normalize_component_type("capacitor") == "Capacitor"
        assert normalize_component_type("CapacitorCeramic") == "CapacitorCeramic"


class TestNormalizeNetRole:
    def test_ground_variants(self) -> None:
        assert normalize_net_role("GND") == "ground"
        assert normalize_net_role("ground") == "ground"
        assert normalize_net_role("0V") == "ground"

    def test_power_variants(self) -> None:
        assert normalize_net_role("VCC") == "power"
        assert normalize_net_role("VEE") == "power"
        assert normalize_net_role("VSS") == "power"
        assert normalize_net_role("negative_supply") == "power"
        assert normalize_net_role("power") == "power"
        assert normalize_net_role("+5V") == "power"

    def test_signal_default(self) -> None:
        assert normalize_net_role("") == "signal"
        assert normalize_net_role(None) == "signal"
        assert normalize_net_role("random") == "signal"


class TestValidateLogicalReference:
    def test_valid(self) -> None:
        validate_logical_reference(
            {
                "format": "logical_reference_v1",
                "components": [
                    {
                        "ref_id": "R1",
                        "type": "Resistor",
                        "pins": [
                            {"pin": "p1", "net": "N1"},
                            {"pin": "p2", "net": "N2"},
                        ],
                    }
                ],
                "nets": [{"net": "N1"}, {"net": "N2"}],
            }
        )

    def test_missing_format(self) -> None:
        with pytest.raises(ValueError, match="logical_reference_v1"):
            validate_logical_reference({"components": []})

    def test_empty_components(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_logical_reference(
                {"format": "logical_reference_v1", "components": []}
            )

    def test_duplicate_ref_id(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            validate_logical_reference(
                {
                    "format": "logical_reference_v1",
                    "components": [
                        {
                            "ref_id": "R1",
                            "type": "Resistor",
                            "pins": [{"pin": "p1", "net": "N1"}],
                        },
                        {
                            "ref_id": "R1",
                            "type": "Resistor",
                            "pins": [{"pin": "p1", "net": "N1"}],
                        },
                    ],
                    "nets": [{"net": "N1"}],
                }
            )

    def test_unknown_net(self) -> None:
        with pytest.raises(ValueError, match="unknown net"):
            validate_logical_reference(
                {
                    "format": "logical_reference_v1",
                    "components": [
                        {
                            "ref_id": "R1",
                            "type": "Resistor",
                            "pins": [{"pin": "p1", "net": "UNKNOWN"}],
                        }
                    ],
                    "nets": [{"net": "N1"}],
                }
            )


class TestLogicalReferenceToGraph:
    def test_basic(self) -> None:
        graph = logical_reference_to_graph(
            {
                "format": "logical_reference_v1",
                "reference_id": "test",
                "components": [
                    {
                        "ref_id": "R1",
                        "type": "Resistor",
                        "pins": [
                            {"pin": "p1", "net": "N1"},
                            {"pin": "p2", "net": "N2"},
                        ],
                    }
                ],
                "nets": [
                    {"net": "N1", "role": "input"},
                    {"net": "N2", "role": "ground"},
                ],
            }
        )
        assert graph.number_of_nodes() == 3  # 1 comp + 2 nets
        assert graph.number_of_edges() == 2
        assert graph.nodes["ref_comp:R1"]["kind"] == "comp"
        assert graph.nodes["ref_comp:R1"]["ctype"] == "Resistor"
        assert graph.nodes["ref_net:N1"]["role"] == "input"
        assert graph.nodes["ref_net:N2"]["role"] == "ground"


class TestCurrentNetlistV2ToGraph:
    def test_basic(self) -> None:
        graph = current_netlist_v2_to_graph(
            {
                "components": [
                    {
                        "component_id": "R1",
                        "component_type": "Resistor",
                        "pins": [
                            {"pin_name": "p1", "electrical_net_id": "NET_001"},
                            {"pin_name": "p2", "electrical_net_id": "NET_002"},
                        ],
                    }
                ],
                "nets": [
                    {"electrical_net_id": "NET_001", "power_role": "VCC"},
                    {"electrical_net_id": "NET_002", "power_role": ""},
                ],
            }
        )
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2
        assert graph.nodes["cur_comp:R1"]["ctype"] == "Resistor"
        assert graph.nodes["cur_net:NET_001"]["role"] == "power"
        assert graph.nodes["cur_net:NET_002"]["role"] == "signal"

    def test_keeps_wire_as_node(self) -> None:
        """R8 fix (RISK_REGISTER §5): Wire components are no longer
        silently dropped during netlist→graph conversion. They land
        as regular ``cur_comp:*`` nodes carrying their pin edges so
        the rule comparator can detect stray jumper wires that
        bridge role-critical nets (previously: 100% false_pass on
        ``extra_wire_bridge``; see ``docs/SIM_TO_REAL.md``)."""

        graph = current_netlist_v2_to_graph(
            {
                "components": [
                    {
                        "component_id": "W1",
                        "component_type": "Wire",
                        "pins": [
                            {"pin_name": "p1", "electrical_net_id": "NET_001"},
                            {"pin_name": "p2", "electrical_net_id": "NET_002"},
                        ],
                    },
                    {
                        "component_id": "R1",
                        "component_type": "Resistor",
                        "pins": [
                            {"pin_name": "p1", "electrical_net_id": "NET_001"},
                            {"pin_name": "p2", "electrical_net_id": "NET_002"},
                        ],
                    },
                ],
                "nets": [
                    {"electrical_net_id": "NET_001"},
                    {"electrical_net_id": "NET_002"},
                ],
            }
        )
        comp_nodes = [n for n, d in graph.nodes(data=True) if d.get("kind") == "comp"]
        assert sorted(comp_nodes) == ["cur_comp:R1", "cur_comp:W1"]
        # Each component contributes 2 edges (one per pin); shared net
        # nodes collapse them at the nx layer but cur_comp:W1 still
        # has degree 2 (one edge to each net).
        assert graph.degree("cur_comp:W1") == 2
        assert graph.degree("cur_comp:R1") == 2
