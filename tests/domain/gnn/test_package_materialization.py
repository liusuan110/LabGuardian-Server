"""P0.6 · Package port materialization + symmetry / connection policy tests.

Covers:
- Floating port materialization for unconnected package pins (UA741 NC pins)
- ConnectionPolicy assignment (REQUIRED / OPTIONAL / FORBIDDEN)
- symmetry_class_id grouping (R/Cap/Pot terminals share; LED/Diode anode≠cathode;
  UA741 offset_null_1 ↔ offset_null_2 share)
- ComponentNode.pin_symmetry_groups auto-derived
- pin_number propagation (numeric pins → int; "anode" → None)
- cur-side floating pin handling: netlist_v2 with electrical_net_id=None
  yields a floating PortNode (does not vanish)
- Zero regression on simple non-IC fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from app.domain.gnn import (
    ConnectionPolicy,
    PortType,
    build_from_logical_reference,
    build_from_netlist_v2,
    build_hetero_circuit_graph,
    get_expected_pin_specs,
)

FIXTURE_OPAMP = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_opamp_buffer_v1.json"
)


# ---------------------------------------------------------------------------
# UA741 — full 8-pin materialization
# ---------------------------------------------------------------------------


def test_ua741_buffer_materializes_eight_ports() -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text(encoding="utf-8")))
    assert hcg.summary()["n_ports"] == 8
    keys = {p.port_key for p in hcg.ports.values()}
    assert keys == {"1", "2", "3", "4", "5", "6", "7", "8"}


@pytest.mark.parametrize(
    ("pin_key", "expected_policy"),
    [
        ("1", ConnectionPolicy.OPTIONAL.value),
        ("5", ConnectionPolicy.OPTIONAL.value),
        ("8", ConnectionPolicy.FORBIDDEN.value),
        ("2", ConnectionPolicy.REQUIRED.value),
        ("3", ConnectionPolicy.REQUIRED.value),
        ("4", ConnectionPolicy.REQUIRED.value),
        ("6", ConnectionPolicy.REQUIRED.value),
        ("7", ConnectionPolicy.REQUIRED.value),
    ],
)
def test_ua741_connection_policy_per_pin(pin_key: str, expected_policy: str) -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text(encoding="utf-8")))
    port = next(p for p in hcg.ports.values() if p.port_key == pin_key)
    assert port.connection_policy == expected_policy


def test_ua741_nc_and_offset_pins_are_floating_others_connected() -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text(encoding="utf-8")))
    by_key = {p.port_key: p for p in hcg.ports.values()}
    for nc_key in ("1", "5", "8"):
        assert by_key[nc_key].is_floating is True, nc_key
    for connected_key in ("2", "3", "4", "6", "7"):
        assert by_key[connected_key].is_floating is False, connected_key


def test_ua741_offset_null_pins_share_symmetry_class() -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text(encoding="utf-8")))
    by_key = {p.port_key: p for p in hcg.ports.values()}
    assert by_key["1"].symmetry_class_id == by_key["5"].symmetry_class_id
    # Each remaining IC pin is in its own class — strictly different from the
    # offset_null pair.
    others = {by_key[k].symmetry_class_id for k in ("2", "3", "4", "6", "7", "8")}
    assert by_key["1"].symmetry_class_id not in others
    assert len(others) == 6  # six other pins, each its own class


def test_ua741_component_pin_symmetry_groups_reports_offset_null_pair() -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text(encoding="utf-8")))
    comp = next(iter(hcg.components.values()))
    # Should contain exactly one non-trivial group: {"1", "5"}
    assert comp.pin_symmetry_groups == (("1", "5"),)
    # pin_count covers all 8 package pins after materialization
    assert comp.pin_count == 8


def test_ua741_pin_numbers_assigned() -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text(encoding="utf-8")))
    by_key = {p.port_key: p for p in hcg.ports.values()}
    for n in range(1, 9):
        assert by_key[str(n)].pin_number == n, n


# ---------------------------------------------------------------------------
# Resistor / Capacitor (symmetric two-pin) — should share symmetry class
# ---------------------------------------------------------------------------


def _build_resistor_only_payload() -> dict:
    return {
        "format": "logical_reference_v1",
        "reference_id": "test_r_only",
        "components": [
            {
                "ref_id": "R1",
                "type": "Resistor",
                "pins": [
                    {"pin": "pin1", "net": "NA"},
                    {"pin": "pin2", "net": "NB"},
                ],
            }
        ],
        "nets": [
            {"net": "NA", "role": "signal"},
            {"net": "NB", "role": "signal"},
        ],
    }


def test_resistor_pins_share_symmetry_class() -> None:
    hcg = build_from_logical_reference(_build_resistor_only_payload())
    p1 = hcg.ports["ref_port:R1.pin1"]
    p2 = hcg.ports["ref_port:R1.pin2"]
    assert p1.symmetry_class_id == p2.symmetry_class_id
    comp = hcg.components[p1.parent_component_id]
    assert comp.pin_symmetry_groups == (("pin1", "pin2"),)


def test_resistor_pin_numbers_propagated() -> None:
    hcg = build_from_logical_reference(_build_resistor_only_payload())
    assert hcg.ports["ref_port:R1.pin1"].pin_number == 1
    assert hcg.ports["ref_port:R1.pin2"].pin_number == 2


# ---------------------------------------------------------------------------
# LED / Diode / electrolytic Cap — polarized, distinct symmetry classes
# ---------------------------------------------------------------------------


def test_led_anode_cathode_distinct_symmetry_classes() -> None:
    payload = {
        "format": "logical_reference_v1",
        "reference_id": "led_only",
        "components": [
            {
                "ref_id": "D1",
                "type": "LED",
                "pins": [
                    {"pin": "anode", "net": "NA"},
                    {"pin": "cathode", "net": "NB"},
                ],
            }
        ],
        "nets": [
            {"net": "NA", "role": "signal"},
            {"net": "NB", "role": "ground"},
        ],
    }
    hcg = build_from_logical_reference(payload)
    anode = hcg.ports["ref_port:D1.anode"]
    cathode = hcg.ports["ref_port:D1.cathode"]
    assert anode.symmetry_class_id != cathode.symmetry_class_id
    # No swap group should be reported (each class size == 1)
    comp = hcg.components[anode.parent_component_id]
    assert comp.pin_symmetry_groups == ()
    # pin_number assigned per spec (anode=1, cathode=2)
    assert anode.pin_number == 1
    assert cathode.pin_number == 2


# ---------------------------------------------------------------------------
# Potentiometer — terminal_a/b swap, wiper alone
# ---------------------------------------------------------------------------


def test_potentiometer_terminals_share_class_wiper_alone() -> None:
    payload = {
        "format": "logical_reference_v1",
        "reference_id": "pot",
        "components": [
            {
                "ref_id": "RV1",
                "type": "Potentiometer",
                "pins": [
                    {"pin": "wiper", "net": "VOUT"},
                    {"pin": "terminal_a", "net": "VIN"},
                    {"pin": "terminal_b", "net": "GND"},
                ],
            }
        ],
        "nets": [
            {"net": "VIN", "role": "input"},
            {"net": "VOUT", "role": "output"},
            {"net": "GND", "role": "ground"},
        ],
    }
    hcg = build_from_logical_reference(payload)
    by_key = {p.port_key: p for p in hcg.ports.values()}
    assert by_key["terminal_a"].symmetry_class_id == by_key["terminal_b"].symmetry_class_id
    assert by_key["wiper"].symmetry_class_id != by_key["terminal_a"].symmetry_class_id
    comp = hcg.components[by_key["wiper"].parent_component_id]
    assert comp.pin_symmetry_groups == (("terminal_a", "terminal_b"),)


# ---------------------------------------------------------------------------
# Transistor — three distinct classes
# ---------------------------------------------------------------------------


def test_transistor_three_pins_distinct_classes() -> None:
    payload = {
        "format": "logical_reference_v1",
        "reference_id": "bjt",
        "components": [
            {
                "ref_id": "Q1",
                "type": "Transistor",
                "pins": [
                    {"pin": "base", "net": "NB"},
                    {"pin": "collector", "net": "NC"},
                    {"pin": "emitter", "net": "GND"},
                ],
            }
        ],
        "nets": [
            {"net": "NB", "role": "signal"},
            {"net": "NC", "role": "signal"},
            {"net": "GND", "role": "ground"},
        ],
    }
    hcg = build_from_logical_reference(payload)
    classes = {p.symmetry_class_id for p in hcg.ports.values()}
    assert len(classes) == 3
    comp = next(iter(hcg.components.values()))
    assert comp.pin_symmetry_groups == ()


# ---------------------------------------------------------------------------
# pin_number for named-only pins is None
# ---------------------------------------------------------------------------


def test_named_only_pin_has_no_pin_number_when_spec_says_so() -> None:
    # Spec assigns 1/2 to LED anode/cathode for ordering. So they DO have
    # pin_number even though they're named. This test instead checks that a
    # component with no spec at all (e.g., Sensor) yields pin_number=None.
    g = nx.Graph()
    g.add_node("ref_comp:S1", kind="comp", ctype="Sensor", source_id="S1")
    g.add_node("ref_net:N1", kind="net", role="signal", source_id="N1")
    g.add_edge(
        "ref_comp:S1",
        "ref_net:N1",
        pin="signal_out",
        pin_role="signal_out",
        comp_type="Sensor",
    )
    hcg = build_hetero_circuit_graph(g, side="ref")
    port = next(iter(hcg.ports.values()))
    # Sensor has no PACKAGE_PIN_SPECS entry → falls back to None / unique cls.
    assert port.pin_number is None
    assert port.connection_policy == ConnectionPolicy.REQUIRED.value
    assert port.symmetry_class_id == 0


# ---------------------------------------------------------------------------
# cur-side floating pin (netlist_v2 with electrical_net_id=None) and IC NC
# ---------------------------------------------------------------------------


def test_cur_side_resistor_with_unmapped_pin_materializes_floating_port() -> None:
    """When a student left one terminal of a resistor in the air (vision
    sees the pin but the topology stage failed to assign electrical_net_id),
    we must surface that as a floating PortNode rather than drop it."""

    netlist_v2 = {
        "scene_id": "test_floating",
        "board_schema_id": "breadboard_legacy_v1",
        "components": [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "package_type": "axial_2pin",
                "polarity": "none",
                "pins": [
                    {
                        "pin_id": 1,
                        "pin_name": "pin1",
                        "hole_id": "B12",
                        "electrical_net_id": "NET_000",
                        "confidence": 1.0,
                        "is_ambiguous": False,
                    },
                    {
                        # Floating: vision saw the pin but topology stage
                        # couldn't assign an electrical_net_id.
                        "pin_id": 2,
                        "pin_name": "pin2",
                        "hole_id": "F12",
                        "electrical_net_id": None,
                        "confidence": 1.0,
                        "is_ambiguous": False,
                    },
                ],
                "confidence": 1.0,
            }
        ],
        "nets": [
            {
                "electrical_net_id": "NET_000",
                "member_node_ids": ["ROW_12_L"],
                "labels": [],
            }
        ],
        "node_index": {"ROW_12_L": ["B12"]},
    }

    hcg = build_from_netlist_v2(netlist_v2)
    by_key = {p.port_key: p for p in hcg.ports.values()}
    assert "pin1" in by_key
    assert "pin2" in by_key
    assert by_key["pin1"].is_floating is False
    assert by_key["pin2"].is_floating is True
    # Connected edge only for pin1
    assert len(hcg.edges) == 1
    assert hcg.edges[0].src_port_id == "cur_port:R1.pin1"


def test_cur_side_ic_with_no_subtype_still_materializes_via_default_spec() -> None:
    """An IC instance lacking part_subtype should still get *some* materializa-
    tion only when a spec is known. Without subtype, we currently can't tell
    how many pins it has, so we skip materialization (graceful, no crash)."""

    netlist_v2 = {
        "scene_id": "ic_no_subtype",
        "board_schema_id": "breadboard_legacy_v1",
        "components": [
            {
                "component_id": "U1",
                "component_type": "IC",
                "package_type": "DIP8",
                "polarity": "none",
                "pins": [
                    {
                        "pin_id": 3,
                        "pin_name": "3",
                        "hole_id": "A05",
                        "electrical_net_id": "NET_VIN",
                        "confidence": 1.0,
                        "is_ambiguous": False,
                    }
                ],
                "confidence": 1.0,
            }
        ],
        "nets": [
            {"electrical_net_id": "NET_VIN", "member_node_ids": [], "labels": []}
        ],
        "node_index": {},
    }
    hcg = build_from_netlist_v2(netlist_v2)
    # No subtype → no spec → only the one connected pin appears.
    assert hcg.summary()["n_ports"] == 1
    assert hcg.summary()["n_edges"] == 1
    port = next(iter(hcg.ports.values()))
    assert port.is_floating is False


# ---------------------------------------------------------------------------
# get_expected_pin_specs API surface
# ---------------------------------------------------------------------------


def test_get_expected_pin_specs_for_resistor() -> None:
    specs = get_expected_pin_specs("Resistor")
    assert specs is not None
    assert {s.pin_key for s in specs} == {"pin1", "pin2"}
    assert all(s.connection_policy == ConnectionPolicy.REQUIRED.value for s in specs)
    # Same symmetry_class for both pins → interchangeable
    assert specs[0].symmetry_class == specs[1].symmetry_class


def test_get_expected_pin_specs_for_ic_requires_subtype() -> None:
    # IC without subtype → no spec
    assert get_expected_pin_specs("IC") is None
    assert get_expected_pin_specs("IC", "") is None
    assert get_expected_pin_specs("IC", "UNKNOWN_SUBTYPE") is None
    # IC + UA741 → 8 specs
    specs = get_expected_pin_specs("IC", "UA741")
    assert specs is not None
    assert len(specs) == 8
    pin8 = next(s for s in specs if s.pin_key == "8")
    assert pin8.connection_policy == ConnectionPolicy.FORBIDDEN.value
    assert pin8.port_type == PortType.NC.value


def test_get_expected_pin_specs_unknown_ctype_returns_none() -> None:
    assert get_expected_pin_specs("UNKNOWN") is None
    assert get_expected_pin_specs("Sensor") is None  # no spec yet
