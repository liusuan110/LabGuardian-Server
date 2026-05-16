"""Tests for ``app.domain.gnn.port_graph`` — NetworkX → HeteroCircuitGraph.

Acceptance: existing fixtures all convert to component+port+net schema with
the expected node / edge counts and metadata.
"""

from __future__ import annotations

import pytest

from app.domain.gnn import (
    HeteroCircuitGraph,
    build_from_logical_reference,
    build_from_netlist_v2,
    build_hetero_circuit_graph,
)
from app.domain.gnn.graph_schema import PolarityClass, PortType
from app.domain.gnn.hetero_circuit import PortConnectsNetEdge, PortNode

# ---------------------------------------------------------------------------
# Reference (logical_reference_v1) → side="ref"
# ---------------------------------------------------------------------------


def test_build_from_rc_reference_basic_shape(rc_reference_payload) -> None:
    hcg = build_from_logical_reference(rc_reference_payload)
    summary = hcg.summary()
    assert summary == {
        "n_components": 2,  # R1, C1
        "n_ports": 4,  # 2 pins each
        "n_nets": 3,  # VIN, VC, GND
        "n_edges": 4,  # one (port,net) per pin
    }
    # invariants hold
    hcg.assert_invariants()


def test_port_node_ids_are_namespaced_and_unique(rc_reference_payload) -> None:
    hcg = build_from_logical_reference(rc_reference_payload)
    port_ids = list(hcg.ports.keys())
    assert len(port_ids) == len(set(port_ids)), "duplicate port node_id"
    assert all(pid.startswith("ref_port:") for pid in port_ids), port_ids
    # R1 should expose pin1 / pin2 ports
    assert "ref_port:R1.pin1" in port_ids
    assert "ref_port:R1.pin2" in port_ids
    assert "ref_port:C1.pin1" in port_ids
    assert "ref_port:C1.pin2" in port_ids


def test_components_pin_count_backfilled(rc_reference_payload) -> None:
    hcg = build_from_logical_reference(rc_reference_payload)
    for comp in hcg.components.values():
        assert comp.pin_count == 2, f"{comp.node_id}: pin_count={comp.pin_count}"


def test_metadata_propagated(rc_reference_payload) -> None:
    hcg = build_from_logical_reference(rc_reference_payload)
    assert hcg.metadata.get("format") == "logical_reference_v1"
    assert hcg.metadata.get("reference_id") == "test_rc_v1"


def test_nets_carry_role_and_power_rail_flag(rc_reference_payload) -> None:
    hcg = build_from_logical_reference(rc_reference_payload)
    nets_by_source = {n.source_id: n for n in hcg.nets.values()}
    assert nets_by_source["GND"].role == "ground"
    assert nets_by_source["GND"].is_power_rail is True
    assert nets_by_source["VIN"].role == "input"
    assert nets_by_source["VIN"].is_power_rail is False
    assert nets_by_source["VC"].role == "signal"


def test_edges_have_default_dsl_source_type_on_ref_side(rc_reference_payload) -> None:
    hcg = build_from_logical_reference(rc_reference_payload)
    assert all(isinstance(e, PortConnectsNetEdge) for e in hcg.edges)
    assert all(e.source_type == "dsl" for e in hcg.edges)
    assert all(e.is_observed_in_cur is False for e in hcg.edges)
    assert all(e.connection_confidence == pytest.approx(1.0) for e in hcg.edges)


# ---------------------------------------------------------------------------
# Polarity-sensitive flag on LED fixture
# ---------------------------------------------------------------------------


def test_led_anode_cathode_are_polarity_sensitive(led_reference_payload) -> None:
    hcg = build_from_logical_reference(led_reference_payload)
    by_id = hcg.ports

    led_anode = by_id["ref_port:LED1.anode"]
    led_cathode = by_id["ref_port:LED1.cathode"]
    assert led_anode.port_type == PortType.ANODE.value
    assert led_cathode.port_type == PortType.CATHODE.value
    assert led_anode.polarity_sensitive is True
    assert led_cathode.polarity_sensitive is True

    # parent component polarity_class is two_polar
    led_comp = hcg.components[led_anode.parent_component_id]
    assert led_comp.polarity_class == PolarityClass.TWO_POLAR.value


def test_resistor_pins_are_not_polarity_sensitive(led_reference_payload) -> None:
    hcg = build_from_logical_reference(led_reference_payload)
    r_pin1: PortNode = hcg.ports["ref_port:R1.pin1"]
    r_pin2: PortNode = hcg.ports["ref_port:R1.pin2"]
    assert r_pin1.polarity_sensitive is False
    assert r_pin2.polarity_sensitive is False
    assert hcg.components[r_pin1.parent_component_id].polarity_class == "none"


# ---------------------------------------------------------------------------
# All fixture references survive conversion
# ---------------------------------------------------------------------------


def test_all_reference_fixtures_convert(all_reference_payloads) -> None:
    assert all_reference_payloads, "fixture directory should not be empty"
    for ref_id, payload in all_reference_payloads.items():
        hcg: HeteroCircuitGraph = build_from_logical_reference(payload)
        s = hcg.summary()
        # 每条 (comp, net) 边对应一个 port，所以 n_edges == n_ports
        assert s["n_edges"] == s["n_ports"], f"{ref_id}: {s}"
        assert s["n_ports"] >= s["n_components"], f"{ref_id}: {s}"
        hcg.assert_invariants()


# ---------------------------------------------------------------------------
# Side="cur" path via netlist_v2
# ---------------------------------------------------------------------------


def test_build_from_netlist_v2_smoke(simple_netlist_v2) -> None:
    hcg = build_from_netlist_v2(simple_netlist_v2)
    assert hcg.side == "cur"
    # All node_ids must be cur_*-prefixed
    for nid in list(hcg.components) + list(hcg.ports) + list(hcg.nets):
        assert nid.startswith("cur_"), nid
    s = hcg.summary()
    assert s["n_components"] >= 1
    assert s["n_ports"] == s["n_edges"]
    # cur-side edges default to source_type=vision
    if hcg.edges:
        assert hcg.edges[0].source_type == "vision"
        assert hcg.edges[0].is_observed_in_cur is True
    hcg.assert_invariants()


# ---------------------------------------------------------------------------
# Robustness: directly handing a hand-built nx graph works
# ---------------------------------------------------------------------------


def test_build_from_handcrafted_nx_graph() -> None:
    import networkx as nx

    g = nx.Graph()
    g.add_node("ref_comp:R1", kind="comp", ctype="Resistor", source_id="R1")
    g.add_node("ref_net:N1", kind="net", role="signal", source_id="N1")
    g.add_node("ref_net:N2", kind="net", role="ground", source_id="N2")
    g.add_edge("ref_comp:R1", "ref_net:N1", pin="pin1", pin_role="pin1", comp_type="Resistor")
    g.add_edge("ref_comp:R1", "ref_net:N2", pin="pin2", pin_role="pin2", comp_type="Resistor")
    g.graph["format"] = "logical_reference_v1"

    hcg = build_hetero_circuit_graph(g, side="ref")
    assert hcg.summary() == {"n_components": 1, "n_ports": 2, "n_nets": 2, "n_edges": 2}
    hcg.assert_invariants()


def test_build_rejects_invalid_side() -> None:
    import networkx as nx

    with pytest.raises(ValueError, match="side"):
        build_hetero_circuit_graph(nx.Graph(), side="bogus")  # type: ignore[arg-type]
