"""Tests for _fallback_comp_mapping: must use connection signatures, not list order."""

from __future__ import annotations

import networkx as nx

from app.domain.graph_compare import (
    _fallback_comp_mapping,
    _graph_neighbor_signature,
    _neighbor_signature_similarity,
)


def test_fallback_uses_connection_signature_not_list_order() -> None:
    """When graphs are not isomorphic, fallback should pair components by
    neighbor net roles and pin labels, not by their IDs or list order.
    """
    ref_graph = nx.Graph()
    ref_graph.add_node("ref_comp:R1", kind="comp", ctype="Resistor", source_id="R1")
    ref_graph.add_node("ref_comp:R2", kind="comp", ctype="Resistor", source_id="R2")
    ref_graph.add_node("ref_net:VCC", kind="net", role="power", source_id="VCC")
    ref_graph.add_node("ref_net:GND", kind="net", role="ground", source_id="GND")
    ref_graph.add_node("ref_net:SIG1", kind="net", role="signal", source_id="SIG1")
    ref_graph.add_node("ref_net:SIG2", kind="net", role="signal", source_id="SIG2")
    ref_graph.add_edge("ref_comp:R1", "ref_net:VCC", pin="pin1", comp_type="Resistor")
    ref_graph.add_edge("ref_comp:R1", "ref_net:GND", pin="pin2", comp_type="Resistor")
    ref_graph.add_edge("ref_comp:R2", "ref_net:SIG1", pin="pin1", comp_type="Resistor")
    ref_graph.add_edge("ref_comp:R2", "ref_net:SIG2", pin="pin2", comp_type="Resistor")

    # Current list order is R3 then R4, but R3 connects to VCC-GND (like R1)
    # and R4 connects to SIG1-SIG2 (like R2).
    # To force fallback, add a mismatching node so isomorphism fails.
    cur_graph = nx.Graph()
    cur_graph.add_node("cur_comp:R3", kind="comp", ctype="Resistor", source_id="R3")
    cur_graph.add_node("cur_comp:R4", kind="comp", ctype="Resistor", source_id="R4")
    cur_graph.add_node("cur_comp:C1", kind="comp", ctype="CapacitorCeramic", source_id="C1")
    cur_graph.add_node("cur_net:N0", kind="net", role="power", source_id="N0")
    cur_graph.add_node("cur_net:N1", kind="net", role="ground", source_id="N1")
    cur_graph.add_node("cur_net:N2", kind="net", role="signal", source_id="N2")
    cur_graph.add_node("cur_net:N3", kind="net", role="signal", source_id="N3")
    cur_graph.add_edge("cur_comp:R3", "cur_net:N0", pin="pin1", comp_type="Resistor")
    cur_graph.add_edge("cur_comp:R3", "cur_net:N1", pin="pin2", comp_type="Resistor")
    cur_graph.add_edge("cur_comp:R4", "cur_net:N2", pin="pin1", comp_type="Resistor")
    cur_graph.add_edge("cur_comp:R4", "cur_net:N3", pin="pin2", comp_type="Resistor")
    # Extra capacitor breaks full isomorphism but not subgraph isomorphism.
    # Add an edge to the capacitor so subgraph isomorphism also differs enough.
    cur_graph.add_edge("cur_comp:C1", "cur_net:N0", pin="pin1", comp_type="CapacitorCeramic")
    cur_graph.add_edge("cur_comp:C1", "cur_net:N2", pin="pin2", comp_type="CapacitorCeramic")

    comp_map = _fallback_comp_mapping(ref_graph, cur_graph)

    # R1 should map to R3 (both connect to power+ground)
    # R2 should map to R4 (both connect to signal+signal)
    assert comp_map.get("R1") == "R3", f"Expected R1->R3 by signature, got R1->{comp_map.get('R1')}"
    assert comp_map.get("R2") == "R4", f"Expected R2->R4 by signature, got R2->{comp_map.get('R2')}"


def test_fallback_swapped_order_still_matches_by_signature() -> None:
    """Even when current components appear in reverse list order,
    fallback should match by topology, not by position.
    """
    ref_graph = nx.Graph()
    ref_graph.add_node("ref_comp:R1", kind="comp", ctype="Resistor", source_id="R1")
    ref_graph.add_node("ref_net:VCC", kind="net", role="power", source_id="VCC")
    ref_graph.add_node("ref_net:GND", kind="net", role="ground", source_id="GND")
    ref_graph.add_edge("ref_comp:R1", "ref_net:VCC", pin="pin1", comp_type="Resistor")
    ref_graph.add_edge("ref_comp:R1", "ref_net:GND", pin="pin2", comp_type="Resistor")

    cur_graph = nx.Graph()
    cur_graph.add_node("cur_comp:R9", kind="comp", ctype="Resistor", source_id="R9")
    cur_graph.add_node("cur_net:N0", kind="net", role="power", source_id="N0")
    cur_graph.add_node("cur_net:N1", kind="net", role="ground", source_id="N1")
    cur_graph.add_edge("cur_comp:R9", "cur_net:N0", pin="pin1", comp_type="Resistor")
    cur_graph.add_edge("cur_comp:R9", "cur_net:N1", pin="pin2", comp_type="Resistor")

    comp_map = _fallback_comp_mapping(ref_graph, cur_graph)
    assert comp_map.get("R1") == "R9"


def test_neighbor_signature_similarity_scoring() -> None:
    """Direct unit test for signature similarity scoring."""
    sig_a = (2, ("ground", "power"), ("pin1", "pin2"))
    sig_b = (2, ("ground", "power"), ("pin1", "pin2"))
    sig_c = (2, ("signal", "signal"), ("pin1", "pin2"))
    sig_d = (3, ("ground", "power", "signal"), ("pin1", "pin2", "pin3"))

    assert _neighbor_signature_similarity(sig_a, sig_b) > _neighbor_signature_similarity(sig_a, sig_c)
    assert _neighbor_signature_similarity(sig_a, sig_b) > _neighbor_signature_similarity(sig_a, sig_d)


def test_graph_neighbor_signature_extraction() -> None:
    g = nx.Graph()
    g.add_node("comp:R1", kind="comp", ctype="Resistor")
    g.add_node("net:VCC", kind="net", role="power")
    g.add_node("net:GND", kind="net", role="ground")
    g.add_edge("comp:R1", "net:VCC", pin="pin1")
    g.add_edge("comp:R1", "net:GND", pin="pin2")

    sig = _graph_neighbor_signature("comp:R1", g)
    assert sig[0] == 2
    assert set(sig[1]) == {"ground", "power"}
    assert set(sig[2]) == {"pin1", "pin2"}
