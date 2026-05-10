from __future__ import annotations

import networkx as nx

from app.domain.graph_compare import compare_logical_graphs


def _build_ref() -> nx.Graph:
    """R1 between VIN and VC, C1 between VC and GND"""
    g = nx.Graph()
    g.add_node("ref_comp:R1", kind="comp", ctype="Resistor")
    g.add_node("ref_comp:C1", kind="comp", ctype="CapacitorCeramic")
    g.add_node("ref_net:VIN", kind="net", role="input")
    g.add_node("ref_net:VC", kind="net", role="signal")
    g.add_node("ref_net:GND", kind="net", role="ground")
    g.add_edge("ref_comp:R1", "ref_net:VIN")
    g.add_edge("ref_comp:R1", "ref_net:VC")
    g.add_edge("ref_comp:C1", "ref_net:VC")
    g.add_edge("ref_comp:C1", "ref_net:GND")
    return g


def _build_cur_match() -> nx.Graph:
    """Same topology but different node names and net IDs"""
    g = nx.Graph()
    g.add_node("cur_comp:R1", kind="comp", ctype="Resistor")
    g.add_node("cur_comp:C1", kind="comp", ctype="CapacitorCeramic")
    g.add_node("cur_net:NET_000", kind="net", role="input")
    g.add_node("cur_net:NET_001", kind="net", role="signal")
    g.add_node("cur_net:NET_002", kind="net", role="ground")
    g.add_edge("cur_comp:R1", "cur_net:NET_000")
    g.add_edge("cur_comp:R1", "cur_net:NET_001")
    g.add_edge("cur_comp:C1", "cur_net:NET_001")
    g.add_edge("cur_comp:C1", "cur_net:NET_002")
    return g


def _build_cur_missing_cap() -> nx.Graph:
    """Only R1"""
    g = nx.Graph()
    g.add_node("cur_comp:R1", kind="comp", ctype="Resistor")
    g.add_node("cur_net:NET_000", kind="net", role="input")
    g.add_node("cur_net:NET_001", kind="net", role="signal")
    g.add_edge("cur_comp:R1", "cur_net:NET_000")
    g.add_edge("cur_comp:R1", "cur_net:NET_001")
    return g


def _build_cur_extra_resistor() -> nx.Graph:
    """R1, C1 plus an extra resistor R2"""
    g = nx.Graph()
    g.add_node("cur_comp:R1", kind="comp", ctype="Resistor")
    g.add_node("cur_comp:C1", kind="comp", ctype="CapacitorCeramic")
    g.add_node("cur_comp:R2", kind="comp", ctype="Resistor")
    g.add_node("cur_net:NET_000", kind="net", role="input")
    g.add_node("cur_net:NET_001", kind="net", role="signal")
    g.add_node("cur_net:NET_002", kind="net", role="ground")
    g.add_node("cur_net:NET_003", kind="net", role="signal")
    g.add_edge("cur_comp:R1", "cur_net:NET_000")
    g.add_edge("cur_comp:R1", "cur_net:NET_001")
    g.add_edge("cur_comp:C1", "cur_net:NET_001")
    g.add_edge("cur_comp:C1", "cur_net:NET_002")
    g.add_edge("cur_comp:R2", "cur_net:NET_001")
    g.add_edge("cur_comp:R2", "cur_net:NET_003")
    return g


def _build_cur_wrong_connection() -> nx.Graph:
    """R1 and C1 both connect to VIN and GND (parallel instead of series)"""
    g = nx.Graph()
    g.add_node("cur_comp:R1", kind="comp", ctype="Resistor")
    g.add_node("cur_comp:C1", kind="comp", ctype="CapacitorCeramic")
    g.add_node("cur_net:NET_000", kind="net", role="input")
    g.add_node("cur_net:NET_001", kind="net", role="ground")
    g.add_edge("cur_comp:R1", "cur_net:NET_000")
    g.add_edge("cur_comp:R1", "cur_net:NET_001")
    g.add_edge("cur_comp:C1", "cur_net:NET_000")
    g.add_edge("cur_comp:C1", "cur_net:NET_001")
    return g


class TestCompareLogicalGraphs:
    def test_full_isomorphism(self) -> None:
        result = compare_logical_graphs(_build_ref(), _build_cur_match())
        assert result["logic_correct"] is True
        assert result["similarity"] == 1.0
        assert result["progress"] == 1.0
        assert result["details"]["match_type"] == "full_isomorphism"

    def test_missing_component(self) -> None:
        result = compare_logical_graphs(_build_ref(), _build_cur_missing_cap())
        assert result["logic_correct"] is False
        assert result["details"]["match_type"] == "current_subgraph_in_reference"
        items = result["report"]["items"]
        assert any(i["error_code"] == "COMPONENT_MISSING" for i in items)
        assert any(i["error_code"] == "INCOMPLETE_CIRCUIT" for i in items)

    def test_extra_component(self) -> None:
        result = compare_logical_graphs(_build_ref(), _build_cur_extra_resistor())
        assert result["logic_correct"] is False
        assert result["details"]["match_type"] == "reference_subgraph_in_current"
        items = result["report"]["items"]
        assert any(i["error_code"] == "COMPONENT_EXTRA" for i in items)

    def test_wrong_connection(self) -> None:
        result = compare_logical_graphs(_build_ref(), _build_cur_wrong_connection())
        assert result["logic_correct"] is False
        assert result["details"]["match_type"] == "graph_edit_distance_or_fallback"
        items = result["report"]["items"]
        assert any(i["error_code"] == "WRONG_CONNECTION" for i in items)

    def test_no_hole_mismatch(self) -> None:
        result = compare_logical_graphs(_build_ref(), _build_cur_match())
        assert not any(i.get("category") == "hole_errors" for i in result["report"]["items"])
