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

    def test_different_component_ids_same_topology(self) -> None:
        """Ref R1+C1 vs Cur R7+C9 — different IDs, same topology."""
        g_ref = nx.Graph()
        g_ref.add_node("ref_comp:R1", kind="comp", ctype="Resistor")
        g_ref.add_node("ref_comp:C1", kind="comp", ctype="CapacitorCeramic")
        g_ref.add_node("ref_net:VIN", kind="net", role="input")
        g_ref.add_node("ref_net:VC", kind="net", role="signal")
        g_ref.add_node("ref_net:GND", kind="net", role="ground")
        g_ref.add_edge("ref_comp:R1", "ref_net:VIN")
        g_ref.add_edge("ref_comp:R1", "ref_net:VC")
        g_ref.add_edge("ref_comp:C1", "ref_net:VC")
        g_ref.add_edge("ref_comp:C1", "ref_net:GND")

        g_cur = nx.Graph()
        g_cur.add_node("cur_comp:R7", kind="comp", ctype="Resistor")
        g_cur.add_node("cur_comp:C9", kind="comp", ctype="CapacitorCeramic")
        g_cur.add_node("cur_net:NET_000", kind="net", role="input")
        g_cur.add_node("cur_net:NET_001", kind="net", role="signal")
        g_cur.add_node("cur_net:NET_002", kind="net", role="ground")
        g_cur.add_edge("cur_comp:R7", "cur_net:NET_000")
        g_cur.add_edge("cur_comp:R7", "cur_net:NET_001")
        g_cur.add_edge("cur_comp:C9", "cur_net:NET_001")
        g_cur.add_edge("cur_comp:C9", "cur_net:NET_002")

        result = compare_logical_graphs(g_ref, g_cur)
        assert result["logic_correct"] is True
        assert result["details"]["match_type"] == "full_isomorphism"

    def test_passive_pin_swap_resistor(self) -> None:
        """Resistor pin1↔pin2 swap should still be isomorphic."""
        g_ref = nx.Graph()
        g_ref.add_node("ref_comp:R1", kind="comp", ctype="Resistor")
        g_ref.add_node("ref_net:A", kind="net", role="signal")
        g_ref.add_node("ref_net:B", kind="net", role="signal")
        g_ref.add_edge("ref_comp:R1", "ref_net:A", pin="pin1", comp_type="Resistor")
        g_ref.add_edge("ref_comp:R1", "ref_net:B", pin="pin2", comp_type="Resistor")

        g_cur = nx.Graph()
        g_cur.add_node("cur_comp:R1", kind="comp", ctype="Resistor")
        g_cur.add_node("cur_net:NET_0", kind="net", role="signal")
        g_cur.add_node("cur_net:NET_1", kind="net", role="signal")
        g_cur.add_edge("cur_comp:R1", "cur_net:NET_0", pin="pin2", comp_type="Resistor")
        g_cur.add_edge("cur_comp:R1", "cur_net:NET_1", pin="pin1", comp_type="Resistor")

        result = compare_logical_graphs(g_ref, g_cur)
        assert result["logic_correct"] is True
        assert result["details"]["match_type"] == "full_isomorphism"

    def test_passive_pin_swap_capacitor(self) -> None:
        """CapacitorCeramic pin1↔pin2 swap should still be isomorphic."""
        g_ref = nx.Graph()
        g_ref.add_node("ref_comp:C1", kind="comp", ctype="CapacitorCeramic")
        g_ref.add_node("ref_net:A", kind="net", role="signal")
        g_ref.add_node("ref_net:B", kind="net", role="signal")
        g_ref.add_edge("ref_comp:C1", "ref_net:A", pin="pin1", comp_type="CapacitorCeramic")
        g_ref.add_edge("ref_comp:C1", "ref_net:B", pin="pin2", comp_type="CapacitorCeramic")

        g_cur = nx.Graph()
        g_cur.add_node("cur_comp:C9", kind="comp", ctype="CapacitorCeramic")
        g_cur.add_node("cur_net:NET_0", kind="net", role="signal")
        g_cur.add_node("cur_net:NET_1", kind="net", role="signal")
        g_cur.add_edge("cur_comp:C9", "cur_net:NET_0", pin="pin2", comp_type="CapacitorCeramic")
        g_cur.add_edge("cur_comp:C9", "cur_net:NET_1", pin="pin1", comp_type="CapacitorCeramic")

        result = compare_logical_graphs(g_ref, g_cur)
        assert result["logic_correct"] is True
        assert result["details"]["match_type"] == "full_isomorphism"

    def test_summary_metadata(self) -> None:
        result = compare_logical_graphs(_build_ref(), _build_cur_match())
        summary = result["report"]["summary"]
        assert summary.get("ignore_component_id") is True
        assert summary.get("ignore_hole_id") is True
        assert summary.get("ignore_passive_pin_order") is True
        assert summary.get("equivalence_rule") == "component_type_and_topology"

    def test_led_polarity_field_ignored(self) -> None:
        """LED polarity 字段不同，但图比较应视为正确（本阶段忽略极性）。"""
        g_ref = nx.Graph()
        g_ref.add_node("ref_comp:D1", kind="comp", ctype="LED")
        g_ref.add_node("ref_net:A", kind="net", role="signal")
        g_ref.add_node("ref_net:B", kind="net", role="signal")
        g_ref.add_edge("ref_comp:D1", "ref_net:A", pin="anode", comp_type="LED")
        g_ref.add_edge("ref_comp:D1", "ref_net:B", pin="cathode", comp_type="LED")

        g_cur = nx.Graph()
        g_cur.add_node("cur_comp:D1", kind="comp", ctype="LED")
        g_cur.add_node("cur_net:NET_0", kind="net", role="signal")
        g_cur.add_node("cur_net:NET_1", kind="net", role="signal")
        g_cur.add_edge("cur_comp:D1", "cur_net:NET_0", pin="anode", comp_type="LED")
        g_cur.add_edge("cur_comp:D1", "cur_net:NET_1", pin="cathode", comp_type="LED")

        result = compare_logical_graphs(g_ref, g_cur)
        assert result["logic_correct"] is True
        assert result["report"]["summary"].get("ignore_polarity") is True
        assert result["report"]["polarity_errors"] == []

    def test_vcc_ground_role_mismatch(self) -> None:
        """VCC/GND 网络角色不匹配时应检测为错误。"""
        g_ref = nx.Graph()
        g_ref.add_node("ref_comp:R1", kind="comp", ctype="Resistor")
        g_ref.add_node("ref_net:VCC", kind="net", role="power")
        g_ref.add_node("ref_net:GND", kind="net", role="ground")
        g_ref.add_edge("ref_comp:R1", "ref_net:VCC", pin="pin1", comp_type="Resistor")
        g_ref.add_edge("ref_comp:R1", "ref_net:GND", pin="pin2", comp_type="Resistor")

        # 当前电路中缺少 power 节点，有一个 signal 节点，导致 VCC 无法映射
        g_cur = nx.Graph()
        g_cur.add_node("cur_comp:R1", kind="comp", ctype="Resistor")
        g_cur.add_node("cur_net:NET_0", kind="net", role="ground")
        g_cur.add_node("cur_net:NET_1", kind="net", role="signal")
        g_cur.add_edge("cur_comp:R1", "cur_net:NET_0", pin="pin1", comp_type="Resistor")
        g_cur.add_edge("cur_comp:R1", "cur_net:NET_1", pin="pin2", comp_type="Resistor")

        result = compare_logical_graphs(g_ref, g_cur)
        assert result["logic_correct"] is False
        items = result["report"]["items"]
        assert any(i["error_family"] in {"wiring_mismatch", "open_circuit", "extra_connection"} for i in items)

    def test_ic_pin_role_checked(self) -> None:
        """多引脚 IC 元件 pin 角色不可互换。"""
        g_ref = nx.Graph()
        g_ref.add_node("ref_comp:U1", kind="comp", ctype="IC")
        g_ref.add_node("ref_net:VCC", kind="net", role="power")
        g_ref.add_node("ref_net:GND", kind="net", role="ground")
        g_ref.add_node("ref_net:OUT", kind="net", role="output")
        g_ref.add_edge("ref_comp:U1", "ref_net:VCC", pin="VCC", comp_type="IC")
        g_ref.add_edge("ref_comp:U1", "ref_net:GND", pin="GND", comp_type="IC")
        g_ref.add_edge("ref_comp:U1", "ref_net:OUT", pin="OUT", comp_type="IC")

        g_cur = nx.Graph()
        g_cur.add_node("cur_comp:U1", kind="comp", ctype="IC")
        g_cur.add_node("cur_net:NET_0", kind="net", role="power")
        g_cur.add_node("cur_net:NET_1", kind="net", role="ground")
        g_cur.add_node("cur_net:NET_2", kind="net", role="output")
        # swap VCC and GND pins
        g_cur.add_edge("cur_comp:U1", "cur_net:NET_0", pin="GND", comp_type="IC")
        g_cur.add_edge("cur_comp:U1", "cur_net:NET_1", pin="VCC", comp_type="IC")
        g_cur.add_edge("cur_comp:U1", "cur_net:NET_2", pin="OUT", comp_type="IC")

        result = compare_logical_graphs(g_ref, g_cur)
        assert result["logic_correct"] is False
        items = result["report"]["items"]
        assert any(i["error_family"] == "wiring_mismatch" for i in items)
