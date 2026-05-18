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
        # 额外元件未合并关键网络时视为无害 extra，logic_correct 可为 true
        assert result["logic_correct"] is True
        assert result["details"]["match_type"] == "equivalent_with_extra"
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

    def test_non_polar_capacitor_class_aliases_match(self) -> None:
        """Generic non-polar Capacitor and CapacitorCeramic should match."""
        g_ref = nx.Graph()
        g_ref.add_node("ref_comp:C1", kind="comp", ctype="Capacitor")
        g_ref.add_node("ref_net:A", kind="net", role="signal")
        g_ref.add_node("ref_net:B", kind="net", role="ground")
        g_ref.add_edge("ref_comp:C1", "ref_net:A", pin="pin1", comp_type="Capacitor")
        g_ref.add_edge("ref_comp:C1", "ref_net:B", pin="pin2", comp_type="Capacitor")

        g_cur = nx.Graph()
        g_cur.add_node("cur_comp:C9", kind="comp", ctype="CapacitorCeramic")
        g_cur.add_node("cur_net:NET_0", kind="net", role="signal")
        g_cur.add_node("cur_net:NET_1", kind="net", role="ground")
        g_cur.add_edge("cur_comp:C9", "cur_net:NET_0", pin="pin2", comp_type="CapacitorCeramic")
        g_cur.add_edge("cur_comp:C9", "cur_net:NET_1", pin="pin1", comp_type="CapacitorCeramic")

        result = compare_logical_graphs(g_ref, g_cur)
        assert result["logic_correct"] is True
        assert result["similarity"] == 1.0
        assert result["details"]["match_type"] == "full_isomorphism"

    def test_electrolytic_capacitor_still_strict(self) -> None:
        """Electrolytic capacitors must not match non-polar capacitors."""
        g_ref = nx.Graph()
        g_ref.add_node("ref_comp:C1", kind="comp", ctype="CapacitorElectrolytic")
        g_ref.add_node("ref_net:VCC", kind="net", role="power", role_label="VCC")
        g_ref.add_node("ref_net:GND", kind="net", role="ground", role_label="GND")
        g_ref.add_edge(
            "ref_comp:C1",
            "ref_net:VCC",
            pin="positive",
            pin_role="positive",
            comp_type="CapacitorElectrolytic",
        )
        g_ref.add_edge(
            "ref_comp:C1",
            "ref_net:GND",
            pin="negative",
            pin_role="negative",
            comp_type="CapacitorElectrolytic",
        )

        g_cur = nx.Graph()
        g_cur.add_node("cur_comp:C9", kind="comp", ctype="CapacitorCeramic")
        g_cur.add_node("cur_net:NET_0", kind="net", role="power", role_label="VCC")
        g_cur.add_node("cur_net:NET_1", kind="net", role="ground", role_label="GND")
        g_cur.add_edge("cur_comp:C9", "cur_net:NET_0", pin="pin1", comp_type="CapacitorCeramic")
        g_cur.add_edge("cur_comp:C9", "cur_net:NET_1", pin="pin2", comp_type="CapacitorCeramic")

        result = compare_logical_graphs(g_ref, g_cur)
        assert result["logic_correct"] is False

    def test_summary_metadata(self) -> None:
        result = compare_logical_graphs(_build_ref(), _build_cur_match())
        summary = result["report"]["summary"]
        assert summary.get("ignore_component_id") is True
        assert summary.get("ignore_hole_id") is True
        assert summary.get("ignore_passive_pin_order") is True
        assert summary.get("strict_functional_pin_roles") is True
        assert summary.get("equivalence_rule") == "logical_topology_with_port_semantics"

    def test_led_polarity_strict(self) -> None:
        """LED anode/cathode 正确连接时应通过；anode 接到 ground 网络时应失败。"""
        g_ref = nx.Graph()
        g_ref.add_node("ref_comp:D1", kind="comp", ctype="LED")
        g_ref.add_node("ref_net:VCC", kind="net", role="power", role_label="VCC")
        g_ref.add_node("ref_net:GND", kind="net", role="ground", role_label="GND")
        g_ref.add_edge("ref_comp:D1", "ref_net:VCC", pin="anode", pin_role="anode", comp_type="LED")
        g_ref.add_edge("ref_comp:D1", "ref_net:GND", pin="cathode", pin_role="cathode", comp_type="LED")

        # 正确连接
        g_cur_ok = nx.Graph()
        g_cur_ok.add_node("cur_comp:D1", kind="comp", ctype="LED")
        g_cur_ok.add_node("cur_net:N0", kind="net", role="power", role_label="VCC")
        g_cur_ok.add_node("cur_net:N1", kind="net", role="ground", role_label="GND")
        g_cur_ok.add_edge("cur_comp:D1", "cur_net:N0", pin="anode", pin_role="anode", comp_type="LED")
        g_cur_ok.add_edge("cur_comp:D1", "cur_net:N1", pin="cathode", pin_role="cathode", comp_type="LED")

        result_ok = compare_logical_graphs(g_ref, g_cur_ok)
        assert result_ok["logic_correct"] is True

        # anode/cathode 接反（anode 接到 GND，cathode 接到 VCC）
        g_cur_bad = nx.Graph()
        g_cur_bad.add_node("cur_comp:D1", kind="comp", ctype="LED")
        g_cur_bad.add_node("cur_net:N0", kind="net", role="power", role_label="VCC")
        g_cur_bad.add_node("cur_net:N1", kind="net", role="ground", role_label="GND")
        g_cur_bad.add_edge("cur_comp:D1", "cur_net:N1", pin="anode", pin_role="anode", comp_type="LED")
        g_cur_bad.add_edge("cur_comp:D1", "cur_net:N0", pin="cathode", pin_role="cathode", comp_type="LED")

        result_bad = compare_logical_graphs(g_ref, g_cur_bad)
        assert result_bad["logic_correct"] is False

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

    def test_unlabeled_port_roles_are_inferred(self) -> None:
        from app.domain.logical_reference import current_netlist_v2_to_graph, logical_reference_to_graph

        ref_payload = {
            "format": "logical_reference_v1",
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
                }
            ],
            "nets": [
                {"net": "VIN", "role": "input", "role_label": "UI1"},
                {"net": "VC", "role": "signal"},
                {"net": "GND", "role": "ground", "role_label": "GND"},
            ],
        }
        cur_netlist = {
            "components": [
                {
                    "component_id": "R7",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "N0"},
                        {"pin_name": "pin2", "electrical_net_id": "N1"},
                    ],
                },
                {
                    "component_id": "C7",
                    "component_type": "CapacitorCeramic",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "N1"},
                        {"pin_name": "pin2", "electrical_net_id": "N2"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "N0"},
                {"electrical_net_id": "N1"},
                {"electrical_net_id": "N2"},
            ],
        }

        result = compare_logical_graphs(
            logical_reference_to_graph(ref_payload),
            current_netlist_v2_to_graph(cur_netlist),
            ref_payload=ref_payload,
            cur_netlist_v2=cur_netlist,
        )
        assert result["logic_correct"] is True
        assert result["details"]["match_type"] == "full_isomorphism_with_inferred_roles"

    def test_existing_mismatched_port_label_is_not_silently_overwritten(self) -> None:
        from app.domain.logical_reference import current_netlist_v2_to_graph, logical_reference_to_graph

        ref_payload = {
            "format": "logical_reference_v1",
            "components": [
                {"ref_id": "U1", "type": "IC", "pins": [{"pin": "IN1", "net": "UI1"}]}
            ],
            "nets": [{"net": "UI1", "role": "input", "role_label": "UI1"}],
        }
        cur_netlist = {
            "components": [
                {
                    "component_id": "U2",
                    "component_type": "IC",
                    "pins": [{"pin_name": "IN1", "electrical_net_id": "N0"}],
                }
            ],
            "nets": [{"electrical_net_id": "N0", "role": "input", "role_label": "UI2"}],
        }

        result = compare_logical_graphs(
            logical_reference_to_graph(ref_payload),
            current_netlist_v2_to_graph(cur_netlist),
            ref_payload=ref_payload,
            cur_netlist_v2=cur_netlist,
        )
        assert result["logic_correct"] is False

    def test_auto_detected_symmetric_ports_can_swap(self) -> None:
        ref = nx.Graph()
        ref.add_node("ref_comp:R1", kind="comp", ctype="Resistor", source_id="R1")
        ref.add_node("ref_comp:R2", kind="comp", ctype="Resistor", source_id="R2")
        ref.add_node("ref_net:UI1", kind="net", role="input", role_label="UI1", source_id="UI1")
        ref.add_node("ref_net:UI2", kind="net", role="input", role_label="UI2", source_id="UI2")
        ref.add_node("ref_net:MID", kind="net", role="signal", source_id="MID")
        ref.add_edge("ref_comp:R1", "ref_net:UI1", pin="pin1", comp_type="Resistor")
        ref.add_edge("ref_comp:R1", "ref_net:MID", pin="pin2", comp_type="Resistor")
        ref.add_edge("ref_comp:R2", "ref_net:UI2", pin="pin1", comp_type="Resistor")
        ref.add_edge("ref_comp:R2", "ref_net:MID", pin="pin2", comp_type="Resistor")

        cur = nx.Graph()
        cur.add_node("cur_comp:R7", kind="comp", ctype="Resistor", source_id="R7")
        cur.add_node("cur_comp:R8", kind="comp", ctype="Resistor", source_id="R8")
        cur.add_node("cur_net:N0", kind="net", role="input", role_label="UI2", source_id="N0")
        cur.add_node("cur_net:N1", kind="net", role="input", role_label="UI1", source_id="N1")
        cur.add_node("cur_net:N2", kind="net", role="signal", source_id="N2")
        cur.add_edge("cur_comp:R7", "cur_net:N0", pin="pin1", comp_type="Resistor")
        cur.add_edge("cur_comp:R7", "cur_net:N2", pin="pin2", comp_type="Resistor")
        cur.add_edge("cur_comp:R8", "cur_net:N1", pin="pin1", comp_type="Resistor")
        cur.add_edge("cur_comp:R8", "cur_net:N2", pin="pin2", comp_type="Resistor")

        result = compare_logical_graphs(ref, cur)
        assert result["logic_correct"] is True
        assert result["details"]["match_type"] == "equivalent_with_allowed_symmetry"


# ---------------------------------------------------------------------------
# R1 Position B — extras on role-critical nets promote to logic_correct=False.
# See app/domain/compare/RULE_SEMANTICS.md §3 + RISK_REGISTER §5 R1.
# ---------------------------------------------------------------------------


def _build_cur_extra_resistor_to_vcc() -> nx.Graph:
    """Same shape as `_build_cur_extra_resistor` but R2 connects to a new
    VCC net (role="power") instead of a signal-internal pair."""

    g = nx.Graph()
    g.add_node("cur_comp:R1", kind="comp", ctype="Resistor")
    g.add_node("cur_comp:C1", kind="comp", ctype="CapacitorCeramic")
    g.add_node("cur_comp:R2", kind="comp", ctype="Resistor")
    g.add_node("cur_net:NET_000", kind="net", role="input")
    g.add_node("cur_net:NET_001", kind="net", role="signal")
    g.add_node("cur_net:NET_002", kind="net", role="ground")
    g.add_node("cur_net:NET_VCC", kind="net", role="power")
    g.add_edge("cur_comp:R1", "cur_net:NET_000")
    g.add_edge("cur_comp:R1", "cur_net:NET_001")
    g.add_edge("cur_comp:C1", "cur_net:NET_001")
    g.add_edge("cur_comp:C1", "cur_net:NET_002")
    # Extra R2 connects signal net → VCC (power, role-critical)
    g.add_edge("cur_comp:R2", "cur_net:NET_001")
    g.add_edge("cur_comp:R2", "cur_net:NET_VCC")
    return g


def _build_cur_extra_wire_on_gnd() -> nx.Graph:
    """An extra Wire node bridging an existing internal net to GND."""

    g = nx.Graph()
    g.add_node("cur_comp:R1", kind="comp", ctype="Resistor")
    g.add_node("cur_comp:C1", kind="comp", ctype="CapacitorCeramic")
    g.add_node("cur_comp:W1", kind="comp", ctype="Wire")
    g.add_node("cur_net:NET_000", kind="net", role="input")
    g.add_node("cur_net:NET_001", kind="net", role="signal")
    g.add_node("cur_net:NET_002", kind="net", role="ground")
    g.add_node("cur_net:NET_003", kind="net", role="signal")
    g.add_edge("cur_comp:R1", "cur_net:NET_000")
    g.add_edge("cur_comp:R1", "cur_net:NET_001")
    g.add_edge("cur_comp:C1", "cur_net:NET_001")
    g.add_edge("cur_comp:C1", "cur_net:NET_002")
    # Extra Wire on GND — degree on ground role goes up.
    g.add_edge("cur_comp:W1", "cur_net:NET_002")
    g.add_edge("cur_comp:W1", "cur_net:NET_003")
    return g


def _build_cur_extra_resistor_on_signal_only() -> nx.Graph:
    """Same as `_build_cur_extra_resistor` — R2 between two signal nets,
    nothing critical. Kept as a regression guard to verify R1 doesn't
    over-fire on benign extras."""

    return _build_cur_extra_resistor()


class TestR1CriticalExtras:
    def test_extra_on_signal_stays_pass(self) -> None:
        """Extra component connecting two signal nets must still pass
        under the lenient ``equivalent_with_extra`` semantics."""

        result = compare_logical_graphs(
            _build_ref(), _build_cur_extra_resistor_on_signal_only(),
        )
        assert result["logic_correct"] is True
        assert result["details"]["match_type"] == "equivalent_with_extra"
        assert not any(
            i["error_code"] == "CRITICAL_EXTRA_CONNECTION"
            for i in result["report"]["items"]
        )

    def test_extra_resistor_to_vcc_fails(self) -> None:
        """Extra resistor connecting signal → VCC (role=power) must
        flip logic_correct to False under R1 Position B."""

        result = compare_logical_graphs(
            _build_ref(), _build_cur_extra_resistor_to_vcc(),
        )
        assert result["logic_correct"] is False
        assert result["is_correct"] is False
        assert result["is_match"] is False
        assert result["details"]["match_type"] == "extra_on_critical_net"
        assert any(
            i["error_code"] == "CRITICAL_EXTRA_CONNECTION"
            and i["severity"] == "error"
            and i["expected"]["role"] == "power"
            for i in result["report"]["items"]
        )
        # details.critical_extras surfaces the offending role(s)
        crit = result["details"]["critical_extras"]
        assert any(c["role"] == "power" and c["extra_edges"] >= 1 for c in crit)

    def test_extra_wire_on_gnd_fails(self) -> None:
        """Extra Wire that bumps degree on a ground net must fail."""

        result = compare_logical_graphs(
            _build_ref(), _build_cur_extra_wire_on_gnd(),
        )
        assert result["logic_correct"] is False
        assert result["details"]["match_type"] == "extra_on_critical_net"
        crit_items = [
            i for i in result["report"]["items"]
            if i["error_code"] == "CRITICAL_EXTRA_CONNECTION"
        ]
        assert any(i["expected"]["role"] == "ground" for i in crit_items)

    def test_critical_extra_marks_report_topology_errors(self) -> None:
        """The promoted item should land in report.topology_errors
        bucket (validator_report_v2 convention)."""

        result = compare_logical_graphs(
            _build_ref(), _build_cur_extra_resistor_to_vcc(),
        )
        report = result["report"]
        # summary mirror updated
        assert report["summary"]["logic_correct"] is False
        assert report["summary"]["match_type"] == "extra_on_critical_net"
        # topology bucket carries the critical item
        assert any(
            i["error_code"] == "CRITICAL_EXTRA_CONNECTION"
            for i in report.get("topology_errors", [])
        )
        # total_item_count reflects the bump
        assert report["summary"]["total_item_count"] >= len(report["items"])
