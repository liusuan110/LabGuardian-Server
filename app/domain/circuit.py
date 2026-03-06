"""
电路拓扑建模 (← src_v2/logic/circuit.py)

将面包板上检测到的元件构建为电气连接图，生成网表描述。
基于 NetworkX 图论，支持极性感知、电源网络识别。
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


# ============================================================
# 元件类型归一化
# ============================================================

def norm_component_type(t: str) -> str:
    """归一化元件类型名 (与 YOLO 训练类名对齐)"""
    if not t:
        return "UNKNOWN"
    u = str(t).strip().upper()
    if "RESIST" in u:
        return "Resistor"
    if "WIRE" in u:
        return "Wire"
    if "LED" in u:
        return "LED"
    return u


# ============================================================
# 枚举定义
# ============================================================

class Polarity(Enum):
    NONE = "none"
    FORWARD = "forward"
    REVERSE = "reverse"
    UNKNOWN = "unknown"


class PinRole(Enum):
    GENERIC = "generic"
    ANODE = "anode"
    CATHODE = "cathode"
    BASE = "base"
    COLLECTOR = "collector"
    EMITTER = "emitter"
    VCC = "vcc"
    GND = "gnd"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    WIPER = "wiper"
    TERMINAL_A = "terminal_a"
    TERMINAL_B = "terminal_b"


@dataclass
class CircuitComponent:
    """电路元件"""

    name: str
    type: str
    pin1_loc: Tuple[str, str]
    pin2_loc: Optional[Tuple[str, str]] = None
    polarity: Polarity = Polarity.NONE
    confidence: float = 1.0
    orientation_deg: float = 0.0

    def __repr__(self):
        pol = f" [{self.polarity.value}]" if self.polarity != Polarity.NONE else ""
        return f"{self.name}({self.pin1_loc}-{self.pin2_loc}{pol})"

    @property
    def is_polarized(self) -> bool:
        return self.polarity not in (Polarity.NONE,)

    @property
    def has_known_polarity(self) -> bool:
        return self.polarity in (Polarity.FORWARD, Polarity.REVERSE)


POLARIZED_TYPES = {"LED"}
THREE_PIN_TYPES: set = set()
CAPACITOR_TYPES: set = set()
NON_POLAR_TYPES = {"Resistor", "Wire"}
IC_TYPES: set = set()
POTENTIOMETER_TYPES: set = set()
POWER_KEYWORDS = {"VCC", "GND", "POWER", "BATTERY"}


class CircuitAnalyzer:
    """电路拓扑分析器

    基于 NetworkX 图论，将面包板上的元件建模为电气连接图。
    面包板规则: 同行 a-e 导通 (Left), f-j 导通 (Right)。
    """

    def __init__(self, rail_track_rows: Optional[Dict[str, tuple]] = None):
        self.graph = nx.Graph()
        self.components: List[CircuitComponent] = []
        self.power_nets: Dict[str, str] = {}
        self._name_counters: Dict[str, int] = {}
        self._rail_track_rows = rail_track_rows or {}
        self._row_to_rail: Dict[int, str] = {}
        for track_id, rows in self._rail_track_rows.items():
            for r in rows:
                self._row_to_rail[r] = track_id
        self.rail_assignments: Dict[str, str] = {}

    def reset(self):
        self.graph.clear()
        self.components = []
        self.power_nets = {}
        self._name_counters = {}

    _TYPE_PREFIX = {
        "Resistor": "R", "Wire": "W", "LED": "LED",
    }

    def _auto_name(self, comp_type: str) -> str:
        norm = self._norm_type(comp_type)
        prefix = self._TYPE_PREFIX.get(norm, norm[:3])
        self._name_counters[norm] = self._name_counters.get(norm, 0) + 1
        return f"{prefix}{self._name_counters[norm]}"

    def add_component(self, comp: CircuitComponent):
        """添加元件到电路图"""
        if comp.name == comp.type or comp.name in ("UNKNOWN", ""):
            comp.name = self._auto_name(comp.type)
        self.components.append(comp)

        node1 = self._get_node_name(comp.pin1_loc)
        if comp.pin2_loc:
            node2 = self._get_node_name(comp.pin2_loc)
            edge_attrs = {
                "component": comp.name,
                "type": comp.type,
                "polarity": comp.polarity.value,
                "confidence": comp.confidence,
                "pin1_role": "generic",
                "pin2_role": "generic",
            }
            self.graph.add_edge(node1, node2, **edge_attrs)
        else:
            self.graph.add_node(node1, component=comp.name)

    def _get_node_name(self, loc: Tuple[str, str]) -> str:
        """面包板导通规则映射"""
        row, col = loc
        if col in ("+", "plus", "P"):
            return "PWR_PLUS"
        if col in ("-", "minus", "N", "GND"):
            return "PWR_MINUS"

        try:
            row_int = int(row)
        except (ValueError, TypeError):
            row_int = -1

        if row_int in self._row_to_rail:
            return self._row_to_rail[row_int]

        if col in ("a", "b", "c", "d", "e"):
            return f"Row{row}_L"
        else:
            return f"Row{row}_R"

    @staticmethod
    def _norm_type(t: str) -> str:
        return norm_component_type(t)

    def build_topology_graph(self) -> nx.Graph:
        """构建布局无关的拓扑图 (Wire 视为理想导体)"""
        conductor = nx.Graph()
        conductor.add_nodes_from(self.graph.nodes())

        for u, v, data in self.graph.edges(data=True):
            if self._norm_type(data.get("type", "")) == "Wire":
                conductor.add_edge(u, v)

        net_groups = list(nx.connected_components(conductor))
        node_to_net = {}
        for i, group in enumerate(net_groups):
            for n in group:
                node_to_net[n] = f"N{i}"

        topo = nx.Graph()
        self._identify_power_nets()
        for i in range(len(net_groups)):
            net_id = f"N{i}"
            attrs = {"kind": "net"}
            for n in net_groups[i]:
                if n in self.power_nets:
                    attrs["power"] = self.power_nets[n]
                    break
            topo.add_node(net_id, **attrs)

        comp_idx = 0
        for comp in self.components:
            ctype = self._norm_type(comp.type)
            if ctype == "Wire":
                continue

            try:
                n1 = node_to_net.get(self._get_node_name(comp.pin1_loc))
                n2 = node_to_net.get(self._get_node_name(comp.pin2_loc)) if comp.pin2_loc else None
            except Exception:
                n1, n2 = None, None

            if n1 is None:
                continue

            cid = f"C{comp_idx}"
            comp_idx += 1
            node_attrs = {
                "kind": "comp",
                "ctype": ctype,
                "polarity": comp.polarity.value,
            }

            if n2 is None:
                node_attrs["pins"] = 1
                topo.add_node(cid, **node_attrs)
                topo.add_edge(cid, n1)
            elif n1 == n2:
                node_attrs["pins"] = 2
                node_attrs["same_net"] = True
                topo.add_node(cid, **node_attrs)
                topo.add_edge(cid, n1)
            else:
                node_attrs["pins"] = 2
                topo.add_node(cid, **node_attrs)
                topo.add_edge(cid, n1)
                topo.add_edge(cid, n2)

        return topo

    def get_circuit_description(self) -> str:
        """生成结构化电路网表描述"""
        if not self.components:
            return "当前未检测到电路元件。"

        self._identify_power_nets()
        connected_groups = list(nx.connected_components(self.graph))

        node_to_net = {}
        for idx, group in enumerate(connected_groups):
            net_id = f"Net_{idx + 1}"
            for n in group:
                node_to_net[n] = net_id

        type_counts = Counter(self._norm_type(c.type) for c in self.components)
        total = len(self.components)
        counts_str = ", ".join(f"{t}×{c}" for t, c in sorted(type_counts.items()))
        desc = f"电路概况: 共 {total} 个元件 ({counts_str}), {len(connected_groups)} 个电气网络\n\n"

        desc += "元件连接:\n"
        for comp in self.components:
            ctype = self._norm_type(comp.type)
            node1 = self._get_node_name(comp.pin1_loc)
            net1 = node_to_net.get(node1, "?")

            role_info = ""
            if comp.polarity == Polarity.FORWARD:
                role_info = " [pin1/pin2]"
            elif comp.polarity == Polarity.REVERSE:
                role_info = " [pin1/pin2 反向]"
            elif comp.polarity == Polarity.UNKNOWN:
                role_info = " [极性未知]"

            if comp.pin2_loc:
                node2 = self._get_node_name(comp.pin2_loc)
                net2 = node_to_net.get(node2, "?")
                desc += (
                    f"  {comp.name} ({ctype}{role_info}): "
                    f"Row{comp.pin1_loc[0]}{comp.pin1_loc[1]}({net1}) — "
                    f"Row{comp.pin2_loc[0]}{comp.pin2_loc[1]}({net2})\n"
                )
            else:
                desc += (
                    f"  {comp.name} ({ctype}{role_info}): "
                    f"Row{comp.pin1_loc[0]}{comp.pin1_loc[1]}({net1})\n"
                )

        desc += "\n电气网络:\n"
        for idx, group in enumerate(connected_groups):
            net_id = f"Net_{idx + 1}"
            nodes = sorted(list(group))

            comps_on_net = []
            for comp in self.components:
                comp_nodes = set()
                comp_nodes.add(self._get_node_name(comp.pin1_loc))
                if comp.pin2_loc:
                    comp_nodes.add(self._get_node_name(comp.pin2_loc))
                if comp_nodes & group:
                    comps_on_net.append(comp.name)

            power_tag = ""
            for n in nodes:
                if n in self.power_nets:
                    power_tag = f" [{self.power_nets[n]}]"
                    break
            comps_str = ", ".join(sorted(set(comps_on_net)))
            desc += f"  {net_id}{power_tag}: {', '.join(nodes)} → 元件: {comps_str}\n"

        if self.power_nets:
            desc += "\n电源:\n"
            for node, ptype in sorted(self.power_nets.items()):
                label = self.rail_assignments.get(node, "")
                extra = f" ({label})" if label else ""
                desc += f"  {node} → {ptype}{extra}\n"

        issues = self._quick_check_issues()
        if issues:
            desc += "\n⚠ 潜在问题:\n"
            for issue in issues:
                desc += f"  - {issue}\n"

        return desc

    def _quick_check_issues(self) -> List[str]:
        issues = []
        has_led = False
        has_resistor_near_led = False

        for comp in self.components:
            ctype = self._norm_type(comp.type)
            if ctype == "LED":
                has_led = True
                led_node1 = self._get_node_name(comp.pin1_loc)
                led_node2 = self._get_node_name(comp.pin2_loc) if comp.pin2_loc else None
                for other in self.components:
                    if self._norm_type(other.type) == "Resistor":
                        r_node1 = self._get_node_name(other.pin1_loc)
                        r_node2 = self._get_node_name(other.pin2_loc) if other.pin2_loc else None
                        if r_node1 in (led_node1, led_node2) or r_node2 in (led_node1, led_node2):
                            has_resistor_near_led = True
                            break

            if ctype in POLARIZED_TYPES and comp.polarity == Polarity.UNKNOWN:
                issues.append(f"{comp.name} ({ctype}) 极性未确定, 请检查安装方向")
            if comp.pin2_loc:
                n1 = self._get_node_name(comp.pin1_loc)
                n2 = self._get_node_name(comp.pin2_loc)
                if n1 == n2 and ctype != "WIRE":
                    issues.append(f"{comp.name} ({ctype}) 两引脚在同一导通组, 可能短路或未跨行")

        if has_led and not has_resistor_near_led:
            issues.append("LED 未检测到相邻限流电阻, 可能缺少限流保护")

        return issues

    def _identify_power_nets(self):
        for track_id, label in self.rail_assignments.items():
            if track_id in self.graph:
                power_type = self._parse_rail_label(label)
                if power_type:
                    self.power_nets[track_id] = power_type
        if "PWR_PLUS" in self.graph:
            self.power_nets["PWR_PLUS"] = "VCC"
        if "PWR_MINUS" in self.graph:
            self.power_nets["PWR_MINUS"] = "GND"

    @staticmethod
    def _parse_rail_label(label: str) -> Optional[str]:
        if not label:
            return None
        u = label.strip().upper()
        if any(kw in u for kw in ("VCC", "VDD", "V+", "+5", "+3.3", "+12", "正电源", "正极", "电源正")):
            return "VCC"
        if u.startswith("+") and any(c.isdigit() for c in u):
            return "VCC"
        if any(kw in u for kw in ("GND", "VSS", "V-", "0V", "地", "负极", "电源负", "接地")):
            return "GND"
        if "V" in u and any(c.isdigit() for c in u):
            return "VCC"
        return None

    def set_rail_assignment(self, track_id: str, label: str):
        if track_id in self._rail_track_rows or track_id.startswith("RAIL_"):
            self.rail_assignments[track_id] = label

    def get_net_count(self) -> int:
        return len(list(nx.connected_components(self.graph)))

    def export_netlist(self) -> Dict:
        """导出结构化网表"""
        self._identify_power_nets()
        connected = list(nx.connected_components(self.graph))
        node_to_net_id = {}
        nets = {}
        for i, group in enumerate(connected):
            net_id = f"N{i}"
            nets[net_id] = sorted(list(group))
            for n in group:
                node_to_net_id[n] = net_id

        comp_list = []
        for comp in self.components:
            entry = {
                "name": comp.name,
                "type": self._norm_type(comp.type),
                "polarity": comp.polarity.value,
                "confidence": comp.confidence,
                "pins": [],
            }
            n1 = self._get_node_name(comp.pin1_loc)
            entry["pins"].append({
                "loc": comp.pin1_loc,
                "role": "generic",
                "net": node_to_net_id.get(n1, "floating"),
            })
            if comp.pin2_loc:
                n2 = self._get_node_name(comp.pin2_loc)
                entry["pins"].append({
                    "loc": comp.pin2_loc,
                    "role": "generic",
                    "net": node_to_net_id.get(n2, "floating"),
                })
            comp_list.append(entry)

        power = {}
        for node, ptype in self.power_nets.items():
            net_id = node_to_net_id.get(node)
            if net_id:
                power[net_id] = ptype

        return {"nets": nets, "components": comp_list, "power": power}
