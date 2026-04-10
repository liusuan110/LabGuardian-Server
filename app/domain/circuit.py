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

from app.domain.board_schema import BoardSchema
from app.domain.netlist_models import ComponentInstance, ElectricalNet, NetlistV2, PinAssignment

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
    if "CAPACIT" in u or u == "CAP":
        return "Capacitor"
    if "WIRE" in u:
        return "Wire"
    if u == "LED":
        return "LED"
    if "DIODE" in u:
        return "Diode"
    if u == "IC":
        return "IC"
    if "POTENTIOMETER" in u or "POT" in u:
        return "Potentiometer"
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
    extra_pins: List[Tuple[str, str]] = field(default_factory=list)
    pin_roles: List[str] = field(default_factory=list)
    polarity: Polarity = Polarity.NONE
    confidence: float = 1.0
    orientation_deg: float = 0.0

    def __repr__(self):
        pol = f" [{self.polarity.value}]" if self.polarity != Polarity.NONE else ""
        return f"{self.name}(pins={len(self.all_pin_locs())}{pol})"

    @property
    def is_polarized(self) -> bool:
        return self.polarity not in (Polarity.NONE,)

    @property
    def has_known_polarity(self) -> bool:
        return self.polarity in (Polarity.FORWARD, Polarity.REVERSE)

    def all_pin_locs(self) -> List[Tuple[str, str]]:
        pins: List[Tuple[str, str]] = []
        if self.pin1_loc is not None:
            pins.append(self.pin1_loc)
        if self.pin2_loc is not None:
            pins.append(self.pin2_loc)
        pins.extend(self.extra_pins or [])
        return pins


POLARIZED_TYPES = {"LED", "Diode"}
THREE_PIN_TYPES: set = set()
CAPACITOR_TYPES = {"Capacitor"}
NON_POLAR_TYPES = {"Resistor", "Wire", "Capacitor", "Potentiometer"}
IC_TYPES = {"IC"}
POTENTIOMETER_TYPES = {"Potentiometer"}
POWER_KEYWORDS = {"VCC", "GND", "POWER", "BATTERY"}


# ============================================================
# 并查集 (Union-Find / Disjoint Set)
# 用于增量合并等电位网络, 比 nx.connected_components 更高效
# 参考: EDA 网表生成的标准数据结构 (KiCad, Altium)
# ============================================================

class UnionFind:
    """并查集 — O(α(n)) 近常数时间合并与查询

    用于 Wire 导线将两端网络合并为等电位网络 (Electrical Net)。
    支持路径压缩 + 按秩合并, 是工业 EDA 底层生成网表 (Netlist) 的标准算法。
    """

    def __init__(self):
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        """查找根节点 (带路径压缩)"""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str):
        """合并两个集合 (按秩合并)"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def connected(self, a: str, b: str) -> bool:
        """判断两个元素是否在同一集合"""
        return self.find(a) == self.find(b)

    def groups(self) -> Dict[str, set]:
        """返回所有连通分组 {root: {members}}"""
        result: Dict[str, set] = {}
        for x in self._parent:
            root = self.find(x)
            result.setdefault(root, set()).add(x)
        return result

    def clear(self):
        """清空"""
        self._parent.clear()
        self._rank.clear()


class CircuitAnalyzer:
    """电路拓扑分析器

    基于 NetworkX 图论，将面包板上的元件建模为电气连接图。
    面包板规则: 同行 a-e 导通 (Left), f-j 导通 (Right)。
    """

    def __init__(
        self,
        rail_track_rows: Optional[Dict[str, tuple]] = None,
        board_schema: BoardSchema | None = None,
    ):
        # 二分图: 左集 U = 元件节点, 右集 V = 网络节点 (等电位点)
        # 参考: circuit_recognizer + NetworkX 二分图建模
        self.graph = nx.Graph()
        self._uf = UnionFind()  # 并查集: Wire 增量合并等电位网络
        self.components: List[CircuitComponent] = []
        self.component_instances: List[ComponentInstance] = []
        self.power_nets: Dict[str, str] = {}
        self._name_counters: Dict[str, int] = {}
        self._rail_track_rows = rail_track_rows or {}
        self.board_schema = board_schema or BoardSchema.default_breadboard()
        self._row_to_rail: Dict[int, str] = {}
        for track_id, rows in self._rail_track_rows.items():
            for r in rows:
                self._row_to_rail[r] = track_id
        self.rail_assignments: Dict[str, str] = {}

    def reset(self):
        self.graph.clear()
        self._uf.clear()
        self.components = []
        self.component_instances = []
        self.power_nets = {}
        self._name_counters = {}

    _TYPE_PREFIX = {
        "Resistor": "R", "Capacitor": "C", "Wire": "W", "LED": "LED",
        "Diode": "D", "IC": "IC", "Potentiometer": "POT",
    }

    def _auto_name(self, comp_type: str) -> str:
        norm = self._norm_type(comp_type)
        prefix = self._TYPE_PREFIX.get(norm, norm[:3])
        self._name_counters[norm] = self._name_counters.get(norm, 0) + 1
        return f"{prefix}{self._name_counters[norm]}"

    def add_component(self, comp: CircuitComponent):
        """添加元件到二分图 (U=元件, V=网络) + 并查集登记

        二分图结构:
          元件节点 (bipartite=0) ←边→ 网络节点 (bipartite=1)
        Wire 元件额外触发 Union-Find 合并两端网络。
        参考: mahmut-aksakalli/circuit_recognizer 的 Node-Endpoint 建模
        """
        if comp.name == comp.type or comp.name in ("UNKNOWN", ""):
            comp.name = self._auto_name(comp.type)
        self.components.append(comp)

        ctype = self._norm_type(comp.type)
        pin_locs = comp.all_pin_locs()
        if not pin_locs:
            return

        net_names: List[str] = []
        for loc in pin_locs:
            net_name = self._get_node_name(loc)
            net_names.append(net_name)
            if not self.graph.has_node(net_name):
                self.graph.add_node(net_name, bipartite=1, kind="net")
            self._uf.find(net_name)

        # 元件节点 (U 集)
        self.graph.add_node(comp.name, bipartite=0, kind="comp",
                            ctype=ctype, polarity=comp.polarity.value,
                            confidence=comp.confidence)

        # 边: 元件 ↔ 网络 (携带引脚信息)
        for i, net_name in enumerate(net_names):
            role = "generic"
            if i < len(comp.pin_roles):
                role = str(comp.pin_roles[i])
            self.graph.add_edge(
                comp.name,
                net_name,
                pin=f"pin{i + 1}",
                role=role,
                pin_role=role,
                component=comp.name,
                type=comp.type,
            )

        # Wire → 并查集合并两端网络 (面包板先验: 导线 = 等电位连接)
        if ctype == "Wire" and len(net_names) >= 2:
            self._uf.union(net_names[0], net_names[1])

    def add_component_instance(self, comp: ComponentInstance):
        """接收新一代组件中心化输入，并兼容注入旧图模型。"""
        component_id = comp.component_id or self._auto_name(comp.component_type)
        normalized_pins: List[PinAssignment] = []
        for pin in comp.pins:
            hole_id = self.board_schema.normalize_hole_id(pin.hole_id)
            normalized_pins.append(
                PinAssignment(
                    pin_id=pin.pin_id,
                    pin_name=pin.pin_name,
                    hole_id=hole_id,
                    electrical_node_id=pin.electrical_node_id or self.board_schema.resolve_hole_to_node(hole_id),
                    electrical_net_id=pin.electrical_net_id,
                    observations=list(pin.observations or []),
                    confidence=pin.confidence,
                    is_ambiguous=pin.is_ambiguous,
                    metadata=dict(pin.metadata or {}),
                )
            )

        instance = ComponentInstance(
            component_id=component_id,
            component_type=comp.component_type,
            package_type=comp.package_type,
            part_subtype=comp.part_subtype,
            polarity=comp.polarity,
            orientation=comp.orientation,
            symmetry_group=[list(group) for group in comp.symmetry_group],
            pins=normalized_pins,
            confidence=comp.confidence,
            metadata=dict(comp.metadata or {}),
        )
        self.component_instances.append(instance)

        pin_locs = [
            self.board_schema.hole_id_to_logic_loc(pin.hole_id)
            for pin in normalized_pins
        ]
        pin_locs = [loc for loc in pin_locs if loc is not None]
        if not pin_locs:
            return

        legacy = CircuitComponent(
            name=component_id,
            type=comp.component_type,
            pin1_loc=pin_locs[0],
            pin2_loc=pin_locs[1] if len(pin_locs) > 1 else None,
            extra_pins=pin_locs[2:],
            pin_roles=[pin.pin_name for pin in normalized_pins],
            polarity=self._polarity_from_value(comp.polarity),
            confidence=comp.confidence,
            orientation_deg=comp.orientation,
        )
        self.add_component(legacy)

    def _get_node_name(self, loc: Tuple[str, str]) -> str:
        """面包板导通规则映射"""
        hole_id = self.board_schema.logic_loc_to_hole_id(loc)
        node_name = self.board_schema.resolve_hole_to_node(hole_id)
        if not node_name.startswith(("PWR_", "RAIL_")):
            return node_name

        row, col = loc
        if col in ("+", "plus", "P"):
            return "PWR_PLUS"
        if col in ("-", "minus", "N", "GND"):
            return "PWR_MINUS"

        # 识别校准器返回的电轨列名: rail_top+, rail_top-, rail_bot+, rail_bot-
        if col.startswith("rail_"):
            # 如果有 rail_assignments, 用用户指定的名称
            rail_key = col.replace("rail_", "").replace("+", "_plus").replace("-", "_minus")
            if rail_key in self.rail_assignments:
                label = self.rail_assignments[rail_key]
                return f"PWR_{label}"
            # 默认: + → VCC, - → GND
            if "+" in col:
                return "PWR_PLUS"
            elif "-" in col:
                return "PWR_MINUS"
            return f"RAIL_{col}"

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

    @staticmethod
    def _polarity_from_value(value: str | Polarity | None) -> Polarity:
        if isinstance(value, Polarity):
            return value
        if value is None:
            return Polarity.NONE
        try:
            return Polarity(str(value))
        except ValueError:
            return Polarity.NONE

    def build_topology_graph(self) -> nx.Graph:
        """构建布局无关的拓扑图 — Union-Find 合并 Wire 等电位网络

        使用并查集 (而非 nx.connected_components) 做 Wire 网络合并,
        支持增量更新, 复杂度 O(α(n))。
        输出格式与 SINA netlist generator 对齐: comp 节点 + net 节点的二分图。
        """
        self._identify_power_nets()

        # 从 Union-Find 获取合并后的等电位网络组
        all_nets = {n for n, d in self.graph.nodes(data=True) if d.get("kind") == "net"}
        net_groups: Dict[str, set] = {}
        for net_name in all_nets:
            root = self._uf.find(net_name)
            net_groups.setdefault(root, set()).add(net_name)

        # 为每组分配 net ID, net_name → net_id 映射
        node_to_net: Dict[str, str] = {}
        topo = nx.Graph()
        for i, (root, members) in enumerate(net_groups.items()):
            net_id = f"N{i}"
            for m in members:
                node_to_net[m] = net_id
            attrs: Dict = {"kind": "net"}
            for m in members:
                if m in self.power_nets:
                    attrs["power"] = self.power_nets[m]
                    break
            topo.add_node(net_id, **attrs)

        # 添加非 Wire 元件节点
        comp_idx = 0
        for comp in self.components:
            ctype = self._norm_type(comp.type)
            if ctype == "Wire":
                continue

            try:
                net_ids: List[str] = []
                for loc in comp.all_pin_locs():
                    n_name = self._get_node_name(loc)
                    net_id = node_to_net.get(self._uf.find(n_name))
                    if net_id is not None:
                        net_ids.append(net_id)
            except Exception:
                net_ids = []

            if not net_ids:
                continue

            cid = f"C{comp_idx}"
            comp_idx += 1
            node_attrs = {
                "kind": "comp",
                "ctype": ctype,
                "polarity": comp.polarity.value,
            }

            unique_nets = list(dict.fromkeys(net_ids))
            net_roles: Dict[str, str] = {}
            for i, net_id in enumerate(net_ids):
                if net_id not in net_roles:
                    role = comp.pin_roles[i] if i < len(comp.pin_roles) else "generic"
                    net_roles[net_id] = role
            node_attrs["pins"] = len(net_ids)
            if len(unique_nets) == 1 and len(net_ids) > 1:
                node_attrs["same_net"] = True
            topo.add_node(cid, **node_attrs)
            for net_id in unique_nets:
                role = net_roles.get(net_id, "generic")
                topo.add_edge(cid, net_id, role=role, pin_role=role)

        return topo

    def get_circuit_description(self) -> str:
        """生成结构化电路网表描述 (基于 Union-Find 等电位网络)"""
        if not self.components:
            return "当前未检测到电路元件。"

        self._identify_power_nets()

        # 从 Union-Find 获取合并后的等电位网络组
        all_nets = {n for n, d in self.graph.nodes(data=True) if d.get("kind") == "net"}
        uf_groups: Dict[str, set] = {}
        for net_name in all_nets:
            root = self._uf.find(net_name)
            uf_groups.setdefault(root, set()).add(net_name)
        connected_groups = list(uf_groups.values())

        node_to_net: Dict[str, str] = {}
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
            pins = comp.all_pin_locs()
            pin_nets: List[Tuple[Tuple[str, str], str]] = []
            for loc in pins:
                net_name = self._get_node_name(loc)
                pin_nets.append((loc, node_to_net.get(self._uf.find(net_name), "?")))

            role_info = ""
            if comp.polarity == Polarity.FORWARD:
                role_info = " [pin1/pin2]"
            elif comp.polarity == Polarity.REVERSE:
                role_info = " [pin1/pin2 反向]"
            elif comp.polarity == Polarity.UNKNOWN:
                role_info = " [极性未知]"

            if len(pin_nets) <= 2:
                if len(pin_nets) == 2:
                    p1, n1 = pin_nets[0]
                    p2, n2 = pin_nets[1]
                    desc += (
                        f"  {comp.name} ({ctype}{role_info}): "
                        f"Row{p1[0]}{p1[1]}({n1}) — "
                        f"Row{p2[0]}{p2[1]}({n2})\n"
                    )
                else:
                    p1, n1 = pin_nets[0]
                    desc += f"  {comp.name} ({ctype}{role_info}): Row{p1[0]}{p1[1]}({n1})\n"
            else:
                pins_text = ", ".join([f"Pin{i+1}=Row{loc[0]}{loc[1]}({net})" for i, (loc, net) in enumerate(pin_nets)])
                desc += f"  {comp.name} ({ctype}{role_info}): {pins_text}\n"

        desc += "\n电气网络:\n"
        for idx, group in enumerate(connected_groups):
            net_id = f"Net_{idx + 1}"
            nodes = sorted(list(group))

            # 通过 Union-Find 判断元件是否在此网络上
            group_roots = {self._uf.find(n) for n in group}
            comps_on_net = []
            for comp in self.components:
                comp_roots = set()
                for loc in comp.all_pin_locs():
                    comp_roots.add(self._uf.find(self._get_node_name(loc)))
                if comp_roots & group_roots:
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
        """快速问题检测 — 使用 Union-Find 感知 Wire 合并后的等电位关系"""
        issues = []
        has_led = False
        has_resistor_near_led = False

        for comp in self.components:
            ctype = self._norm_type(comp.type)
            if ctype == "LED":
                has_led = True
                led_nets = set()
                for loc in comp.all_pin_locs():
                    led_nets.add(self._uf.find(self._get_node_name(loc)))
                for other in self.components:
                    if self._norm_type(other.type) == "Resistor":
                        r_nets = set()
                        for loc in other.all_pin_locs():
                            r_nets.add(self._uf.find(self._get_node_name(loc)))
                        # UF 感知: 即使 LED 和 Resistor 不在同一行,
                        # 只要通过 Wire 连接到同一网络也算 "相邻"
                        if led_nets & r_nets:
                            has_resistor_near_led = True
                            break

            if ctype in POLARIZED_TYPES and comp.polarity == Polarity.UNKNOWN:
                issues.append(f"{comp.name} ({ctype}) 极性未确定, 请检查安装方向")
            pins = comp.all_pin_locs()
            if len(pins) >= 2:
                n1 = self._get_node_name(pins[0])
                n2 = self._get_node_name(pins[1])
                # UF 感知: 不仅检查同位置, 还检查 Wire 合并后的等电位
                if self._uf.find(n1) == self._uf.find(n2) and ctype != "Wire":
                    issues.append(f"{comp.name} ({ctype}) 两引脚在同一导通组, 可能短路或未跨行")

        if has_led and not has_resistor_near_led:
            issues.append("LED 未检测到相邻限流电阻, 可能缺少限流保护")

        return issues

    def _identify_power_nets(self):
        for track_id, label in self.rail_assignments.items():
            power_type = self._parse_rail_label(label)
            if not power_type:
                continue
            schema_nodes = self.board_schema.resolve_track_assignment_nodes(track_id)
            for node_id in schema_nodes:
                if node_id in self.graph:
                    self.power_nets[node_id] = power_type
            if track_id in self.graph:
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
        """设置电轨标签, 如 set_rail_assignment("top_plus", "VCC")"""
        self.rail_assignments[track_id] = label

    def get_net_count(self) -> int:
        """Union-Find 计算独立等电位网络数"""
        all_nets = {n for n, d in self.graph.nodes(data=True) if d.get("kind") == "net"}
        if not all_nets:
            return 0
        roots = {self._uf.find(n) for n in all_nets}
        return len(roots)

    def export_netlist(self) -> Dict:
        """导出结构化网表 (Union-Find 合并等电位网络)"""
        self._identify_power_nets()
        all_nets = {n for n, d in self.graph.nodes(data=True) if d.get("kind") == "net"}
        uf_groups: Dict[str, set] = {}
        for net_name in all_nets:
            root = self._uf.find(net_name)
            uf_groups.setdefault(root, set()).add(net_name)

        node_to_net_id: Dict[str, str] = {}
        nets: Dict[str, list] = {}
        for i, (root, members) in enumerate(uf_groups.items()):
            net_id = f"N{i}"
            nets[net_id] = sorted(list(members))
            for m in members:
                node_to_net_id[m] = net_id

        comp_list = []
        for comp in self.components:
            entry = {
                "name": comp.name,
                "type": self._norm_type(comp.type),
                "polarity": comp.polarity.value,
                "confidence": comp.confidence,
                "pins": [],
            }
            for i, loc in enumerate(comp.all_pin_locs()):
                role = comp.pin_roles[i] if i < len(comp.pin_roles) else "generic"
                n2 = self._get_node_name(loc)
                entry["pins"].append({
                    "loc": loc,
                    "role": role,
                    "net": node_to_net_id.get(self._uf.find(n2), "floating"),
                })
            comp_list.append(entry)

        power = {}
        for node, ptype in self.power_nets.items():
            net_id = node_to_net_id.get(self._uf.find(node), None)
            if net_id:
                power[net_id] = ptype

        return {"nets": nets, "components": comp_list, "power": power}

    def export_netlist_v2(self, scene_id: str = "runtime_scene") -> Dict:
        """导出保留 hole_id / pin_name / electrical_net 的新网表格式。"""
        self._identify_power_nets()
        all_nets = {n for n, d in self.graph.nodes(data=True) if d.get("kind") == "net"}
        uf_groups: Dict[str, set] = {}
        for net_name in all_nets:
            root = self._uf.find(net_name)
            uf_groups.setdefault(root, set()).add(net_name)

        node_to_net_id: Dict[str, str] = {}
        nets: List[ElectricalNet] = []
        member_holes_by_net: Dict[str, set] = {}
        member_nodes_by_net: Dict[str, set] = {}

        instances = self.component_instances or self._legacy_instances()
        for idx, (root, members) in enumerate(uf_groups.items()):
            net_id = f"NET_{idx:03d}"
            for member in members:
                node_to_net_id[member] = net_id
            member_holes_by_net[net_id] = set()
            member_nodes_by_net[net_id] = set(members)

        exported_components: List[ComponentInstance] = []
        node_index: Dict[str, List[str]] = {}
        for comp in instances:
            exported_pins: List[PinAssignment] = []
            for pin in comp.pins:
                node_id = pin.electrical_node_id or self.board_schema.resolve_hole_to_node(pin.hole_id)
                net_id = node_to_net_id.get(self._uf.find(node_id))
                if net_id:
                    member_holes_by_net.setdefault(net_id, set()).add(pin.hole_id)
                    member_nodes_by_net.setdefault(net_id, set()).add(node_id)
                node_index.setdefault(node_id, [])
                if pin.hole_id not in node_index[node_id]:
                    node_index[node_id].append(pin.hole_id)
                exported_pins.append(
                    PinAssignment(
                        pin_id=pin.pin_id,
                        pin_name=pin.pin_name,
                        hole_id=pin.hole_id,
                        electrical_node_id=node_id,
                        electrical_net_id=net_id,
                        observations=list(pin.observations or []),
                        confidence=pin.confidence,
                        is_ambiguous=pin.is_ambiguous,
                        metadata=dict(pin.metadata or {}),
                    )
                )
            exported_components.append(
                ComponentInstance(
                    component_id=comp.component_id,
                    component_type=comp.component_type,
                    package_type=comp.package_type,
                    part_subtype=comp.part_subtype,
                    polarity=comp.polarity,
                    orientation=comp.orientation,
                    symmetry_group=[list(group) for group in comp.symmetry_group],
                    pins=exported_pins,
                    confidence=comp.confidence,
                    metadata=dict(comp.metadata or {}),
                )
            )

        for idx, (root, members) in enumerate(uf_groups.items()):
            net_id = f"NET_{idx:03d}"
            power_role = ""
            for member in members:
                if member in self.power_nets:
                    power_role = self.power_nets[member]
                    break
            nets.append(
                ElectricalNet(
                    electrical_net_id=net_id,
                    member_node_ids=sorted(member_nodes_by_net.get(net_id, set())),
                    member_hole_ids=sorted(member_holes_by_net.get(net_id, set())),
                    power_role=power_role,
                )
            )

        netlist = NetlistV2(
            scene_id=scene_id,
            board_schema_id=self.board_schema.schema_id,
            components=exported_components,
            nets=nets,
            node_index=node_index,
        )
        return netlist.to_dict()

    def export_spice_netlist(self) -> str:
        """导出 SPICE 格式网表 (参考 SINA: Circuit Schematic Image-to-Netlist Generator)

        生成兼容 LTspice / ngspice 的网表文本。
        Wire 通过 Union-Find 隐式合并, 不出现在网表中。
        """
        self._identify_power_nets()
        all_nets = {n for n, d in self.graph.nodes(data=True) if d.get("kind") == "net"}
        uf_groups: Dict[str, set] = {}
        for net_name in all_nets:
            root = self._uf.find(net_name)
            uf_groups.setdefault(root, set()).add(net_name)

        root_to_id: Dict[str, str] = {}
        net_idx = 1
        for root, members in uf_groups.items():
            power_name = None
            for m in members:
                if m in self.power_nets:
                    power_name = self.power_nets[m]
                    break
            if power_name:
                root_to_id[root] = power_name
            else:
                root_to_id[root] = f"N{net_idx:03d}"
                net_idx += 1

        lines = ["* LabGuardian Auto-Generated SPICE Netlist"]
        lines.append(f"* Components: {len(self.components)}")
        lines.append(f"* Nets: {len(uf_groups)}")
        lines.append("")

        for comp in self.components:
            ctype = self._norm_type(comp.type)
            if ctype == "Wire":
                continue
            pin_nets = []
            for loc in comp.all_pin_locs():
                n1 = self._get_node_name(loc)
                pin_nets.append(root_to_id.get(self._uf.find(n1), "?"))
            if len(pin_nets) >= 2:
                net1_id, net2_id = pin_nets[0], pin_nets[1]
                if ctype == "Resistor":
                    lines.append(f"{comp.name} {net1_id} {net2_id} 1k")
                elif ctype == "Capacitor":
                    lines.append(f"{comp.name} {net1_id} {net2_id} 100n")
                elif ctype == "LED":
                    lines.append(f"D_{comp.name} {net1_id} {net2_id} LED")
                elif len(pin_nets) > 2:
                    lines.append(f"{comp.name} {' '.join(pin_nets)}")
                else:
                    lines.append(f"{comp.name} {net1_id} {net2_id}")
            else:
                net1_id = pin_nets[0] if pin_nets else "?"
                lines.append(f"* {comp.name} (single-pin): {net1_id}")

        lines.append("")
        lines.append(".end")
        return "\n".join(lines)

    def describe(self) -> str:
        """get_circuit_description 的别名，保持向后兼容"""
        return self.get_circuit_description()

    def to_node_link_data(self) -> dict:
        """将拓扑图序列化为 node-link JSON 格式"""
        topo = self.build_topology_graph()
        return nx.node_link_data(topo)

    def component_count(self) -> int:
        """返回元件总数"""
        return len(self.components)

    def _legacy_instances(self) -> List[ComponentInstance]:
        """把旧组件模型临时映射成 v2 结构，保证迁移期可并行导出。"""
        instances: List[ComponentInstance] = []
        for comp in self.components:
            pins: List[PinAssignment] = []
            for idx, loc in enumerate(comp.all_pin_locs(), start=1):
                hole_id = self.board_schema.logic_loc_to_hole_id(loc)
                pin_name = comp.pin_roles[idx - 1] if idx - 1 < len(comp.pin_roles) else f"pin{idx}"
                pins.append(
                    PinAssignment(
                        pin_id=idx,
                        pin_name=pin_name,
                        hole_id=hole_id,
                        electrical_node_id=self.board_schema.resolve_hole_to_node(hole_id),
                        confidence=comp.confidence,
                    )
                )
            instances.append(
                ComponentInstance(
                    component_id=comp.name,
                    component_type=comp.type,
                    package_type="legacy",
                    polarity=comp.polarity.value,
                    orientation=comp.orientation_deg,
                    pins=pins,
                    confidence=comp.confidence,
                    metadata={"source": "legacy_component"},
                )
            )
        return instances
