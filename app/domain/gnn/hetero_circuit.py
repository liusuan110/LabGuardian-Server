"""GNN 模块 · 中间数据结构 ``HeteroCircuitGraph``（P0 · 纯 Python）

把"NetworkX component-net 二分图"提升为"component / port / net 三元异构图"
的中间表示。该结构 dict-of-frozen-dataclasses，便于：

- O(1) 通过 node_id 查节点
- 按 component 反查其所有 ports
- 在 P2 ``pyg_converter.py`` 中按节点类型分组直接 stack 成 tensor

设计要点（参见 plan §三 与附录 A · 文件 2）：

- 节点 frozen → hashable，未来用作映射 key 不需要复制
- ``HeteroCircuitGraph`` 本身可变 → ``port_graph.build_*`` 增量构造
- 不依赖 torch / torch_geometric
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

NodeKind = Literal["component", "port", "net"]
Side = Literal["ref", "cur"]


@dataclass(frozen=True)
class ComponentNode:
    """元件节点。"""

    node_id: str  # "<side>_comp:<source_id>"
    side: Side
    source_id: str
    ctype: str  # ``ComponentType.value``
    package: str | None
    polarity_class: str  # ``PolarityClass.value``
    pin_count: int
    value: float | None  # 电阻欧姆值 / 电容法拉值；缺失为 None
    confidence: float
    # P0.6: per-component pin 互换组。每组是该 component 内部互换的 port_key
    # tuple。e.g. Resistor → ``(("pin1", "pin2"),)``, Potentiometer →
    # ``(("terminal_a", "terminal_b"),)``, UA741 → ``(("1", "5"),)`` (offset
    # null pair；其它 pin 单独成组在 PortNode.symmetry_class_id 上表示，不
    # 重复列出)。 source: ``graph_schema.get_expected_pin_specs`` + 可选
    # netlist_v2 ``ComponentInstance.symmetry_group`` overlay。
    pin_symmetry_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class PortNode:
    """元件引脚节点 —— GNN-ACLP 范式中的"一等公民"。"""

    node_id: str  # "<side>_port:<comp_source_id>.<port_key>"
    side: Side
    parent_component_id: str  # 指向 ComponentNode.node_id
    port_key: str  # 原始 pin 名（slugify 后），保留可追溯性
    port_type: str  # ``PortType.value``
    parent_ctype: str  # 复制自 component，方便 SEAL 局部使用
    polarity_sensitive: bool
    is_power_port: bool
    is_ground_port: bool
    # P0.6: 真正的"floating"语义 —— True 表示该 port 在当前 side 不连接任何
    # net。ref 侧的 floating 表示 spec 允许（OPTIONAL）或禁止（FORBIDDEN，但
    # 学生没接才正确）；cur 侧的 floating 表示视觉确实观测到 pin 但未映射到
    # net（``electrical_net_id`` 为 None）。
    is_floating: bool
    # P0.6 新增字段 -------------------------------------------------------
    # 1-indexed 物理 pin 号；无位置概念则 None（LED.anode / Pot.wiper 等）。
    pin_number: int | None = None
    # ``ConnectionPolicy.value`` —— P0.6 引入。缺省 REQUIRED 兼容旧调用方。
    connection_policy: str = "required"
    # 0-indexed per-component 互换类 id。同 component 内同 id 的 port 可互
    # 换。spec 缺失时调用方给唯一 id。
    symmetry_class_id: int = 0


@dataclass(frozen=True)
class NetNode:
    """电气网络节点。"""

    node_id: str  # "<side>_net:<source_id>"
    side: Side
    source_id: str
    role: str  # ``NetRole.value``
    role_label: str | None
    is_power_rail: bool
    voltage_hint: float | None  # DSL 标注的预期电压；缺失为 None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortConnectsNetEdge:
    """``(port, connects, net)`` 关系边。"""

    src_port_id: str
    dst_net_id: str
    connection_confidence: float
    source_type: str  # ``SourceType.value``
    is_observed_in_cur: bool


@dataclass
class HeteroCircuitGraph:
    """整张异构图 —— 单边（ref 或 cur）。

    P0 阶段不做 ref/cur 合并；那是 P2 ``pyg_converter`` 的职责（拼成 PyG
    HeteroData 时会按 ``is_reference`` 特征区分两侧并放在同一对象里）。
    """

    side: Side
    components: dict[str, ComponentNode] = field(default_factory=dict)
    ports: dict[str, PortNode] = field(default_factory=dict)
    nets: dict[str, NetNode] = field(default_factory=dict)
    # component_node_id → list[port_node_id]
    port_of_component: dict[str, list[str]] = field(default_factory=dict)
    edges: list[PortConnectsNetEdge] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def summary(self) -> dict[str, int]:
        """节点 / 边计数（测试断言用）。"""

        return {
            "n_components": len(self.components),
            "n_ports": len(self.ports),
            "n_nets": len(self.nets),
            "n_edges": len(self.edges),
        }

    def ports_for(self, component_node_id: str) -> list[PortNode]:
        """按 component 反查其所有 ports（保持插入顺序）。"""

        return [self.ports[pid] for pid in self.port_of_component.get(component_node_id, [])]

    # -- 基本不变量 --------------------------------------------------------

    def assert_invariants(self) -> None:
        """轻量自检：node_id 唯一、边端点合法、port 父亲存在。

        正常构造路径下不会触发；保留作为开发期 sanity check。测试可显式调用。
        """

        # node_id 在三类节点中两两不相交
        comp_ids = set(self.components)
        port_ids = set(self.ports)
        net_ids = set(self.nets)
        assert comp_ids.isdisjoint(port_ids), "component / port id collision"
        assert comp_ids.isdisjoint(net_ids), "component / net id collision"
        assert port_ids.isdisjoint(net_ids), "port / net id collision"

        # 每个 port 的 parent 必须是已注册 component
        for port in self.ports.values():
            assert port.parent_component_id in self.components, (
                f"orphan port {port.node_id}: parent {port.parent_component_id} missing"
            )

        # port_of_component 反向索引必须自洽
        for comp_id, port_ids_for_comp in self.port_of_component.items():
            assert comp_id in self.components, f"unknown comp in index: {comp_id}"
            for pid in port_ids_for_comp:
                assert pid in self.ports, f"port_of_component refers missing port {pid}"
                assert self.ports[pid].parent_component_id == comp_id, (
                    f"port {pid} parent mismatch"
                )

        # 边端点必须是已注册 port / net
        for edge in self.edges:
            assert edge.src_port_id in self.ports, f"edge src port {edge.src_port_id} missing"
            assert edge.dst_net_id in self.nets, f"edge dst net {edge.dst_net_id} missing"


__all__ = [
    "NodeKind",
    "Side",
    "ComponentNode",
    "PortNode",
    "NetNode",
    "PortConnectsNetEdge",
    "HeteroCircuitGraph",
]
