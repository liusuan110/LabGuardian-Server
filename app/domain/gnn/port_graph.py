"""GNN 模块 · NetworkX 二分图 → HeteroCircuitGraph 三元异构图（P0）

本模块的唯一职责：把现有 ``app.domain.logical_reference`` 产出的
``component-net`` bipartite NetworkX 图升级为 GNN-ACLP 风格的
``component / port / net`` 三元异构中间结构 ``HeteroCircuitGraph``。

**重要约束**：
- 不依赖 torch / torch_geometric（向量化推到 P2 ``pyg_converter.py``）。
- 不重新实现 logical_reference / netlist_v2 解析，只消费其输出。
- 边语义保持 1:1（每条原图 (comp, net) edge 对应**新建一个 port 节点** +
  一条 (port, net) 边），所以 ``n_edges == n_ports``。
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple, cast  # noqa: UP035 — keep Tuple for py<3.10 at runtime

import networkx as nx  # type: ignore[import-untyped]  # types-networkx not installed

from app.domain.gnn.graph_schema import (
    GROUND_PORT_TYPES,
    POLARITY_CLASS_OF,
    POLARITY_SENSITIVE_PORT_TYPES,
    POWER_PORT_TYPES,
    ConnectionPolicy,
    PinSpec,
    PolarityClass,
    PortType,
    SourceType,
    get_expected_pin_specs,
    normalize_port_type,
)
from app.domain.gnn.hetero_circuit import (
    ComponentNode,
    HeteroCircuitGraph,
    NetNode,
    PortConnectsNetEdge,
    PortNode,
    Side,
)
from app.domain.logical_reference import (
    current_netlist_v2_to_graph,
    logical_reference_to_graph,
)

# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

# 把 pin 名 slugify 成可放进 node_id 的安全字符串（保留字母数字、下划线、
# 短横线、点；其它替换为下划线）。
_SLUG_RE = re.compile(r"[^A-Za-z0-9_.\-]+")


def _slug(text: str) -> str:
    cleaned = _SLUG_RE.sub("_", text.strip())
    # 折叠连续下划线
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def _comp_node_id(side: Side, source_id: str) -> str:
    return f"{side}_comp:{source_id}"


def _port_node_id(side: Side, comp_source_id: str, port_key: str) -> str:
    return f"{side}_port:{comp_source_id}.{port_key}"


def _net_node_id(side: Side, source_id: str) -> str:
    return f"{side}_net:{source_id}"


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _spec_lookup_by_key(
    specs: list[PinSpec] | None, key: str
) -> PinSpec | None:
    """Find spec entry matching ``key`` (canonical port_key) or numeric
    equivalent. Returns None if no spec or no match."""

    if not specs or not key:
        return None
    for s in specs:
        if s.pin_key == key:
            return s
    # numeric-key fallback: "pin3" → "3"
    if key.startswith("pin"):
        tail = key[3:]
        for s in specs:
            if s.pin_key == tail:
                return s
    return None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


# Tuple shape for raw pin edges (compatible with build_from_logical_reference /
# build_from_netlist_v2 bypass when the underlying nx.Graph collapses parallel
# pin edges — e.g., UA741 pin 2 + pin 6 both wired to VOUT in a unity-gain
# buffer):  (comp_source_id, comp_ctype, pin_raw, pin_role, net_source_id,
# connection_confidence_or_none, source_type_or_none)
RawPinEdge = Tuple[str, str, str, str, str, Optional[float], Optional[str]]  # noqa: UP006, UP045 — runtime value on py<3.10


def build_hetero_circuit_graph(
    nx_graph: nx.Graph,
    side: Side,
    *,
    subtype_by_source_id: dict[str, str] | None = None,
    raw_pin_edges: list[RawPinEdge] | None = None,
) -> HeteroCircuitGraph:
    """把 ``logical_reference_to_graph`` / ``current_netlist_v2_to_graph``
    的产物转为 ``HeteroCircuitGraph``。

    输入 graph 的契约：
    - 节点属性 ``kind`` ∈ {"comp", "net"}
    - "comp" 节点附带 ``ctype: str`` 与 ``source_id: str``
    - "net" 节点附带 ``role: str`` 与 ``source_id: str``，可选
      ``role_label`` / ``aliases``
    - 边附带 ``pin: str`` / ``pin_role: str`` / ``comp_type: str``

    side 决定 node_id 前缀（"ref_" / "cur_"）以及默认的 source_type 字段。

    Args:
        subtype_by_source_id: 可选 ``{component source_id → part_subtype}``
            映射。P0.5 起用于 IC pin map 查表（让 UA741 的 pin "3" 拿到
            ``non_inverting_input`` 而不是 ``pin_n_generic``）。NetworkX
            原图不携带 ``part_subtype``，所以这条信息由
            ``build_from_logical_reference`` / ``build_from_netlist_v2``
            从上游 payload 旁路注入。
        raw_pin_edges: 可选的 ``RawPinEdge`` 列表，**完全替代** 对
            ``nx_graph.edges()`` 的扫描。用于绕过 ``nx.Graph`` 多重边折叠
            的限制 —— 例如运放单位增益缓冲器里 pin2 与 pin6 同接 VOUT，
            nx 只保留一条边、丢失 pin2 端口。``build_from_logical_reference``
            / ``build_from_netlist_v2`` 会从原始 payload 重建此列表传入。
            字段顺序：(comp_source_id, comp_ctype, pin_raw, pin_role,
            net_source_id, connection_confidence|None, source_type|None)。
    """

    if side not in ("ref", "cur"):
        raise ValueError(f"side must be 'ref' or 'cur', got {side!r}")

    subtype_by_source_id = subtype_by_source_id or {}
    hcg = HeteroCircuitGraph(side=side)

    # ---- 1. 注册所有 component / net 节点（先扫节点，再扫边） --------------

    for node_id, data in nx_graph.nodes(data=True):
        kind = data.get("kind")
        source_id = str(data.get("source_id") or node_id)

        if kind == "comp":
            ctype = str(data.get("ctype") or "UNKNOWN")
            polarity_class = POLARITY_CLASS_OF.get(ctype, PolarityClass.NONE).value
            comp_node = ComponentNode(
                node_id=_comp_node_id(side, source_id),
                side=side,
                source_id=source_id,
                ctype=ctype,
                package=data.get("package"),  # 当前 nx 图未必带，None 即可
                polarity_class=polarity_class,
                pin_count=0,  # 第 3 步回填
                value=_coerce_float(data.get("value")),
                confidence=float(data.get("confidence", 1.0)),
            )
            hcg.components[comp_node.node_id] = comp_node
            hcg.port_of_component[comp_node.node_id] = []

        elif kind == "net":
            role = str(data.get("role") or "signal")
            role_label = data.get("role_label") or None
            aliases_raw = data.get("aliases") or ()
            try:
                aliases = tuple(str(a) for a in aliases_raw if a)
            except TypeError:
                aliases = ()
            net_node = NetNode(
                node_id=_net_node_id(side, source_id),
                side=side,
                source_id=source_id,
                role=role,
                role_label=role_label if isinstance(role_label, str) else None,
                is_power_rail=role in {"power", "ground"},
                voltage_hint=_coerce_float(data.get("voltage_hint")),
                aliases=aliases,
            )
            hcg.nets[net_node.node_id] = net_node

        # 其它 kind 一律忽略（保持向前兼容）

    # ---- 2. 扫边：每条 (comp, net) edge 衍生一个 port 节点 + 一条
    #          (port, net) PortConnectsNetEdge -----------------------------

    default_source_type = (
        SourceType.DSL.value if side == "ref" else SourceType.VISION.value
    )

    # 选择边来源：raw_pin_edges（payload 旁路，能保留 parallel pin） vs
    # nx_graph.edges()（nx.Graph 会折叠平行边，对运放反馈连线丢失 pin）。
    if raw_pin_edges is not None:
        edge_records: list[
            tuple[str, str, str, str, float | None, str | None]
        ] = []
        for (
            comp_source_id,
            _comp_ctype,
            pin_raw,
            pin_role,
            net_source_id,
            confidence,
            source_type,
        ) in raw_pin_edges:
            edge_records.append(
                (comp_source_id, net_source_id, pin_raw, pin_role, confidence, source_type)
            )
    else:
        edge_records = []
        for u, v, attrs in nx_graph.edges(data=True):
            u_kind = nx_graph.nodes[u].get("kind")
            v_kind = nx_graph.nodes[v].get("kind")
            if u_kind == "comp" and v_kind == "net":
                comp_nx_id, net_nx_id = u, v
            elif u_kind == "net" and v_kind == "comp":
                comp_nx_id, net_nx_id = v, u
            else:
                continue
            comp_source_id = str(
                nx_graph.nodes[comp_nx_id].get("source_id") or comp_nx_id
            )
            net_source_id = str(nx_graph.nodes[net_nx_id].get("source_id") or net_nx_id)
            pin_raw = str(attrs.get("pin") or "").strip()
            pin_role = str(attrs.get("pin_role") or "").strip().lower()
            confidence = _coerce_float(
                attrs.get("connection_confidence")
            ) or _coerce_float(attrs.get("confidence"))
            source_type = attrs.get("source_type")
            edge_records.append(
                (comp_source_id, net_source_id, pin_raw, pin_role, confidence, source_type)
            )

    for (
        comp_source_id,
        net_source_id,
        pin_raw_str,
        pin_role_str,
        confidence,
        source_type,
    ) in edge_records:
        comp_node_id = _comp_node_id(side, comp_source_id)
        net_node_id = _net_node_id(side, net_source_id)

        # 异常防御：边端点对应的节点未注册（异常 fixture），跳过
        if comp_node_id not in hcg.components or net_node_id not in hcg.nets:
            continue

        comp = hcg.components[comp_node_id]
        pin_raw = (pin_raw_str or "").strip()
        pin_role = (pin_role_str or "").strip().lower()
        part_subtype = subtype_by_source_id.get(comp_source_id, "")
        port_type = normalize_port_type(
            pin_role,
            comp.ctype,
            part_subtype=part_subtype,
            pin_raw=pin_raw,
        )

        # port_key 决定 node_id 唯一性。优先用 pin 原始名（"pin1"/"anode"/
        # "3" 等），缺失则退回 pin_role 或顺序号。
        if pin_raw:
            port_key = _slug(pin_raw)
        elif pin_role:
            port_key = _slug(pin_role)
        else:
            port_key = f"p{len(hcg.port_of_component[comp_node_id]) + 1}"

        port_node_id = _port_node_id(side, comp_source_id, port_key)

        # 防御：同一 component 上同名 port 重复出现 → 加序号后缀
        if port_node_id in hcg.ports:
            i = 2
            while _port_node_id(side, comp_source_id, f"{port_key}_{i}") in hcg.ports:
                i += 1
            port_key = f"{port_key}_{i}"
            port_node_id = _port_node_id(side, comp_source_id, port_key)

        polarity_sensitive = (
            comp.polarity_class != PolarityClass.NONE.value
            and port_type in POLARITY_SENSITIVE_PORT_TYPES
        )

        # P0.6: 查 spec 回填 pin_number / policy / symmetry_class_id；spec 缺
        # 时按 "REQUIRED + 唯一 class" 处理（symmetry_class 顺延 component 当
        # 前 port 数量 —— 保证每个无 spec 的 port 各自一类）。
        specs = get_expected_pin_specs(comp.ctype, part_subtype)
        spec = _spec_lookup_by_key(specs, port_key)
        if spec is not None:
            pin_number_val: int | None = spec.pin_number
            connection_policy_val = spec.connection_policy
            symmetry_class_val = spec.symmetry_class
        else:
            pin_number_val = _maybe_int(port_key) or _maybe_int(pin_raw)
            connection_policy_val = ConnectionPolicy.REQUIRED.value
            symmetry_class_val = len(hcg.port_of_component[comp_node_id])

        port_node = PortNode(
            node_id=port_node_id,
            side=side,
            parent_component_id=comp_node_id,
            port_key=port_key,
            port_type=port_type,
            parent_ctype=comp.ctype,
            polarity_sensitive=polarity_sensitive,
            is_power_port=port_type in POWER_PORT_TYPES,
            is_ground_port=port_type in GROUND_PORT_TYPES,
            # 该 port 来自一条真实的 (port, net) edge → 必然连接，不 floating
            is_floating=False,
            pin_number=pin_number_val,
            connection_policy=connection_policy_val,
            symmetry_class_id=symmetry_class_val,
        )
        hcg.ports[port_node_id] = port_node
        hcg.port_of_component[comp_node_id].append(port_node_id)

        if confidence is None:
            confidence = 1.0

        edge = PortConnectsNetEdge(
            src_port_id=port_node_id,
            dst_net_id=net_node_id,
            connection_confidence=confidence,
            source_type=str(source_type or default_source_type),
            is_observed_in_cur=(side == "cur"),
        )
        hcg.edges.append(edge)

    # ---- 2.5 · Materialize phase (P0.6) ----------------------------------
    # 对每个 component，查 spec → 把 spec 期望但当前未出现的 pin 作为
    # ``is_floating=True`` 的 PortNode 补上，并把 spec 派生的 pin_symmetry
    # 组合记到 ComponentNode 上。

    for comp_id in list(hcg.components):
        comp = hcg.components[comp_id]
        part_subtype = subtype_by_source_id.get(comp.source_id, "")
        specs = get_expected_pin_specs(comp.ctype, part_subtype)
        if not specs:
            continue
        existing_keys = {
            hcg.ports[pid].port_key for pid in hcg.port_of_component[comp_id]
        }
        # 处理同义 key（"pin3" / "3" 两种写法）：把 existing 的 "pin3" 也算
        # 作覆盖了 spec 里的 "3"。
        for k in list(existing_keys):
            if k.startswith("pin"):
                existing_keys.add(k[3:])

        for spec in specs:
            if spec.pin_key in existing_keys:
                continue
            # Spec 期待该 pin，但当前 side 没观测到 → materialize floating port
            port_node_id = _port_node_id(side, comp.source_id, spec.pin_key)
            polarity_sensitive = (
                comp.polarity_class != PolarityClass.NONE.value
                and spec.port_type in POLARITY_SENSITIVE_PORT_TYPES
            )
            port_node = PortNode(
                node_id=port_node_id,
                side=side,
                parent_component_id=comp_id,
                port_key=spec.pin_key,
                port_type=spec.port_type,
                parent_ctype=comp.ctype,
                polarity_sensitive=polarity_sensitive,
                is_power_port=spec.port_type in POWER_PORT_TYPES,
                is_ground_port=spec.port_type in GROUND_PORT_TYPES,
                is_floating=True,
                pin_number=spec.pin_number,
                connection_policy=spec.connection_policy,
                symmetry_class_id=spec.symmetry_class,
            )
            hcg.ports[port_node_id] = port_node
            hcg.port_of_component[comp_id].append(port_node_id)

    # ---- 3. 回填 ComponentNode.pin_count + pin_symmetry_groups ----------

    for comp_id, port_ids in hcg.port_of_component.items():
        original = hcg.components[comp_id]
        # Group port_keys by symmetry_class_id, emit only groups of size ≥ 2.
        by_class: dict[int, list[str]] = {}
        for pid in port_ids:
            port = hcg.ports[pid]
            by_class.setdefault(port.symmetry_class_id, []).append(port.port_key)
        sym_groups = tuple(
            tuple(sorted(keys))
            for cid, keys in sorted(by_class.items())
            if len(keys) >= 2
        )

        if (
            original.pin_count != len(port_ids)
            or original.pin_symmetry_groups != sym_groups
        ):
            hcg.components[comp_id] = ComponentNode(
                node_id=original.node_id,
                side=original.side,
                source_id=original.source_id,
                ctype=original.ctype,
                package=original.package,
                polarity_class=original.polarity_class,
                pin_count=len(port_ids),
                value=original.value,
                confidence=original.confidence,
                pin_symmetry_groups=sym_groups,
            )

    # ---- 4. 透传图级元数据 ------------------------------------------------

    graph_meta = dict(nx_graph.graph)
    # 删几个噪音键但保留 format / reference_id / name / symmetry_groups
    for noisy_key in ("node_default", "edge_default"):
        graph_meta.pop(noisy_key, None)
    hcg.metadata.update(graph_meta)

    return hcg


# ---------------------------------------------------------------------------
# 便利入口（直接吃 logical_reference_v1 payload 或 netlist_v2 dict）
# ---------------------------------------------------------------------------


def _extract_subtypes(
    raw_components: list[dict[str, Any]] | None,
    id_keys: tuple[str, ...],
    subtype_keys: tuple[str, ...],
) -> dict[str, str]:
    """Build ``{source_id → part_subtype}`` mapping from a list of component
    dicts. Empty / missing subtypes are skipped."""

    mapping: dict[str, str] = {}
    for comp in raw_components or ():
        if not isinstance(comp, dict):
            continue
        comp_id = next(
            (str(comp[k]).strip() for k in id_keys if comp.get(k)),
            "",
        )
        if not comp_id:
            continue
        subtype = next(
            (str(comp[k]).strip() for k in subtype_keys if comp.get(k)),
            "",
        )
        if subtype:
            mapping[comp_id] = subtype
    return mapping


def _payload_raw_pin_edges_ref(payload: dict[str, Any]) -> list[RawPinEdge]:
    """Build the raw (comp, pin, net) edge list from a logical_reference_v1
    payload, preserving parallel pins on the same net (e.g., op-amp feedback)."""

    # Local import to avoid a top-level dep on the normalize helpers.
    from app.domain.logical_reference import (
        normalize_component_type,
        normalize_pin_role,
    )

    out: list[RawPinEdge] = []
    for comp in payload.get("components") or []:
        if not isinstance(comp, dict):
            continue
        ref_id = str(comp.get("ref_id") or "").strip()
        if not ref_id:
            continue
        ctype = normalize_component_type(comp.get("type"))
        for pin in comp.get("pins") or []:
            if not isinstance(pin, dict):
                continue
            if pin.get("nc") is True:
                continue
            net_name = str(pin.get("net") or "").strip()
            if not net_name:
                continue
            pin_raw = str(pin.get("pin") or "").strip()
            pin_role = normalize_pin_role(ctype, pin)
            out.append(
                (ref_id, ctype, pin_raw, pin_role, net_name, None, None)
            )
    return out


def _payload_raw_pin_edges_cur(netlist_v2: dict[str, Any]) -> list[RawPinEdge]:
    """Same as ``_payload_raw_pin_edges_ref`` but for ``netlist_v2`` dicts."""

    from app.domain.logical_reference import (
        normalize_component_type,
        normalize_pin_role,
    )

    out: list[RawPinEdge] = []
    for comp in netlist_v2.get("components") or []:
        if not isinstance(comp, dict):
            continue
        ctype = normalize_component_type(
            comp.get("component_type") or comp.get("type")
        )
        if ctype == "Wire":
            # current_netlist_v2_to_graph skips Wire; mirror that here.
            continue
        comp_id = str(
            comp.get("component_id") or comp.get("ref_id") or ""
        ).strip()
        if not comp_id:
            continue
        for pin in comp.get("pins") or []:
            if not isinstance(pin, dict):
                continue
            net_id = str(
                pin.get("electrical_net_id") or pin.get("net_id") or ""
            ).strip()
            if not net_id:
                continue
            pin_raw = str(pin.get("pin_name") or pin.get("pin") or "").strip()
            pin_role = normalize_pin_role(ctype, pin)
            confidence = _coerce_float(pin.get("confidence"))
            out.append(
                (comp_id, ctype, pin_raw, pin_role, net_id, confidence, None)
            )
    return out


def build_from_logical_reference(payload: dict[str, Any]) -> HeteroCircuitGraph:
    """``logical_reference_v1`` payload → ``HeteroCircuitGraph(side="ref")``.

    P0.5: 从 ``components[*].subtype`` 抽出 part_subtype 旁路传给 builder
    让 IC 的 pin 拿到精细的 PortType；并直接从 payload 重建原始 pin 边表，
    绕过 ``nx.Graph`` 折叠平行边（如 op-amp 单位增益缓冲器里 pin2 和 pin6
    同接 VOUT，nx 只保留一条）。
    """

    nx_graph = logical_reference_to_graph(payload)
    subtypes = _extract_subtypes(
        payload.get("components"),
        id_keys=("ref_id",),
        subtype_keys=("subtype", "part_subtype"),
    )
    raw_edges = _payload_raw_pin_edges_ref(payload)
    return build_hetero_circuit_graph(
        nx_graph,
        cast(Side, "ref"),
        subtype_by_source_id=subtypes,
        raw_pin_edges=raw_edges,
    )


def build_from_netlist_v2(netlist_v2: dict[str, Any]) -> HeteroCircuitGraph:
    """``netlist_v2`` dict → ``HeteroCircuitGraph(side="cur")``.

    P0.5: 同 ``build_from_logical_reference`` 一样抽 part_subtype + 原始
    pin 边表；保留 IC 多脚接同一 net 的情形。
    """

    nx_graph = current_netlist_v2_to_graph(netlist_v2)
    subtypes = _extract_subtypes(
        netlist_v2.get("components"),
        id_keys=("component_id", "ref_id"),
        subtype_keys=("part_subtype", "subtype"),
    )
    raw_edges = _payload_raw_pin_edges_cur(netlist_v2)
    return build_hetero_circuit_graph(
        nx_graph,
        cast(Side, "cur"),
        subtype_by_source_id=subtypes,
        raw_pin_edges=raw_edges,
    )


__all__ = [
    "build_hetero_circuit_graph",
    "build_from_logical_reference",
    "build_from_netlist_v2",
]


# 暴露 PortType 仅是为了类型重导出方便（避免外部测试再 import schema）。
_ = PortType  # noqa: F841 — keep linter happy without altering __all__
