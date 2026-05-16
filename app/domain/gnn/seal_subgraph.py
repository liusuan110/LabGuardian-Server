"""GNN 模块 · SEAL Enclosing Subgraph + DRNL（P0.7）

GNN-ACLP 范式的核心采样层。对一条候选边 ``(port_u, net_v)`` 抽取它的
**h-hop enclosing subgraph**：

- 底图：``HeteroCircuitGraph`` 的 ``(port, net)`` 二分图（component 节点
  不进入 SEAL 子图，它们的信息已被烤进 port 节点特征 —— parent_ctype /
  polarity_class / connection_policy 等）。
- 候选边本身从 BFS 与最终 edge 集合中**全部剔除**，遵循 Zhang & Chen 2018
  SEAL 守则："the model must not see the link it is predicting"。
- 每个节点附 **DRNL 标签**（Double-Radius Node Labeling）：用节点到两个
  anchor 的最短距离对编码相对结构位置，提供强归纳偏置。

本模块不引入 ``torch`` / ``torch_geometric``；输出是纯 Python 数据结构。
P2 ``pyg_converter`` 负责把它们打包成 ``torch_geometric.data.Data``。

公开 API：
- :class:`SealSubgraph` ：单个子图的不可变快照
- :func:`extract_seal_subgraph` ：单边抽取
- :func:`extract_subgraphs_for_observed_edges` ：批量 — wrong-edge 检测
- :func:`extract_subgraphs_for_floating_ports` ：批量 — suggested-target /
  missing-edge 检测
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

from app.domain.gnn.graph_schema import ConnectionPolicy
from app.domain.gnn.hetero_circuit import HeteroCircuitGraph

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SealSubgraph:
    """一条候选边 ``(target_port_id, target_net_id)`` 周围的 h-hop enclosing
    subgraph，已计算 DRNL 标签。

    Attributes:
        target_port_id: anchor port 节点 id（"u"）。
        target_net_id: anchor net 节点 id（"v"）。
        edge_present: 候选边是否真的存在于源 ``HeteroCircuitGraph.edges`` 中。
            wrong-edge 检测时为 True；missing/suggested 候选时为 False。
        num_hops: 抽取使用的 hop 半径（plan 默认 2）。
        port_ids: 子图内 port 节点 id 元组，**anchor 排第一**，其余按
            ``node_id`` 字典序，确定性。
        net_ids: 同上，net 节点；anchor 排第一。
        edges: 子图内 ``(port_id, net_id)`` 边元组；**候选边本身已被
            剔除**（per SEAL convention）。
        same_component_edges: 子图内同属一个 component 的 ``(port, port)``
            对，仅当 ``extract_seal_subgraph(..., include_same_component_edges=True)``
            才填充；否则为空元组。**DRNL 距离始终在 bipartite (port, net)
            底图上计算**，本字段不影响 ``drnl_labels`` —— 它是 P2 PyG
            converter 与下游模型决定是否消费的"额外结构信号"。
        drnl_labels: ``node_id → int``。anchor → 1；不可达 → 0；其余按
            Zhang & Chen 2018 公式 ``1 + min(d_u, d_v) + d_half * (d_half +
            (d % 2) - 1)``，其中 ``d = d_u + d_v``、``d_half = d // 2``。
        is_target: ``node_id → bool``，True 表示该节点是候选边的一个 anchor。
    """

    target_port_id: str
    target_net_id: str
    edge_present: bool
    num_hops: int
    port_ids: tuple[str, ...]
    net_ids: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    drnl_labels: dict[str, int]
    is_target: dict[str, bool]
    # P0.7 收尾：预留字段，默认空。开启 ``include_same_component_edges``
    # 才填充，承载"同片 IC / 同只 BJT 的兄弟 pin"结构边。
    same_component_edges: tuple[tuple[str, str], ...] = ()

    def num_nodes(self) -> int:
        return len(self.port_ids) + len(self.net_ids)

    def num_edges(self) -> int:
        return len(self.edges)

    def node_ids(self) -> tuple[str, ...]:
        """All node ids in deterministic order: ports first, then nets."""

        return self.port_ids + self.net_ids


# ---------------------------------------------------------------------------
# 私有算法
# ---------------------------------------------------------------------------


def _build_bipartite_adjacency(
    hcg: HeteroCircuitGraph,
) -> dict[str, set[str]]:
    """Build undirected (port ↔ net) adjacency map from
    ``HeteroCircuitGraph.edges``. Floating ports (no edges) are absent from
    the result; callers can treat them as unreachable."""

    adj: dict[str, set[str]] = defaultdict(set)
    for edge in hcg.edges:
        adj[edge.src_port_id].add(edge.dst_net_id)
        adj[edge.dst_net_id].add(edge.src_port_id)
    return adj


def _bfs_distances(
    adjacency: dict[str, set[str]],
    source: str,
    excluded_edges: frozenset[frozenset[str]] = frozenset(),
    max_depth: int | None = None,
) -> dict[str, int]:
    """Breadth-first distances from ``source`` over an undirected adjacency
    map. ``excluded_edges`` is a set of ``frozenset({u, v})`` describing
    edges to ignore (used to drop the candidate edge from the BFS). If
    ``max_depth`` is given, BFS stops expanding beyond that radius (the
    returned dict still includes all nodes up to and including max_depth).
    """

    distances: dict[str, int] = {source: 0}
    queue: deque[str] = deque([source])
    while queue:
        node = queue.popleft()
        if max_depth is not None and distances[node] >= max_depth:
            # Still process incoming neighbors of `node` but do not enqueue
            # them for further expansion — distances dict already covers them
            # at exactly `max_depth + 0` ... actually simpler: just stop.
            continue
        for neighbor in adjacency.get(node, ()):
            if frozenset({node, neighbor}) in excluded_edges:
                continue
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + 1
            queue.append(neighbor)
    return distances


def _drnl_label(d_u: float, d_v: float) -> int:
    """Zhang & Chen 2018 Double-Radius Node Labeling formula.

    Returns 0 for unreachable nodes (either ``d_u`` or ``d_v`` is +inf).
    Caller is responsible for special-casing the two anchors (which by
    convention receive label 1).
    """

    if math.isinf(d_u) or math.isinf(d_v):
        return 0
    du = int(d_u)
    dv = int(d_v)
    d = du + dv
    d_half = d // 2
    return 1 + min(du, dv) + d_half * (d_half + (d % 2) - 1)


def _is_port(hcg: HeteroCircuitGraph, node_id: str) -> bool:
    return node_id in hcg.ports


def _is_net(hcg: HeteroCircuitGraph, node_id: str) -> bool:
    return node_id in hcg.nets


def _enumerate_same_component_edges(
    hcg: HeteroCircuitGraph,
    port_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """For each component that has ≥ 2 of its ports inside ``port_ids``, emit
    every ``(port_i, port_j)`` pair (i < j by sorted port id) as an
    undirected structural edge.

    Determinism: components processed in ``parent_component_id`` order; within
    each component, ports sorted by id; pairs emitted in lexicographic order.
    This guarantees stable test diffs and reproducible cache keys.
    """

    by_comp: dict[str, list[str]] = defaultdict(list)
    for pid in port_ids:
        port = hcg.ports.get(pid)
        if port is None:
            continue
        by_comp[port.parent_component_id].append(pid)

    edges: list[tuple[str, str]] = []
    for comp_id in sorted(by_comp):
        ports_in_comp = sorted(by_comp[comp_id])
        for i in range(len(ports_in_comp)):
            for j in range(i + 1, len(ports_in_comp)):
                edges.append((ports_in_comp[i], ports_in_comp[j]))
    return tuple(edges)


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------


def extract_seal_subgraph(
    hcg: HeteroCircuitGraph,
    port_node_id: str,
    net_node_id: str,
    num_hops: int = 2,
    *,
    edge_present: bool | None = None,
    include_same_component_edges: bool = False,
) -> SealSubgraph:
    """Extract a SEAL h-hop enclosing subgraph around the candidate
    ``(port_node_id, net_node_id)`` edge.

    Args:
        hcg: 源异构图。
        port_node_id: 必须存在于 ``hcg.ports``。
        net_node_id: 必须存在于 ``hcg.nets``。
        num_hops: BFS / enclosing 半径。GNN-ACLP 论文默认 2，本项目沿用。
        edge_present: 候选边是否真存在。``None`` → 自动从 ``hcg.edges`` 推断。
            显式传 False 用于 missing/suggested 候选（即便恰好 hcg 中存在
            这条边，也按"不存在"处理 —— 用于负采样场景）。
        include_same_component_edges: 是否填充 ``same_component_edges`` 字段。
            默认 False。**开启不影响 DRNL 距离计算** —— 距离始终在 bipartite
            (port, net) 底图上 BFS。开启同片边只是给下游模型多一路结构信号
            （"这两个 pin 在同一片 IC 上"），是否消费由 P2 / P3 决定。

    Raises:
        KeyError: anchor 节点不在 ``hcg`` 中。
    """

    if port_node_id not in hcg.ports:
        raise KeyError(f"port node not found in HeteroCircuitGraph: {port_node_id!r}")
    if net_node_id not in hcg.nets:
        raise KeyError(f"net node not found in HeteroCircuitGraph: {net_node_id!r}")

    adj = _build_bipartite_adjacency(hcg)

    # 自动判定 edge_present
    if edge_present is None:
        edge_present = net_node_id in adj.get(port_node_id, set())

    # 候选边从 BFS 与 edge list 中剔除
    excluded: frozenset[frozenset[str]] = frozenset(
        {frozenset({port_node_id, net_node_id})} if edge_present else set()
    )

    # BFS from both anchors (含 anchor 各自 distance=0)
    d_u = _bfs_distances(adj, port_node_id, excluded, max_depth=num_hops)
    d_v = _bfs_distances(adj, net_node_id, excluded, max_depth=num_hops)

    # Enclosing subgraph 节点集：union of nodes within num_hops of either
    # anchor. Anchors themselves are always included.
    in_subgraph: set[str] = {port_node_id, net_node_id}
    for node, dist in d_u.items():
        if dist <= num_hops:
            in_subgraph.add(node)
    for node, dist in d_v.items():
        if dist <= num_hops:
            in_subgraph.add(node)

    # DRNL labels (anchors → 1，不可达 → 0，其它走公式)
    drnl: dict[str, int] = {}
    for w in in_subgraph:
        if w == port_node_id or w == net_node_id:
            drnl[w] = 1
            continue
        du = d_u.get(w, math.inf)
        dv = d_v.get(w, math.inf)
        drnl[w] = _drnl_label(du, dv)

    # Edge 集合：子图内已观测边，但**剔除候选边本身**
    subgraph_edges: list[tuple[str, str]] = []
    for edge in hcg.edges:
        if edge.src_port_id not in in_subgraph:
            continue
        if edge.dst_net_id not in in_subgraph:
            continue
        if (
            edge_present
            and edge.src_port_id == port_node_id
            and edge.dst_net_id == net_node_id
        ):
            continue
        subgraph_edges.append((edge.src_port_id, edge.dst_net_id))
    # 同一 (port, net) 多次出现 (理论上不应该) → 去重保持顺序
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for e in subgraph_edges:
        if e not in seen:
            seen.add(e)
            deduped.append(e)
    subgraph_edges = deduped

    # 确定性排序：anchor 排第一，其余字典序
    other_ports = sorted(
        n for n in in_subgraph if _is_port(hcg, n) and n != port_node_id
    )
    other_nets = sorted(
        n for n in in_subgraph if _is_net(hcg, n) and n != net_node_id
    )
    port_ids = (port_node_id, *other_ports)
    net_ids = (net_node_id, *other_nets)

    is_target = {n: (n == port_node_id or n == net_node_id) for n in in_subgraph}

    if include_same_component_edges:
        same_comp_edges = _enumerate_same_component_edges(hcg, port_ids)
    else:
        same_comp_edges = ()

    return SealSubgraph(
        target_port_id=port_node_id,
        target_net_id=net_node_id,
        edge_present=bool(edge_present),
        num_hops=num_hops,
        port_ids=port_ids,
        net_ids=net_ids,
        edges=tuple(subgraph_edges),
        drnl_labels=drnl,
        is_target=is_target,
        same_component_edges=same_comp_edges,
    )


# ---------------------------------------------------------------------------
# 批量入口
# ---------------------------------------------------------------------------


def extract_subgraphs_for_observed_edges(
    hcg: HeteroCircuitGraph,
    num_hops: int = 2,
    *,
    include_same_component_edges: bool = False,
) -> list[SealSubgraph]:
    """Enumerate every observed (port, net) edge in ``hcg`` and extract one
    SealSubgraph per edge with ``edge_present=True``. This is the input to
    the SEAL **wrong-edge detection** head: each subgraph is scored, and
    edges whose P(correct) falls below ``τ_wrong`` are flagged as
    wrong-pin candidates."""

    out: list[SealSubgraph] = []
    for edge in hcg.edges:
        out.append(
            extract_seal_subgraph(
                hcg,
                edge.src_port_id,
                edge.dst_net_id,
                num_hops=num_hops,
                edge_present=True,
                include_same_component_edges=include_same_component_edges,
            )
        )
    return out


# Default policy set for ``extract_subgraphs_for_floating_ports``: only
# ``REQUIRED`` pins are surfaced as suggested-target / missing-edge candidates.
# Rationale (P0.7 follow-up audit): OPTIONAL pins (e.g., UA741 offset_null)
# legitimately may stay floating, so including them would inject systemic
# label noise into P1 synthetic dataset construction (a positive ground-truth
# label for "should connect" is ambiguous when the spec says "either-way").
# Callers wanting OPTIONAL coverage must opt in explicitly.
_DEFAULT_FLOATING_POLICIES: frozenset[ConnectionPolicy] = frozenset(
    {ConnectionPolicy.REQUIRED}
)


def extract_subgraphs_for_floating_ports(
    hcg: HeteroCircuitGraph,
    num_hops: int = 2,
    candidate_nets: list[str] | None = None,
    *,
    policies: frozenset[ConnectionPolicy] = _DEFAULT_FLOATING_POLICIES,
    include_same_component_edges: bool = False,
) -> list[SealSubgraph]:
    """For every floating port (``is_floating=True``) whose
    ``connection_policy`` is in ``policies``, pair it with each candidate net
    and extract a SealSubgraph (``edge_present=False``). This is the input
    to the SEAL **suggested-target** head (and to missing-connection
    detection when run on the ref-mapped current graph).

    Args:
        candidate_nets: list of ``net_node_id``. Defaults to **all** nets in
            ``hcg``. Caller may pass a narrower set (e.g., the cur side's
            nets mapped from the ref side).
        policies: which ``ConnectionPolicy`` values to include. Default is
            ``frozenset({ConnectionPolicy.REQUIRED})`` —— only "must be
            connected but isn't" pins. To also surface OPTIONAL or FORBIDDEN
            candidates (typically for diagnostic / auditing pipelines), pass
            an explicit set, e.g. ``frozenset({REQUIRED, OPTIONAL})``.
        include_same_component_edges: forwarded to
            :func:`extract_seal_subgraph`. Default False.
    """

    if candidate_nets is None:
        candidate_nets = list(hcg.nets.keys())

    # Normalize policies to a set of string values so callers may pass either
    # enum members or plain strings interchangeably.
    allowed_values: set[str] = {
        p.value if isinstance(p, ConnectionPolicy) else str(p) for p in policies
    }

    out: list[SealSubgraph] = []
    for port_id, port in hcg.ports.items():
        if not port.is_floating:
            continue
        if port.connection_policy not in allowed_values:
            continue
        for net_id in candidate_nets:
            if net_id not in hcg.nets:
                continue
            out.append(
                extract_seal_subgraph(
                    hcg,
                    port_id,
                    net_id,
                    num_hops=num_hops,
                    edge_present=False,
                    include_same_component_edges=include_same_component_edges,
                )
            )
    return out


__all__ = [
    "SealSubgraph",
    "extract_seal_subgraph",
    "extract_subgraphs_for_observed_edges",
    "extract_subgraphs_for_floating_ports",
]
