"""GNN 模块 · ref ↔ cur 模糊组件对齐（Phase E · S1）

替换两条死代码路径：
- ``app/domain/net_normalization.infer_net_aliases_from_reference``
- ``app/domain/compare/role_inference._infer_current_net_roles_from_reference``

两者都用 ``GraphMatcher.is_isomorphic()``，对真实学生板（永远多 1-N 根
跳线导致节点数不等）100% 返回 None。本模块改用 Hungarian + ctype 分桶的
模糊对齐，对部分缺组件 / 多余跳线 / 跨拓扑变体都鲁棒。

**Wire 特殊性（项目根本约束 · 用户明示）**：

跳线在本项目里**不是常规元件**，是 net 的物理延伸：
1. 电学上 no-op（两端在同 net）
2. 每块板数量随机（学生路由习惯决定）
3. 会污染图签名（节点数、邻居签名都被吹大）

因此 Wire 在本模块里：
- 不进 component 对齐池（不与真实元件竞争对齐槽位）
- 不参与签名计算（neighbor ctypes 过滤 wire）
- 同 net wire 在 ``_build_wire_collapsed_net_union`` 时 no-op
- 跨 net wire 在 ``_build_wire_collapsed_net_union`` 时 union 两端 net
  并在 alignment.notes["wire_collapsed_groups"] 留痕

**算法（Hungarian + signature 多数票）**：

1. ``_build_wire_collapsed_net_union(cur_hcg)`` — Union-Find：把被 wire
   bridged 的 cur net 合并成代表 net
2. ``_bucket_components(hcg)`` — 按 (ctype, IC.subtype, pin_count) 分桶；
   wire 不入桶
3. 对每个桶内的 (ref_comps, cur_comps) 算 cost matrix（基于 pin 邻居 ctype
   多集距离），跑 ``scipy.optimize.linear_sum_assignment``
4. cost > ``max_match_cost`` 的对舍弃，分别记 ``unmatched_ref/cur``
5. ``_derive_net_alignment`` — 已对齐元件的 pin-net 对应关系做多数票，得到
   ``ref_to_cur_net``

输入：``HeteroCircuitGraph`` × 2；输出：:class:`ComponentAlignment`
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.domain.gnn.alignment import ComponentAlignment
from app.domain.gnn.hetero_circuit import HeteroCircuitGraph

# Wire-class ctypes are treated as net extensions, not components.
WIRE_CTYPES: frozenset[str] = frozenset({"Wire"})

# Default Hungarian cost cutoff — pairs with cost above this become
# unmatched. Distance is roughly "number of pin neighbor-ctype mismatches"
# so 5.0 is generous (allows up to ~5 mismatched neighbors).
DEFAULT_MAX_MATCH_COST: float = 5.0

# Net-vote majority threshold for deriving ref_to_cur_net mapping from
# component alignment. 0.5 = simple majority.
DEFAULT_NET_VOTE_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Wire-collapse (Union-Find over cur nets bridged by wire components)
# ---------------------------------------------------------------------------


def _build_wire_collapsed_net_union(
    hcg: HeteroCircuitGraph,
    *,
    wire_ctypes: frozenset[str] = WIRE_CTYPES,
) -> dict[str, str]:
    """Return ``{net_source_id → representative_net_source_id}`` map.

    For same-net wires this is a no-op (net maps to itself). For cross-net
    wires (cur has a wire connecting two distinct nets) the two nets get
    unioned and share a single representative.
    """
    parent: dict[str, str] = {n.source_id: n.source_id for n in hcg.nets.values()}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # For every wire component, union the nets that all its pins touch
    for comp in hcg.components.values():
        if comp.ctype not in wire_ctypes:
            continue
        wire_nets: list[str] = []
        for edge in hcg.edges:
            port = hcg.ports[edge.src_port_id]
            if port.parent_component_id != comp.node_id:
                continue
            net = hcg.nets.get(edge.dst_net_id)
            if net is not None:
                wire_nets.append(net.source_id)
        # Union all nets this wire touches
        for i in range(1, len(wire_nets)):
            union(wire_nets[0], wire_nets[i])

    # Path-compress all entries
    return {n: find(n) for n in parent}


def _wire_collapsed_groups(
    net_union: dict[str, str],
) -> list[list[str]]:
    """Pretty-print groups (only groups with > 1 member)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for child, root in net_union.items():
        groups[root].append(child)
    return [sorted(members) for members in groups.values() if len(members) > 1]


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def _bucket_key(
    comp: Any,
    hcg: HeteroCircuitGraph,
) -> tuple[str, str, int]:
    """Bucket key = (ctype, ic_subtype, pin_count).

    ic_subtype is taken from ``hcg.metadata["subtype_by_source_id"]`` for IC
    components, empty string otherwise — this prevents UA741 ↔ LM358 cross
    matching even though they're both "IC" ctype.
    """
    subtype = ""
    if comp.ctype == "IC":
        subtype = (hcg.metadata.get("subtype_by_source_id") or {}).get(comp.source_id, "")
    return (comp.ctype, subtype.upper(), comp.pin_count)


def _bucket_components(
    hcg: HeteroCircuitGraph,
    *,
    wire_ctypes: frozenset[str] = WIRE_CTYPES,
) -> dict[tuple[str, str, int], list[Any]]:
    """Group components by alignment-key. Wires excluded entirely."""
    buckets: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
    for comp in hcg.components.values():
        if comp.ctype in wire_ctypes:
            continue
        buckets[_bucket_key(comp, hcg)].append(comp)
    # Sort by source_id within each bucket for deterministic Hungarian input
    for key in buckets:
        buckets[key].sort(key=lambda c: c.source_id)
    return dict(buckets)


# ---------------------------------------------------------------------------
# Signature + cost
# ---------------------------------------------------------------------------


def _canonical_pin_key(ctype: str, pin_key: str) -> str:
    """Normalize pin_key across data sources.

    Why: ``build_from_logical_reference`` uses the raw ``pin`` field from the
    DSL (e.g. UA741 → ``"2"``/``"3"``/``"4"``), while ``build_from_netlist_v2``
    uses ``pin_name`` from runtime-scene payloads (e.g. ``"pin2"``/``"pin3"``).
    Same physical IC pin → different ``port_key`` → signature mismatch and
    Hungarian failure. This normalizer strips the ``"pin"`` prefix from IC
    numeric pins so both sides yield ``"2"``.

    Non-IC components already use canonical keys on both paths
    (Resistor ``pin1/pin2``, Transistor ``base/collector/emitter``,
    LED ``anode/cathode``, Pot ``terminal_a/terminal_b/wiper``), so they
    return unchanged.
    """
    if ctype == "IC":
        lowered = pin_key.lower()
        if lowered.startswith("pin"):
            lowered = lowered[3:]
        return lowered
    return pin_key


def _comp_pin_to_net(
    hcg: HeteroCircuitGraph,
    comp: Any,
) -> dict[str, str]:
    """Return ``{canonical_pin_key → net.source_id}`` for one component."""
    out: dict[str, str] = {}
    for edge in hcg.edges:
        port = hcg.ports[edge.src_port_id]
        if port.parent_component_id != comp.node_id:
            continue
        net = hcg.nets.get(edge.dst_net_id)
        if net is None:
            continue
        out[_canonical_pin_key(comp.ctype, port.port_key)] = net.source_id
    return out


def _net_neighbor_ctypes(
    hcg: HeteroCircuitGraph,
    net_repr_sid: str,
    exclude_comp_source_id: str,
    net_union: dict[str, str],
    wire_ctypes: frozenset[str] = WIRE_CTYPES,
) -> Counter[str]:
    """For a (wire-collapsed) net (identified by **source_id**), return
    Counter of neighbor ctypes, excluding the calling component and any
    wire components.

    Note: ``edge.dst_net_id`` is the **node_id** (e.g. ``ref_net:INV``)
    while ``net_union`` is keyed by **source_id** (e.g. ``INV``). We must
    convert via ``hcg.nets[node_id].source_id`` before lookup.
    """
    out: Counter[str] = Counter()
    for edge in hcg.edges:
        net_node = hcg.nets.get(edge.dst_net_id)
        if net_node is None:
            continue
        edge_net_sid = net_node.source_id
        if net_union.get(edge_net_sid, edge_net_sid) != net_repr_sid:
            continue
        port = hcg.ports[edge.src_port_id]
        other_comp = hcg.components.get(port.parent_component_id)
        if other_comp is None:
            continue
        if other_comp.source_id == exclude_comp_source_id:
            continue
        if other_comp.ctype in wire_ctypes:
            continue
        out[other_comp.ctype] += 1
    return out


def _component_signature(
    hcg: HeteroCircuitGraph,
    comp: Any,
    net_union: dict[str, str],
    *,
    wire_ctypes: frozenset[str] = WIRE_CTYPES,
) -> dict[str, Counter[str]]:
    """Return ``{port_key → Counter(neighbor_ctype)}``. Used for cost matrix.

    Two components with similar topological role will have similar per-pin
    neighbor Counter signatures.
    """
    sig: dict[str, Counter[str]] = {}
    pin_to_net = _comp_pin_to_net(hcg, comp)
    for pin_key, net_sid in pin_to_net.items():
        net_repr = net_union.get(net_sid, net_sid)
        sig[pin_key] = _net_neighbor_ctypes(
            hcg, net_repr, comp.source_id, net_union, wire_ctypes
        )
    return sig


def _signature_distance(
    sig_a: dict[str, Counter[str]],
    sig_b: dict[str, Counter[str]],
) -> float:
    """L1 distance over per-pin neighbor Counters.

    Pins that exist in one signature but not the other contribute the L1
    norm of the present Counter (worst case).
    """
    pin_keys = set(sig_a) | set(sig_b)
    dist = 0.0
    for pk in pin_keys:
        a = sig_a.get(pk, Counter())
        b = sig_b.get(pk, Counter())
        # L1 distance on Counters: sum |a[c] - b[c]| over all ctypes
        all_ctypes = set(a) | set(b)
        dist += sum(abs(a.get(c, 0) - b.get(c, 0)) for c in all_ctypes)
    return dist


def _build_cost_matrix(
    ref_hcg: HeteroCircuitGraph,
    cur_hcg: HeteroCircuitGraph,
    ref_comps: list[Any],
    cur_comps: list[Any],
    cur_net_union: dict[str, str],
    wire_ctypes: frozenset[str],
) -> np.ndarray:
    """Build len(ref_comps) × len(cur_comps) cost matrix.

    Hungarian needs square matrix — caller pads with large sentinel cost
    (DEFAULT_MAX_MATCH_COST × 10) so unmatched padding rows/cols are
    auto-deprioritized.
    """
    ref_sigs = [
        _component_signature(ref_hcg, c, {n.source_id: n.source_id for n in ref_hcg.nets.values()}, wire_ctypes=wire_ctypes)
        for c in ref_comps
    ]
    cur_sigs = [
        _component_signature(cur_hcg, c, cur_net_union, wire_ctypes=wire_ctypes)
        for c in cur_comps
    ]
    n = max(len(ref_comps), len(cur_comps))
    if n == 0:
        return np.zeros((0, 0))
    PADDING_COST = DEFAULT_MAX_MATCH_COST * 10
    cost = np.full((n, n), PADDING_COST, dtype=float)
    for i, ref_sig in enumerate(ref_sigs):
        for j, cur_sig in enumerate(cur_sigs):
            cost[i, j] = _signature_distance(ref_sig, cur_sig)
    return cost


# ---------------------------------------------------------------------------
# Net alignment derivation (from component alignment + voting)
# ---------------------------------------------------------------------------


def _derive_net_alignment(
    ref_hcg: HeteroCircuitGraph,
    cur_hcg: HeteroCircuitGraph,
    ref_to_cur_component: dict[str, str],
    cur_net_union: dict[str, str],
    *,
    vote_threshold: float = DEFAULT_NET_VOTE_THRESHOLD,
) -> dict[str, str]:
    """For each ref net, vote across aligned components for the cur net
    counterpart. Returns ``ref_net.source_id → cur_net.source_id`` map.

    Voting only uses non-wire aligned components. Pins on the same key
    in (ref_comp, cur_comp) tell us "ref_comp.pinK is on ref_net X,
    cur_comp.pinK is on cur_net Y → vote X→Y".
    """
    # Build comp source_id → ComponentNode lookups
    ref_comp_by_sid = {c.source_id: c for c in ref_hcg.components.values()}
    cur_comp_by_sid = {c.source_id: c for c in cur_hcg.components.values()}

    # Accumulate votes: ref_net_sid → Counter(cur_net_repr_sid)
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for ref_sid, cur_sid in ref_to_cur_component.items():
        ref_comp = ref_comp_by_sid.get(ref_sid)
        cur_comp = cur_comp_by_sid.get(cur_sid)
        if ref_comp is None or cur_comp is None:
            continue
        ref_pin_to_net = _comp_pin_to_net(ref_hcg, ref_comp)
        cur_pin_to_net = _comp_pin_to_net(cur_hcg, cur_comp)
        for pin_key, ref_net_sid in ref_pin_to_net.items():
            cur_net_sid = cur_pin_to_net.get(pin_key)
            if cur_net_sid is None:
                continue
            cur_net_repr = cur_net_union.get(cur_net_sid, cur_net_sid)
            votes[ref_net_sid][cur_net_repr] += 1

    # Resolve each ref_net by majority vote
    ref_to_cur_net: dict[str, str] = {}
    for ref_net_sid, vote_counter in votes.items():
        if not vote_counter:
            continue
        winner, winning_count = vote_counter.most_common(1)[0]
        total = sum(vote_counter.values())
        if winning_count / total >= vote_threshold:
            ref_to_cur_net[ref_net_sid] = winner
    return ref_to_cur_net


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def align_components_by_signature(
    ref_hcg: HeteroCircuitGraph,
    cur_hcg: HeteroCircuitGraph,
    *,
    wire_ctypes: frozenset[str] = WIRE_CTYPES,
    max_match_cost: float = DEFAULT_MAX_MATCH_COST,
    vote_threshold: float = DEFAULT_NET_VOTE_THRESHOLD,
) -> ComponentAlignment:
    """Produce a :class:`ComponentAlignment` from two HCGs via fuzzy matching.

    Robust to:
    - cur has extra wires (collapsed via Union-Find before alignment)
    - cur is missing components (those ref components land in unmatched_ref)
    - cur has extra components (those land in unmatched_cur)
    - cross-net wires (unioned + flagged in notes)

    Args:
        ref_hcg: clean reference HCG (from ``build_from_logical_reference``).
        cur_hcg: student-board HCG (from ``build_from_netlist_v2``).
        wire_ctypes: ctypes treated as "wire / net extension"; default
            ``frozenset({"Wire"})``. Override for project-specific wire
            taxonomies.
        max_match_cost: Hungarian pairs with cost above this become unmatched.
        vote_threshold: net-vote majority needed to derive ref→cur net edge.

    Returns:
        ComponentAlignment with:
        - ``ref_to_cur_component``: source_id → source_id (only matched)
        - ``ref_to_cur_net``: derived from voting
        - ``notes``:
          - ``constructor``: "align_components_by_signature"
          - ``unmatched_ref_components``: ref comps with no cur counterpart
          - ``unmatched_cur_components``: cur comps with no ref counterpart
          - ``wire_collapsed_groups``: list of cur net groups unioned by wires
          - ``match_costs``: dict[(ref_sid, cur_sid)] → cost
    """
    # Wire collapse cur (ref is assumed clean — no wires in reference fixtures)
    cur_net_union = _build_wire_collapsed_net_union(cur_hcg, wire_ctypes=wire_ctypes)

    # Bucket components by (ctype, ic_subtype, pin_count); wires excluded
    ref_buckets = _bucket_components(ref_hcg, wire_ctypes=wire_ctypes)
    cur_buckets = _bucket_components(cur_hcg, wire_ctypes=wire_ctypes)

    ref_to_cur_component: dict[str, str] = {}
    unmatched_ref: list[str] = []
    unmatched_cur: list[str] = []
    match_costs: dict[tuple[str, str], float] = {}

    # Process every bucket key that appears in either side
    for key in set(ref_buckets) | set(cur_buckets):
        ref_comps = ref_buckets.get(key, [])
        cur_comps = cur_buckets.get(key, [])
        if not ref_comps:
            unmatched_cur.extend(c.source_id for c in cur_comps)
            continue
        if not cur_comps:
            unmatched_ref.extend(c.source_id for c in ref_comps)
            continue

        cost = _build_cost_matrix(
            ref_hcg, cur_hcg, ref_comps, cur_comps, cur_net_union, wire_ctypes
        )
        row_ind, col_ind = linear_sum_assignment(cost)

        for ri, ci in zip(row_ind, col_ind, strict=False):
            # Skip Hungarian-padded ghost rows/cols (cost matrix padded to square)
            is_pad_row = ri >= len(ref_comps)
            is_pad_col = ci >= len(cur_comps)
            if is_pad_row and is_pad_col:
                continue
            if is_pad_row:
                unmatched_cur.append(cur_comps[ci].source_id)
                continue
            if is_pad_col:
                unmatched_ref.append(ref_comps[ri].source_id)
                continue
            pair_cost = float(cost[ri, ci])
            if pair_cost > max_match_cost:
                unmatched_ref.append(ref_comps[ri].source_id)
                unmatched_cur.append(cur_comps[ci].source_id)
                continue
            ref_to_cur_component[ref_comps[ri].source_id] = cur_comps[ci].source_id
            match_costs[(ref_comps[ri].source_id, cur_comps[ci].source_id)] = pair_cost

    # Derive net alignment via voting on aligned components
    ref_to_cur_net = _derive_net_alignment(
        ref_hcg, cur_hcg, ref_to_cur_component, cur_net_union,
        vote_threshold=vote_threshold,
    )

    return ComponentAlignment(
        ref_to_cur_component=ref_to_cur_component,
        ref_to_cur_net=ref_to_cur_net,
        notes={
            "constructor": "align_components_by_signature",
            "unmatched_ref_components": sorted(unmatched_ref),
            "unmatched_cur_components": sorted(unmatched_cur),
            "wire_collapsed_groups": _wire_collapsed_groups(cur_net_union),
            "match_costs": {f"{r}->{c}": v for (r, c), v in match_costs.items()},
            "wire_ctypes": sorted(wire_ctypes),
            "max_match_cost": max_match_cost,
            "vote_threshold": vote_threshold,
        },
    )


__all__ = [
    "align_components_by_signature",
    "WIRE_CTYPES",
    "DEFAULT_MAX_MATCH_COST",
    "DEFAULT_NET_VOTE_THRESHOLD",
]
