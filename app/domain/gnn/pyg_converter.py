"""GNN 模块 · PyG Converter（P2）

把 :class:`HeteroCircuitGraph` / :class:`SealSubgraph` 翻译成
``torch_geometric`` 的 :class:`HeteroData` / :class:`Data`，并提供从
P1 dataset_builder 写盘的 ``labels/<ref_id>/<sample_id>.json`` 出发
的 flat SEAL dataset 入口供 P3 训练循环消费。

公开 API：

- :func:`to_hetero_data(hcg)` —— HCG → ``HeteroData``，三类节点 +
  ``(component,has_port,port)`` 与 ``(port,connects,net)`` 两类边，
  reverse 边由 ``T.ToUndirected()`` 在外部添加（converter 默认产单向）
- :func:`encode_component_features(hcg)` / :func:`encode_port_features(hcg)` /
  :func:`encode_net_features(hcg)` —— 三类节点向量化（plan §三 严格对齐）
- :func:`encode_port_net_edge_features(hcg)` —— ``(port,connects,net)`` 边特征
- :func:`seal_subgraph_to_pyg_data(sg, cur_hcg, *, label, source, task_type)``
  —— 一个 SealSubgraph → 一个 ``Data``：节点特征 = DRNL[17] ⊕
  原 port/net feat ⊕ target_flag[1]；边集复用 SealSubgraph.edges。

**约定**：本模块需要 ``torch`` + ``torch_geometric``（pyproject 的
``[gnn]`` extra）。import 期不立刻 import torch，但所有公开函数调用
torch —— 调用者必须装好 extras。

设计参见 plan §二 (modules table) 与 §三 (schema) 与 §三.6 (SEAL)。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data, HeteroData  # type: ignore[import-untyped]

from app.domain.gnn.graph_schema import (
    COMPONENT_FEAT_DIM,
    CONNECTION_POLICY_TO_INDEX,
    CTYPE_TO_INDEX,
    DRNL_LABEL_DIM,
    NET_FEAT_DIM,
    NET_ROLE_TO_INDEX,
    PACKAGE_TO_INDEX,
    POLARITY_CLASS_TO_INDEX,
    PORT_FEAT_DIM,
    PORT_NET_EDGE_FEAT_DIM,
    PORT_TYPE_TO_INDEX,
    SOURCE_TYPE_TO_INDEX,
)

if TYPE_CHECKING:  # pragma: no cover
    from app.domain.gnn.hetero_circuit import HeteroCircuitGraph
    from app.domain.gnn.seal_subgraph import SealSubgraph


# ---------------------------------------------------------------------------
# Node feature encoders
# ---------------------------------------------------------------------------


def _is_ref(side: str) -> float:
    return 1.0 if side == "ref" else 0.0


def encode_component_features(hcg: HeteroCircuitGraph) -> tuple[torch.Tensor, list[str]]:
    """``components`` dict → ``(x, node_ids)`` where x is
    ``[Nc, COMPONENT_FEAT_DIM]``. node_ids preserves the dict iteration
    order (insertion-order; deterministic across processes for the same
    HCG construction)."""

    node_ids = list(hcg.components)
    x = torch.zeros((len(node_ids), COMPONENT_FEAT_DIM), dtype=torch.float32)
    n_ct = len(CTYPE_TO_INDEX)
    n_pkg = len(PACKAGE_TO_INDEX)
    n_pol = len(POLARITY_CLASS_TO_INDEX)
    for i, node_id in enumerate(node_ids):
        c = hcg.components[node_id]
        # ctype one-hot
        idx = CTYPE_TO_INDEX.get(c.ctype, CTYPE_TO_INDEX["UNKNOWN"])
        x[i, idx] = 1.0
        # package one-hot (empty/unknown → all zero)
        if c.package and c.package in PACKAGE_TO_INDEX:
            x[i, n_ct + PACKAGE_TO_INDEX[c.package]] = 1.0
        # polarity_class one-hot
        pol_idx = POLARITY_CLASS_TO_INDEX.get(c.polarity_class, 0)
        x[i, n_ct + n_pkg + pol_idx] = 1.0
        cursor = n_ct + n_pkg + n_pol
        # pin_count_log
        x[i, cursor] = math.log1p(max(0, c.pin_count)) / math.log(1 + 64)
        # value_log10 + mask
        if c.value is not None and c.value > 0:
            x[i, cursor + 1] = math.log10(c.value)
            x[i, cursor + 2] = 1.0
        # confidence
        x[i, cursor + 3] = float(c.confidence)
        # is_reference
        x[i, cursor + 4] = _is_ref(c.side)
    return x, node_ids


def encode_port_features(hcg: HeteroCircuitGraph) -> tuple[torch.Tensor, list[str]]:
    """Per plan §三 schema (PORT_FEAT_DIM = 50)."""

    node_ids = list(hcg.ports)
    x = torch.zeros((len(node_ids), PORT_FEAT_DIM), dtype=torch.float32)
    n_pt = len(PORT_TYPE_TO_INDEX)
    n_ct = len(CTYPE_TO_INDEX)

    # symmetry class size: count ports sharing the same (parent_component_id,
    # symmetry_class_id) — needed for symmetry_class_size_inverse feature
    sym_class_size: dict[tuple[str, int], int] = {}
    for p in hcg.ports.values():
        key = (p.parent_component_id, p.symmetry_class_id)
        sym_class_size[key] = sym_class_size.get(key, 0) + 1

    for i, node_id in enumerate(node_ids):
        p = hcg.ports[node_id]
        # port_type one-hot
        pt_idx = PORT_TYPE_TO_INDEX.get(p.port_type, PORT_TYPE_TO_INDEX["generic"])
        x[i, pt_idx] = 1.0
        # parent_ctype one-hot
        pct_idx = CTYPE_TO_INDEX.get(p.parent_ctype, CTYPE_TO_INDEX["UNKNOWN"])
        x[i, n_pt + pct_idx] = 1.0
        cursor = n_pt + n_ct
        x[i, cursor + 0] = 1.0 if p.polarity_sensitive else 0.0
        x[i, cursor + 1] = 1.0 if p.is_power_port else 0.0
        x[i, cursor + 2] = 1.0 if p.is_ground_port else 0.0
        x[i, cursor + 3] = 1.0 if p.is_floating else 0.0
        x[i, cursor + 4] = _is_ref(p.side)
        cursor += 5
        # connection_policy one-hot
        cp_idx = CONNECTION_POLICY_TO_INDEX.get(p.connection_policy, 0)
        x[i, cursor + cp_idx] = 1.0
        cursor += len(CONNECTION_POLICY_TO_INDEX)
        # has_pin_number + pin_number_log
        if p.pin_number is not None:
            x[i, cursor + 0] = 1.0
            x[i, cursor + 1] = min(
                1.0, math.log1p(p.pin_number) / math.log(1 + 64)
            )
        cursor += 2
        # symmetry_class_size_inverse
        sz = sym_class_size.get(
            (p.parent_component_id, p.symmetry_class_id), 1
        )
        x[i, cursor + 0] = 1.0 / max(1, sz)
    return x, node_ids


def encode_net_features(hcg: HeteroCircuitGraph) -> tuple[torch.Tensor, list[str]]:
    """Per plan §三 schema (NET_FEAT_DIM = 11)."""

    node_ids = list(hcg.nets)
    x = torch.zeros((len(node_ids), NET_FEAT_DIM), dtype=torch.float32)
    n_role = len(NET_ROLE_TO_INDEX)

    # degree per net
    net_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    for e in hcg.edges:
        if e.dst_net_id in net_degree:
            net_degree[e.dst_net_id] += 1

    for i, node_id in enumerate(node_ids):
        n = hcg.nets[node_id]
        role_idx = NET_ROLE_TO_INDEX.get(n.role, NET_ROLE_TO_INDEX["unknown"])
        x[i, role_idx] = 1.0
        cursor = n_role
        x[i, cursor + 0] = math.log1p(net_degree[node_id]) / math.log(1 + 32)
        x[i, cursor + 1] = 1.0 if n.is_power_rail else 0.0
        if n.voltage_hint is not None:
            x[i, cursor + 2] = float(n.voltage_hint)
            x[i, cursor + 3] = 1.0
        x[i, cursor + 4] = _is_ref(n.side)
    return x, node_ids


def encode_port_net_edge_features(
    hcg: HeteroCircuitGraph,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(edge_index[2,E], edge_attr[E, PORT_NET_EDGE_FEAT_DIM])``
    for the ``(port, connects, net)`` edge type, indexed against the
    canonical port/net node_id order returned by encode_port_features /
    encode_net_features."""

    port_ids = list(hcg.ports)
    net_ids = list(hcg.nets)
    port_idx = {nid: i for i, nid in enumerate(port_ids)}
    net_idx = {nid: i for i, nid in enumerate(net_ids)}

    src: list[int] = []
    dst: list[int] = []
    attrs: list[list[float]] = []
    n_st = len(SOURCE_TYPE_TO_INDEX)
    for e in hcg.edges:
        if e.src_port_id not in port_idx or e.dst_net_id not in net_idx:
            continue
        src.append(port_idx[e.src_port_id])
        dst.append(net_idx[e.dst_net_id])
        v = [0.0] * PORT_NET_EDGE_FEAT_DIM
        v[0] = float(e.connection_confidence)
        st_idx = SOURCE_TYPE_TO_INDEX.get(e.source_type, 0)
        v[1 + st_idx] = 1.0
        v[1 + n_st + 0] = 1.0 if e.is_observed_in_cur else 0.0
        attrs.append(v)

    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    edge_attr = (
        torch.tensor(attrs, dtype=torch.float32)
        if attrs
        else torch.zeros((0, PORT_NET_EDGE_FEAT_DIM), dtype=torch.float32)
    )
    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# HeteroCircuitGraph → HeteroData
# ---------------------------------------------------------------------------


def to_hetero_data(hcg: HeteroCircuitGraph) -> HeteroData:
    """Single-side HCG → PyG ``HeteroData``.

    Layout:
        data['component'].x           [Nc, COMPONENT_FEAT_DIM]
        data['port'].x                [Np, PORT_FEAT_DIM]
        data['net'].x                 [Nn, NET_FEAT_DIM]
        ('component', 'has_port', 'port').edge_index   [2, Np]
        ('port', 'connects', 'net').edge_index         [2, E]
        ('port', 'connects', 'net').edge_attr          [E, PORT_NET_EDGE_FEAT_DIM]

    ``node_id`` mapping is attached as ``data['<kind>'].node_ids`` (list)
    so callers can map index → original string id when debugging or
    aligning with SEAL subgraphs.

    Reverse edges are **not** added — caller does
    ``T.ToUndirected()(data)`` or builds the reverse manually when
    feeding a HeteroConv.
    """

    data = HeteroData()
    x_c, comp_ids = encode_component_features(hcg)
    x_p, port_ids = encode_port_features(hcg)
    x_n, net_ids = encode_net_features(hcg)
    data["component"].x = x_c
    data["port"].x = x_p
    data["net"].x = x_n
    data["component"].node_ids = comp_ids
    data["port"].node_ids = port_ids
    data["net"].node_ids = net_ids

    # component → port structural edges (one per port)
    comp_idx = {nid: i for i, nid in enumerate(comp_ids)}
    port_idx = {nid: i for i, nid in enumerate(port_ids)}
    cp_src: list[int] = []
    cp_dst: list[int] = []
    for port_node_id, port in hcg.ports.items():
        if port.parent_component_id in comp_idx and port_node_id in port_idx:
            cp_src.append(comp_idx[port.parent_component_id])
            cp_dst.append(port_idx[port_node_id])
    data["component", "has_port", "port"].edge_index = torch.tensor(
        [cp_src, cp_dst], dtype=torch.long
    ).reshape(2, -1)

    # port → net edges
    pn_edge_index, pn_edge_attr = encode_port_net_edge_features(hcg)
    data["port", "connects", "net"].edge_index = pn_edge_index
    data["port", "connects", "net"].edge_attr = pn_edge_attr

    return data


# ---------------------------------------------------------------------------
# SealSubgraph → PyG Data (DRNL + features + target flag)
# ---------------------------------------------------------------------------


def _drnl_one_hot(label: int) -> list[float]:
    """``DRNL_LABEL_DIM`` (=17) buckets: 0..15 + overflow."""
    v = [0.0] * DRNL_LABEL_DIM
    if label < 0:
        idx = 0  # unreachable / sentinel — fold into 0
    elif label >= DRNL_LABEL_DIM:
        idx = DRNL_LABEL_DIM - 1  # overflow bucket
    else:
        idx = label
    v[idx] = 1.0
    return v


def seal_subgraph_to_pyg_data(
    sg: SealSubgraph,
    cur_hcg: HeteroCircuitGraph,
    *,
    label: int | None = None,
    label_source: str | None = None,
    task_type: str | None = None,
    group_id: str | None = None,
    drop_drnl: bool = False,
) -> Data:
    """One enclosing subgraph → one PyG :class:`Data` ready for the SEAL
    DGCNN head.

    Per plan §三.6 / §四 L2:

    - Node feature = DRNL[17] ⊕ raw port-or-net feat ⊕ target_flag[1]
      (note: port and net features have different widths — we pad to
      ``DRNL_LABEL_DIM + max(PORT_FEAT_DIM, NET_FEAT_DIM) + 1`` so the
      tensor is rectangular. The first 17 dims are DRNL, the last dim is
      target_flag; in between is the type-specific feat with the rest of
      the row left zero, allowing DGCNN to learn type discrimination
      from feature locality.)
    - Edge index is over the subgraph's (port, net) bipartite slice;
      undirected for SAGE-style aggregation (caller may decide otherwise).
    - ``data.target_port_idx`` / ``data.target_net_idx`` mark the two
      anchor nodes (always at position 0 for port-anchor, 1 for
      net-anchor by plan §三.6 step 4).
    - ``data.y`` / ``data.label_source`` / ``data.task_type`` /
      ``data.group_id`` carry the label_builder ground truth.
    """

    node_ids = list(sg.port_ids) + list(sg.net_ids)
    n_total = len(node_ids)
    feat_width = DRNL_LABEL_DIM + max(PORT_FEAT_DIM, NET_FEAT_DIM) + 1
    x = torch.zeros((n_total, feat_width), dtype=torch.float32)

    # DRNL one-hot. ``drop_drnl=True`` (P3.1 "去 DRNL" ablation) leaves
    # the DRNL slice as all-zeros — the model still sees a 68-d input
    # but the first 17 dims carry no info, isolating the contribution
    # of DRNL labels to the SEAL head.
    if not drop_drnl:
        for i, nid in enumerate(node_ids):
            drnl = sg.drnl_labels.get(nid, 0)
            x[i, :DRNL_LABEL_DIM] = torch.tensor(_drnl_one_hot(drnl))

    # Fill type-specific features (ports first, then nets)
    for i, nid in enumerate(sg.port_ids):
        if nid in cur_hcg.ports:
            p = cur_hcg.ports[nid]
            single_x, _ = _encode_single_port(p, cur_hcg)
            x[i, DRNL_LABEL_DIM : DRNL_LABEL_DIM + PORT_FEAT_DIM] = single_x
    offset = len(sg.port_ids)
    for j, nid in enumerate(sg.net_ids):
        if nid in cur_hcg.nets:
            n = cur_hcg.nets[nid]
            single_x = _encode_single_net(n, cur_hcg)
            x[offset + j, DRNL_LABEL_DIM : DRNL_LABEL_DIM + NET_FEAT_DIM] = single_x

    # target flag (last column)
    for i, nid in enumerate(node_ids):
        x[i, -1] = 1.0 if sg.is_target.get(nid, False) else 0.0

    # Edges — index into the joined [ports..., nets...] node list
    node_idx = {nid: i for i, nid in enumerate(node_ids)}
    src: list[int] = []
    dst: list[int] = []
    for src_id, dst_id in sg.edges:
        if src_id not in node_idx or dst_id not in node_idx:
            continue
        a, b = node_idx[src_id], node_idx[dst_id]
        # undirected — emit both directions
        src.append(a)
        dst.append(b)
        src.append(b)
        dst.append(a)
    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)

    data = Data(x=x, edge_index=edge_index)
    data.target_port_idx = (
        torch.tensor([node_idx.get(sg.target_port_id, -1)], dtype=torch.long)
    )
    data.target_net_idx = (
        torch.tensor([node_idx.get(sg.target_net_id, -1)], dtype=torch.long)
    )
    data.edge_present = torch.tensor([sg.edge_present], dtype=torch.bool)
    data.node_ids = node_ids
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float32)
    # PyG ``Data`` collation requires consistent attribute presence across
    # all samples in a batch. Always set string fields (defaulting to "")
    # and group_id (defaulting to "") so ``DataLoader`` can stack any mix
    # of WRONG_EDGE (group_id=None) and MISSING_EDGE (group_id=<id>) rows.
    data.label_source = label_source or ""
    data.task_type = task_type or ""
    data.group_id = group_id or ""
    return data


def _encode_single_port(
    p, hcg: HeteroCircuitGraph
) -> tuple[torch.Tensor, dict]:
    """Tiny per-port encoder used by SEAL Data construction —— factored
    out so encode_port_features doesn't have to rebuild sym_class_size
    for every subgraph. Returns a 1-D tensor of length PORT_FEAT_DIM."""

    x = torch.zeros(PORT_FEAT_DIM, dtype=torch.float32)
    n_pt = len(PORT_TYPE_TO_INDEX)
    n_ct = len(CTYPE_TO_INDEX)
    pt_idx = PORT_TYPE_TO_INDEX.get(p.port_type, PORT_TYPE_TO_INDEX["generic"])
    x[pt_idx] = 1.0
    pct_idx = CTYPE_TO_INDEX.get(p.parent_ctype, CTYPE_TO_INDEX["UNKNOWN"])
    x[n_pt + pct_idx] = 1.0
    cursor = n_pt + n_ct
    x[cursor + 0] = 1.0 if p.polarity_sensitive else 0.0
    x[cursor + 1] = 1.0 if p.is_power_port else 0.0
    x[cursor + 2] = 1.0 if p.is_ground_port else 0.0
    x[cursor + 3] = 1.0 if p.is_floating else 0.0
    x[cursor + 4] = _is_ref(p.side)
    cursor += 5
    cp_idx = CONNECTION_POLICY_TO_INDEX.get(p.connection_policy, 0)
    x[cursor + cp_idx] = 1.0
    cursor += len(CONNECTION_POLICY_TO_INDEX)
    if p.pin_number is not None:
        x[cursor + 0] = 1.0
        x[cursor + 1] = min(
            1.0, math.log1p(p.pin_number) / math.log(1 + 64)
        )
    cursor += 2
    # symmetry_class_size_inverse — count siblings on the same component
    sz = sum(
        1
        for q in hcg.ports.values()
        if q.parent_component_id == p.parent_component_id
        and q.symmetry_class_id == p.symmetry_class_id
    )
    x[cursor + 0] = 1.0 / max(1, sz)
    return x, {}


def _encode_single_net(n, hcg: HeteroCircuitGraph) -> torch.Tensor:
    x = torch.zeros(NET_FEAT_DIM, dtype=torch.float32)
    n_role = len(NET_ROLE_TO_INDEX)
    role_idx = NET_ROLE_TO_INDEX.get(n.role, NET_ROLE_TO_INDEX["unknown"])
    x[role_idx] = 1.0
    cursor = n_role
    degree = sum(1 for e in hcg.edges if e.dst_net_id == n.node_id)
    x[cursor + 0] = math.log1p(degree) / math.log(1 + 32)
    x[cursor + 1] = 1.0 if n.is_power_rail else 0.0
    if n.voltage_hint is not None:
        x[cursor + 2] = float(n.voltage_hint)
        x[cursor + 3] = 1.0
    x[cursor + 4] = _is_ref(n.side)
    return x


__all__ = [
    "encode_component_features",
    "encode_port_features",
    "encode_net_features",
    "encode_port_net_edge_features",
    "to_hetero_data",
    "seal_subgraph_to_pyg_data",
]
