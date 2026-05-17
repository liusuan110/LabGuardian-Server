"""GNN 模块 · L1 HeteroConv Backbone（P3.1 · plan §四 L1）

按 plan §四 L1 描述的"共享 backbone"：

    NodeEncoder.comp  [Nc, 30]  → [Nc, 128]
    NodeEncoder.port  [Np, 50]  → [Np, 128]
    NodeEncoder.net   [Nn, 11]  → [Nn, 128]
              ↓
    HeteroConv(SAGEConv) × 3   with residual + LayerNorm
        edges: comp↔port, port↔net  (后者通过 T.ToUndirected
        自动得到反向)
              ↓
       z_comp / z_port / z_net  (128-d each)

L1 输出被 L2 SEAL DGCNN 主头消费（plan §四 L2 节点输入 =
``DRNL[17] ⊕ z[128] ⊕ target_flag[1]``）。**P3.1 仅交付独立 backbone
模块 + 测试**；与 SEAL head 的端到端拼装推到 P3.2，原因：

- L1 把 SEAL 节点输入从 ``68``-d 变成 ``146``-d，P2.5 backbone 权重
  无法直接迁移（需要重新预训练或新增 projection 层）
- L2 当前的 row-level ``FlatSealDataset`` 不知道"哪些 row 来自同一个
  cur_hcg"，要让 L1 forward 不重复跑就得换成 sample-level batching。
  那是单独一块工程（plan 中没明确，我们先把 backbone 模块完整交付
  作为 P3.2 集成的依赖）

依赖 torch + torch_geometric (``[gnn]`` extra)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData  # type: ignore[import-untyped]
from torch_geometric.nn import HeteroConv, SAGEConv  # type: ignore[import-untyped]

from app.domain.gnn.graph_schema import (
    COMPONENT_FEAT_DIM,
    NET_FEAT_DIM,
    PORT_FEAT_DIM,
)


class HeteroNodeEncoder(nn.Module):
    """Per-type linear encoder that lifts raw component/port/net features
    to a shared hidden dim.

    Layout (plan §四 L1):
        comp [Nc, 30] → linear → tanh → [Nc, hidden_dim]
        port [Np, 50] → linear → tanh → [Np, hidden_dim]
        net  [Nn, 11] → linear → tanh → [Nn, hidden_dim]

    Tanh activation keeps activations bounded so the downstream SAGE
    layers don't blow up at init.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enc_comp = nn.Linear(COMPONENT_FEAT_DIM, hidden_dim)
        self.enc_port = nn.Linear(PORT_FEAT_DIM, hidden_dim)
        self.enc_net = nn.Linear(NET_FEAT_DIM, hidden_dim)

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        return {
            "component": torch.tanh(self.enc_comp(data["component"].x)),
            "port": torch.tanh(self.enc_port(data["port"].x)),
            "net": torch.tanh(self.enc_net(data["net"].x)),
        }


class HeteroSAGEBackbone(nn.Module):
    """Plan §四 L1 — 3-layer HeteroConv(SAGE) stack with residual +
    LayerNorm on every layer.

    Edge types consumed:
        ('component', 'has_port', 'port')   — structural
        ('port', 'connects', 'net')         — electrical
        ('port', 'rev_has_port', 'component')  ← added by ToUndirected
        ('net', 'rev_connects', 'port')        ← added by ToUndirected

    The caller is expected to apply :class:`torch_geometric.transforms.ToUndirected`
    to the input HeteroData before forward (or wire reverse edges manually).

    Output: dict ``{node_type: tensor[N_type, hidden_dim]}`` — the same
    shape contract as :class:`HeteroNodeEncoder`, ready to feed L2 SEAL.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = HeteroNodeEncoder(hidden_dim=hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("component", "has_port", "port"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("port", "connects", "net"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("port", "rev_has_port", "component"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("net", "rev_connects", "port"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)
            # One LayerNorm per node type per layer
            self.norms.append(
                nn.ModuleDict(
                    {
                        kind: nn.LayerNorm(hidden_dim)
                        for kind in ("component", "port", "net")
                    }
                )
            )

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = self.encoder(data)
        # Build the edge_index_dict that HeteroConv needs
        edge_index_dict = {
            et: data[et].edge_index
            for et in data.edge_types
        }
        for conv, norm in zip(self.convs, self.norms):
            new_x = conv(x_dict, edge_index_dict)
            # Residual + LayerNorm per node type
            out: dict[str, torch.Tensor] = {}
            for kind in x_dict:
                if kind not in new_x:
                    # Node type with no incoming edges — keep prev embedding
                    out[kind] = x_dict[kind]
                    continue
                h = new_x[kind] + x_dict[kind]
                h = norm[kind](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                out[kind] = h
            x_dict = out
        return x_dict


def embeddings_for_subgraph(
    backbone_out: dict[str, torch.Tensor],
    port_node_ids: list[str],
    net_node_ids: list[str],
    hetero_data_node_ids: dict[str, list[str]],
) -> torch.Tensor:
    """Look up the L1 ``z`` embeddings for the ports and nets that appear
    in one :class:`SealSubgraph`.

    Args:
        backbone_out: ``HeteroSAGEBackbone.forward`` output
            (``{node_type: [N, hidden]}``).
        port_node_ids / net_node_ids: the ``SealSubgraph.port_ids /
            net_ids`` lists.
        hetero_data_node_ids: the ``data['<kind>'].node_ids`` lists
            saved by :func:`pyg_converter.to_hetero_data` — needed to
            map string node_id back to row index.

    Returns:
        Tensor of shape ``[len(port_node_ids) + len(net_node_ids), hidden_dim]``
        in the canonical SEAL order (ports first, then nets), ready to
        be concatenated with DRNL + target_flag and fed into the
        :class:`SealDGCNN` main head.
    """

    port_idx = {nid: i for i, nid in enumerate(hetero_data_node_ids["port"])}
    net_idx = {nid: i for i, nid in enumerate(hetero_data_node_ids["net"])}

    rows: list[torch.Tensor] = []
    for pid in port_node_ids:
        if pid in port_idx:
            rows.append(backbone_out["port"][port_idx[pid]])
        else:
            rows.append(torch.zeros(backbone_out["port"].shape[1]))
    for nid in net_node_ids:
        if nid in net_idx:
            rows.append(backbone_out["net"][net_idx[nid]])
        else:
            rows.append(torch.zeros(backbone_out["net"].shape[1]))
    return torch.stack(rows, dim=0)


__all__ = [
    "HeteroNodeEncoder",
    "HeteroSAGEBackbone",
    "embeddings_for_subgraph",
]
