"""GNN 模块 · SEAL DGCNN 主头（plan §四 L2 · P2.5 + P3 共用）

按 plan §四 L2 描述的"主任务 · 哪根线接错了"模型：

    每个节点特征 = DRNL_label[17] ⊕ z_(port|net)[..] ⊕ target_flag[1]
                              ↓
              3-layer GCN (tanh, hidden=64) with concat residual
                              ↓
                  SortPooling (k = 30)
                              ↓
                  1-D Conv (output_channels=32 → 1)
                              ↓
                          Sigmoid (in BCEWithLogits)
              ⇒ P(edge_correct) ∈ [0, 1]

Input: ``Data(x=[N, feat_width], edge_index=[2,E], batch=[N])``，feat_width
由 :func:`pyg_converter.seal_subgraph_to_pyg_data` 决定（当前 = DRNL_LABEL_DIM
+ max(PORT_FEAT_DIM, NET_FEAT_DIM) + 1 = 68）。

输出: ``logits`` shape ``[B]``（per-graph 概率前的 logit；训练用
``BCEWithLogitsLoss``）。

依赖 torch + torch_geometric。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_sort_pool  # type: ignore[import-untyped]


class SealDGCNN(nn.Module):
    """SEAL link-prediction backbone (Zhang & Chen 2018, DGCNN variant).

    Args:
        in_channels: input node feature width. Default 68 = 17 (DRNL) +
            50 (PORT_FEAT_DIM) + 1 (target flag).
        hidden_channels: per-GCN-layer width (default 32 — small enough
            for CPU pretraining on the 155-circuit SpiceNetlist).
        num_layers: number of GCN layers (default 3, per DGCNN).
        sort_k: SortPooling cap (top-k nodes by last-layer activation,
            default 30 — matches plan §四 L2).
        conv1_channels: width of the first 1-D Conv after SortPooling.
        dense_hidden: hidden width of the final MLP head.
        dropout: dropout probability before the final logit.
    """

    def __init__(
        self,
        in_channels: int = 68,
        hidden_channels: int = 32,
        num_layers: int = 3,
        sort_k: int = 30,
        conv1_channels: int = 16,
        dense_hidden: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.sort_k = sort_k

        # GCN stack. We follow the original DGCNN trick of using
        # 1-channel output for the LAST GCN layer so the SortPooling
        # "score" is a single scalar per node, then concatenating all
        # layer outputs as the per-node descriptor.
        gcn_layers: list[GCNConv] = []
        widths = [in_channels] + [hidden_channels] * (num_layers - 1) + [1]
        for i in range(num_layers):
            gcn_layers.append(GCNConv(widths[i], widths[i + 1]))
        self.gcns = nn.ModuleList(gcn_layers)

        # Concatenated descriptor width = sum of post-GCN widths
        total_channels = sum(widths[1:])  # hidden_channels*(L-1) + 1

        # SortPool gives [B, sort_k * total_channels] when flattened
        # (PyG returns [B*sort_k, total_channels] then we reshape).
        # We then 1-D conv over the sort_k dimension.
        self.conv1d = nn.Conv1d(
            in_channels=total_channels,
            out_channels=conv1_channels,
            kernel_size=2,
            stride=1,
        )
        conv1_out_len = (sort_k - 2) + 1  # default stride 1 / kernel 2 / no pad

        self.dense = nn.Sequential(
            nn.Linear(conv1_channels * conv1_out_len, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``logits`` of shape ``[B]`` (apply BCEWithLogits)."""

        # 1. GCN stack — collect per-layer activations for concat.
        layer_outs: list[torch.Tensor] = []
        h = x
        for conv in self.gcns:
            h = conv(h, edge_index)
            h = torch.tanh(h)
            layer_outs.append(h)
        concat = torch.cat(layer_outs, dim=-1)  # [N, sum(widths)]

        # 2. SortPooling sorts nodes per-graph by the LAST column (DGCNN
        #    convention) and keeps top sort_k.
        pooled = global_sort_pool(concat, batch, k=self.sort_k)
        # pooled shape: [B, sort_k * total_channels]
        b = pooled.size(0)
        pooled = pooled.view(b, self.sort_k, -1)  # [B, sort_k, total_channels]
        pooled = pooled.permute(0, 2, 1)  # [B, total_channels, sort_k]

        # 3. 1-D Conv across the sort_k axis
        h = F.relu(self.conv1d(pooled))  # [B, conv1_channels, sort_k - 1]
        h = h.flatten(start_dim=1)        # [B, conv1_channels * (sort_k - 1)]

        # 4. Dense head
        logits: torch.Tensor = self.dense(h).squeeze(-1)  # [B]
        return logits


def predict_prob(model: SealDGCNN, batch) -> torch.Tensor:
    """Convenience: forward + sigmoid. Returns ``P(edge_correct) ∈ [0, 1]``
    of shape ``[B]``."""

    logits = model(batch.x, batch.edge_index, batch.batch)
    return torch.sigmoid(logits)


__all__ = ["SealDGCNN", "predict_prob"]
