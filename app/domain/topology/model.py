"""GNN-A topology classifier model — GraphSAGE + global mean pool + MLP head.

## Architecture rationale (v2)

For a 7-way classification on small bipartite graphs (10-20 nodes typical)
with 23-dim node features, we use a 3-layer model:

  * 3× ``SAGEConv`` layers with hidden dim 96 — 3 hops covers the
    full op-amp feedback loop (INV → R/C → VOUT → opamp → INV) so
    each INV node's embedding sees its complete feedback signature.
    v1 used 2 layers + dim 64; v2 deepened to better distinguish
    the UA741 three-tribe (inverting / summing / integrator).
  * ``global_mean_pool`` → graph-level vector (no max/sum bias).
  * 2-layer MLP (96 → 48 → 7) with ReLU + dropout.

Total parameters: ~50K. Forward latency on Intel iGPU INT8 (target Phase 3
deployment): well under 5ms even with batch=1.

## Why GraphSAGE (vs GCN / GIN / GAT)

  * **GraphSAGE** — mean aggregator, no normalization weights, OpenVINO
    has first-class support. Robust to varying graph sizes.
  * GCN needs Laplacian normalization which means we'd need to add
    self-loops + degree normalization in preprocessing; more places to
    drift between PyTorch and OpenVINO.
  * GIN is a touch more expressive but adds another linear layer per
    aggregation; not worth the parameter cost for 7-way classification
    on bipartite graphs.
  * GAT requires attention kernels which on NPU INT8 are flaky.

## Future evolution

  * Phase 1.5: replace mean pool with a small set-transformer if
    diff_pair vs common_emitter accuracy plateaus.
  * Phase 2: add edge features (pin labels) when we have new topologies
    that genuinely need pin-level discrimination.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool

from app.domain.topology.features import FEATURE_DIM
from app.domain.topology.labels import TOPOLOGY_LABELS


# Hidden dim chosen so total params land near 50K (well under 100K
# budget). v1 used 64; v2 widened to 96 alongside deepening to 3
# SAGEConv layers, to give the model more capacity for distinguishing
# the UA741 three-tribe.
DEFAULT_HIDDEN_DIM = 96
DEFAULT_DROPOUT = 0.2
# v2 — 3 layers covers the op-amp feedback loop in full (INV →
# feedback element → VOUT → opamp.pin6 → opamp → opamp.pin2 → INV).
DEFAULT_NUM_LAYERS = 3

NUM_CLASSES = len(TOPOLOGY_LABELS)


class TopologyClassifier(nn.Module):
    """GraphSAGE-based circuit topology classifier (GNN-A).

    Args:
        in_dim: Per-node input feature dimension. Defaults to
            :data:`app.domain.topology.features.FEATURE_DIM` (21).
        hidden_dim: SAGEConv + first-MLP-layer width.
        num_classes: Output softmax dimension. Defaults to
            ``len(TOPOLOGY_LABELS)``.
        dropout: Dropout probability between MLP layers.
    """

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
        num_layers: int = DEFAULT_NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_layers = num_layers

        # Stack of SAGEConv layers. First maps in_dim → hidden_dim,
        # subsequent layers stay at hidden_dim → hidden_dim.
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim_i = in_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_dim_i, hidden_dim, aggr="mean"))

        # Classification head: graph_embedding -> hidden -> num_classes.
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: ``(N, in_dim)`` node features.
            edge_index: ``(2, E)`` edge index (PyG convention).
            batch: ``(N,)`` int64 batch assignment vector.

        Returns:
            ``(B, num_classes)`` logits (un-softmaxed). Wrap with
            ``F.cross_entropy`` for training loss or ``F.softmax`` for
            inference confidences.
        """
        # Message passing — N layers; dropout between layers (except last).
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.relu(h)
            if i < len(self.convs) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Graph-level pooling
        g = global_mean_pool(h, batch)  # (B, hidden_dim)

        # Classification head
        g = self.fc1(g)
        g = F.relu(g)
        g = F.dropout(g, p=self.dropout, training=self.training)
        return self.fc_out(g)

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Inference helper that returns softmax probabilities."""
        self.eval()
        logits = self.forward(x, edge_index, batch)
        return F.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Return the total number of learnable parameters (sanity check
        for ``< 100K`` Phase 1 design goal)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = [
    "DEFAULT_DROPOUT",
    "DEFAULT_HIDDEN_DIM",
    "NUM_CLASSES",
    "TopologyClassifier",
]
