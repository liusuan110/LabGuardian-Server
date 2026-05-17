"""GNN 模块 · CircuitMatchNet（P3 MVP）

P3 多任务模型。**当前 MVP 范围**（详见 plan §四 + 协商决定）：

- **L2 SEAL DGCNN 主头** —— 复用 :class:`SealDGCNN`，输入为
  :func:`pyg_converter.seal_subgraph_to_pyg_data` 的 ``Data``，输出
  per-edge logit。**同一份权重同时**服务：
  - 点式 ``WRONG_EDGE``（P(edge_correct) 二分类）
  - 组式 ``MISSING_EDGE``（同 group_id 内 softmax-style 排序 → top-k）
- 多任务 loss：BCE 主头（共享权重，task_type 不切分）+ 推理时按
  task_type 分桶评估。

**MVP 不实现**（plan §四 完整 / 后续 P3 迭代）：

- L1 ``HeteroConv`` 共享 port/net embedding backbone
- L4 辅助 head（graph_similarity / error_type / hotspot / progress）
- 主头多 loss 加权（``λ_seal=1.5, λ_tgt=1.0, λ_hot=0.5, λ_sim=0.4, λ_err=0.3``）

这些 head 加入后**不需要**改 CircuitMatchNet 的 forward 接口 ——
:meth:`forward` 已返回一个 dict，新增 head 时往 dict 加键即可。

加载 P2.5 backbone：

>>> from app.domain.gnn import CircuitMatchNet
>>> model = CircuitMatchNet.from_pretrained_backbone(
...     "checkpoints/pretrain_v1/backbone.pt"
... )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from app.domain.gnn.seal_dgcnn import SealDGCNN


class CircuitMatchNet(nn.Module):
    """Multi-task wrapper. **MVP**: just the SEAL DGCNN head."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int = 32,
        sort_k: int = 30,
        num_layers: int = 3,
        dense_hidden: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.seal_head = SealDGCNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            sort_k=sort_k,
            dense_hidden=dense_hidden,
            dropout=dropout,
        )
        # Save constructor kwargs so checkpoints can self-describe
        self.config = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "sort_k": sort_k,
            "num_layers": num_layers,
            "dense_hidden": dense_hidden,
            "dropout": dropout,
        }

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Returns a dict so future heads can add keys without breaking
        the public interface.

        Keys present in MVP:
            ``"seal_logits"`` — shape ``[B]``, raw logits for both
            WRONG_EDGE and MISSING_EDGE rows. Apply ``BCEWithLogitsLoss``
            for the main head; apply ``sigmoid`` for ranking.
        """

        return {
            "seal_logits": self.seal_head(x, edge_index, batch),
        }

    # ----- Pretrained checkpoint loading ------------------------------------

    @classmethod
    def from_pretrained_backbone(
        cls,
        checkpoint_path: str | Path,
        *,
        strict: bool = True,
        override_in_channels: int | None = None,
    ) -> CircuitMatchNet:
        """Build a :class:`CircuitMatchNet` and load P2.5
        ``SealDGCNN`` weights into its main head.

        The P2.5 ``scripts/gnn_pretrain_seal.py`` writes checkpoints of
        the form::

            {"state_dict": <SealDGCNN state>,
             "hidden": int, "sort_k": int, "in_channels": int,
             "best_val_auc": float, "fold": int}

        Args:
            strict: if False, allow size mismatches (e.g. when finetuning
                with a different ``in_channels``).
            override_in_channels: if provided, use this instead of the
                pretrain's ``in_channels``. The first GCN layer's
                ``lin.weight`` is re-initialised (P2.5 used SpiceNetlist
                feature width which matches P1; this is for future use).
        """

        ckpt = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        in_channels = override_in_channels or int(ckpt["in_channels"])
        model = cls(
            in_channels=in_channels,
            hidden_channels=int(ckpt.get("hidden", 32)),
            sort_k=int(ckpt.get("sort_k", 30)),
        )
        # The state_dict keys in the pretrain are SealDGCNN's (no
        # `seal_head.` prefix), so re-key them.
        sd = ckpt["state_dict"]
        renamed = {f"seal_head.{k}": v for k, v in sd.items()}

        # If the first GCN layer's width differs, drop it from the load
        # set so it stays freshly initialised.
        own_state = model.state_dict()
        if not strict:
            renamed = {
                k: v for k, v in renamed.items()
                if k in own_state and v.shape == own_state[k].shape
            }
        model.load_state_dict(renamed, strict=strict)
        return model

    # ----- Checkpointing ----------------------------------------------------

    def save(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
                **(extra or {}),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> CircuitMatchNet:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model


__all__ = ["CircuitMatchNet"]
