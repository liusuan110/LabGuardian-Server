"""GNN 模块 · SpiceNetlist masked-edge SEAL pretraining dataset（P2.5）

self-supervised link prediction：对每个 SpiceNetlist 电路，
- positive 样本：每条真实 ``(port, net)`` 边 → SealSubgraph(label=1)
- negative 样本：随机一对未连接的 ``(port, net)`` → SealSubgraph(label=0)

masked-edge 训练范式（GNN-ACLP 论文 §4.1）：positive 子图在抽取时**会
把自己排除**（plan §三.6 SEAL 守则）—— 我们的
:func:`extract_seal_subgraph` 已经保证这一点。

依赖 torch + torch_geometric (``[gnn]`` extra)。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data  # type: ignore[import-untyped]

from app.domain.gnn.pyg_converter import seal_subgraph_to_pyg_data
from app.domain.gnn.seal_subgraph import extract_seal_subgraph

if TYPE_CHECKING:  # pragma: no cover
    from app.domain.gnn.hetero_circuit import HeteroCircuitGraph
    from app.domain.gnn.spicenetlist_loader import SpiceNetlistCircuit


@dataclass(frozen=True)
class _RowSpec:
    """One row of the pretrain dataset before tensorisation."""

    circuit_id: str
    port_id: str
    net_id: str
    label: int  # 1 positive, 0 negative


# ---------------------------------------------------------------------------
# Positive / negative sampling per circuit
# ---------------------------------------------------------------------------


def _enumerate_pos_neg_pairs(
    hcg: HeteroCircuitGraph,
    rng: random.Random,
    *,
    negatives_per_positive: float = 1.0,
    max_pairs_per_circuit: int | None = None,
) -> list[tuple[str, str, int]]:
    """Returns ``[(port_id, net_id, label), ...]``.

    Positives: every observed ``(port, net)`` edge (label=1).
    Negatives: ``negatives_per_positive × N_pos`` randomly sampled non-edges
    over the cartesian (port × net) space, excluding existing edges.

    Cap per-circuit pairs at ``max_pairs_per_circuit`` (None = no cap) so
    huge circuits (237 edges) don't dominate training.
    """

    existing: set[tuple[str, str]] = {
        (e.src_port_id, e.dst_net_id) for e in hcg.edges
    }
    positives = list(existing)
    n_pos = len(positives)
    n_neg = int(round(n_pos * negatives_per_positive))

    port_ids = list(hcg.ports)
    net_ids = list(hcg.nets)
    negatives: list[tuple[str, str]] = []
    # Sample without replacement; bound attempts to avoid infinite loops
    # on dense circuits where most pairs are edges.
    max_attempts = max(64, 4 * n_neg)
    attempts = 0
    seen_neg: set[tuple[str, str]] = set()
    while len(negatives) < n_neg and attempts < max_attempts:
        attempts += 1
        if not port_ids or not net_ids:
            break
        pair = (rng.choice(port_ids), rng.choice(net_ids))
        if pair in existing or pair in seen_neg:
            continue
        seen_neg.add(pair)
        negatives.append(pair)

    out: list[tuple[str, str, int]] = []
    for port_id, net_id in positives:
        out.append((port_id, net_id, 1))
    for port_id, net_id in negatives:
        out.append((port_id, net_id, 0))

    if max_pairs_per_circuit is not None and len(out) > max_pairs_per_circuit:
        rng.shuffle(out)
        out = out[:max_pairs_per_circuit]

    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SpiceNetlistPretrainDataset(TorchDataset):
    """In-memory PyG dataset of SEAL Data rows for self-supervised pretrain.

    Args:
        circuits: list of :class:`SpiceNetlistCircuit` (typically all 155
            from the SpiceNetlist JSON dir, or a fold subset).
        negatives_per_positive: ratio of negative to positive samples
            per circuit (default 1.0 for balanced BCE).
        num_hops: SEAL enclosing subgraph radius (default 2; GNN-ACLP §3).
        max_pairs_per_circuit: cap each circuit's contribution so a giant
            circuit doesn't dominate (default None = uncapped).
        seed: deterministic RNG seed for negative sampling.

    Notes:
        - We do NOT call ``super().__init__()`` with a root because we
          process in-memory (no on-disk PyG cache; the SpiceNetlist
          source is already on disk and parsing is cheap).
        - cur_hcg in :func:`seal_subgraph_to_pyg_data` is the same hcg
          (each SpiceNetlist circuit IS the ground-truth graph; pretrain
          doesn't separate ref / cur).
    """

    def __init__(
        self,
        circuits: list[SpiceNetlistCircuit],
        *,
        negatives_per_positive: float = 1.0,
        num_hops: int = 2,
        max_pairs_per_circuit: int | None = 60,
        seed: int = 0,
    ):
        super().__init__()
        self._data_list: list[Data] = []
        rng = random.Random(seed)
        for circ in circuits:
            pairs = _enumerate_pos_neg_pairs(
                circ.hcg,
                rng,
                negatives_per_positive=negatives_per_positive,
                max_pairs_per_circuit=max_pairs_per_circuit,
            )
            for port_id, net_id, label in pairs:
                try:
                    sg = extract_seal_subgraph(
                        circ.hcg,
                        port_id,
                        net_id,
                        num_hops=num_hops,
                        edge_present=bool(label),
                    )
                except Exception:
                    # Skip subgraphs we can't extract (rare malformed pairs)
                    continue
                d = seal_subgraph_to_pyg_data(sg, circ.hcg, label=label)
                d.circuit_id = circ.circuit_id
                self._data_list.append(d)

    # PyG Dataset API ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Data:
        if isinstance(idx, slice):
            sub = SpiceNetlistPretrainDataset.__new__(
                SpiceNetlistPretrainDataset
            )
            sub._data_list = self._data_list[idx]
            return sub  # type: ignore[return-value]
        return self._data_list[idx]


__all__ = [
    "SpiceNetlistPretrainDataset",
]
