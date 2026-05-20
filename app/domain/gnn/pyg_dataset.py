"""GNN 模块 · PyG Dataset / DataLoader（P2）

把 P1 dataset_builder 写盘的 ``labels/<ref_id>/<sample_id>.json`` +
``splits/{train,val,test}.json`` 喂给 PyG 训练循环。

**核心思路**：训练时的最小可消费单位是**一个 SealSubgraph → 一个 PyG Data**
（plan §四 L2 主头按候选边逐条评分）。所以 dataset 需要把"label 文件级"
的样本展平成"row 级"，每个 row 就是一个 PyG Data。

API:

- :class:`RefRegistry` —— 把 ref_id → (payload, subtype dict) 注入；
  支持 reconstruct cur_hcg = apply_perturbation(op, ref, seed)
- :class:`FlatSealDataset` —— PyG ``Dataset`` 子类；
  - ``__init__(labels_dir, refs_registry, split_entries)``
  - ``__getitem__(i)`` 返回一个 PyG ``Data``，
  - ``len()`` = 全 split 内所有 sample 总和 row 数

确定性：同 (labels_dir, splits 文件) → 同 row 顺序。Random 抽样由
``torch.utils.data.DataLoader`` 的 sampler 控制。

依赖：torch + torch_geometric (pyproject ``[gnn]`` extra)。
"""

from __future__ import annotations

import json
import random as _random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from torch_geometric.data import Data, Dataset  # type: ignore[import-untyped]

from app.domain.gnn.label_builder import deserialize_label_build_result
from app.domain.gnn.perturbation import get_perturbation
from app.domain.gnn.port_graph import build_from_logical_reference
from app.domain.gnn.pyg_converter import seal_subgraph_to_pyg_data

if TYPE_CHECKING:  # pragma: no cover
    from app.domain.gnn.hetero_circuit import HeteroCircuitGraph
    from app.domain.gnn.label_builder import LabelBuildResult


# ---------------------------------------------------------------------------
# RefRegistry — ref_id → payload + subtype dict (mirrors RefSpec for loader)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefEntry:
    ref_id: str
    payload_path: Path
    subtype_by_source_id: dict[str, str] = field(default_factory=dict)


@dataclass
class RefRegistry:
    """In-memory map ref_id → :class:`RefEntry` + cached ref_hcg.

    The cache is per-instance and bounded to the number of refs (usually
    small — 4 for the P1 acceptance dataset). For huge ref pools, wrap
    ``ref_hcg(ref_id)`` in ``functools.lru_cache``.
    """

    entries: dict[str, RefEntry] = field(default_factory=dict)
    _ref_hcg_cache: dict[str, HeteroCircuitGraph] = field(
        default_factory=dict, repr=False
    )

    def register(self, entry: RefEntry) -> None:
        self.entries[entry.ref_id] = entry

    def from_config_dict(self, refs_cfg: list[dict]) -> None:
        """Populate from the JSON config used by scripts.gnn_generate_dataset
        (each entry has ``ref_id`` / ``payload_path`` / optional subtypes
        via the sibling ``subtypes_by_ref_id`` map).

        Usage::

            reg = RefRegistry()
            reg.from_config_dict(cfg["refs"], cfg.get("subtypes_by_ref_id", {}))
        """

        raise NotImplementedError(
            "use ``populate(refs, subtypes_by_ref_id)`` directly"
        )

    def populate(
        self,
        refs: list[dict],
        subtypes_by_ref_id: dict[str, dict[str, str]] | None = None,
    ) -> None:
        st = subtypes_by_ref_id or {}
        for r in refs:
            self.register(
                RefEntry(
                    ref_id=r["ref_id"],
                    payload_path=Path(r["payload_path"]),
                    subtype_by_source_id=dict(st.get(r["ref_id"], {})),
                )
            )

    def ref_hcg(self, ref_id: str) -> HeteroCircuitGraph:
        cached = self._ref_hcg_cache.get(ref_id)
        if cached is not None:
            return cached
        entry = self.entries[ref_id]
        payload = json.loads(entry.payload_path.read_text(encoding="utf-8"))
        hcg = build_from_logical_reference(
            payload,
            extra_subtypes_by_source_id=entry.subtype_by_source_id or None,
        )
        if entry.subtype_by_source_id:
            hcg.metadata["subtype_by_source_id"] = dict(
                entry.subtype_by_source_id
            )
        self._ref_hcg_cache[ref_id] = hcg
        return hcg


# ---------------------------------------------------------------------------
# Cur HCG reconstruction (replay perturbation from cur_metadata)
# ---------------------------------------------------------------------------


def _perturbation_name_from_chain(chain: list[str]) -> str:
    """Best-effort extraction of the original op name from
    ``perturbation_chain[0]``. Chain entries are like
    ``"wrong_connection:R1.pin1:VIN→GND"`` or ``"chained:"``.

    Falls back to ``"identity"`` if the chain is empty (perturbation
    fell through to identity)."""

    if not chain:
        return "identity"
    head = chain[0]
    return head.split(":", 1)[0] if ":" in head else head


def reconstruct_cur_hcg(
    ref_hcg: HeteroCircuitGraph,
    cur_metadata: dict,
    subtype_by_source_id: dict[str, str] | None = None,
) -> HeteroCircuitGraph:
    """Replay the perturbation recorded in ``cur_metadata`` to get the
    same cur_hcg the label_builder saw. Bit-exact because we use the
    same ``seed`` field saved by dataset_builder."""

    seed = int(cur_metadata.get("seed", 0))
    pname = cur_metadata.get("perturbation_name") or _perturbation_name_from_chain(
        cur_metadata.get("perturbation_chain", []) or []
    )
    op = get_perturbation(pname)
    rng = _random.Random(seed)
    perturbed = op.apply(
        ref_hcg,
        rng,
        subtype_by_source_id=subtype_by_source_id or None,
    )
    return perturbed.cur_hcg


# ---------------------------------------------------------------------------
# Flat SEAL Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RowKey:
    """Identifier for one row in the flat dataset."""

    ref_id: str
    sample_id: str
    row_idx: int


class FlatSealDataset(Dataset):
    """Iterate every SealSample as a PyG ``Data`` row.

    Designed for the SEAL main head training loop (plan §四 L2): given
    a row, the model produces ``P(edge_correct)`` for that one candidate
    edge.

    Construction is cheap (just builds the row index by scanning split
    entries' file headers). Actual feature tensorization is lazy in
    :meth:`__getitem__`, with cur_hcg replay cached via
    :func:`functools.lru_cache` (default 64 most-recent samples).

    Args:
        labels_dir: ``<dataset>/labels`` directory.
        refs: :class:`RefRegistry` carrying ref payload paths + subtypes.
        split_entries: list of ``"<ref_id>/<sample_id>"`` strings (as
            written by ``app.domain.gnn.splits.write_splits``).
        cur_cache_size: LRU cache of cur HCGs (key = (ref_id, sample_id)).
            Larger = faster batching at the cost of memory.
    """

    def __init__(
        self,
        labels_dir: Path,
        refs: RefRegistry,
        split_entries: list[str],
        *,
        cur_cache_size: int = 64,
        drop_drnl: bool = False,
    ):
        super().__init__()
        self.labels_dir = labels_dir
        self.refs = refs
        self.drop_drnl = drop_drnl
        # Per-sample row count index — read each JSON header once.
        self._rows: list[_RowKey] = []
        self._sample_paths: dict[tuple[str, str], Path] = {}
        for entry in split_entries:
            ref_id, sample_id = entry.split("/", 1)
            path = labels_dir / ref_id / f"{sample_id}.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"split entry references missing label file: {path}"
                )
            self._sample_paths[(ref_id, sample_id)] = path
            n_rows = _count_samples_in_json(path)
            for row_idx in range(n_rows):
                self._rows.append(_RowKey(ref_id, sample_id, row_idx))

        # Build the LRU cur loader (closure over self.refs)
        @lru_cache(maxsize=cur_cache_size)
        def _load_cur_and_result(
            ref_id: str, sample_id: str
        ) -> tuple[HeteroCircuitGraph, LabelBuildResult, dict]:
            path = self._sample_paths[(ref_id, sample_id)]
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = deserialize_label_build_result(payload)
            ref_hcg = self.refs.ref_hcg(ref_id)
            cur_hcg = reconstruct_cur_hcg(
                ref_hcg,
                payload["cur_metadata"],
                subtype_by_source_id=(
                    self.refs.entries[ref_id].subtype_by_source_id or None
                ),
            )
            return cur_hcg, result, payload

        self._load_cur_and_result = _load_cur_and_result

    # PyG Dataset API ----------------------------------------------------

    def len(self) -> int:  # noqa: D401 — PyG Dataset uses snake-case len()
        return len(self._rows)

    def get(self, idx: int) -> Data:
        key = self._rows[idx]
        cur_hcg, result, _payload = self._load_cur_and_result(
            key.ref_id, key.sample_id
        )
        sample = result.samples[key.row_idx]
        sg = sample.subgraph
        data = seal_subgraph_to_pyg_data(
            sg,
            cur_hcg,
            label=sample.label,
            label_source=sample.label_source,
            task_type=sample.task_type,
            group_id=sample.group_id,
            drop_drnl=self.drop_drnl,
        )
        # Anchor identifiers for analytics / debug
        data.ref_id = key.ref_id
        data.sample_id = key.sample_id
        data.row_idx = key.row_idx
        return data

    # Convenience --------------------------------------------------------

    def by_task_type(self, task_type: str) -> FlatSealDataset:
        """Return a new dataset subset filtered to a single task_type.

        Done by inspecting each sample header — same I/O cost as init.
        Use case: PyG's `DataLoader` over only `WRONG_EDGE` rows for the
        SEAL main head, or only `MISSING_EDGE` rows for suggested_target.
        """

        filtered_entries: list[str] = []
        for (ref_id, sample_id), path in self._sample_paths.items():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if any(
                s.get("task_type") == task_type for s in payload.get("samples", [])
            ):
                filtered_entries.append(f"{ref_id}/{sample_id}")
        # Build a fresh dataset; it re-reads counts (cheap) but only over
        # the kept files. Row-level filtering happens during __getitem__
        # by checking sample.task_type in caller's collate if needed.
        return FlatSealDataset(
            self.labels_dir,
            self.refs,
            filtered_entries,
        )


def _count_samples_in_json(path: Path) -> int:
    """Cheap header read: parse JSON and return ``len(samples)``.

    Used by ``FlatSealDataset.__init__`` to size the row index without
    materialising every PyG ``Data``."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return len(payload.get("samples", []))


__all__ = [
    "RefEntry",
    "RefRegistry",
    "FlatSealDataset",
    "reconstruct_cur_hcg",
]
