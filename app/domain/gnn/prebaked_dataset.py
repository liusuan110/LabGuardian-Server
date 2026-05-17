"""GNN 模块 · Prebaked SEAL Dataset（P3.2 数据管线加速）

把训练循环里"每个 row 都重放 perturbation + 重建 cur_hcg + tensorize"
的开销一次性预跑掉，序列化成一个单 ``.pt`` blob；训练时只做
``torch.load`` + index 查找。

**为什么需要**: :class:`FlatSealDataset.__getitem__` 在 Windows CPU 上
~250 s/epoch（cur_hcg replay + 95% LRU miss）。Prebake 后实测 < 10 s/epoch
（pure tensor lookup）。一次预算 ~80 s + 60 MB 磁盘换 25 × epoch
speedup。Plan §九 ablation gate 在 23k-row 全数据集上才能稳定显现。

**保留的灵活性**: ``drop_drnl`` ablation 不需要重 bake —— 加载时按需把
首 17 维清零即可（DRNL 在 ``x[:, :17]``，layout 由
:func:`pyg_converter.seal_subgraph_to_pyg_data` 固定）。

格式 (PREBAKED_SCHEMA_VERSION = ``1``):

    {
      "version": 1,
      "n_rows": int,
      "feature_width": int,
      "entries": list[str],          # 与 data_list 平行；``f"{ref}/{sample}"``
      "row_indices": list[int],      # 同上；sample 内的 row 索引
      "data_list": list[Data],       # PyG Data，每个 Data 一行
      "config": {                    # 重现 prebake 时的关键参数
        "drop_drnl_at_bake": bool,
        "num_hops": int,
        "schema_version": str,       # label_builder.SCHEMA_VERSION
      },
    }

依赖 torch + torch_geometric（``[gnn]`` extra）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data  # type: ignore[import-untyped]

from app.domain.gnn.graph_schema import DRNL_LABEL_DIM
from app.domain.gnn.label_builder import (
    SCHEMA_VERSION,
    deserialize_label_build_result,
)
from app.domain.gnn.pyg_converter import seal_subgraph_to_pyg_data
from app.domain.gnn.pyg_dataset import RefRegistry, reconstruct_cur_hcg

if TYPE_CHECKING:  # pragma: no cover
    pass


log = logging.getLogger(__name__)

PREBAKED_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Bake side — convert labels/ directory → single .pt blob
# ---------------------------------------------------------------------------


@dataclass
class PrebakeStats:
    """Counters returned by :func:`prebake_to_disk` so callers can verify
    bake health (e.g. no silent drop of malformed samples)."""

    n_samples_processed: int = 0
    n_samples_failed_to_load: int = 0
    n_samples_failed_to_replay: int = 0
    n_samples_failed_to_extract: int = 0
    n_rows_baked: int = 0
    n_rows_dropped: int = 0


def prebake_to_disk(
    labels_dir: Path,
    refs: RefRegistry,
    entries: list[str],
    output_path: Path,
    *,
    drop_drnl: bool = False,
    num_hops: int = 2,
    log_every: int = 200,
) -> PrebakeStats:
    """Walk ``entries`` (each = ``"<ref_id>/<sample_id>"``), load each
    label JSON, replay cur_hcg **once**, tensorise every SealSample row,
    and persist the whole list to ``output_path`` as a single torch.save
    blob.

    Args:
        drop_drnl: bake **without** DRNL one-hot (rare; prefer to bake
            DRNL ON and toggle off at load time via
            ``PrebakedSealDataset(..., drop_drnl=True)``).
        num_hops: kept for parity with the label_builder; doesn't affect
            the bake since SealSubgraph extraction already happened in
            P1.
        log_every: progress log frequency (samples).

    Returns:
        :class:`PrebakeStats` — every counter must be inspected at scale
        so silent failures don't slip through.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = PrebakeStats()
    data_list: list[Data] = []
    out_entries: list[str] = []
    out_row_indices: list[int] = []
    feature_width: int | None = None

    for i, entry in enumerate(entries):
        ref_id, sample_id = entry.split("/", 1)
        label_file = labels_dir / ref_id / f"{sample_id}.json"
        if not label_file.is_file():
            log.warning("missing label file: %s", label_file)
            stats.n_samples_failed_to_load += 1
            continue
        try:
            payload = json.loads(label_file.read_text())
            result = deserialize_label_build_result(payload)
        except Exception as e:  # noqa: BLE001
            log.warning("failed to deserialize %s: %r", label_file, e)
            stats.n_samples_failed_to_load += 1
            continue

        try:
            ref_hcg = refs.ref_hcg(ref_id)
        except Exception as e:  # noqa: BLE001
            log.warning("failed to load ref %s: %r", ref_id, e)
            stats.n_samples_failed_to_replay += 1
            continue

        try:
            cur_hcg = reconstruct_cur_hcg(
                ref_hcg,
                payload["cur_metadata"],
                subtype_by_source_id=(
                    refs.entries[ref_id].subtype_by_source_id or None
                ),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("failed to replay cur for %s: %r", entry, e)
            stats.n_samples_failed_to_replay += 1
            continue

        for row_idx, sample in enumerate(result.samples):
            try:
                data = seal_subgraph_to_pyg_data(
                    sample.subgraph,
                    cur_hcg,
                    label=sample.label,
                    label_source=sample.label_source,
                    task_type=sample.task_type,
                    group_id=sample.group_id,
                    drop_drnl=drop_drnl,
                )
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "failed to tensorise %s row %d: %r", entry, row_idx, e
                )
                stats.n_samples_failed_to_extract += 1
                stats.n_rows_dropped += 1
                continue
            # Carry the entry/row_idx on the Data so downstream filtering /
            # debugging stays trivial (these are tiny string attrs — PyG
            # collates them as Python lists).
            data.ref_id = ref_id
            data.sample_id = sample_id
            data.row_idx = row_idx
            data_list.append(data)
            out_entries.append(entry)
            out_row_indices.append(row_idx)
            stats.n_rows_baked += 1
            if feature_width is None:
                feature_width = int(data.x.shape[1])

        stats.n_samples_processed += 1
        if (i + 1) % log_every == 0:
            log.info(
                "prebake progress: %d/%d samples, %d rows baked",
                i + 1, len(entries), stats.n_rows_baked,
            )

    blob: dict[str, Any] = {
        "version": PREBAKED_SCHEMA_VERSION,
        "n_rows": len(data_list),
        "feature_width": feature_width,
        "entries": out_entries,
        "row_indices": out_row_indices,
        "data_list": data_list,
        "config": {
            "drop_drnl_at_bake": drop_drnl,
            "num_hops": num_hops,
            "label_schema_version": SCHEMA_VERSION,
        },
    }
    torch.save(blob, output_path)
    log.info(
        "prebake done: %d rows from %d samples → %s (%.1f MB)",
        stats.n_rows_baked, stats.n_samples_processed, output_path,
        output_path.stat().st_size / 1e6,
    )
    return stats


# ---------------------------------------------------------------------------
# Read side — drop-in replacement for FlatSealDataset
# ---------------------------------------------------------------------------


class PrebakedSealDataset(TorchDataset):
    """Drop-in replacement for :class:`FlatSealDataset` that loads a
    pre-tensorised ``.pt`` blob written by :func:`prebake_to_disk`.

    `__getitem__` is now O(1) tensor indexing — no cur_hcg replay, no
    JSON parsing, no per-row tensorisation. Trade-off: ~60 MB of RAM
    for the full 2400-sample × 14-rows P1 dataset.

    Args:
        prebaked_path: path to the .pt blob.
        split_entries: optional ``list[str]`` of ``"<ref>/<sample>"`` to
            filter; if None, all rows are exposed.
        drop_drnl: ablation flag. Applied at load time by zeroing the
            first :data:`DRNL_LABEL_DIM` columns of ``x``. Bake the
            dataset ONCE with ``drop_drnl=False`` and toggle here for
            cheap ablations.
        verify_schema: if True, raise on version / config mismatch.
            Default True (catches stale blobs after schema bumps).
    """

    def __init__(
        self,
        prebaked_path: Path,
        split_entries: list[str] | None = None,
        *,
        drop_drnl: bool = False,
        verify_schema: bool = True,
    ):
        super().__init__()
        if not prebaked_path.is_file():
            raise FileNotFoundError(
                f"prebaked blob not found: {prebaked_path} — run "
                f"`scripts/gnn_prebake_dataset.py` first"
            )
        blob = torch.load(prebaked_path, weights_only=False)
        if verify_schema and blob.get("version") != PREBAKED_SCHEMA_VERSION:
            raise ValueError(
                f"prebaked blob version {blob.get('version')} != expected "
                f"{PREBAKED_SCHEMA_VERSION}; re-bake with the current code"
            )
        if verify_schema and blob["config"]["label_schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"label_builder schema {blob['config']['label_schema_version']} "
                f"!= current {SCHEMA_VERSION}; re-bake"
            )

        self._data_list: list[Data] = blob["data_list"]
        self._entries: list[str] = blob["entries"]
        self._row_indices: list[int] = blob["row_indices"]
        self.drop_drnl = drop_drnl
        self.feature_width = blob["feature_width"]

        # Index map: which rows belong to the requested split?
        if split_entries is None:
            self._indices = list(range(len(self._data_list)))
        else:
            wanted = set(split_entries)
            self._indices = [
                i for i, e in enumerate(self._entries) if e in wanted
            ]
            if not self._indices:
                raise ValueError(
                    "split_entries selected zero rows from the prebaked "
                    "blob; check ref_id formatting (expected `<ref>/<sample>`)"
                )

    # Torch Dataset API --------------------------------------------------

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> Data:
        data = self._data_list[self._indices[i]]
        if not self.drop_drnl:
            return data
        # Cheap clone so we don't mutate the cached tensor (PyG batches
        # share underlying storage when not cloned, leading to surprises).
        cloned = data.clone()
        cloned.x[:, :DRNL_LABEL_DIM] = 0
        return cloned

    # Convenience --------------------------------------------------------

    @property
    def entries(self) -> list[str]:
        """Filtered ``"<ref>/<sample>"`` entries (parallel to len)."""
        return [self._entries[i] for i in self._indices]


__all__ = [
    "PrebakeStats",
    "PrebakedSealDataset",
    "prebake_to_disk",
    "PREBAKED_SCHEMA_VERSION",
]
