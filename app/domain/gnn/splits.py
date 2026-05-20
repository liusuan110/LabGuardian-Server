"""GNN 模块 · Train/Val/Test Splits（P1 Phase C）

把 :func:`generate_dataset` 写盘后的 labels 目录切成 train / val / test 三个
JSON 名单。**plan §五硬约束**：test 必须保留**整条 ref 拓扑**，绝不能让
test 里的样本与 train/val 来自同一个 ref（否则模型只是在做记忆，不是泛化）。

公开 API（与 dataset_builder 解耦 —— 任何时候 labels 目录就绪都可调用）：

- :class:`SplitSpec` —— 切分配方
- :class:`DatasetSplits` —— 切分结果
- :func:`discover_samples` —— 扫盘列出 ``{ref_id: [sample_id, ...]}``
- :func:`build_splits` —— 按 spec 实际分到 train/val/test
- :func:`write_splits` / :func:`load_splits` —— JSON 持久化

切分单位是 ``"<ref_id>/<sample_id>"`` 相对路径字符串，可直接拼到
``<dataset>/labels/`` 后还原文件名。

不引入 torch / torch_geometric。
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SplitsError(ValueError):
    """Raised by :func:`build_splits` when the spec is unsatisfiable —
    e.g. ``test_ref_ids`` references a ref not present on disk, or
    ``val_fraction`` is out of [0, 1)."""


# ---------------------------------------------------------------------------
# Spec / Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitSpec:
    """切分配方。

    Attributes:
        test_ref_ids: **整条** 留出的 ref id 集合。test 的所有样本都来自这些
            ref；train/val 的样本绝不来自这些 ref。**plan §五 硬约束**：
            至少留 1 个新拓扑（理想 = 1 个 MVP 内 + 1 个全新）以测泛化。
        val_fraction: 在 train+val 候选池中划给 val 的比例。0 表示无 val
            split，全部进 train。
        seed: 决定 val 抽样的随机性。同 seed + 同 labels 目录 → 同 splits。
    """

    test_ref_ids: tuple[str, ...]
    val_fraction: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.val_fraction < 1.0):
            raise SplitsError(
                f"val_fraction must be in [0, 1), got {self.val_fraction}"
            )


@dataclass(frozen=True)
class DatasetSplits:
    """train / val / test 的 ``"<ref_id>/<sample_id>"`` 字符串列表。

    确定性：同 spec + 同 labels 目录每次 build 出的列表完全一致（按 ref/
    sample 字典序排序）。

    ``stats`` 记录每个 split 的样本数 + ref 分布，便于 logging 与 CI 检查。
    """

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    stats: dict = field(default_factory=dict)

    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


# ---------------------------------------------------------------------------
# Disk discovery
# ---------------------------------------------------------------------------


def discover_samples(labels_dir: Path) -> dict[str, list[str]]:
    """扫描 ``labels_dir/<ref_id>/<sample_id>.json``，返回
    ``{ref_id: [sample_id, ...]}``。

    ``sample_id`` 是 .json 文件名去后缀。空目录返回空 dict（不抛）。
    每个 ref 内 sample_id 排序确定性。
    """

    if not labels_dir.is_dir():
        return {}
    out: dict[str, list[str]] = {}
    for ref_dir in sorted(labels_dir.iterdir()):
        if not ref_dir.is_dir():
            continue
        sample_ids = sorted(
            f.stem for f in ref_dir.iterdir()
            if f.is_file() and f.suffix == ".json"
        )
        if sample_ids:
            out[ref_dir.name] = sample_ids
    return out


# ---------------------------------------------------------------------------
# Build / load / write
# ---------------------------------------------------------------------------


def build_splits(
    samples_by_ref: dict[str, list[str]],
    spec: SplitSpec,
) -> DatasetSplits:
    """按 :class:`SplitSpec` 把 samples 分到三组。

    算法：
    1. 先按 ``spec.test_ref_ids`` 把全部 sample 一刀切成 ``trainval_pool`` /
       ``test_pool``。绝不在 ref 内混合（plan §五 硬约束）。
    2. ``trainval_pool`` 内部按 ``spec.val_fraction`` + ``spec.seed`` 随机
       抽 val，其余进 train。**val 抽样在每个 ref 内独立**，保证每个
       train ref 在 train 与 val 都有覆盖（否则小 ref 容易全跑掉）。
    3. 三个列表都按 ``f"{ref_id}/{sample_id}"`` 字典序排序，确定性输出。

    Raises:
        SplitsError: ``test_ref_ids`` 含 labels 目录中不存在的 ref id；
            ``samples_by_ref`` 为空；val_fraction 越界（spec 自身已校验）。
    """

    if not samples_by_ref:
        raise SplitsError("samples_by_ref is empty; nothing to split")

    available_refs = set(samples_by_ref)
    missing = [r for r in spec.test_ref_ids if r not in available_refs]
    if missing:
        raise SplitsError(
            f"test_ref_ids contains refs not in labels dir: {missing}; "
            f"available: {sorted(available_refs)}"
        )

    test_ids: list[str] = []
    train_ids: list[str] = []
    val_ids: list[str] = []
    rng = random.Random(spec.seed)

    test_set = set(spec.test_ref_ids)
    n_per_ref: dict[str, dict[str, int]] = {}

    for ref_id in sorted(samples_by_ref):
        sample_ids = list(samples_by_ref[ref_id])
        if ref_id in test_set:
            test_ids.extend(f"{ref_id}/{sid}" for sid in sample_ids)
            n_per_ref[ref_id] = {"test": len(sample_ids)}
            continue
        # train/val split within this ref.
        # ⚠️ MUST use a stable hash (sha256) — Python builtin ``hash()`` on
        # strings/tuples is randomised per-process via PYTHONHASHSEED, which
        # used to break the "same seed → same splits across processes"
        # contract. Switched to sha256 to fix the cross-process regression.
        ref_seed_bytes = hashlib.sha256(
            f"{spec.seed}::{ref_id}".encode()
        ).digest()
        ref_seed = int.from_bytes(ref_seed_bytes[:8], "big")
        ref_rng = random.Random(ref_seed)
        shuffled = sample_ids.copy()
        ref_rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * spec.val_fraction))
        # Guarantee at least 1 train sample if ref has any
        if n_val == len(shuffled) and shuffled:
            n_val = len(shuffled) - 1
        val_picks = shuffled[:n_val]
        train_picks = shuffled[n_val:]
        train_ids.extend(f"{ref_id}/{sid}" for sid in train_picks)
        val_ids.extend(f"{ref_id}/{sid}" for sid in val_picks)
        n_per_ref[ref_id] = {
            "train": len(train_picks),
            "val": len(val_picks),
        }
    _ = rng  # reserved for future global shuffle hook

    train_ids.sort()
    val_ids.sort()
    test_ids.sort()

    stats = {
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_total": len(train_ids) + len(val_ids) + len(test_ids),
        "by_ref": n_per_ref,
        "test_ref_ids": list(spec.test_ref_ids),
        "val_fraction": spec.val_fraction,
        "seed": spec.seed,
    }
    return DatasetSplits(
        train=tuple(train_ids),
        val=tuple(val_ids),
        test=tuple(test_ids),
        stats=stats,
    )


def write_splits(splits: DatasetSplits, output_dir: Path) -> Path:
    """把三个 split 文件写到 ``output_dir/splits/{train,val,test}.json``
    + ``splits/stats.json``。返回 splits 子目录路径。"""

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.json").write_text(
        json.dumps(list(splits.train), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (splits_dir / "val.json").write_text(
        json.dumps(list(splits.val), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (splits_dir / "test.json").write_text(
        json.dumps(list(splits.test), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (splits_dir / "stats.json").write_text(
        json.dumps(splits.stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return splits_dir


def load_splits(output_dir: Path) -> DatasetSplits:
    """从 ``output_dir/splits/`` 读回 :class:`DatasetSplits`。"""

    splits_dir = output_dir / "splits"
    if not splits_dir.is_dir():
        raise SplitsError(f"splits dir not found: {splits_dir}")
    train = tuple(json.loads((splits_dir / "train.json").read_text(encoding="utf-8")))
    val = tuple(json.loads((splits_dir / "val.json").read_text(encoding="utf-8")))
    test = tuple(json.loads((splits_dir / "test.json").read_text(encoding="utf-8")))
    stats_path = splits_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    return DatasetSplits(train=train, val=val, test=test, stats=stats)


__all__ = [
    "SplitSpec",
    "DatasetSplits",
    "SplitsError",
    "discover_samples",
    "build_splits",
    "write_splits",
    "load_splits",
]
