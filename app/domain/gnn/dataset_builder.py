"""GNN 模块 · Dataset Builder（P1 Phase A）

把"参考电路 + 扰动配方"转成磁盘上的 labeled dataset。是 P1 perturbation
pipeline 与 P0.8 label_builder 的合体调度层。

**严格遵守 P0.8 收尾给出的契约**：

1. 每个样本调 ``build_seal_samples_with_coverage_check`` —— 不裸调
   ``build_seal_samples``，coverage gap 自动抛 :class:`CoverageError`
2. :class:`LabelManifest` 累积分布，每 100 个样本调 ``checkpoint(every=100)``
   打印健康状况
3. 最后调 ``assert_manifest_healthy`` 守住分布健康，违反会 raise；
   ``manifest.json`` 已先写盘以供诊断（违规快照 + 已生成 labels 不丢）

**Phase A 范围**：
- :class:`RefSpec` —— 一个参考电路（payload + optional subtype dict）
- :class:`PerturbationPlan` —— "每个 ref 跑多少次每个 perturbation"
- :class:`DatasetSpec` —— refs + plan + output dir + seeds
- :func:`generate_dataset(spec)` —— 主入口；返回 :class:`LabelManifest`

**Phase B 待补**：splits（train/val/test 写盘）、并发执行、断点续传。

不引入 torch / torch_geometric。
"""

from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.domain.gnn.alignment import ComponentAlignment
from app.domain.gnn.label_builder import (
    CoverageError,
    build_seal_samples_with_coverage_check,
    deserialize_label_build_result,
    serialize_label_build_result,
)
from app.domain.gnn.label_manifest import LabelManifest
from app.domain.gnn.perturbation import (
    PERTURBATION_REGISTRY,
    PerturbedCur,
    get_perturbation,
)
from app.domain.gnn.port_graph import build_from_logical_reference

if TYPE_CHECKING:  # pragma: no cover
    from app.domain.gnn.hetero_circuit import HeteroCircuitGraph

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatasetSpecError(ValueError):
    """Raised by :func:`validate_dataset_spec` when the spec is malformed
    or references invalid resources. Per P1 audit: configuration errors
    (bad ref path / unknown perturbation name) must surface **before** any
    sample work begins, not as per-sample failures."""


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefSpec:
    """单个参考电路的规格。

    Attributes:
        ref_id: 唯一字符串 id，作为 labels/<ref_id>/ 目录名。
        payload_path: ``logical_reference_v1`` JSON 文件路径。
        subtype_by_source_id: IC 子类型 dict（如 ``{"U1": "UA741"}``）。
            **同时**应用在 ref HCG 构建（覆盖 payload 中的 subtype 字段，
            如果 fixture 没写）与 perturbation 生成 cur HCG 两个阶段，
            保证 ref/cur pin 语义一致（如 UA741 pin 8 = FORBIDDEN 在两侧
            都生效）。优先级：本 dict 覆盖 payload 中的 ``subtype`` 字段。
    """

    ref_id: str
    payload_path: Path
    subtype_by_source_id: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PerturbationPlan:
    """每个 ref 跑多少次每个 perturbation。

    Attributes:
        counts: ``{perturbation_name: n_samples}``。所有 name 必须在
            :data:`PERTURBATION_REGISTRY` 中。
    """

    counts: dict[str, int]

    def total_per_ref(self) -> int:
        return sum(self.counts.values())


@dataclass(frozen=True)
class DatasetSpec:
    """完整的 dataset 生成配置。"""

    refs: tuple[RefSpec, ...]
    plan: PerturbationPlan
    output_dir: Path
    base_seed: int = 0
    # P0.8 label builder kwargs（透传）
    negatives_per_positive: float = 1.0
    forbidden_negative_samples: int = 4
    missing_edge_group_size: int = 5
    include_optional: bool = False
    num_hops: int = 2
    # Manifest 与 checkpoint 频率
    checkpoint_every: int = 100
    # 健康检查（生成完后自动调用 assert_manifest_healthy）
    enforce_healthy: bool = True

    def total_samples(self) -> int:
        return len(self.refs) * self.plan.total_per_ref()


# ---------------------------------------------------------------------------
# Sample id 生成
# ---------------------------------------------------------------------------


def _make_sample_id(ref_id: str, perturbation_name: str, idx: int) -> str:
    """E.g. ``"voltage_divider__pin_swap_symmetric_0042"``."""

    return f"{ref_id}__{perturbation_name}_{idx:04d}"


def _make_seed(base_seed: int, ref_id: str, sample_id: str) -> int:
    """确定性的 per-sample seed：base_seed XOR hash(sample_id)。"""

    # Python hash is randomized per-process; use a stable hash via hashlib
    # to ensure same (base_seed, sample_id) → same seed across runs.
    import hashlib

    h = hashlib.sha256(f"{ref_id}::{sample_id}".encode()).digest()
    # First 8 bytes as int
    salt = int.from_bytes(h[:8], "big") & 0x7FFFFFFF
    return (base_seed ^ salt) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Upfront validation (P1 audit fix — separates config errors from sample errors)
# ---------------------------------------------------------------------------


def validate_dataset_spec(spec: DatasetSpec) -> None:
    """Validate spec before any sample work. Raises :class:`DatasetSpecError`
    with a consolidated list of issues if any of:

    - ref payload file doesn't exist or isn't readable
    - perturbation name not in registry
    - duplicate ref_id (causes label dir collisions)
    - empty refs / empty plan

    Configuration errors (above) are distinct from per-sample errors
    (CoverageError, etc.) which are recorded in manifest at run time.
    """

    issues: list[str] = []

    if not spec.refs:
        issues.append("DatasetSpec.refs is empty")
    if not spec.plan.counts:
        issues.append("PerturbationPlan.counts is empty")

    # ref_id uniqueness
    seen_ref_ids: set[str] = set()
    for ref_spec in spec.refs:
        if ref_spec.ref_id in seen_ref_ids:
            issues.append(f"duplicate ref_id: {ref_spec.ref_id!r}")
        seen_ref_ids.add(ref_spec.ref_id)
        # File existence
        if not ref_spec.payload_path.is_file():
            issues.append(
                f"ref_id={ref_spec.ref_id!r}: payload not found at "
                f"{ref_spec.payload_path}"
            )
            continue
        # Try parse JSON
        try:
            json.loads(ref_spec.payload_path.read_text())
        except Exception as e:
            issues.append(
                f"ref_id={ref_spec.ref_id!r}: payload not valid JSON: {e!r}"
            )

    # Perturbation names
    available = set(PERTURBATION_REGISTRY)
    for name, count in spec.plan.counts.items():
        if name not in available:
            issues.append(
                f"unknown perturbation {name!r}; available: {sorted(available)}"
            )
        if count < 0:
            issues.append(f"perturbation {name!r}: negative count {count}")

    if issues:
        raise DatasetSpecError(
            "DatasetSpec validation failed:\n  - " + "\n  - ".join(issues)
        )


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def generate_dataset(
    spec: DatasetSpec,
    *,
    progress: bool = False,
    resume: bool = False,
    workers: int = 1,
) -> LabelManifest:
    """主入口：跑 spec 中所有 (ref, perturbation, idx) 组合，写 labels
    JSON 到磁盘，返回累积的 :class:`LabelManifest`。

    **Pipeline 流程**：
    1. ``validate_dataset_spec(spec)`` —— 配置错误（bad ref path / unknown
       perturbation / 重复 ref_id）**在生成任何样本之前** 立刻 raise
       :class:`DatasetSpecError`。
    2. 每个 (ref, perturbation, idx) 独立 try/except 跑 perturbation +
       label_builder，**单个样本失败计入 manifest，pipeline 不中断**。
    3. 末尾 ``assert_manifest_healthy``（若 ``enforce_healthy=True``）守住
       数据集级分布健康，违反 raise ValueError 但 manifest 与已写 labels
       仍保留供诊断。

    Args:
        progress: True 时打印每 ``checkpoint_every`` 个样本的进度（仅
            stdout，不依赖 tqdm；P1 后续可换 tqdm）。
        resume: True 时**复用已存在的 label 文件**。Pipeline 跳过已 emit
            的 ``<labels>/<ref_id>/<sample_id>.json``，从磁盘 deserialize
            其 LabelBuildResult 并 ``manifest.add(...)``，再继续生成缺失的。
            适用于长任务被中断 / 增量补样本的场景。**前提**：output_dir
            不能含有跨 spec 残留（resume 不会主动清理）。
        workers: >1 时启用 :class:`ProcessPoolExecutor` 并发跑 per-sample
            perturbation + label_builder。==1 走串行路径，行为与旧版完全
            一致（适用于调试 / 小数据集）。Workers 之间通过 dict 任务
            描述 + ``LabelStats`` 回传通信，避免 pickle 整个
            ``LabelBuildResult``（含 SealSample 列表，单样本可能 KB 级）。
            **resume 与 workers 兼容**：resume 在主进程串行 replay（廉价
            的反序列化），剩余待生成的样本才进入 pool。
    """

    validate_dataset_spec(spec)  # P1 audit fix: fail fast on config errors

    manifest = LabelManifest()
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    labels_root = spec.output_dir / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)

    # 1) Build the task list. Resume replays happen in the main process
    #    (cheap json.loads) so they don't waste worker slots.
    pending: list[_WorkerTask] = []
    for ref_spec in spec.refs:
        ref_payload = json.loads(ref_spec.payload_path.read_text())
        # P1 audit fix: subtype override **at ref build time** —— let
        # FORBIDDEN/OPTIONAL spec apply to ref ports too. Previously this
        # only flowed to perturbation, causing ref/cur semantic mismatch
        # when fixture JSON omitted the subtype field.
        ref_hcg = build_from_logical_reference(
            ref_payload,
            extra_subtypes_by_source_id=ref_spec.subtype_by_source_id or None,
        )
        # Still stash in metadata so perturbation fallback path can read it.
        if ref_spec.subtype_by_source_id:
            ref_hcg.metadata["subtype_by_source_id"] = dict(
                ref_spec.subtype_by_source_id
            )
        ref_labels_dir = labels_root / ref_spec.ref_id
        ref_labels_dir.mkdir(parents=True, exist_ok=True)

        for perturbation_name, n_samples in spec.plan.counts.items():
            for idx in range(n_samples):
                sample_id = _make_sample_id(ref_spec.ref_id, perturbation_name, idx)
                seed = _make_seed(spec.base_seed, ref_spec.ref_id, sample_id)
                label_file = ref_labels_dir / f"{sample_id}.json"
                if resume and label_file.is_file():
                    if _try_resume_sample(sample_id, label_file, manifest):
                        continue
                    # On resume failure, fall through to regenerate (logged)
                pending.append(
                    _WorkerTask(
                        ref_hcg=ref_hcg,
                        ref_id=ref_spec.ref_id,
                        subtype_by_source_id=dict(
                            ref_spec.subtype_by_source_id or {}
                        ),
                        op_name=perturbation_name,
                        seed=seed,
                        sample_id=sample_id,
                        label_file=label_file,
                        negatives_per_positive=spec.negatives_per_positive,
                        include_optional=spec.include_optional,
                        forbidden_negative_samples=spec.forbidden_negative_samples,
                        missing_edge_group_size=spec.missing_edge_group_size,
                        num_hops=spec.num_hops,
                    )
                )

    # 2) Execute pending tasks — either serial or via ProcessPoolExecutor.
    if workers <= 1:
        for t in pending:
            outcome = _worker_run_sample(t)
            _absorb_worker_outcome(outcome, manifest)
            _maybe_checkpoint(manifest, spec, progress)
    else:
        # ProcessPoolExecutor: pickle each task, fan out, collect stats
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker_run_sample, t): t for t in pending}
            for fut in as_completed(futures):
                outcome = fut.result()
                _absorb_worker_outcome(outcome, manifest)
                _maybe_checkpoint(manifest, spec, progress)

    # Final manifest write (always, even on health failure → diagnostic)
    manifest.to_json(spec.output_dir / "manifest.json")

    if spec.enforce_healthy:
        # Lazy import to avoid circular
        from app.domain.gnn.label_manifest import assert_manifest_healthy

        assert_manifest_healthy(manifest)

    return manifest


# ---------------------------------------------------------------------------
# Per-sample worker
# ---------------------------------------------------------------------------


def _try_resume_sample(
    sample_id: str, label_file: Path, manifest: LabelManifest
) -> bool:
    """Replay an on-disk sample into the manifest. Returns True on success,
    False on any error (caller then re-generates).

    P1 Phase C: keeps long runs idempotent. Sample files that fail to
    deserialize are logged + regenerated rather than silently dropped."""

    try:
        payload = json.loads(label_file.read_text())
        result = deserialize_label_build_result(payload)
    except Exception as e:
        log.warning(
            "resume: failed to replay %s (%r); regenerating", sample_id, e
        )
        return False
    manifest.add(sample_id, result)
    return True


# ---------------------------------------------------------------------------
# Picklable per-sample task (Phase C parallel execution)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkerTask:
    """Self-contained description of one (ref, perturbation, idx) sample.

    All fields are picklable so ``ProcessPoolExecutor`` can fan tasks out.
    ``ref_hcg`` is the most expensive piece (a HeteroCircuitGraph of frozen
    dataclasses) — empirically tiny (<10 KB for MVP fixtures), so we send
    it per task rather than reconstruct in the worker.
    """

    ref_hcg: HeteroCircuitGraph
    ref_id: str
    subtype_by_source_id: dict[str, str]
    op_name: str
    seed: int
    sample_id: str
    label_file: Path
    negatives_per_positive: float
    include_optional: bool
    forbidden_negative_samples: int
    missing_edge_group_size: int
    num_hops: int


def _worker_run_sample(task: _WorkerTask) -> dict[str, Any]:
    """Top-level (picklable) worker — applies one perturbation, runs the
    label builder, writes the label JSON to disk, and returns **only the
    LabelStats** for the main process to absorb. Returning the full
    LabelBuildResult would pickle the entire SealSample list (KB / sample
    × N tasks), so we keep the inter-process payload minimal.

    Return shape::

        {"status": "ok",       "sample_id": ..., "stats": LabelStats}
        {"status": "coverage", "sample_id": ..., "reason": str}
        {"status": "error",    "sample_id": ..., "reason": str}
    """

    try:
        op = get_perturbation(task.op_name)
        rng = random.Random(task.seed)
        perturbed: PerturbedCur = op.apply(
            task.ref_hcg,
            rng,
            subtype_by_source_id=task.subtype_by_source_id or None,
        )
        result = build_seal_samples_with_coverage_check(
            task.ref_hcg,
            perturbed.cur_hcg,
            perturbed.alignment,
            negatives_per_positive=task.negatives_per_positive,
            include_optional=task.include_optional,
            forbidden_negative_samples=task.forbidden_negative_samples,
            missing_edge_group_size=task.missing_edge_group_size,
            seed=task.seed,
            num_hops=task.num_hops,
        )
        cur_metadata = {
            "perturbation_chain": list(perturbed.perturbation_chain),
            "expected_outcome": perturbed.expected_outcome,
            "alignment": _alignment_to_safe_dict(perturbed.alignment),
            "perturbation_notes": _coerce_json_safe(perturbed.notes),
            "seed": task.seed,
        }
        payload = serialize_label_build_result(
            result,
            sample_id=task.sample_id,
            ref_id=task.ref_id,
            cur_metadata=cur_metadata,
        )
        task.label_file.write_text(json.dumps(payload, ensure_ascii=False))
        return {
            "status": "ok",
            "sample_id": task.sample_id,
            "stats": result.stats,
        }
    except CoverageError as e:
        return {
            "status": "coverage",
            "sample_id": task.sample_id,
            "reason": str(e),
        }
    except Exception as e:  # noqa: BLE001 — worker boundary, never crash pool
        return {
            "status": "error",
            "sample_id": task.sample_id,
            "reason": repr(e),
        }


def _absorb_worker_outcome(
    outcome: dict[str, Any], manifest: LabelManifest
) -> None:
    sample_id = outcome["sample_id"]
    status = outcome["status"]
    if status == "ok":
        manifest.add_stats(sample_id, outcome["stats"])
    elif status == "coverage":
        manifest.record_failure(sample_id, f"coverage: {outcome['reason']}")
    else:
        manifest.record_failure(sample_id, f"unexpected: {outcome['reason']}")


def _maybe_checkpoint(
    manifest: LabelManifest, spec: DatasetSpec, progress: bool
) -> None:
    periodic = manifest.checkpoint(every=spec.checkpoint_every)
    if not periodic:
        return
    if progress:
        _print_progress(periodic)
    log.info(
        "label distribution at %d samples: %s",
        manifest.n_processed,
        {
            k: periodic[k]
            for k in (
                "n_skipped_failures",
                "pos_neg_ratio",
                "by_source",
            )
        },
    )


def _alignment_to_safe_dict(align: ComponentAlignment) -> dict:
    """Serializable subset (drop heavy reverse caches)."""

    return {
        "ref_to_cur_component": dict(align.ref_to_cur_component),
        "ref_to_cur_net": dict(align.ref_to_cur_net),
        "notes": _coerce_json_safe(align.notes),
    }


def _coerce_json_safe(value):
    """Recursive coercion: tuples → lists, sets → sorted lists, frozensets
    → sorted lists; other values pass through. Used to ensure manifest
    write doesn't fail on non-JSON-native containers."""

    if isinstance(value, dict):
        return {k: _coerce_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_coerce_json_safe(v) for v in value)
    return value


def _print_progress(snap: dict) -> None:
    """Minimal stdout progress line; P1 Phase C may upgrade to tqdm."""

    print(
        f"[dataset] n={snap['n_processed']} "
        f"failures={snap['n_skipped_failures']} "
        f"pos/neg={snap['pos_neg_ratio']:.2f} "
        f"groups={snap['n_groups']} "
        f"by_source={snap['by_source']}"
    )


__all__ = [
    "RefSpec",
    "PerturbationPlan",
    "DatasetSpec",
    "DatasetSpecError",
    "validate_dataset_spec",
    "generate_dataset",
]
