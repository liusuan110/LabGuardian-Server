"""GNN 模块 · 大规模 dataset 生成期的标签分布监控（P0.8 收尾）

P1 perturbation pipeline 将一次生成 5 × 600 = 3000 个 (ref, cur) pair。
即便每个 sample 内部通过 :func:`build_seal_samples_with_coverage_check`
做了 coverage 检查，**跨样本的分布**仍可能悄无声息退化：

- 某种扰动生成器突然只产生 OPTIONAL pin → ``n_skipped_optional_pin``
  突涨，正样本骤减
- perturbation 重命名 net 时漏对齐 → ``n_skipped_missing_component``
  比例飙升
- 某个 ref 电路缺 FORBIDDEN pin → ``forbidden_violated`` / ``forbidden_negative``
  归零，模型学不到关键负

:class:`LabelManifest` 累积所有样本的 LabelStats 并支持**每 N 个样本一次
periodic checkpoint**，让 dataset_builder 实时打印 / 日志，**避免跑完
3 小时才发现数据集已经坏了**。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.domain.gnn.label_builder import (
    LabelBuildResult,
    LabelSource,
    LabelStats,
    TaskType,
)


@dataclass
class LabelManifest:
    """跨样本的累积统计 + 失败追踪。

    使用模式（plan §附录 A.8 给 P1 dataset_builder 的契约）：

    ```python
    manifest = LabelManifest()
    for sample_id, ref, cur, alignment in tqdm(samples_to_build):
        try:
            result = build_seal_samples_with_coverage_check(
                ref, cur, alignment, seed=hash(sample_id),
            )
            manifest.add(sample_id, result)
            (LABELS_DIR / ref.id / f"{sample_id}.json").write_text(
                json.dumps(serialize_label_build_result(
                    result, sample_id=sample_id, ref_id=ref.id,
                ))
            )
        except CoverageError as e:
            manifest.record_failure(sample_id, f"coverage: {e}")
        except Exception as e:
            manifest.record_failure(sample_id, f"unexpected: {e!r}")

        # Periodic health check —— 每 100 个样本打印 by_source / pos_neg_ratio
        if periodic := manifest.checkpoint(every=100):
            log.info("label distribution after %d samples: %s",
                     manifest.n_processed, periodic)

    final = manifest.summary()
    (DATASET_DIR / "manifest.json").write_text(json.dumps(final, indent=2))
    if final["n_skipped_failures"] > 0:
        log.warning("%d samples failed coverage; see manifest.failures",
                    final["n_skipped_failures"])
    ```
    """

    n_processed: int = 0
    n_skipped_failures: int = 0
    total_samples: int = 0
    total_positives: int = 0
    total_negatives: int = 0
    n_groups: int = 0
    n_groups_without_positive: int = 0
    n_skipped_missing_component: int = 0
    n_skipped_optional_pin: int = 0
    n_skipped_forbidden_pin_no_violation: int = 0
    n_skipped_extract_error: int = 0
    by_source: dict[str, int] = field(
        default_factory=lambda: {src.value: 0 for src in LabelSource}
    )
    by_task_type: dict[str, int] = field(
        default_factory=lambda: {t.value: 0 for t in TaskType}
    )
    # 每个失败样本：(sample_id, reason)
    failures: list[tuple[str, str]] = field(default_factory=list)
    # 追踪本 manifest 已 emit 过 checkpoint 的样本计数（n_processed 值），
    # 用于支持 every=N 的"恰好每 N 次"语义。
    _checkpointed_at: set[int] = field(default_factory=set)

    # -- mutation --------------------------------------------------------

    def add(self, sample_id: str, result: LabelBuildResult) -> None:
        """累积一个成功的 LabelBuildResult 到 running totals."""

        self.add_stats(sample_id, result.stats)

    def add_stats(self, sample_id: str, stats: LabelStats) -> None:
        """直接用 LabelStats 累积 —— 用于跨进程 worker，只回传 stats
        而不是完整 LabelBuildResult（picking 全 SealSample 列表代价高）。

        Phase C 并发路径：worker 内部 build_seal_samples + 写文件后只把
        ``result.stats`` 通过 ProcessPoolExecutor 回传，主进程 dispatch
        进这里。
        """

        self.n_processed += 1
        self.total_samples += stats.total_samples
        self.total_positives += stats.n_positives
        self.total_negatives += stats.n_negatives
        self.n_groups += stats.n_groups
        self.n_groups_without_positive += stats.n_groups_without_positive
        self.n_skipped_missing_component += stats.n_skipped_missing_component
        self.n_skipped_optional_pin += stats.n_skipped_optional_pin
        self.n_skipped_forbidden_pin_no_violation += (
            stats.n_skipped_forbidden_pin_no_violation
        )
        self.n_skipped_extract_error += stats.n_skipped_extract_error
        for k, v in stats.by_source.items():
            self.by_source[k] = self.by_source.get(k, 0) + v
        for k, v in stats.by_task_type.items():
            self.by_task_type[k] = self.by_task_type.get(k, 0) + v
        # sample_id 暂不记录到 manifest（避免巨大文件），P1 dataset_builder
        # 可独立维护 sample_id → file_path 索引

    def record_failure(self, sample_id: str, reason: str) -> None:
        """记录一个失败样本。CoverageError / 任何异常都走这里。"""

        self.n_processed += 1
        self.n_skipped_failures += 1
        self.failures.append((sample_id, reason))

    # -- introspection ---------------------------------------------------

    @property
    def pos_neg_ratio(self) -> float:
        return self.total_positives / max(1, self.total_negatives)

    @property
    def failure_rate(self) -> float:
        return self.n_skipped_failures / max(1, self.n_processed)

    @property
    def avg_samples_per_build(self) -> float:
        successful = max(1, self.n_processed - self.n_skipped_failures)
        return self.total_samples / successful

    def checkpoint(self, *, every: int = 100) -> dict | None:
        """如果 ``n_processed`` 是 ``every`` 的整数倍且未对此次报过，返回
        当前 snapshot；否则返回 None。"""

        if every <= 0:
            raise ValueError("every must be positive")
        if self.n_processed == 0:
            return None
        if self.n_processed % every != 0:
            return None
        if self.n_processed in self._checkpointed_at:
            return None
        self._checkpointed_at.add(self.n_processed)
        return self._snapshot()

    def summary(self) -> dict:
        """最终 manifest（写盘前最后一次调用）。"""

        snap = self._snapshot()
        snap["failures"] = [
            {"sample_id": sid, "reason": reason}
            for sid, reason in self.failures
        ]
        return snap

    def to_json(self, path: Path) -> None:
        """便利：写最终 manifest 到磁盘。"""

        path.write_text(json.dumps(self.summary(), indent=2))

    # -- internals -------------------------------------------------------

    def _snapshot(self) -> dict:
        return {
            "n_processed": self.n_processed,
            "n_skipped_failures": self.n_skipped_failures,
            "failure_rate": self.failure_rate,
            "total_samples": self.total_samples,
            "total_positives": self.total_positives,
            "total_negatives": self.total_negatives,
            "pos_neg_ratio": self.pos_neg_ratio,
            "avg_samples_per_build": self.avg_samples_per_build,
            "n_groups": self.n_groups,
            "n_groups_without_positive": self.n_groups_without_positive,
            "n_skipped_missing_component": self.n_skipped_missing_component,
            "n_skipped_optional_pin": self.n_skipped_optional_pin,
            "n_skipped_forbidden_pin_no_violation": (
                self.n_skipped_forbidden_pin_no_violation
            ),
            "n_skipped_extract_error": self.n_skipped_extract_error,
            "by_source": dict(self.by_source),
            "by_task_type": dict(self.by_task_type),
        }


__all__ = ["LabelManifest"]


# ---------------------------------------------------------------------------
# 健康度断言（dataset_builder 末尾调用，避免分布坏掉的 dataset 进训练）
# ---------------------------------------------------------------------------


def assert_manifest_healthy(
    manifest: LabelManifest,
    *,
    max_failure_rate: float = 0.05,
    min_pos_neg_ratio: float = 0.3,
    max_pos_neg_ratio: float = 3.0,
    require_sources: tuple[str, ...] = (
        "ref_present",
        "wrong_observed",
        "negative_random",
    ),
) -> None:
    """跑完所有样本后调用，违反任意一条 raise ValueError。

    默认阈值适配 MVP 5 电路 × 600 perturbation 的预期分布。具体训练前
    可按 dataset 实际特征调整。
    """

    issues: list[str] = []
    if manifest.failure_rate > max_failure_rate:
        issues.append(
            f"failure_rate={manifest.failure_rate:.3f} > {max_failure_rate}"
        )
    if (
        manifest.total_negatives > 0
        and not (min_pos_neg_ratio <= manifest.pos_neg_ratio <= max_pos_neg_ratio)
    ):
        issues.append(
            f"pos_neg_ratio={manifest.pos_neg_ratio:.3f} outside "
            f"[{min_pos_neg_ratio}, {max_pos_neg_ratio}]"
        )
    for src in require_sources:
        if manifest.by_source.get(src, 0) == 0:
            issues.append(f"required source '{src}' has zero samples")
    if issues:
        raise ValueError(
            "label manifest unhealthy:\n  - " + "\n  - ".join(issues)
        )


__all__.append("assert_manifest_healthy")
