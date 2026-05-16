"""GNN 模块 · Label Builder（P0.8）

把 ``(ref_hcg, cur_hcg, alignment)`` 三元组转化为一组带 0/1 标签的训练
样本 :class:`SealSample`，配合 :class:`SealSampleGroup` 支持 ranking 任务。

**P0.8 二轮 audit 与第三轮收尾**：

- 加入 `WRONG_OBSERVED` 强负样本（100% 覆盖 cur 中所有"非 ref-correct"
  的观测边），杜绝 WRONG_EDGE 主头被弱监督。
- MISSING_EDGE group 同时覆盖 ``floating`` 与 ``wrong_redirect`` 两种来源。
- ``symmetry_class_id`` 在 sibling pin swap 中首次被实质消费（"R.pin1↔pin2
  swap 后两边都打 label=1"）。
- ConnectionPolicy 全程被尊重：OPTIONAL 默认排除，FORBIDDEN 不进 sym
  / random / missing 候选。
- ``label_stats`` 与 ``label_build_result.serialize()`` 是 P1 dataset
  builder 的对接面。

**永不 import torch / torch_geometric**；纯 Python。
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from app.domain.gnn.alignment import ComponentAlignment
from app.domain.gnn.graph_schema import ConnectionPolicy
from app.domain.gnn.seal_subgraph import SealSubgraph, extract_seal_subgraph

if TYPE_CHECKING:  # pragma: no cover
    from app.domain.gnn.hetero_circuit import HeteroCircuitGraph

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 枚举与数据结构
# ---------------------------------------------------------------------------


class TaskType(str, Enum):  # noqa: UP042 — keep (str, Enum) for py<3.11 compat
    """一个 SealSample 服务哪个模型 head（决定 P3 loss 形态）。

    **WRONG_EDGE** 覆盖**所有**点式 P(edge_correct) 监督，**不仅是 cur 中实
    际存在的边**：含 ref-present 正样本、cur 实际错边（WRONG_OBSERVED 强
    负）、FORBIDDEN-violated、FORBIDDEN-合成、随机负、ref-absent-REQUIRED
    （正样本，告诉模型该有边但 cur 没接）。

    **MISSING_EDGE** 覆盖 per-port N-way ranking / softmax：给 REQUIRED-floating
    或 REQUIRED-wrong_redirect 的 port 在候选 net 集合中选正确目标。推理时
    suggested_target head 复用此训练。
    """

    WRONG_EDGE = "wrong_edge"
    MISSING_EDGE = "missing_edge"


class LabelSource(str, Enum):  # noqa: UP042 — keep (str, Enum) for py<3.11 compat
    """label=1 / label=0 各自的"来源说明"，用于 stats 健康监控。"""

    REF_PRESENT = "ref_present"
    REF_SYMMETRIC_SWAP = "ref_symmetric_swap"
    REF_ABSENT_REQUIRED = "ref_absent_required"
    WRONG_OBSERVED = "wrong_observed"  # 强负，cur 中实际接但 ref 不期望
    FORBIDDEN_VIOLATED = "forbidden_violated"
    FORBIDDEN_NEGATIVE = "forbidden_negative"
    NEGATIVE_RANDOM = "negative_random"
    NEGATIVE_HARD = "negative_hard"  # 预留 slot；P0.8 不生成


# 同 component 内可互换的 port pair：sym_swap 展开候选
_REF_POSITIVE_SOURCES = frozenset(
    {
        LabelSource.REF_PRESENT.value,
        LabelSource.REF_SYMMETRIC_SWAP.value,
        LabelSource.REF_ABSENT_REQUIRED.value,
    }
)


@dataclass(frozen=True)
class SealSample:
    """单个 (subgraph, label) 训练样本。"""

    subgraph: SealSubgraph
    label: int  # 0 or 1
    label_source: str  # LabelSource.value
    task_type: str  # TaskType.value
    candidate_edge: tuple[str, str]  # (cur_port_id, cur_net_id)
    expected_edge: tuple[str, str] | None = None
    ref_edge_origin: tuple[str, str] | None = None
    confidence: float = 1.0
    is_symmetric_equivalent: bool = False
    group_id: str | None = None


@dataclass(frozen=True)
class SealSampleGroup:
    """N-way ranking 候选集（MISSING_EDGE 专用）。"""

    group_id: str
    task_type: str
    query_port_id: str
    query_origin: str  # "floating" | "wrong_redirect"
    sample_indices: tuple[int, ...]
    correct_index: int | None


@dataclass(frozen=True)
class LabelStats:
    total_samples: int
    n_positives: int
    n_negatives: int
    pos_neg_ratio: float
    by_source: dict[str, int]
    by_task_type: dict[str, int]
    n_groups: int
    n_groups_without_positive: int
    n_skipped_missing_component: int
    n_skipped_optional_pin: int
    n_skipped_forbidden_pin_no_violation: int
    n_skipped_extract_error: int
    n_unique_ports_covered: int
    n_unique_nets_covered: int


@dataclass(frozen=True)
class LabelBuildResult:
    samples: tuple[SealSample, ...]
    groups: tuple[SealSampleGroup, ...]
    stats: LabelStats


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _sym_class_siblings(
    cur_hcg: HeteroCircuitGraph, cur_port_id: str
) -> list[str]:
    """同 component 中 ``symmetry_class_id`` 相同且 != self 的 port id 列表。

    决定 "R.pin1 ↔ pin2 swap" 是否合法的核心查表。
    """

    port = cur_hcg.ports.get(cur_port_id)
    if port is None:
        return []
    siblings: list[str] = []
    for other_id in cur_hcg.port_of_component.get(port.parent_component_id, []):
        if other_id == cur_port_id:
            continue
        other = cur_hcg.ports[other_id]
        if other.symmetry_class_id == port.symmetry_class_id:
            siblings.append(other_id)
    return siblings


def _compute_sym_aware_correct_cur_edges(
    ref_hcg: HeteroCircuitGraph,
    cur_hcg: HeteroCircuitGraph,
    alignment: ComponentAlignment,
) -> set[tuple[str, str]]:
    """``{(cur_port_id, cur_net_id)}`` 集合：所有 ref 期望的边，按 alignment
    映射到 cur 侧并按 symmetry_class 展开 sibling。

    这是判断 "cur 中一条边算正还是算错" 的唯一事实源。
    """

    result: set[tuple[str, str]] = set()
    for ref_edge in ref_hcg.edges:
        cur_port = alignment.map_ref_port_to_cur_port_id(
            ref_edge.src_port_id, ref_hcg, cur_hcg
        )
        cur_net = alignment.map_ref_net_to_cur_net_id(
            ref_edge.dst_net_id, cur_hcg
        )
        if cur_port is None or cur_net is None:
            continue
        result.add((cur_port, cur_net))
        for sib in _sym_class_siblings(cur_hcg, cur_port):
            result.add((sib, cur_net))
    return result


def _infer_expected_net_for_port(
    cur_port_id: str,
    ref_hcg: HeteroCircuitGraph,
    cur_hcg: HeteroCircuitGraph,
    alignment: ComponentAlignment,
) -> tuple[str, str] | None:
    """对 cur_port_id 反查 ref 中它**应该**接到哪个 cur_net。

    若该 ref port 在 ref 中接多于 1 个 net（罕见但合法），返回 None 表示
    "expected 不唯一"。
    """

    ref_port_id = alignment.map_cur_port_to_ref_port_id(
        cur_port_id, ref_hcg, cur_hcg
    )
    if ref_port_id is None:
        return None
    expected_cur_nets: list[str] = []
    for ref_edge in ref_hcg.edges:
        if ref_edge.src_port_id != ref_port_id:
            continue
        cur_net = alignment.map_ref_net_to_cur_net_id(
            ref_edge.dst_net_id, cur_hcg
        )
        if cur_net is not None and cur_net not in expected_cur_nets:
            expected_cur_nets.append(cur_net)
    if len(expected_cur_nets) == 1:
        return (cur_port_id, expected_cur_nets[0])
    return None


def _safe_extract(
    cur_hcg: HeteroCircuitGraph,
    port_id: str,
    net_id: str,
    *,
    edge_present: bool,
    num_hops: int,
    include_same_component_edges: bool,
    builder: _Builder,
) -> SealSubgraph | None:
    """抽 SEAL 子图，KeyError 时 None + 计数（不 raise，保 silent skip）。"""

    try:
        return extract_seal_subgraph(
            cur_hcg,
            port_id,
            net_id,
            num_hops=num_hops,
            edge_present=edge_present,
            include_same_component_edges=include_same_component_edges,
        )
    except KeyError as e:
        builder.n_skipped_extract_error += 1
        log.debug("seal extract failed for (%s, %s): %s", port_id, net_id, e)
        return None


# ---------------------------------------------------------------------------
# Builder 上下文（可变累积；finalize 时冻结为 LabelBuildResult）
# ---------------------------------------------------------------------------


@dataclass
class _Builder:
    samples: list[SealSample] = field(default_factory=list)
    groups: list[SealSampleGroup] = field(default_factory=list)
    # 任务级 (port, net) 去重 —— Step 5 random negative 用
    _emitted_pairs_by_task: dict[str, set[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    n_skipped_missing_component: int = 0
    n_skipped_optional_pin: int = 0
    n_skipped_forbidden_pin_no_violation: int = 0
    n_skipped_extract_error: int = 0
    _ports_covered: set[str] = field(default_factory=set)
    _nets_covered: set[str] = field(default_factory=set)

    def add_sample(self, sample: SealSample) -> int:
        idx = len(self.samples)
        self.samples.append(sample)
        self._emitted_pairs_by_task[sample.task_type].add(sample.candidate_edge)
        self._ports_covered.add(sample.candidate_edge[0])
        self._nets_covered.add(sample.candidate_edge[1])
        return idx

    def add_group(self, group: SealSampleGroup) -> None:
        self.groups.append(group)

    def already_emitted(
        self, task: str, candidate_edge: tuple[str, str]
    ) -> bool:
        return candidate_edge in self._emitted_pairs_by_task[task]

    def finalize(self) -> LabelBuildResult:
        by_source: dict[str, int] = defaultdict(int)
        by_task: dict[str, int] = defaultdict(int)
        n_positives = 0
        n_negatives = 0
        for s in self.samples:
            by_source[s.label_source] += 1
            by_task[s.task_type] += 1
            if s.label == 1:
                n_positives += 1
            else:
                n_negatives += 1
        # 所有 LabelSource value 都建键，便于 stats 表静态结构稳定
        for v in (e.value for e in LabelSource):
            by_source.setdefault(v, 0)
        for v in (e.value for e in TaskType):
            by_task.setdefault(v, 0)
        n_groups_without_positive = sum(
            1 for g in self.groups if g.correct_index is None
        )
        stats = LabelStats(
            total_samples=len(self.samples),
            n_positives=n_positives,
            n_negatives=n_negatives,
            pos_neg_ratio=n_positives / max(1, n_negatives),
            by_source=dict(by_source),
            by_task_type=dict(by_task),
            n_groups=len(self.groups),
            n_groups_without_positive=n_groups_without_positive,
            n_skipped_missing_component=self.n_skipped_missing_component,
            n_skipped_optional_pin=self.n_skipped_optional_pin,
            n_skipped_forbidden_pin_no_violation=self.n_skipped_forbidden_pin_no_violation,
            n_skipped_extract_error=self.n_skipped_extract_error,
            n_unique_ports_covered=len(self._ports_covered),
            n_unique_nets_covered=len(self._nets_covered),
        )
        return LabelBuildResult(
            samples=tuple(self.samples),
            groups=tuple(self.groups),
            stats=stats,
        )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def build_seal_samples(
    ref_hcg: HeteroCircuitGraph,
    cur_hcg: HeteroCircuitGraph,
    alignment: ComponentAlignment,
    *,
    negatives_per_positive: float = 1.0,
    include_optional: bool = False,
    forbidden_negative_samples: int = 4,
    missing_edge_group_size: int = 5,
    enable_hard_negative_mining: bool = False,
    seed: int = 0,
    num_hops: int = 2,
    include_same_component_edges: bool = False,
) -> LabelBuildResult:
    """从 (ref, cur, alignment) 构造一组 :class:`SealSample` + 配套
    :class:`SealSampleGroup` 与 :class:`LabelStats`。

    6 步算法：
    - **Step 1** ref→positive (含 sym swap 双正)
    - **Step 2** MISSING_EDGE group (floating + wrong_redirect 两种来源)
    - **Step 2.5** WRONG_OBSERVED 强负 (100% 覆盖 cur 中所有非正观测边)
    - **Step 3** FORBIDDEN_VIOLATED
    - **Step 4** FORBIDDEN_NEGATIVE (默认 N=4)
    - **Step 5** NEGATIVE_RANDOM 补差到 ``negatives_per_positive``

    见 plan §附录 A.8。
    """

    if enable_hard_negative_mining:
        # P0.8 留 slot；P3 启用时再实现 _step_hard_negative
        raise NotImplementedError(
            "Hard-negative mining is reserved for P3 model training. "
            "See plan §附录 A.8 hard-negative strategies table."
        )

    builder = _Builder()
    rng = random.Random(seed)
    sym_aware_correct = _compute_sym_aware_correct_cur_edges(
        ref_hcg, cur_hcg, alignment
    )
    cur_edge_set: set[tuple[str, str]] = {
        (e.src_port_id, e.dst_net_id) for e in cur_hcg.edges
    }

    # ---- Step 1 · ref-driven WRONG_EDGE positives + sym expansion ---------
    for ref_edge in ref_hcg.edges:
        cur_port = alignment.map_ref_port_to_cur_port_id(
            ref_edge.src_port_id, ref_hcg, cur_hcg
        )
        cur_net = alignment.map_ref_net_to_cur_net_id(
            ref_edge.dst_net_id, cur_hcg
        )
        if cur_port is None:
            # missing_component perturbation — decision: silently skip + log
            builder.n_skipped_missing_component += 1
            log.debug(
                "missing_component: ref port %s has no cur counterpart",
                ref_edge.src_port_id,
            )
            continue
        if cur_net is None:
            builder.n_skipped_missing_component += 1
            continue
        port = cur_hcg.ports[cur_port]
        if port.connection_policy == ConnectionPolicy.OPTIONAL.value:
            if not include_optional:
                builder.n_skipped_optional_pin += 1
                continue
        if port.connection_policy == ConnectionPolicy.FORBIDDEN.value:
            log.warning(
                "spec contradiction: ref expects edge on FORBIDDEN port %s",
                cur_port,
            )
            continue

        actually_present = (cur_port, cur_net) in cur_edge_set
        siblings = _sym_class_siblings(cur_hcg, cur_port)
        anchors = [cur_port] + siblings
        ref_origin = (
            _strip_prefix(ref_edge.src_port_id, "ref_port:"),
            _strip_prefix(ref_edge.dst_net_id, "ref_net:"),
        )
        for anchor in anchors:
            if builder.already_emitted(TaskType.WRONG_EDGE.value, (anchor, cur_net)):
                continue
            is_sym = anchor != cur_port
            sib_present = (anchor, cur_net) in cur_edge_set
            sg = _safe_extract(
                cur_hcg,
                anchor,
                cur_net,
                edge_present=sib_present,
                num_hops=num_hops,
                include_same_component_edges=include_same_component_edges,
                builder=builder,
            )
            if sg is None:
                continue
            if is_sym:
                source = LabelSource.REF_SYMMETRIC_SWAP.value
            elif actually_present:
                source = LabelSource.REF_PRESENT.value
            else:
                source = LabelSource.REF_ABSENT_REQUIRED.value
            builder.add_sample(
                SealSample(
                    subgraph=sg,
                    label=1,
                    label_source=source,
                    task_type=TaskType.WRONG_EDGE.value,
                    candidate_edge=(anchor, cur_net),
                    expected_edge=(cur_port, cur_net),
                    ref_edge_origin=ref_origin,
                    is_symmetric_equivalent=is_sym,
                )
            )

    # ---- Step 2 · MISSING_EDGE groups（floating + wrong_redirect 双触发）--
    # 收集 cur 中每个 port 的实际邻居 net 集合
    cur_nets_by_port: dict[str, set[str]] = defaultdict(set)
    for e in cur_hcg.edges:
        cur_nets_by_port[e.src_port_id].add(e.dst_net_id)

    for ref_edge in ref_hcg.edges:
        cur_port = alignment.map_ref_port_to_cur_port_id(
            ref_edge.src_port_id, ref_hcg, cur_hcg
        )
        if cur_port is None:
            continue
        port = cur_hcg.ports[cur_port]
        if port.connection_policy != ConnectionPolicy.REQUIRED.value:
            continue
        cur_net_correct = alignment.map_ref_net_to_cur_net_id(
            ref_edge.dst_net_id, cur_hcg
        )
        if cur_net_correct is None:
            continue
        cur_nets_actual = cur_nets_by_port.get(cur_port, set())
        if cur_net_correct in cur_nets_actual:
            continue  # 已接对，跳过
        query_origin = "floating" if not cur_nets_actual else "wrong_redirect"

        must_include = {cur_net_correct, *cur_nets_actual}
        all_nets = list(cur_hcg.nets)
        pool = [n for n in all_nets if n not in must_include]
        rng.shuffle(pool)
        n_distractors = max(0, missing_edge_group_size - len(must_include))
        distractors = pool[:n_distractors]
        candidate_nets = list(must_include) + distractors

        group_id = (
            f"miss_{cur_port}_"
            f"{ref_edge.src_port_id}_{ref_edge.dst_net_id}"
        )
        ref_origin = (
            _strip_prefix(ref_edge.src_port_id, "ref_port:"),
            _strip_prefix(ref_edge.dst_net_id, "ref_net:"),
        )
        group_sample_indices: list[int] = []
        correct_idx: int | None = None
        for i, net_id in enumerate(candidate_nets):
            is_correct = net_id == cur_net_correct
            edge_present = net_id in cur_nets_actual
            sg = _safe_extract(
                cur_hcg,
                cur_port,
                net_id,
                edge_present=edge_present,
                num_hops=num_hops,
                include_same_component_edges=include_same_component_edges,
                builder=builder,
            )
            if sg is None:
                continue
            source = (
                LabelSource.REF_ABSENT_REQUIRED.value
                if is_correct
                else LabelSource.NEGATIVE_RANDOM.value
            )
            idx = builder.add_sample(
                SealSample(
                    subgraph=sg,
                    label=int(is_correct),
                    label_source=source,
                    task_type=TaskType.MISSING_EDGE.value,
                    candidate_edge=(cur_port, net_id),
                    expected_edge=(cur_port, cur_net_correct),
                    ref_edge_origin=ref_origin,
                    group_id=group_id,
                )
            )
            if is_correct:
                correct_idx = len(group_sample_indices)
            group_sample_indices.append(idx)
        if group_sample_indices:
            builder.add_group(
                SealSampleGroup(
                    group_id=group_id,
                    task_type=TaskType.MISSING_EDGE.value,
                    query_port_id=cur_port,
                    query_origin=query_origin,
                    sample_indices=tuple(group_sample_indices),
                    correct_index=correct_idx,
                )
            )

    # ---- Step 2.5 · WRONG_OBSERVED — cur 中实际存在但非 ref-sym-correct ---
    # 关键：100% 覆盖，不留给 NEGATIVE_RANDOM 偶然采样。
    for e in cur_hcg.edges:
        candidate = (e.src_port_id, e.dst_net_id)
        if candidate in sym_aware_correct:
            continue  # 已在 Step 1 计为 positive
        port = cur_hcg.ports[e.src_port_id]
        if port.connection_policy == ConnectionPolicy.FORBIDDEN.value:
            continue  # 留给 Step 3 FORBIDDEN_VIOLATED
        if port.connection_policy == ConnectionPolicy.OPTIONAL.value:
            if not include_optional:
                builder.n_skipped_optional_pin += 1
                continue
        if builder.already_emitted(TaskType.WRONG_EDGE.value, candidate):
            continue  # 防御性去重
        sg = _safe_extract(
            cur_hcg,
            e.src_port_id,
            e.dst_net_id,
            edge_present=True,
            num_hops=num_hops,
            include_same_component_edges=include_same_component_edges,
            builder=builder,
        )
        if sg is None:
            continue
        expected = _infer_expected_net_for_port(
            e.src_port_id, ref_hcg, cur_hcg, alignment
        )
        builder.add_sample(
            SealSample(
                subgraph=sg,
                label=0,
                label_source=LabelSource.WRONG_OBSERVED.value,
                task_type=TaskType.WRONG_EDGE.value,
                candidate_edge=candidate,
                expected_edge=expected,
            )
        )

    # ---- Step 3 · FORBIDDEN_VIOLATED ------------------------------------
    forbidden_ports = [
        p for p in cur_hcg.ports.values()
        if p.connection_policy == ConnectionPolicy.FORBIDDEN.value
    ]
    edges_by_src: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in cur_hcg.edges:
        edges_by_src[e.src_port_id].append((e.src_port_id, e.dst_net_id))
    for port in forbidden_ports:
        violations = edges_by_src.get(port.node_id, [])
        if not violations:
            builder.n_skipped_forbidden_pin_no_violation += 1
            continue
        for candidate in violations:
            if builder.already_emitted(TaskType.WRONG_EDGE.value, candidate):
                continue
            sg = _safe_extract(
                cur_hcg,
                candidate[0],
                candidate[1],
                edge_present=True,
                num_hops=num_hops,
                include_same_component_edges=include_same_component_edges,
                builder=builder,
            )
            if sg is None:
                continue
            builder.add_sample(
                SealSample(
                    subgraph=sg,
                    label=0,
                    label_source=LabelSource.FORBIDDEN_VIOLATED.value,
                    task_type=TaskType.WRONG_EDGE.value,
                    candidate_edge=candidate,
                    expected_edge=None,
                )
            )

    # ---- Step 4 · FORBIDDEN_NEGATIVE — 每个 FORBIDDEN pin 合成 N 条非边 --
    for port in forbidden_ports:
        already_paired = {n for (_, n) in edges_by_src.get(port.node_id, [])}
        candidate_nets = [n for n in cur_hcg.nets if n not in already_paired]
        n_sample = min(forbidden_negative_samples, len(candidate_nets))
        if n_sample == 0:
            continue
        sampled = rng.sample(candidate_nets, k=n_sample)
        for net_id in sampled:
            candidate = (port.node_id, net_id)
            if builder.already_emitted(TaskType.WRONG_EDGE.value, candidate):
                continue
            sg = _safe_extract(
                cur_hcg,
                port.node_id,
                net_id,
                edge_present=False,
                num_hops=num_hops,
                include_same_component_edges=include_same_component_edges,
                builder=builder,
            )
            if sg is None:
                continue
            builder.add_sample(
                SealSample(
                    subgraph=sg,
                    label=0,
                    label_source=LabelSource.FORBIDDEN_NEGATIVE.value,
                    task_type=TaskType.WRONG_EDGE.value,
                    candidate_edge=candidate,
                    expected_edge=None,
                )
            )

    # ---- Step 5 · NEGATIVE_RANDOM — 凑齐 negatives_per_positive 比例 -----
    wrong_edge_pos = sum(
        1
        for s in builder.samples
        if s.task_type == TaskType.WRONG_EDGE.value and s.label == 1
    )
    wrong_edge_neg = sum(
        1
        for s in builder.samples
        if s.task_type == TaskType.WRONG_EDGE.value and s.label == 0
    )
    n_need = max(0, int(wrong_edge_pos * negatives_per_positive) - wrong_edge_neg)

    if n_need > 0:
        candidates: list[tuple[str, str]] = []
        for p_id, p in cur_hcg.ports.items():
            if p.connection_policy != ConnectionPolicy.REQUIRED.value:
                continue
            for n_id in cur_hcg.nets:
                pair = (p_id, n_id)
                if pair in sym_aware_correct:
                    continue
                if builder.already_emitted(TaskType.WRONG_EDGE.value, pair):
                    continue
                candidates.append(pair)
        rng.shuffle(candidates)
        for pair in candidates[:n_need]:
            sg = _safe_extract(
                cur_hcg,
                pair[0],
                pair[1],
                edge_present=pair in cur_edge_set,
                num_hops=num_hops,
                include_same_component_edges=include_same_component_edges,
                builder=builder,
            )
            if sg is None:
                continue
            builder.add_sample(
                SealSample(
                    subgraph=sg,
                    label=0,
                    label_source=LabelSource.NEGATIVE_RANDOM.value,
                    task_type=TaskType.WRONG_EDGE.value,
                    candidate_edge=pair,
                    expected_edge=None,
                )
            )

    return builder.finalize()


# ---------------------------------------------------------------------------
# 强约束：cur 中每条非 OPTIONAL 边要么是 ref-correct 正样本，要么是 WRONG_EDGE 负样本
# ---------------------------------------------------------------------------


class CoverageError(AssertionError):
    """Raised when cur edges lack expected WRONG_EDGE sample coverage.

    Inherits from ``AssertionError`` so legacy ``pytest.raises(AssertionError)``
    patterns still catch it; ``except CoverageError`` is the preferred form
    in dataset_builder for actionable error handling.

    Carries the list of uncovered ``(port, net)`` pairs so dataset_builder
    can log them, drop the sample, or escalate. **P1 dataset_builder MUST**
    catch this and exclude the failing sample from the dataset (never write
    a coverage-broken JSON to disk)."""

    def __init__(self, missing: list[tuple[str, str]]):
        self.missing = missing
        n = len(missing)
        preview = missing[:5]
        super().__init__(
            f"WRONG_EDGE coverage gap: {n} cur edges have no corresponding "
            f"WRONG_EDGE sample (first 5: {preview})"
        )


def assert_observed_edges_covered(
    result: LabelBuildResult,
    cur_hcg: HeteroCircuitGraph,
    ref_hcg: HeteroCircuitGraph,
    alignment: ComponentAlignment,
    *,
    include_optional: bool = False,
) -> None:
    """断言 cur.edges 中每条非 OPTIONAL 非 ref-sym-correct 的边都有
    WRONG_EDGE 负样本对应（plan DoD 关键不变量）。

    若违反则 raise :class:`CoverageError`（``AssertionError`` 的子类，
    便于 pytest.raises 仍可捕获）。dataset_builder 应捕获后丢样本。
    """

    sym_aware_correct = _compute_sym_aware_correct_cur_edges(
        ref_hcg, cur_hcg, alignment
    )
    wrong_edge_negatives_pairs = {
        s.candidate_edge
        for s in result.samples
        if s.task_type == TaskType.WRONG_EDGE.value and s.label == 0
    }
    wrong_edge_positives_pairs = {
        s.candidate_edge
        for s in result.samples
        if s.task_type == TaskType.WRONG_EDGE.value and s.label == 1
    }
    missing: list[tuple[str, str]] = []
    for e in cur_hcg.edges:
        pair = (e.src_port_id, e.dst_net_id)
        port = cur_hcg.ports[e.src_port_id]
        if (
            port.connection_policy == ConnectionPolicy.OPTIONAL.value
            and not include_optional
        ):
            continue
        if pair in sym_aware_correct:
            # 正确边 —— 应在 wrong_edge_positives_pairs 中
            if pair not in wrong_edge_positives_pairs:
                missing.append(pair)
            continue
        # 错边 —— 应在 wrong_edge_negatives_pairs 中
        if pair not in wrong_edge_negatives_pairs:
            missing.append(pair)
    if missing:
        raise CoverageError(missing)


def build_seal_samples_with_coverage_check(
    ref_hcg: HeteroCircuitGraph,
    cur_hcg: HeteroCircuitGraph,
    alignment: ComponentAlignment,
    **kwargs,
) -> LabelBuildResult:
    """便利包装：build_seal_samples + assert_observed_edges_covered。

    **P1 dataset_builder 必须使用此函数**（而非裸 build_seal_samples），
    以免把 coverage gap 带进训练数据。失败时抛 :class:`CoverageError`，
    dataset_builder 用 try/except 跳过 + manifest 记录失败原因即可。

    ``include_optional`` 同步透传到 ``assert_observed_edges_covered``，
    保持两步一致性。
    """

    include_optional = kwargs.get("include_optional", False)
    result = build_seal_samples(ref_hcg, cur_hcg, alignment, **kwargs)
    assert_observed_edges_covered(
        result, cur_hcg, ref_hcg, alignment, include_optional=include_optional
    )
    return result


# ---------------------------------------------------------------------------
# Serialization (P1 dataset_builder 接口)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"


def _strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix):] if s.startswith(prefix) else s


def _seal_subgraph_to_dict(sg: SealSubgraph) -> dict:
    return {
        "target_port_id": sg.target_port_id,
        "target_net_id": sg.target_net_id,
        "edge_present": sg.edge_present,
        "num_hops": sg.num_hops,
        "port_ids": list(sg.port_ids),
        "net_ids": list(sg.net_ids),
        "edges": [list(e) for e in sg.edges],
        "same_component_edges": [list(e) for e in sg.same_component_edges],
        "drnl_labels": dict(sg.drnl_labels),
        "is_target": dict(sg.is_target),
    }


def _seal_subgraph_from_dict(payload: dict) -> SealSubgraph:
    return SealSubgraph(
        target_port_id=payload["target_port_id"],
        target_net_id=payload["target_net_id"],
        edge_present=payload["edge_present"],
        num_hops=payload["num_hops"],
        port_ids=tuple(payload["port_ids"]),
        net_ids=tuple(payload["net_ids"]),
        edges=tuple(tuple(e) for e in payload["edges"]),
        drnl_labels=dict(payload["drnl_labels"]),
        is_target=dict(payload["is_target"]),
        same_component_edges=tuple(
            tuple(e) for e in payload.get("same_component_edges", [])
        ),
    )


def _sample_to_dict(sample: SealSample, index: int) -> dict:
    return {
        "index": index,
        "label": sample.label,
        "label_source": sample.label_source,
        "task_type": sample.task_type,
        "candidate_edge": list(sample.candidate_edge),
        "expected_edge": (
            list(sample.expected_edge) if sample.expected_edge else None
        ),
        "ref_edge_origin": (
            list(sample.ref_edge_origin) if sample.ref_edge_origin else None
        ),
        "confidence": sample.confidence,
        "is_symmetric_equivalent": sample.is_symmetric_equivalent,
        "group_id": sample.group_id,
        "subgraph": _seal_subgraph_to_dict(sample.subgraph),
    }


def _sample_from_dict(payload: dict) -> SealSample:
    return SealSample(
        subgraph=_seal_subgraph_from_dict(payload["subgraph"]),
        label=payload["label"],
        label_source=payload["label_source"],
        task_type=payload["task_type"],
        candidate_edge=tuple(payload["candidate_edge"]),
        expected_edge=(
            tuple(payload["expected_edge"])
            if payload.get("expected_edge")
            else None
        ),
        ref_edge_origin=(
            tuple(payload["ref_edge_origin"])
            if payload.get("ref_edge_origin")
            else None
        ),
        confidence=payload.get("confidence", 1.0),
        is_symmetric_equivalent=payload.get("is_symmetric_equivalent", False),
        group_id=payload.get("group_id"),
    )


def _group_to_dict(group: SealSampleGroup) -> dict:
    return {
        "group_id": group.group_id,
        "task_type": group.task_type,
        "query_port_id": group.query_port_id,
        "query_origin": group.query_origin,
        "sample_indices": list(group.sample_indices),
        "correct_index": group.correct_index,
    }


def _group_from_dict(payload: dict) -> SealSampleGroup:
    return SealSampleGroup(
        group_id=payload["group_id"],
        task_type=payload["task_type"],
        query_port_id=payload["query_port_id"],
        query_origin=payload["query_origin"],
        sample_indices=tuple(payload["sample_indices"]),
        correct_index=payload["correct_index"],
    )


def _stats_to_dict(stats: LabelStats) -> dict:
    return {
        "total_samples": stats.total_samples,
        "n_positives": stats.n_positives,
        "n_negatives": stats.n_negatives,
        "pos_neg_ratio": stats.pos_neg_ratio,
        "by_source": dict(stats.by_source),
        "by_task_type": dict(stats.by_task_type),
        "n_groups": stats.n_groups,
        "n_groups_without_positive": stats.n_groups_without_positive,
        "n_skipped_missing_component": stats.n_skipped_missing_component,
        "n_skipped_optional_pin": stats.n_skipped_optional_pin,
        "n_skipped_forbidden_pin_no_violation": stats.n_skipped_forbidden_pin_no_violation,
        "n_skipped_extract_error": stats.n_skipped_extract_error,
        "n_unique_ports_covered": stats.n_unique_ports_covered,
        "n_unique_nets_covered": stats.n_unique_nets_covered,
    }


def _stats_from_dict(payload: dict) -> LabelStats:
    return LabelStats(
        total_samples=payload["total_samples"],
        n_positives=payload["n_positives"],
        n_negatives=payload["n_negatives"],
        pos_neg_ratio=payload["pos_neg_ratio"],
        by_source=dict(payload["by_source"]),
        by_task_type=dict(payload["by_task_type"]),
        n_groups=payload["n_groups"],
        n_groups_without_positive=payload["n_groups_without_positive"],
        n_skipped_missing_component=payload["n_skipped_missing_component"],
        n_skipped_optional_pin=payload["n_skipped_optional_pin"],
        n_skipped_forbidden_pin_no_violation=payload[
            "n_skipped_forbidden_pin_no_violation"
        ],
        n_skipped_extract_error=payload["n_skipped_extract_error"],
        n_unique_ports_covered=payload["n_unique_ports_covered"],
        n_unique_nets_covered=payload["n_unique_nets_covered"],
    )


def serialize_label_build_result(
    result: LabelBuildResult,
    *,
    sample_id: str,
    ref_id: str,
    cur_metadata: dict | None = None,
) -> dict:
    """LabelBuildResult → JSON-friendly dict (plan §附录 A.8 schema v1.0)。"""

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "ref_id": ref_id,
        "cur_metadata": cur_metadata or {},
        "stats": _stats_to_dict(result.stats),
        "samples": [
            _sample_to_dict(s, idx) for idx, s in enumerate(result.samples)
        ],
        "groups": [_group_to_dict(g) for g in result.groups],
    }


def deserialize_label_build_result(payload: dict) -> LabelBuildResult:
    """``serialize_label_build_result`` 的逆。

    schema_version 不匹配会 raise，避免 silent corruption。
    """

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported label schema_version: {version!r} "
            f"(this build expects {SCHEMA_VERSION!r})"
        )
    samples = tuple(_sample_from_dict(s) for s in payload["samples"])
    groups = tuple(_group_from_dict(g) for g in payload["groups"])
    stats = _stats_from_dict(payload["stats"])
    return LabelBuildResult(samples=samples, groups=groups, stats=stats)


__all__ = [
    # enums
    "TaskType",
    "LabelSource",
    # dataclasses
    "SealSample",
    "SealSampleGroup",
    "LabelStats",
    "LabelBuildResult",
    # main API
    "build_seal_samples",
    "build_seal_samples_with_coverage_check",
    "assert_observed_edges_covered",
    "CoverageError",
    # serialization
    "SCHEMA_VERSION",
    "serialize_label_build_result",
    "deserialize_label_build_result",
]
