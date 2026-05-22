"""Canonical topology label set — single source of truth for GNN-A.

## Why this lives in its own module

The labels are referenced from at least four different places:
  * **Dataset generation** (``scripts/cadx/build_topology_dataset.py``) —
    every sample is tagged with one of these labels.
  * **Model output head** (``app/domain/topology/model.py``) — the
    softmax dimension equals ``len(TOPOLOGY_LABELS)``.
  * **Template registry** (``app/domain/templates/registry/*.py``) — each
    template's ``topology_label`` field must match one of these strings.
  * **API responses** (``app/api/v1/topology/suggest`` — Phase 1) — the
    classifier's output is serialized using these labels.

Centralizing them here prevents drift: if a label is added/renamed,
``mypy`` immediately surfaces every call site that needs updating.

## Label conventions

Labels follow ``snake_case`` and are scoped by IC family when ambiguous
(``inverting_amp_ua741`` rather than just ``inverting_amp``). This leaves
room for ``inverting_amp_lm358`` etc. in future demo extensions without
breaking the ordinal contract.

The integer index of each label (defined by tuple position in
``TOPOLOGY_LABELS``) is the **model's class id**. Reordering the tuple
breaks loaded checkpoints — always append new labels at the end.

## Verification

``TopologyLabelSpec.validate()`` is exercised by
``tests/domain/topology/test_labels.py``. Every template in
``app/domain/templates/registry/`` must have its ``topology_label`` in
this set — enforced by ``test_template_labels_match_label_spec``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final


# ============================================================================
# Canonical label ordering — DO NOT REORDER; only APPEND.
# ============================================================================
#
# The integer index of each label is the class id used by the trained
# GNN-A model. Reordering breaks any checkpoint trained against the old
# order. New labels (Phase 2+) must be appended at the end.
#
TOPOLOGY_LABELS: Final[tuple[str, ...]] = (
    "rc_first_order",         # idx 0 — 一阶 RC 滤波器
    "common_emitter",         # idx 1 — 共射放大电路
    "differential_pair",      # idx 2 — BJT 差分放大器
    "inverting_amp_ua741",    # idx 3 — UA741 反相放大器
    "summing_amp_ua741",      # idx 4 — UA741 反相加法器
    "integrator_ua741",       # idx 5 — UA741 反相积分器 (含漏放)
    "unknown",                # idx 6 — open-set catch-all
)

DEFAULT_UNKNOWN_LABEL: Final[str] = "unknown"


# ============================================================================
# Label metadata
# ============================================================================


@dataclass(frozen=True)
class TopologyLabelSpec:
    """Static metadata for one topology label.

    The ``template_id`` and ``reference_id`` cross-reference fields exist
    so downstream code (e.g. API responses, frontend ReferenceSelector
    AI-recommendation hook) can map from a model output back to the
    correct reference circuit & symbolic template without hard-coded
    dictionaries scattered around the codebase.

    Attributes:
        label: The canonical label string (``snake_case``).
        index: Integer class id used by the model.
        display_name_zh: Chinese display name shown in UI.
        display_name_en: English display name (logs, telemetry).
        template_id: The matching symbolic template's id, or ``None``
            for the ``unknown`` class.
        reference_id: The matching reference DSL's id, or ``None``.
        description: One-paragraph human description.
    """

    label: str
    index: int
    display_name_zh: str
    display_name_en: str
    template_id: str | None
    reference_id: str | None
    description: str


_LABEL_SPECS: Final[tuple[TopologyLabelSpec, ...]] = (
    TopologyLabelSpec(
        label="rc_first_order",
        index=0,
        display_name_zh="一阶 RC 滤波器",
        display_name_en="First-order RC filter",
        template_id="rc_first_order_v1",
        reference_id="rc_first_order_v1",
        description=(
            "无源单极点 RC 滤波器（LPF / HPF / BPF 变体）。"
            "结构最简，节点 ~4 个。"
        ),
    ),
    TopologyLabelSpec(
        label="common_emitter",
        index=1,
        display_name_zh="共射放大电路",
        display_name_en="BJT common-emitter amplifier",
        template_id="common_emitter_v1",
        reference_id="ce_amp_fixed_bias_v1",
        description=(
            "单 BJT 共射放大。覆盖发射极直接接地 / 含 R_E / 含 R_E+C_E "
            "三种变体；偏置可为单支固定或分压。"
        ),
    ),
    TopologyLabelSpec(
        label="differential_pair",
        index=2,
        display_name_zh="BJT 差分放大器",
        display_name_en="BJT differential pair amplifier",
        template_id="differential_pair_v1",
        reference_id="diff_pair_current_source_ref_split_potentiometer",
        description=(
            "BJT 长尾差分对。变体覆盖共发射极短接 / 分压电位器拆分 / "
            "VT3 恒流源尾部三种实现。"
        ),
    ),
    TopologyLabelSpec(
        label="inverting_amp_ua741",
        index=3,
        display_name_zh="UA741 反相放大器",
        display_name_en="UA741 inverting amplifier",
        template_id="inverting_amp_ua741_v1",
        reference_id="ua741_inverting_amp_gain10_v1",
        description=(
            "UA741 反相放大：R_g + R_f 标准配置，可选 R_p 偏置补偿。"
            "反馈路径仅电阻，与积分器的关键判别在反馈支路元件类型。"
        ),
    ),
    TopologyLabelSpec(
        label="summing_amp_ua741",
        index=4,
        display_name_zh="UA741 反相加法器",
        display_name_en="UA741 inverting summing amplifier",
        template_id="summing_amp_ua741_v1",
        reference_id="ua741_inverting_summing_amp_v1",
        description=(
            "UA741 反相加法器：2-5 路输入电阻汇入虚地 SUM 节点。"
            "VOUT = -R_f · Σ(VINi/R_ini)。"
        ),
    ),
    TopologyLabelSpec(
        label="integrator_ua741",
        index=5,
        display_name_zh="UA741 反相积分器",
        display_name_en="UA741 inverting integrator",
        template_id="integrator_ua741_v1",
        reference_id="ua741_integrator_v1",
        description=(
            "UA741 反相积分器：C_f 反馈电容，R_leak (可选) 漏放防直流饱和。"
            "等价于反相 LPF / lossy integrator — 教学上按设计意图归类。"
        ),
    ),
    TopologyLabelSpec(
        label="unknown",
        index=6,
        display_name_zh="未识别拓扑",
        display_name_en="Unknown topology",
        template_id=None,
        reference_id=None,
        description=(
            "Open-set 兜底类。Phase 1 训练时不显式生成该类样本，仅在"
            "推理时由 softmax 置信度阈值（< 0.4）触发，避免对未知拓扑"
            "强行归类。"
        ),
    ),
)


# Aliases for backward compat — ``TopologyLabel`` reads more naturally
# as a type annotation than the spec class itself.
TopologyLabel = str


# ============================================================================
# Lookup helpers
# ============================================================================


def list_labels(include_unknown: bool = True) -> list[str]:
    """Return the canonical label list, optionally without ``unknown``.

    Args:
        include_unknown: When ``False``, omit the ``unknown`` catch-all.
            Useful for dataset generation loops where we only iterate over
            "real" classes.

    Returns:
        Ordered list of label strings.
    """
    if include_unknown:
        return list(TOPOLOGY_LABELS)
    return [label for label in TOPOLOGY_LABELS if label != DEFAULT_UNKNOWN_LABEL]


@lru_cache(maxsize=None)
def get_label_spec(label: str) -> TopologyLabelSpec:
    """Look up a label's full spec record.

    Args:
        label: One of :data:`TOPOLOGY_LABELS`.

    Raises:
        KeyError: When the label is not registered. Misspellings should
            surface as test failures rather than silent ``unknown`` returns.
    """
    for spec in _LABEL_SPECS:
        if spec.label == label:
            return spec
    raise KeyError(
        f"unknown topology label: {label!r}; "
        f"valid labels: {list(TOPOLOGY_LABELS)}"
    )


def label_to_index(label: str) -> int:
    """Convert label string to its integer class id (for model output)."""
    return get_label_spec(label).index


def index_to_label(index: int) -> str:
    """Convert a model's integer class id back to its label string."""
    if not (0 <= index < len(TOPOLOGY_LABELS)):
        raise IndexError(
            f"topology label index out of range: {index} "
            f"(must be 0..{len(TOPOLOGY_LABELS) - 1})"
        )
    return TOPOLOGY_LABELS[index]


# ============================================================================
# Self-validation
# ============================================================================


def validate_label_spec() -> list[str]:
    """Return a list of structural errors in the label table (empty if ok).

    Sanity checks:
      * ``TOPOLOGY_LABELS`` and ``_LABEL_SPECS`` are aligned 1:1.
      * Indices are contiguous 0..N-1 in declaration order.
      * Display names are non-empty.
      * Non-``unknown`` labels carry a ``template_id`` and ``reference_id``.

    This is run from :func:`tests.domain.topology.test_labels` so any
    drift between the label table and spec records is caught in CI.
    """
    errors: list[str] = []

    if len(TOPOLOGY_LABELS) != len(_LABEL_SPECS):
        errors.append(
            f"TOPOLOGY_LABELS has {len(TOPOLOGY_LABELS)} entries but "
            f"_LABEL_SPECS has {len(_LABEL_SPECS)}"
        )

    for i, (label, spec) in enumerate(zip(TOPOLOGY_LABELS, _LABEL_SPECS)):
        if spec.label != label:
            errors.append(
                f"_LABEL_SPECS[{i}].label={spec.label!r} but "
                f"TOPOLOGY_LABELS[{i}]={label!r}"
            )
        if spec.index != i:
            errors.append(
                f"_LABEL_SPECS[{i}].index={spec.index} but expected {i}"
            )
        if not spec.display_name_zh or not spec.display_name_en:
            errors.append(f"label {label!r} has empty display name")
        if label != DEFAULT_UNKNOWN_LABEL:
            if not spec.template_id:
                errors.append(f"label {label!r} missing template_id")
            if not spec.reference_id:
                errors.append(f"label {label!r} missing reference_id")

    return errors
