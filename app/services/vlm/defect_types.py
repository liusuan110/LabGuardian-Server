"""Micro-defect taxonomy for white-box-gated VLM inspection.

These defect classes are the kinds of failures the deterministic vision +
topology pipeline cannot judge from netlist alone (e.g. burn marks, exposed
copper, cold solder joints). The classifier (`verify_draft_answer`) flips
`needs_micro_inspection=True` when validator confidence is low and the error
context plausibly maps to one of these defects, then the VLM is asked to
inspect — strictly under white-box gating.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class MicroDefectType(str, Enum):
    BURN_MARK = "BURN_MARK"
    UNSTRIPPED_WIRE = "UNSTRIPPED_WIRE"
    COLD_SOLDER = "COLD_SOLDER"


# Per-defect VLM prompt fragments. Kept short so they cost few tokens on edge
# (Qwen-VL-Int4 on iGPU/NPU is sensitive to context length).
DEFECT_TYPE_PROMPTS: dict[MicroDefectType, str] = {
    MicroDefectType.BURN_MARK: (
        "请重点观察元件本体和导线表面是否有焦黑、变色或炭化痕迹。"
        "若存在，给出具体位置（如靠近 R1、电源接线端等）。"
    ),
    MicroDefectType.UNSTRIPPED_WIRE: (
        "请检查跳线两端是否完整剥皮、铜芯是否裸露并插入孔位。"
        "若发现绝缘皮未剥或剥皮长度不足，请给出涉及的导线位置。"
    ),
    MicroDefectType.COLD_SOLDER: (
        "请观察焊点（如有）是否光泽良好、无裂纹、无虚焊（粗糙、暗灰、形状不规则）。"
        "若疑似冷焊或虚焊，请指出具体引脚。"
    ),
}


# Error tags that suggest a likely micro defect type. Used by the gate to set
# `suspected_defect_types` so the VLM node knows which prompt to use first.
SUSPICIOUS_TAGS_BY_TYPE: dict[MicroDefectType, tuple[str, ...]] = {
    MicroDefectType.BURN_MARK: (
        "burn_mark",
        "scorch",
        "overheat",
        "焦黑",
        "烧焦",
    ),
    MicroDefectType.UNSTRIPPED_WIRE: (
        "unstripped_wire",
        "wire_insulation",
        "未剥皮",
        "导线绝缘",
    ),
    MicroDefectType.COLD_SOLDER: (
        "cold_solder",
        "solder_joint",
        "虚焊",
        "冷焊",
    ),
}


def suggest_defect_types(error_tags: Iterable[str]) -> list[MicroDefectType]:
    """Return defect types whose suspicious tags overlap the evidence's tags."""
    tags_lower = {str(tag).strip().lower() for tag in error_tags if tag}
    suggestions: list[MicroDefectType] = []
    for defect_type, suspicious in SUSPICIOUS_TAGS_BY_TYPE.items():
        if any(suspect.lower() in tags_lower for suspect in suspicious):
            suggestions.append(defect_type)
    return suggestions
