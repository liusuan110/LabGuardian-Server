"""
风险分级引擎 (← shared/risk.py)

将 CircuitValidator.diagnose() 输出映射到风险等级
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class RiskLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"


_DANGER_KEYWORDS: List[str] = [
    "短路",
    "烧毁",
    "无限流电阻",
    "同一导通组",
    "可能损坏",
]

_WARNING_KEYWORDS: List[str] = [
    "极性未确定",
    "极性未知",
    "引脚缺失",
    "浮空",
    "孤立",
    "开路",
    "未正确跨行",
    "方向",
]


def classify_risk(diagnostics: List[str]) -> Tuple[RiskLevel, List[str]]:
    """将诊断文本列表分类为风险等级。"""
    if not diagnostics:
        return RiskLevel.SAFE, []

    max_level = RiskLevel.SAFE
    reasons: List[str] = []

    for diag in diagnostics:
        matched_level = _match_single(diag)
        if matched_level is not None:
            reasons.append(diag)
            if _level_priority(matched_level) > _level_priority(max_level):
                max_level = matched_level

    return max_level, reasons


def _match_single(diag: str) -> RiskLevel | None:
    for keyword in _DANGER_KEYWORDS:
        if keyword in diag:
            return RiskLevel.DANGER
    for keyword in _WARNING_KEYWORDS:
        if keyword in diag:
            return RiskLevel.WARNING
    return None


def _level_priority(level: RiskLevel) -> int:
    return {
        RiskLevel.SAFE: 0,
        RiskLevel.WARNING: 1,
        RiskLevel.DANGER: 2,
    }.get(level, 0)
