"""
引脚电气约束工具 (← src_v2/vision/pin_utils.py)

Top-K 引脚候选选择 + 电气约束评分
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# 每种元件的物理先验
COMPONENT_PIN_PROPS: Dict[str, Dict] = {
    "Resistor": {"extension_ratio": 1.2, "search_multiplier": 1.5, "min_span": 2},
    "LED": {"extension_ratio": 1.3, "search_multiplier": 1.5, "min_span": 1},
    "Wire": {"extension_ratio": 1.0, "search_multiplier": 2.0, "min_span": 0},
}


def score_electrical_constraints(
    loc1: Tuple[str, str],
    loc2: Tuple[str, str],
    comp_type: str = "",
) -> float:
    """评分电气约束 (0.0=违规, 1.0=理想)"""
    row1, col1 = loc1
    row2, col2 = loc2

    try:
        r1, r2 = int(row1), int(row2)
    except (ValueError, TypeError):
        return 0.5

    score = 1.0

    # 同行同侧 → 短路
    side1 = "L" if col1 in "abcde" else "R"
    side2 = "L" if col2 in "abcde" else "R"
    if r1 == r2 and side1 == side2:
        score *= 0.1  # 严重惩罚

    # 跨沟槽 (a-e → f-j) → 正常
    if side1 != side2:
        score *= 1.0

    # 行距过大
    distance = abs(r1 - r2)
    props = COMPONENT_PIN_PROPS.get(comp_type, {})
    min_span = props.get("min_span", 1)
    if distance < min_span:
        score *= 0.3

    return score


def select_best_pin_pair(
    candidates1: List[Tuple[str, str]],
    candidates2: List[Tuple[str, str]],
    comp_type: str = "",
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """从 Top-K 候选中选择最佳引脚对"""
    best_score = -1.0
    best_pair = (candidates1[0], candidates2[0])

    for c1 in candidates1:
        for c2 in candidates2:
            s = score_electrical_constraints(c1, c2, comp_type)
            if s > best_score:
                best_score = s
                best_pair = (c1, c2)

    return best_pair
