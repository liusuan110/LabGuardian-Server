"""
引脚电气约束工具 (← src_v2/vision/pin_utils.py)

Top-K 引脚候选选择 + 电气约束评分
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# 每种元件的物理先验
COMPONENT_PIN_PROPS: Dict[str, Dict] = {
    "Resistor": {"extension_ratio": 1.2, "search_multiplier": 1.5, "min_span": 2},
    "Capacitor": {"extension_ratio": 1.1, "search_multiplier": 1.5, "min_span": 1},
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

    # 电轨引脚: Wire 连电轨是合理的, 非 Wire 元件两端都在电轨是短路
    col1_is_rail = col1.startswith("rail_")
    col2_is_rail = col2.startswith("rail_")
    if col1_is_rail and col2_is_rail:
        # 两端都在电轨 → Wire 可能 (但不理想), 其他元件则是短路
        return 0.05 if comp_type.lower() != "wire" else 0.2
    if col1_is_rail or col2_is_rail:
        # 一端在电轨, 一端在主 grid → Wire 的正常接法
        return 0.9 if comp_type.lower() == "wire" else 0.6

    score = 1.0

    # 判断主 grid 侧
    side1 = "L" if col1 in "abcde" else "R"
    side2 = "L" if col2 in "abcde" else "R"

    # 同行同侧 → 短路
    if r1 == r2 and side1 == side2:
        score *= 0.1  # 严重惩罚

    # 跨沟槽 (a-e → f-j) → 正常; 对 Resistor/LED 等跨沟槽是典型接法
    if side1 != side2 and comp_type in ("Resistor", "LED", "Capacitor"):
        score *= 1.2  # 鼓励跨沟槽

    # 行距过小 (低于元件物理最小跨度)
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
    """从 Top-K 候选中选择最佳引脚对

    综合考虑:
    1. 电气约束 (避免短路等)
    2. 候选排名 (越靠前表示像素距离越近, 物理上更可信)
    """
    best_score = -1.0
    best_pair = (candidates1[0], candidates2[0])

    for i, c1 in enumerate(candidates1):
        for j, c2 in enumerate(candidates2):
            s = score_electrical_constraints(c1, c2, comp_type)
            # 候选排名衰减: 排名越靠后 (离像素位置越远) 得分越低
            # 第0个候选权重=1.0, 第1个=0.7, 第2个=0.49, ...
            rank_penalty = (0.7 ** i) * (0.7 ** j)
            s *= rank_penalty
            if s > best_score:
                best_score = s
                best_pair = (c1, c2)

    return best_pair
