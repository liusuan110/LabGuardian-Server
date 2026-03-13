"""
IC 多引脚建模 — DIP-8 (UA741 等) 引脚布局推算

从 YOLO 检测到的两个端点 (pin1, pin2) 推算 DIP-8 封装的全部 8 个引脚在面包板上的逻辑坐标。
此模块被 S3 (拓扑构建) 和 S4 (验证重建) 共享。
"""

from __future__ import annotations

from app.domain.circuit import CircuitComponent, Polarity

UA741_PIN_ROLES = [
    "offset_null_1",
    "inverting_input",
    "non_inverting_input",
    "v_minus",
    "offset_null_2",
    "output",
    "v_plus",
    "nc",
]

_COL_PAIR = {
    "a": "j", "b": "i", "c": "h", "d": "g", "e": "f",
    "f": "e", "g": "d", "h": "c", "i": "b", "j": "a",
}


def paired_col(col: str) -> str:
    """返回面包板中央沟槽对面的列名"""
    return _COL_PAIR.get(col, "f")


def build_dip8_component(
    class_name: str,
    pin1: tuple[str, str],
    pin2: tuple[str, str],
    confidence: float,
) -> CircuitComponent:
    """从两端引脚推算 DIP-8 封装的全部 8 脚逻辑坐标"""
    r1, c1 = int(pin1[0]), str(pin1[1])
    r2, c2 = int(pin2[0]), str(pin2[1])
    top = min(r1, r2)
    bottom = max(r1, r2)
    if bottom - top < 3:
        bottom = top + 3

    rows = [int(round(top + i * (bottom - top) / 3.0)) for i in range(4)]
    for i in range(1, len(rows)):
        if rows[i] <= rows[i - 1]:
            rows[i] = rows[i - 1] + 1

    side1 = "L" if c1 in "abcde" else "R"
    side2 = "L" if c2 in "abcde" else "R"
    if side1 != side2:
        left_col = c1 if side1 == "L" else c2
        right_col = c1 if side1 == "R" else c2
    else:
        if side1 == "L":
            left_col = c1
            right_col = paired_col(c1)
        else:
            right_col = c1
            left_col = paired_col(c1)

    pin_locs = [
        (str(rows[0]), left_col),
        (str(rows[1]), left_col),
        (str(rows[2]), left_col),
        (str(rows[3]), left_col),
        (str(rows[3]), right_col),
        (str(rows[2]), right_col),
        (str(rows[1]), right_col),
        (str(rows[0]), right_col),
    ]
    return CircuitComponent(
        name="",
        type=class_name,
        pin1_loc=pin_locs[0],
        pin2_loc=pin_locs[1],
        extra_pins=pin_locs[2:],
        pin_roles=UA741_PIN_ROLES,
        polarity=Polarity.NONE,
        confidence=confidence,
    )
