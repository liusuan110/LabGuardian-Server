"""
IC 多引脚建模 — DIP-8 (UA741 等) 引脚布局推算。

当前模块只负责根据锚点推算封装引脚位置，不再返回旧的内部组件对象。
"""

from __future__ import annotations

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
    """返回面包板中央沟槽对面的列名。"""
    return _COL_PAIR.get(col, "f")


def build_dip8_pin_locs(
    pin1: tuple[str, str],
    pin2: tuple[str, str],
) -> list[tuple[str, str]]:
    """从 pin1/pin2 锚点推算 DIP-8 全部 8 脚逻辑坐标。

    俯视 DIP 封装时，从缺口端的 pin1 开始逆时针编号。面包板上常见的
    左缺口横放 IC 会让 pin1/pin2 落在同一侧、相邻数字列；另一侧按
    pin8→pin5 反向回到缺口端。
    """
    r1, c1 = int(pin1[0]), str(pin1[1])
    r2 = int(pin2[0])
    step = 1 if r2 >= r1 else -1
    if abs(r2 - r1) > 1:
        rows = [int(round(r1 + i * (r2 - r1) / 3.0)) for i in range(4)]
    else:
        rows = [r1 + i * step for i in range(4)]

    pin_side_col = c1
    opposite_col = paired_col(c1)
    return [
        (str(rows[0]), pin_side_col),
        (str(rows[1]), pin_side_col),
        (str(rows[2]), pin_side_col),
        (str(rows[3]), pin_side_col),
        (str(rows[3]), opposite_col),
        (str(rows[2]), opposite_col),
        (str(rows[1]), opposite_col),
        (str(rows[0]), opposite_col),
    ]
