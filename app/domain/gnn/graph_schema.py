"""GNN 模块 · 图 Schema 常量与枚举（P0 · 纯 Python，无 torch 依赖）

本文件定义 GNN 比较模块的"图语言"：

- 节点类型枚举（component / port / net）的取值集合
- 节点 / 边的特征维度与布局（供 P2 `pyg_converter.py` 按 slice 取子特征）
- 类型 → one-hot 索引的映射
- 极性元数据表（对齐 ``app.domain.circuit`` 现有规范）

**规则**：
1. 该模块禁止 import torch / torch_geometric —— 必须能在无 GPU、无 PyG
   的开发环境单独导入并通过单元测试。
2. ComponentType / PortType / NetRole 的字符串值与 ``app.domain.circuit``
   及 ``app.domain.logical_reference`` 中已有的规范化输出严格对齐，下游
   port_graph.py 的查表必须命中。

设计参见 plan §三 与附录 A · 文件 1。
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from app.domain.circuit import (
    NON_POLAR_TYPES,
    POLARIZED_TYPES,
    THREE_PIN_TYPES,
    PinRole,
)

# ---------------------------------------------------------------------------
# 节点类型枚举
# ---------------------------------------------------------------------------


class ComponentType(str, Enum):  # noqa: UP042 — keep (str, Enum) for py<3.11 compat
    """16 类 component 类型 (one-hot 编码用)。

    取值与 ``app.domain.circuit.norm_component_type`` 的输出对齐；额外预留
    几个目前规范化器尚不直接产出但 schema 上需要支持的类型（OpAmp /
    VoltageSource / CurrentSource / Switch / Sensor），方便后续扩展不破坏
    维度。
    """

    RESISTOR = "Resistor"
    CAPACITOR = "Capacitor"
    CAPACITOR_CERAMIC = "CapacitorCeramic"
    CAPACITOR_ELECTROLYTIC = "CapacitorElectrolytic"
    WIRE = "Wire"
    LED = "LED"
    DIODE = "Diode"
    TRANSISTOR = "Transistor"
    POTENTIOMETER = "Potentiometer"
    IC = "IC"
    OPAMP = "OpAmp"
    VOLTAGE_SOURCE = "VoltageSource"
    CURRENT_SOURCE = "CurrentSource"
    SWITCH = "Switch"
    SENSOR = "Sensor"
    UNKNOWN = "UNKNOWN"


class PortType(str, Enum):  # noqa: UP042 — keep (str, Enum) for py<3.11 compat
    """23 类 port 类型 (one-hot 编码用) — P0.5 IC + Pot 语义稳定化后。

    覆盖：
    - ``app.domain.circuit.PinRole`` 的全部 13 个取值（含 wiper /
      terminal_a / terminal_b for Potentiometer）。
    - ``app.domain.ic_models.UA741_PIN_ROLES`` 的 5 个运放专属角色：
      ``INVERTING_INPUT`` / ``NON_INVERTING_INPUT`` / ``OUTPUT`` /
      ``OFFSET_NULL``（合并 offset_null_1/2）/ ``NC``，以及通用 IC 电源
      ``V_PLUS`` / ``V_MINUS`` 区分于 VCC/GND（运放可能用双电源）。
    - ``PIN1`` / ``PIN2`` / ``PIN_N_GENERIC`` 兜底，对应 normalize_pin_role
      fallback 的 ``pin1`` / ``pin2`` / 数字 pin。
    """

    GENERIC = "generic"  # PinRole.GENERIC fallback
    ANODE = "anode"
    CATHODE = "cathode"
    BASE = "base"
    COLLECTOR = "collector"
    EMITTER = "emitter"
    VCC = "vcc"
    GND = "gnd"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    WIPER = "wiper"
    TERMINAL_A = "terminal_a"
    TERMINAL_B = "terminal_b"
    PIN1 = "pin1"
    PIN2 = "pin2"
    PIN_N_GENERIC = "pin_n_generic"
    # ---- P0.5 新增：IC / op-amp 专属角色 ----
    INVERTING_INPUT = "inverting_input"
    NON_INVERTING_INPUT = "non_inverting_input"
    OUTPUT = "output"
    OFFSET_NULL = "offset_null"
    NC = "nc"
    V_PLUS = "v_plus"
    V_MINUS = "v_minus"


class NetRole(str, Enum):  # noqa: UP042 — keep (str, Enum) for py<3.11 compat
    """6 类 net 角色 (one-hot 编码用)。

    与 ``normalize_net_role`` 的返回值集合对齐：
    {"input", "output", "power", "ground", "signal"} + "unknown" 兜底。

    注意：plan §三 文档把 "power" 写作 "vcc"、"ground" 写作 "gnd" 是同义
    映射；本枚举沿用 logical_reference 现有字符串以避免下游再做翻译。
    """

    INPUT = "input"
    OUTPUT = "output"
    POWER = "power"  # ↔ plan §三 的 "vcc"
    GROUND = "ground"  # ↔ plan §三 的 "gnd"
    SIGNAL = "signal"  # ↔ plan §三 的 "internal"
    UNKNOWN = "unknown"


class PolarityClass(str, Enum):  # noqa: UP042 — keep (str, Enum) for py<3.11 compat
    """元件极性敏感度。"""

    NONE = "none"  # 完全对称 (Resistor, 陶瓷 Capacitor, Wire)
    TWO_POLAR = "two_polar"  # 两脚极性元件 (LED, Diode, 电解 Cap)
    MULTI_ASYMMETRIC = "multi_asymmetric"  # 多脚不可任意互换 (BJT, IC, OpAmp, Pot)


class SourceType(str, Enum):  # noqa: UP042 — keep (str, Enum) for py<3.11 compat
    """边来源 —— 区分 DSL 标注 / 视觉识别 / 角色推断。"""

    DSL = "dsl"
    VISION = "vision"
    INFERRED = "inferred"


class ConnectionPolicy(str, Enum):  # noqa: UP042 — keep (str, Enum) for py<3.11 compat
    """每个 package pin 的连接义务（P0.6 引入）。

    - ``REQUIRED``：必须连接到某个 net；缺连为 ``missing_connection`` 错误。
    - ``OPTIONAL``：可连可不连；不连不算错（如 UA741 offset_null 引脚）。
    - ``FORBIDDEN``：必须不连任何 net；学生若接到任何 net 视为
      ``extra_connection`` / wrong_connection 错误（如 UA741 pin 8 NC）。
    """

    REQUIRED = "required"
    OPTIONAL = "optional"
    FORBIDDEN = "forbidden"


# ---------------------------------------------------------------------------
# 维度常量（plan §三）
# ---------------------------------------------------------------------------

COMPONENT_FEAT_DIM = 30
# P0   : 37 = 16 (port_type) + 16 (parent_ctype) + 5 flags
# P0.5 : 44 = expanded PortType to 23 members
# P0.6 : 50 = + 3 ConnectionPolicy one-hot + 1 has_pin_number + 1 pin_number_log
#              + 1 symmetry_class_size_inverse
PORT_FEAT_DIM = 50
NET_FEAT_DIM = 11
PORT_NET_EDGE_FEAT_DIM = 5
DRNL_LABEL_DIM = 17  # 0..15 + overflow bucket，由 P0.7 SEAL pipeline 使用

# Package 类型 one-hot 槽位（包级元数据，目前后端尚未规范化输出，留 6 槽位
# 占位，未知则全 0）。
PACKAGE_VOCAB = ("DIP8", "DIP14", "SMD", "THT", "AXIAL", "RADIAL")

# ---------------------------------------------------------------------------
# 特征布局（每个 (字段名, 宽度) 元组按声明顺序首尾相接拼成最终向量）。
# 维度宽度的总和必须等于上面声明的 *_FEAT_DIM 常量；test_graph_schema.py 会
# 在 import 期对此自检。
# ---------------------------------------------------------------------------

COMPONENT_FEAT_LAYOUT: list[tuple[str, int]] = [
    ("ctype_one_hot", len(ComponentType)),  # 16
    ("package_one_hot", len(PACKAGE_VOCAB)),  # 6
    ("polarity_class_one_hot", len(PolarityClass)),  # 3
    ("pin_count_log", 1),
    ("value_log10", 1),
    ("value_mask", 1),
    ("confidence", 1),
    ("is_reference", 1),
]

PORT_FEAT_LAYOUT: list[tuple[str, int]] = [
    ("port_type_one_hot", len(PortType)),  # 23
    ("parent_ctype_one_hot", len(ComponentType)),  # 16
    ("polarity_sensitive", 1),
    ("is_power_port", 1),
    ("is_ground_port", 1),
    ("is_floating", 1),
    ("is_reference", 1),
    # ---- P0.6 additions ----
    ("connection_policy_one_hot", len(ConnectionPolicy)),  # 3
    ("has_pin_number", 1),
    # 归一化 pin_number ∈ [0, 1]：log(1 + n) / log(1 + 64)；64 个 pin 已覆盖
    # 几乎所有 DIP/SOP 封装。超出则截到 1.0。
    ("pin_number_log", 1),
    # 1 / symmetry_class_size：1.0 表示该 pin 独立（如 IC 第 3 脚），0.5 表
    # 示与另一个 pin 互换（如 R 的 pin1/pin2），0.33... 表示 3 pin 互换组。
    ("symmetry_class_size_inverse", 1),
]

NET_FEAT_LAYOUT: list[tuple[str, int]] = [
    ("role_one_hot", len(NetRole)),  # 6
    ("degree_log", 1),
    ("is_power_rail", 1),
    ("voltage_hint", 1),
    ("voltage_hint_mask", 1),
    ("is_reference", 1),
]

PORT_NET_EDGE_FEAT_LAYOUT: list[tuple[str, int]] = [
    ("connection_confidence", 1),
    ("source_type_one_hot", len(SourceType)),  # 3
    ("is_observed_in_cur", 1),
]

# ---------------------------------------------------------------------------
# 类型 → 索引映射（one-hot 用）
# ---------------------------------------------------------------------------

CTYPE_TO_INDEX: dict[str, int] = {ct.value: idx for idx, ct in enumerate(ComponentType)}
PORT_TYPE_TO_INDEX: dict[str, int] = {pt.value: idx for idx, pt in enumerate(PortType)}
NET_ROLE_TO_INDEX: dict[str, int] = {nr.value: idx for idx, nr in enumerate(NetRole)}
POLARITY_CLASS_TO_INDEX: dict[str, int] = {
    pc.value: idx for idx, pc in enumerate(PolarityClass)
}
CONNECTION_POLICY_TO_INDEX: dict[str, int] = {
    cp.value: idx for idx, cp in enumerate(ConnectionPolicy)
}

# Pin-number 归一化常量：log(1+pin_number) / log(1+PIN_NUMBER_LOG_BASE)
PIN_NUMBER_LOG_BASE = 64
SOURCE_TYPE_TO_INDEX: dict[str, int] = {
    st.value: idx for idx, st in enumerate(SourceType)
}
PACKAGE_TO_INDEX: dict[str, int] = {pkg: idx for idx, pkg in enumerate(PACKAGE_VOCAB)}

# ---------------------------------------------------------------------------
# 极性元数据 —— 每种 component 的极性敏感度
#
# 与 ``app.domain.circuit`` 的 POLARIZED_TYPES / THREE_PIN_TYPES /
# NON_POLAR_TYPES 一致。注意 circuit.py 把 Transistor 同时放进 NON_POLAR_TYPES
# 和 THREE_PIN_TYPES —— 那是为了在某些规则路径上跳过"两脚极性 swap"检测；
# 在 GNN 语义层面 BJT 三脚是真实非对称（base / collector / emitter 不可互
# 换），因此这里把它归入 MULTI_ASYMMETRIC。
# ---------------------------------------------------------------------------

POLARITY_CLASS_OF: dict[str, PolarityClass] = {}


def _seed_polarity_table() -> None:
    """在模块加载期一次性填充 POLARITY_CLASS_OF；test 覆盖完整性。"""

    multi_asymmetric = set(THREE_PIN_TYPES) | {
        ComponentType.IC.value,
        ComponentType.OPAMP.value,
    }
    for ctype in ComponentType:
        v = ctype.value
        if v in POLARIZED_TYPES:
            POLARITY_CLASS_OF[v] = PolarityClass.TWO_POLAR
        elif v in multi_asymmetric:
            POLARITY_CLASS_OF[v] = PolarityClass.MULTI_ASYMMETRIC
        elif v in NON_POLAR_TYPES:
            # NON_POLAR_TYPES 在 circuit.py 中也包含 Transistor / Potentiometer，
            # 但已被上面的 multi_asymmetric 优先吃掉。
            POLARITY_CLASS_OF[v] = PolarityClass.NONE
        else:
            POLARITY_CLASS_OF[v] = PolarityClass.NONE


_seed_polarity_table()


# ---------------------------------------------------------------------------
# 极性敏感的 PortType 集合（在 polarity_sensitive=True 判定中作为充分条件之
# 一）。
# ---------------------------------------------------------------------------

POLARITY_SENSITIVE_PORT_TYPES: frozenset[str] = frozenset(
    {
        # 两脚极性元件
        PortType.ANODE.value,
        PortType.CATHODE.value,
        PortType.POSITIVE.value,
        PortType.NEGATIVE.value,
        # BJT 三脚（不可任意互换）
        PortType.BASE.value,
        PortType.COLLECTOR.value,
        PortType.EMITTER.value,
        # Potentiometer wiper（与 terminal 不可互换；terminal_a ↔ terminal_b
        # 通常可互换，故仅 wiper 列入）
        PortType.WIPER.value,
        # Op-amp 核心引脚 —— 反相 / 同相输入互换会反转极性
        PortType.INVERTING_INPUT.value,
        PortType.NON_INVERTING_INPUT.value,
        PortType.OUTPUT.value,
        # IC 电源引脚 —— 接反 = 烧片
        PortType.V_PLUS.value,
        PortType.V_MINUS.value,
    }
)

# 电源 / 地相关的 PortType（is_power_port / is_ground_port 判定）。
# V_PLUS / V_MINUS 均视为 power 端口（V_MINUS 在双电源场景是负供电而非地）。
POWER_PORT_TYPES: frozenset[str] = frozenset(
    {PortType.VCC.value, PortType.V_PLUS.value, PortType.V_MINUS.value}
)
GROUND_PORT_TYPES: frozenset[str] = frozenset({PortType.GND.value})


# ---------------------------------------------------------------------------
# 自检：维度常量与布局总和必须一致。任何修改触发 ImportError，避免在测试
# 期才发现。
# ---------------------------------------------------------------------------


def _check_layout_sum(name: str, layout: list[tuple[str, int]], expected: int) -> None:
    actual = sum(width for _, width in layout)
    if actual != expected:
        raise AssertionError(
            f"{name} layout width sum = {actual} but {name}_FEAT_DIM = {expected}"
        )


_check_layout_sum("COMPONENT", COMPONENT_FEAT_LAYOUT, COMPONENT_FEAT_DIM)
_check_layout_sum("PORT", PORT_FEAT_LAYOUT, PORT_FEAT_DIM)
_check_layout_sum("NET", NET_FEAT_LAYOUT, NET_FEAT_DIM)
_check_layout_sum("PORT_NET_EDGE", PORT_NET_EDGE_FEAT_LAYOUT, PORT_NET_EDGE_FEAT_DIM)


# ---------------------------------------------------------------------------
# Pin role 字符串 → PortType 的工具映射
# ---------------------------------------------------------------------------

# 直接映射 PinRole.value → PortType.value（13 项）。
_PIN_ROLE_TO_PORT_TYPE: dict[str, str] = {
    PinRole.GENERIC.value: PortType.GENERIC.value,
    PinRole.ANODE.value: PortType.ANODE.value,
    PinRole.CATHODE.value: PortType.CATHODE.value,
    PinRole.BASE.value: PortType.BASE.value,
    PinRole.COLLECTOR.value: PortType.COLLECTOR.value,
    PinRole.EMITTER.value: PortType.EMITTER.value,
    PinRole.VCC.value: PortType.VCC.value,
    PinRole.GND.value: PortType.GND.value,
    PinRole.POSITIVE.value: PortType.POSITIVE.value,
    PinRole.NEGATIVE.value: PortType.NEGATIVE.value,
    PinRole.WIPER.value: PortType.WIPER.value,
    PinRole.TERMINAL_A.value: PortType.TERMINAL_A.value,
    PinRole.TERMINAL_B.value: PortType.TERMINAL_B.value,
    "pin1": PortType.PIN1.value,
    "pin2": PortType.PIN2.value,
}

# ---------------------------------------------------------------------------
# IC pin maps —— part_subtype → {pin name/number → PortType.value}
#
# 这是 P0.5 的核心：让 GNN 在看到 UA741 的 pin "3" 时不再得到无意义的
# pin_n_generic，而是直接得到 non_inverting_input。
#
# 引用 ``app.domain.ic_models.UA741_PIN_ROLES``（顺序 = pin1..pin8）使得 IC
# 模板的唯一事实源仍在 ic_models.py；本表只是把那里的字符串重映射到 GNN
# 的 PortType vocabulary，并合并 offset_null_1 / offset_null_2 为 OFFSET_NULL。
# ---------------------------------------------------------------------------


def _build_ua741_pin_map() -> dict[str, str]:
    """Construct UA741 pin → PortType map from ``ic_models.UA741_PIN_ROLES``.

    Imports locally to keep ``graph_schema`` free of upstream domain
    dependencies during module load order.
    """

    # Late import: ic_models 不依赖 gnn 模块；此处仅延迟加载以减小耦合。
    from app.domain.ic_models import UA741_PIN_ROLES

    # ic_models 字符串 → PortType.value（offset_null_1/2 合并）
    _alias = {
        "offset_null_1": PortType.OFFSET_NULL.value,
        "offset_null_2": PortType.OFFSET_NULL.value,
        "inverting_input": PortType.INVERTING_INPUT.value,
        "non_inverting_input": PortType.NON_INVERTING_INPUT.value,
        "v_minus": PortType.V_MINUS.value,
        "v_plus": PortType.V_PLUS.value,
        "output": PortType.OUTPUT.value,
        "nc": PortType.NC.value,
    }
    pin_map: dict[str, str] = {}
    for idx, role in enumerate(UA741_PIN_ROLES, start=1):
        pt = _alias.get(role, PortType.PIN_N_GENERIC.value)
        # 同时接受 "3" 和 "pin3" 两种 pin 写法
        pin_map[str(idx)] = pt
        pin_map[f"pin{idx}"] = pt
    return pin_map


def _build_lm358_pin_map() -> dict[str, str]:
    """LM358 是双 op-amp，DIP-8 引脚：

        1 = OUT_A     2 = INV_A     3 = NON_INV_A     4 = V-
        5 = NON_INV_B 6 = INV_B     7 = OUT_B         8 = V+

    所有引脚的 PortType 都已存在于现有 enum（与 UA741 重用），所以加
    LM358 不会触发 PORT_FEAT_DIM 变化，prebaked.pt + P2.5 backbone
    继续兼容。P3 follow-up 用它做 IC subtype 多样性实验。
    """

    return {
        "1": PortType.OUTPUT.value,
        "2": PortType.INVERTING_INPUT.value,
        "3": PortType.NON_INVERTING_INPUT.value,
        "4": PortType.V_MINUS.value,
        "5": PortType.NON_INVERTING_INPUT.value,
        "6": PortType.INVERTING_INPUT.value,
        "7": PortType.OUTPUT.value,
        "8": PortType.V_PLUS.value,
        "pin1": PortType.OUTPUT.value,
        "pin2": PortType.INVERTING_INPUT.value,
        "pin3": PortType.NON_INVERTING_INPUT.value,
        "pin4": PortType.V_MINUS.value,
        "pin5": PortType.NON_INVERTING_INPUT.value,
        "pin6": PortType.INVERTING_INPUT.value,
        "pin7": PortType.OUTPUT.value,
        "pin8": PortType.V_PLUS.value,
    }


IC_PIN_MAPS: dict[str, dict[str, str]] = {
    "UA741": _build_ua741_pin_map(),
    "LM358": _build_lm358_pin_map(),
    # 占位：NE555 待 PortType 扩张后补（需要 TRIGGER / RESET / CONTROL
    # / THRESHOLD / DISCHARGE 等新 PortType，会触发 PORT_FEAT_DIM 升级 +
    # prebaked.pt 失效；放到独立 PR 做）。
}


# Op-amp 友好别名 —— 当 DSL 直接写人话而非 pin 号时（"in-" / "INV" / "v+"）
# 映射到 PortType.value。键统一小写。
_OPAMP_PIN_ALIASES: dict[str, str] = {
    "in-": PortType.INVERTING_INPUT.value,
    "-in": PortType.INVERTING_INPUT.value,
    "inv": PortType.INVERTING_INPUT.value,
    "minus": PortType.INVERTING_INPUT.value,
    "inverting": PortType.INVERTING_INPUT.value,
    "inverting_input": PortType.INVERTING_INPUT.value,
    "in+": PortType.NON_INVERTING_INPUT.value,
    "+in": PortType.NON_INVERTING_INPUT.value,
    "noninv": PortType.NON_INVERTING_INPUT.value,
    "plus": PortType.NON_INVERTING_INPUT.value,
    "non_inverting": PortType.NON_INVERTING_INPUT.value,
    "non_inverting_input": PortType.NON_INVERTING_INPUT.value,
    "out": PortType.OUTPUT.value,
    "output": PortType.OUTPUT.value,
    "v+": PortType.V_PLUS.value,
    "vplus": PortType.V_PLUS.value,
    "v_plus": PortType.V_PLUS.value,
    "vs+": PortType.V_PLUS.value,
    "v-": PortType.V_MINUS.value,
    "vminus": PortType.V_MINUS.value,
    "v_minus": PortType.V_MINUS.value,
    "vee": PortType.V_MINUS.value,
    "vs-": PortType.V_MINUS.value,
    "offset": PortType.OFFSET_NULL.value,
    "offset_null": PortType.OFFSET_NULL.value,
    "nc": PortType.NC.value,
}

# 哪些 ctype 视为 op-amp / IC 上下文（触发 IC_PIN_MAPS + alias 查询）。
_IC_LIKE_CTYPES: frozenset[str] = frozenset(
    {ComponentType.IC.value, ComponentType.OPAMP.value}
)


def normalize_port_type(
    pin_role: str | None,
    component_type: str | None = None,
    *,
    part_subtype: str | None = None,
    pin_raw: str | None = None,
) -> str:
    """把"原始 pin 描述"映射到 ``PortType.value``。

    优先级（P0.5）：

    1. **IC pin map**: 若 ``component_type`` 是 IC/OpAmp 且 ``part_subtype``
       命中 ``IC_PIN_MAPS``，按 ``pin_raw`` （pin 名或编号，原始未小写也
       OK）查表。这是最强证据。
    2. **Op-amp 别名**: 若 ``component_type`` 是 IC/OpAmp（即便 subtype 未
       知），按 ``pin_role`` / ``pin_raw`` 查 ``_OPAMP_PIN_ALIASES``。
    3. **PinRole 直查 / pin1 / pin2**：兼容现有 ``normalize_pin_role`` 输出。
    4. **数字 pin → ``pin_n_generic``**：未知 IC 的高 pin 号兜底。
    5. **其它 → ``generic``**。

    Args:
        pin_role: 来自 ``logical_reference.normalize_pin_role`` 的归一化字符串。
        component_type: 元件类型字符串（``ComponentType.value`` 之一），用
            于触发 IC / op-amp 路径。位置参数，保持向后兼容。
        part_subtype: IC 具体型号，区分大小写无关。例如 ``"UA741"`` /
            ``"LM358"``。**仅在 IC pin map 查表时使用**。
        pin_raw: 原始 pin 名字（如 ``"2"`` 或 ``"pin2"`` 或 ``"in-"``）。
            优先于 ``pin_role`` 用于 IC 查表与 op-amp 别名 —— 因为
            normalize_pin_role 已经会把数字 pin 改写为 ``"pin2"`` 之类的
            语义占位，丢失了原始数字。
    """

    ctype_norm = (component_type or "").strip()
    subtype_key = (part_subtype or "").strip().upper()
    raw_key = (pin_raw or "").strip().lower()
    role_key = (pin_role or "").strip().lower()

    is_ic_like = ctype_norm in _IC_LIKE_CTYPES

    # 1. IC pin map（最强证据）
    if is_ic_like and subtype_key in IC_PIN_MAPS:
        pinmap = IC_PIN_MAPS[subtype_key]
        # 优先 raw（保留 "2" / "pin2" 形态），其次 role
        for candidate in (raw_key, role_key):
            if candidate and candidate in pinmap:
                return pinmap[candidate]

    # 2. Op-amp 别名（subtype 未知或 pin 名是人话写法）
    if is_ic_like:
        for candidate in (raw_key, role_key):
            if candidate and candidate in _OPAMP_PIN_ALIASES:
                return _OPAMP_PIN_ALIASES[candidate]

    # 3. PinRole / pin1 / pin2 直查
    if role_key and role_key in _PIN_ROLE_TO_PORT_TYPE:
        return _PIN_ROLE_TO_PORT_TYPE[role_key]
    if raw_key and raw_key in _PIN_ROLE_TO_PORT_TYPE:
        return _PIN_ROLE_TO_PORT_TYPE[raw_key]

    # 4. 数字 pin
    if role_key.isdigit() or raw_key.isdigit():
        return PortType.PIN_N_GENERIC.value

    # 5. 兜底
    return PortType.GENERIC.value


# ---------------------------------------------------------------------------
# Package pin spec —— 每个 component type 的预期 pin 清单（P0.6 引入）。
#
# 这是 SEAL "next stop" 与 missing_connection 检测的事实源：
# - port_graph.py 用它 materialize 未在 DSL/netlist_v2 中显式连接的 package
#   pin，作为 floating PortNode；
# - candidate-edge generator 用它枚举"该 component 还差哪些 pin 没接"。
# ---------------------------------------------------------------------------


class PinSpec(NamedTuple):
    """单个 package pin 的设计期约束。

    Attributes:
        pin_key: 该 component 上对该 pin 的"规范键"。port_graph 会用它做
            port_key（slug 后）与 spec 匹配 —— 因此命名必须与
            ``logical_reference.normalize_pin_role`` 的输出 / DSL 作者常用
            名字相容（"pin1" / "anode" / "1" / "wiper" / "terminal_a"）。
        port_type: ``PortType.value``。
        connection_policy: ``ConnectionPolicy.value``。
        symmetry_class: 该 component 内部 0-indexed 互换类 id。同 component 内
            ``symmetry_class`` 相同的 pin 视为电气可互换（如 R.pin1 / R.pin2）。
        pin_number: 1-indexed 物理位置；无位置概念则 None（如 LED 的
            "anode" / "cathode"）。IC 与多脚封装强烈推荐填，便于 SEAL 子图
            按位置加 prior。
    """

    pin_key: str
    port_type: str
    connection_policy: str
    symmetry_class: int
    pin_number: int | None


# 非 IC 的静态 spec —— 键为 ``ComponentType.value``。
PACKAGE_PIN_SPECS: dict[str, list[PinSpec]] = {
    ComponentType.RESISTOR.value: [
        PinSpec("pin1", PortType.PIN1.value, ConnectionPolicy.REQUIRED.value, 0, 1),
        PinSpec("pin2", PortType.PIN2.value, ConnectionPolicy.REQUIRED.value, 0, 2),
    ],
    ComponentType.CAPACITOR.value: [
        PinSpec("pin1", PortType.PIN1.value, ConnectionPolicy.REQUIRED.value, 0, 1),
        PinSpec("pin2", PortType.PIN2.value, ConnectionPolicy.REQUIRED.value, 0, 2),
    ],
    ComponentType.CAPACITOR_CERAMIC.value: [
        PinSpec("pin1", PortType.PIN1.value, ConnectionPolicy.REQUIRED.value, 0, 1),
        PinSpec("pin2", PortType.PIN2.value, ConnectionPolicy.REQUIRED.value, 0, 2),
    ],
    ComponentType.CAPACITOR_ELECTROLYTIC.value: [
        PinSpec("positive", PortType.POSITIVE.value, ConnectionPolicy.REQUIRED.value, 0, 1),
        PinSpec("negative", PortType.NEGATIVE.value, ConnectionPolicy.REQUIRED.value, 1, 2),
    ],
    ComponentType.WIRE.value: [
        PinSpec("pin1", PortType.PIN1.value, ConnectionPolicy.REQUIRED.value, 0, 1),
        PinSpec("pin2", PortType.PIN2.value, ConnectionPolicy.REQUIRED.value, 0, 2),
    ],
    ComponentType.LED.value: [
        PinSpec("anode", PortType.ANODE.value, ConnectionPolicy.REQUIRED.value, 0, 1),
        PinSpec("cathode", PortType.CATHODE.value, ConnectionPolicy.REQUIRED.value, 1, 2),
    ],
    ComponentType.DIODE.value: [
        PinSpec("anode", PortType.ANODE.value, ConnectionPolicy.REQUIRED.value, 0, 1),
        PinSpec("cathode", PortType.CATHODE.value, ConnectionPolicy.REQUIRED.value, 1, 2),
    ],
    ComponentType.TRANSISTOR.value: [
        PinSpec("base", PortType.BASE.value, ConnectionPolicy.REQUIRED.value, 0, 1),
        PinSpec("collector", PortType.COLLECTOR.value, ConnectionPolicy.REQUIRED.value, 1, 2),
        PinSpec("emitter", PortType.EMITTER.value, ConnectionPolicy.REQUIRED.value, 2, 3),
    ],
    ComponentType.POTENTIOMETER.value: [
        PinSpec("wiper", PortType.WIPER.value, ConnectionPolicy.REQUIRED.value, 0, 1),
        # terminal_a / terminal_b 在线性 pot 中可互换 → 同 symmetry_class。
        PinSpec("terminal_a", PortType.TERMINAL_A.value, ConnectionPolicy.REQUIRED.value, 1, 2),
        PinSpec("terminal_b", PortType.TERMINAL_B.value, ConnectionPolicy.REQUIRED.value, 1, 3),
    ],
}

# IC 覆盖：connection policy 优先于默认（默认 REQUIRED）。键为
# ``part_subtype.upper()``，值为 ``{pin_key → ConnectionPolicy}``。
IC_PIN_POLICIES: dict[str, dict[str, ConnectionPolicy]] = {
    "UA741": {
        # offset_null 一般可不接（仅在需要消除直流失调时挂一个 trim pot）
        "1": ConnectionPolicy.OPTIONAL,
        "5": ConnectionPolicy.OPTIONAL,
        # pin 8 在 UA741 上是 NC（厂家文档标注 No Connect）—— 必须不接
        "8": ConnectionPolicy.FORBIDDEN,
    },
}

# IC 内部互换组：每组 pin_key 列表表示这些 pin 在该 IC 上可互换。例如
# UA741 的 offset_null_1 / offset_null_2 通常接同一只 trim pot 的两个外端，
# 上下游互换不改电路功能。
IC_PIN_SYMMETRY: dict[str, list[list[str]]] = {
    "UA741": [["1", "5"]],
}


def _ua741_pin_keys(subtype_upper: str) -> list[str]:
    """Return canonical pin keys ("1".."8") for an IC subtype that lives in
    ``IC_PIN_MAPS``. IC pin map keys come in both "N" and "pinN" forms; we
    pick the "N" form as canonical for PinSpec."""

    pinmap = IC_PIN_MAPS.get(subtype_upper, {})
    keys = sorted({k for k in pinmap if k.isdigit()}, key=int)
    return keys


def make_ic_pin_specs(subtype: str | None) -> list[PinSpec] | None:
    """Compose a PinSpec list for an IC instance, combining IC_PIN_MAPS
    (port_type) + IC_PIN_POLICIES (policy overlay) + IC_PIN_SYMMETRY
    (互换组). Returns None if ``subtype`` is unknown."""

    if not subtype:
        return None
    key = subtype.strip().upper()
    if key not in IC_PIN_MAPS:
        return None
    pin_keys = _ua741_pin_keys(key)

    policy_overlay = IC_PIN_POLICIES.get(key, {})
    sym_groups = IC_PIN_SYMMETRY.get(key, [])

    # Build pin_key → symmetry_class: each pin starts in its own class; pins
    # listed in the same sym group are then merged into the lowest class id.
    pin_to_class: dict[str, int] = {pk: idx for idx, pk in enumerate(pin_keys)}
    for group in sym_groups:
        if len(group) < 2:
            continue
        anchor = min(pin_to_class[g] for g in group if g in pin_to_class)
        for g in group:
            if g in pin_to_class:
                pin_to_class[g] = anchor
    # 重新规范化 class id 到 0-indexed 连续空间，便于下游 one-hot / 索引。
    canonical_ids = {cid: idx for idx, cid in enumerate(sorted(set(pin_to_class.values())))}
    pin_to_class = {pk: canonical_ids[cid] for pk, cid in pin_to_class.items()}

    pinmap = IC_PIN_MAPS[key]
    specs: list[PinSpec] = []
    for pk in pin_keys:
        port_type = pinmap[pk]
        policy = policy_overlay.get(pk, ConnectionPolicy.REQUIRED).value
        specs.append(
            PinSpec(
                pin_key=pk,
                port_type=port_type,
                connection_policy=policy,
                symmetry_class=pin_to_class[pk],
                pin_number=int(pk),
            )
        )
    return specs


def get_expected_pin_specs(
    component_type: str,
    part_subtype: str | None = None,
) -> list[PinSpec] | None:
    """统一入口：查 ``component_type`` 与可选 ``part_subtype`` 对应的预期
    pin spec 清单。返回 ``None`` 表示无 spec（GNN 在 materialize phase 跳过
    该 component —— 保持向后兼容，不强行制造 floating port）。"""

    ctype = (component_type or "").strip()
    if ctype in (ComponentType.IC.value, ComponentType.OPAMP.value):
        return make_ic_pin_specs(part_subtype)
    return PACKAGE_PIN_SPECS.get(ctype)


__all__ = [
    # enums
    "ComponentType",
    "PortType",
    "NetRole",
    "PolarityClass",
    "SourceType",
    "ConnectionPolicy",
    # dims
    "COMPONENT_FEAT_DIM",
    "PORT_FEAT_DIM",
    "NET_FEAT_DIM",
    "PORT_NET_EDGE_FEAT_DIM",
    "DRNL_LABEL_DIM",
    # vocabs / layouts
    "PACKAGE_VOCAB",
    "COMPONENT_FEAT_LAYOUT",
    "PORT_FEAT_LAYOUT",
    "NET_FEAT_LAYOUT",
    "PORT_NET_EDGE_FEAT_LAYOUT",
    # index tables
    "CTYPE_TO_INDEX",
    "PORT_TYPE_TO_INDEX",
    "NET_ROLE_TO_INDEX",
    "POLARITY_CLASS_TO_INDEX",
    "SOURCE_TYPE_TO_INDEX",
    "PACKAGE_TO_INDEX",
    "CONNECTION_POLICY_TO_INDEX",
    "PIN_NUMBER_LOG_BASE",
    # metadata
    "POLARITY_CLASS_OF",
    "POLARITY_SENSITIVE_PORT_TYPES",
    "POWER_PORT_TYPES",
    "GROUND_PORT_TYPES",
    "IC_PIN_MAPS",
    "IC_PIN_POLICIES",
    "IC_PIN_SYMMETRY",
    # package specs (P0.6)
    "PinSpec",
    "PACKAGE_PIN_SPECS",
    "make_ic_pin_specs",
    "get_expected_pin_specs",
    # tools
    "normalize_port_type",
]
