"""
Vision label mapping helpers.

统一维护:
- 视觉模型原始类别名 -> 后端标准 component_type
- component_type -> package_type / pin_schema_id / symmetry / pin names / id prefix

这样 S1 / S1.5 / S2 不需要各自写一份类别语义。
"""

from __future__ import annotations


MODEL_CLASS_TO_COMPONENT_TYPE = {
    # Current model labels
    "capacitor_ceramic": "CapacitorCeramic",
    "capacitor_electrolytic": "CapacitorElectrolytic",
    "diode": "Diode",
    "jumper_wire": "Wire",
    "led": "LED",
    "resistor": "Resistor",
    "transistor_3pin": "Transistor",
    # Backward-compatible aliases kept during migration
    "capacitor": "Capacitor",
    "wire": "Wire",
    "ic": "IC",
    "ic_dip8": "IC",
    "ic_dip14": "IC",
    "dip8": "IC",
    "dip14": "IC",
    # detect_components_v2 真实标签格式 (大写 IC + 连字符 + 引脚数).
    # 不加这几条, v2 输出的 IC-8 / IC-14 会在 is_supported_component_type 处被静默过滤.
    "IC-8": "IC",
    "IC-14": "IC",
    "ic-8": "IC",
    "ic-14": "IC",
    "potentiometer": "Potentiometer",
}


COMPONENT_TYPE_TO_PACKAGE_TYPE = {
    "Resistor": "axial_2pin",
    "Wire": "jumper_wire_2pin",
    "LED": "led_2pin",
    "Diode": "diode_2pin",
    "Capacitor": "capacitor_2pin",
    "CapacitorCeramic": "capacitor_ceramic_2pin",
    "CapacitorElectrolytic": "capacitor_electrolytic_2pin",
    "Transistor": "transistor_3pin",
    # IC 默认不写死, 留给 S1 的封装识别 (ic_package_inference) 决定 dip8 / dip14 / unknown.
    "IC": "unknown",
    "Potentiometer": "potentiometer_3pin",
}


COMPONENT_TYPE_TO_PIN_SCHEMA_ID = {
    "CapacitorElectrolytic": "polarized_2pin",
    "Transistor": "fixed_3pins",
    "IC": "ic_dip_ef_bridge",
}


COMPONENT_TYPE_TO_PIN_COUNT = {
    "Resistor": 2,
    "Wire": 2,
    "LED": 2,
    "Diode": 2,
    "Capacitor": 2,
    "CapacitorCeramic": 2,
    "CapacitorElectrolytic": 2,
    "Transistor": 3,
    "IC": 8,
    "Potentiometer": 3,
}


COMPONENT_TYPE_TO_SYMMETRY_GROUP = {
    "Resistor": [["pin1", "pin2"]],
    "Wire": [["pin1", "pin2"]],
    "Capacitor": [["pin1", "pin2"]],
    "CapacitorCeramic": [["pin1", "pin2"]],
}


COMPONENT_TYPE_TO_PREFIX = {
    "Resistor": "R",
    "Capacitor": "C",
    "CapacitorCeramic": "CC",
    "CapacitorElectrolytic": "CE",
    "Wire": "W",
    "LED": "LED",
    "Diode": "D",
    "Transistor": "Q",
    "IC": "IC",
    "Potentiometer": "POT",
}


def normalize_component_type(raw_class_name: str) -> str:
    raw = str(raw_class_name or "").strip()
    if not raw:
        return "UNKNOWN"
    return MODEL_CLASS_TO_COMPONENT_TYPE.get(raw, MODEL_CLASS_TO_COMPONENT_TYPE.get(raw.lower(), raw))


def supported_component_types() -> set[str]:
    return set(COMPONENT_TYPE_TO_PACKAGE_TYPE.keys())


def is_supported_component_type(component_type: str) -> bool:
    return component_type in supported_component_types()


def default_package_type(component_type: str) -> str:
    return COMPONENT_TYPE_TO_PACKAGE_TYPE.get(component_type, "generic")


def default_pin_schema_id(component_type: str, package_type: str) -> str:
    return COMPONENT_TYPE_TO_PIN_SCHEMA_ID.get(component_type, "fixed_pins")


def default_symmetry_group(component_type: str) -> list[list[str]]:
    return [list(group) for group in COMPONENT_TYPE_TO_SYMMETRY_GROUP.get(component_type, [])]


def default_pin_names(component_type: str, pin_count: int) -> list[str]:
    if component_type == "LED" and pin_count >= 2:
        names = ["anode", "cathode"]
        return names[:pin_count] + [f"pin{i}" for i in range(3, pin_count + 1)]
    if component_type == "Diode" and pin_count >= 2:
        names = ["anode", "cathode"]
        return names[:pin_count] + [f"pin{i}" for i in range(3, pin_count + 1)]
    if component_type == "CapacitorElectrolytic" and pin_count >= 2:
        names = ["positive", "negative"]
        return names[:pin_count] + [f"pin{i}" for i in range(3, pin_count + 1)]
    return [f"pin{i}" for i in range(1, pin_count + 1)]


def component_id_prefix(component_type: str) -> str:
    return COMPONENT_TYPE_TO_PREFIX.get(component_type, (component_type[:3].upper() or "CMP"))


def default_pin_count(component_type: str, package_type: str) -> int:
    pkg = (package_type or "").lower()
    if pkg == "dip8":
        return 8
    if pkg == "dip14":
        return 14
    return COMPONENT_TYPE_TO_PIN_COUNT.get(component_type, 2)


def is_pin_order_exchangeable(component_type: str) -> bool:
    return component_type in {"Resistor", "Wire", "Capacitor", "CapacitorCeramic"}
