"""
Pin schema helpers.

为组件检测结果补齐封装类型、pin schema 和默认 pin 命名。
"""

from __future__ import annotations


def default_package_type(component_type: str) -> str:
    c = component_type.lower()
    if c == "resistor":
        return "axial_2pin"
    if c == "wire":
        return "jumper_wire_2pin"
    if c == "led":
        return "led_2pin"
    if c == "diode":
        return "diode_2pin"
    if c == "capacitor":
        return "capacitor_2pin"
    if c == "potentiometer":
        return "potentiometer_3pin"
    if c == "ic":
        return "dip8"
    return "generic"


def default_pin_schema_id(component_type: str, package_type: str) -> str:
    c = component_type.lower()
    if c == "ic" and package_type == "dip8":
        return "dip8_anchor_pair"
    return "fixed_pins"


def default_symmetry_group(component_type: str) -> list[list[str]]:
    c = component_type.lower()
    if c in ("resistor", "wire", "capacitor"):
        return [["pin1", "pin2"]]
    return []


def default_pin_names(component_type: str, pin_count: int) -> list[str]:
    c = component_type.lower()
    if c == "ic":
        return [f"anchor_pin{i}" for i in range(1, pin_count + 1)]
    return [f"pin{i}" for i in range(1, pin_count + 1)]
