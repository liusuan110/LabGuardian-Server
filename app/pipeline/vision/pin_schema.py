"""
Pin schema helpers.

为组件检测结果补齐封装类型、pin schema 和默认 pin 命名。
"""

from __future__ import annotations

from typing import Dict


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


def roi_crop_profile(
    component_type: str,
    package_type: str,
    *,
    view_id: str = "top",
) -> Dict[str, float | int | str]:
    """按封装返回 ROI 裁剪策略.

    目标不是统一 margin, 而是让 ROI 更接近元件真实可能覆盖范围:
    - 轴向器件: 沿主轴更多保留引脚延伸空间
    - DIP: 主体沿长轴紧, 短轴给 pin 排留更多空间
    - side 视图: 在没有真实 side bbox 时允许更保守的正交扩展
    """
    c = component_type.lower()
    p = package_type.lower()
    is_side = view_id != "top"

    profile: Dict[str, float | int | str] = {
        "profile_name": "generic",
        "major_pad_ratio": 0.18,
        "minor_pad_ratio": 0.18,
        "min_major_pad_px": 6,
        "min_minor_pad_px": 6,
        "min_roi_w": 32,
        "min_roi_h": 32,
    }

    if p in {"axial_2pin", "led_2pin", "diode_2pin", "jumper_wire_2pin"} or c in {"resistor", "led", "diode", "wire"}:
        profile.update(
            {
                "profile_name": "axial_lead_extended",
                "major_pad_ratio": 0.42 if not is_side else 0.28,
                "minor_pad_ratio": 0.20 if not is_side else 0.34,
                "min_major_pad_px": 10,
                "min_minor_pad_px": 6,
                "min_roi_w": 64,
                "min_roi_h": 32,
            }
        )
    elif p in {"capacitor_2pin"} or c == "capacitor":
        profile.update(
            {
                "profile_name": "radial_body_with_short_leads",
                "major_pad_ratio": 0.24 if not is_side else 0.18,
                "minor_pad_ratio": 0.30 if not is_side else 0.36,
                "min_major_pad_px": 8,
                "min_minor_pad_px": 8,
                "min_roi_w": 40,
                "min_roi_h": 40,
            }
        )
    elif p in {"potentiometer_3pin"} or c == "potentiometer":
        profile.update(
            {
                "profile_name": "three_pin_body_fanout",
                "major_pad_ratio": 0.24 if not is_side else 0.20,
                "minor_pad_ratio": 0.30 if not is_side else 0.36,
                "min_major_pad_px": 10,
                "min_minor_pad_px": 8,
                "min_roi_w": 56,
                "min_roi_h": 40,
            }
        )
    elif p in {"dip8"} or c == "ic":
        profile.update(
            {
                "profile_name": "dip_body_with_side_pins",
                "major_pad_ratio": 0.14 if not is_side else 0.12,
                "minor_pad_ratio": 0.32 if not is_side else 0.40,
                "min_major_pad_px": 8,
                "min_minor_pad_px": 12,
                "min_roi_w": 72,
                "min_roi_h": 48,
            }
        )

    return profile
