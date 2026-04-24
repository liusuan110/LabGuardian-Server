"""
Pin schema helpers.

为组件检测结果补齐封装类型、pin schema 和默认 pin 命名。
"""

from __future__ import annotations

from typing import Dict

from app.pipeline.vision.label_mapping import (
    default_package_type as mapped_default_package_type,
    default_pin_names as mapped_default_pin_names,
    default_pin_schema_id as mapped_default_pin_schema_id,
    default_symmetry_group as mapped_default_symmetry_group,
)


def default_package_type(component_type: str) -> str:
    return mapped_default_package_type(component_type)


def default_pin_schema_id(component_type: str, package_type: str) -> str:
    return mapped_default_pin_schema_id(component_type, package_type)


def default_symmetry_group(component_type: str) -> list[list[str]]:
    return mapped_default_symmetry_group(component_type)


def default_pin_names(component_type: str, pin_count: int) -> list[str]:
    return mapped_default_pin_names(component_type, pin_count)


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
        "expand_mode": "body_bbox_expand",
        "major_pad_ratio": 0.18,
        "minor_pad_ratio": 0.18,
        "major_pad_before_ratio": 0.18,
        "major_pad_after_ratio": 0.18,
        "minor_pad_before_ratio": 0.18,
        "minor_pad_after_ratio": 0.18,
        "min_major_pad_px": 6,
        "min_minor_pad_px": 6,
        "min_major_span_px": 48,
        "min_minor_span_px": 48,
        "min_roi_w": 32,
        "min_roi_h": 32,
    }

    if p in {"led_2pin"} or c in {"led"}:
        profile.update(
            {
                "profile_name": "led_body_with_extended_leads",
                "major_pad_ratio": 1.10 if not is_side else 0.64,
                "minor_pad_ratio": 0.42 if not is_side else 0.52,
                "major_pad_before_ratio": 1.10 if not is_side else 0.64,
                "major_pad_after_ratio": 1.10 if not is_side else 0.64,
                "minor_pad_before_ratio": 0.42 if not is_side else 0.52,
                "minor_pad_after_ratio": 0.42 if not is_side else 0.52,
                "min_major_pad_px": 28,
                "min_minor_pad_px": 12,
                "min_major_span_px": 300 if not is_side else 180,
                "min_minor_span_px": 180 if not is_side else 120,
                "min_roi_w": 144,
                "min_roi_h": 64,
            }
        )
    elif p in {"axial_2pin"} or c in {"resistor"}:
        profile.update(
            {
                "profile_name": "axial_resistor_body_with_leads",
                "major_pad_ratio": 1.15 if not is_side else 0.58,
                "minor_pad_ratio": 0.26 if not is_side else 0.44,
                "major_pad_before_ratio": 1.15 if not is_side else 0.58,
                "major_pad_after_ratio": 1.15 if not is_side else 0.58,
                "minor_pad_before_ratio": 0.26 if not is_side else 0.44,
                "minor_pad_after_ratio": 0.26 if not is_side else 0.44,
                "min_major_pad_px": 26,
                "min_minor_pad_px": 10,
                "min_major_span_px": 320 if not is_side else 180,
                "min_minor_span_px": 140 if not is_side else 110,
                "min_roi_w": 140,
                "min_roi_h": 56,
            }
        )
    elif p in {"diode_2pin"} or c in {"diode"}:
        profile.update(
            {
                "profile_name": "diode_body_with_leads",
                "major_pad_ratio": 1.05 if not is_side else 0.56,
                "minor_pad_ratio": 0.24 if not is_side else 0.42,
                "major_pad_before_ratio": 1.05 if not is_side else 0.56,
                "major_pad_after_ratio": 1.05 if not is_side else 0.56,
                "minor_pad_before_ratio": 0.24 if not is_side else 0.42,
                "minor_pad_after_ratio": 0.24 if not is_side else 0.42,
                "min_major_pad_px": 22,
                "min_minor_pad_px": 10,
                "min_major_span_px": 300 if not is_side else 180,
                "min_minor_span_px": 150 if not is_side else 120,
                "min_roi_w": 132,
                "min_roi_h": 56,
            }
        )
    elif p in {"jumper_wire_2pin"} or c in {"wire", "jumper_wire"}:
        profile.update(
            {
                "profile_name": "jumper_segment_with_terminals",
                "major_pad_ratio": 0.62 if not is_side else 0.42,
                "minor_pad_ratio": 0.26 if not is_side else 0.36,
                "min_major_pad_px": 14,
                "min_minor_pad_px": 8,
                "min_major_span_px": 220 if not is_side else 150,
                "min_minor_span_px": 120 if not is_side else 90,
                "min_roi_w": 104,
                "min_roi_h": 40,
            }
        )
    elif p in {"capacitor_ceramic_2pin"} or c in {"capacitorceramic", "capacitor_ceramic"}:
        profile.update(
            {
                "profile_name": "ceramic_cap_body_with_extended_leads",
                "major_pad_ratio": 1.10 if not is_side else 0.32,
                "minor_pad_ratio": 0.26 if not is_side else 0.56,
                "major_pad_before_ratio": 1.10 if not is_side else 0.32,
                "major_pad_after_ratio": 1.10 if not is_side else 0.32,
                "minor_pad_before_ratio": 0.26 if not is_side else 0.56,
                "minor_pad_after_ratio": 0.26 if not is_side else 0.56,
                "min_major_pad_px": 18,
                "min_minor_pad_px": 12,
                "min_major_span_px": 240 if not is_side else 130,
                "min_minor_span_px": 120 if not is_side else 130,
                "min_roi_w": 84,
                "min_roi_h": 72,
            }
        )
    elif p in {"capacitor_electrolytic_2pin"} or c in {"capacitorelectrolytic", "capacitor_electrolytic"}:
        profile.update(
            {
                "profile_name": "electrolytic_cap_body_with_leads",
                "major_pad_ratio": 0.58 if not is_side else 0.36,
                "minor_pad_ratio": 0.60 if not is_side else 0.62,
                "min_major_pad_px": 16,
                "min_minor_pad_px": 14,
                "min_major_span_px": 220 if not is_side else 160,
                "min_minor_span_px": 220 if not is_side else 160,
                "min_roi_w": 96,
                "min_roi_h": 84,
            }
        )
    elif p in {"capacitor_2pin"} or c in {"capacitor"}:
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
    elif p in {"potentiometer_3pin"} or c in {"potentiometer"}:
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
    elif p in {"transistor_3pin"} or c in {"transistor", "transistor_3pin"}:
        profile.update(
            {
                "profile_name": "three_pin_semiconductor",
                "major_pad_ratio": 0.56 if not is_side else 0.38,
                "minor_pad_ratio": 0.62 if not is_side else 0.68,
                "min_major_pad_px": 16,
                "min_minor_pad_px": 14,
                "min_major_span_px": 240 if not is_side else 180,
                "min_minor_span_px": 220 if not is_side else 180,
                "min_roi_w": 96,
                "min_roi_h": 96,
            }
        )

    return profile
