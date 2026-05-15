"""
Pin schema helpers.

为组件检测结果补齐封装类型、pin schema 和默认 pin 命名。
"""

from __future__ import annotations

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
