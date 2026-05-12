from __future__ import annotations

from typing import Any

SUPPORTED_REFERENCE_FORMAT = "logical_reference_v1"
DSL_REFERENCE_SOURCE_TYPE = "dsl_python_v1"
LEGACY_REFERENCE_FORMATS = {"labguardian_ref_v4", "netlist_v2"}


def get_reference_format(payload: Any) -> str | None:
    """Return the declared or inferred reference format."""
    if not isinstance(payload, dict):
        return None
    explicit = payload.get("format")
    if explicit:
        return str(explicit)
    meta = payload.get("meta")
    if isinstance(meta, dict) and meta.get("format"):
        return str(meta.get("format"))
    if "netlist_v2" in payload:
        return "labguardian_ref_v4"
    if "components" in payload and "nets" in payload:
        return "netlist_v2"
    return None


def unsupported_reference_format_message(actual_format: str | None) -> str:
    actual = actual_format or "unknown"
    if actual in LEGACY_REFERENCE_FORMATS:
        return (
            f"不支持旧参考电路格式: {actual}。"
            f"当前仅支持 {SUPPORTED_REFERENCE_FORMAT}；请先迁移参考文件。"
        )
    return (
        f"不支持的参考电路格式: {actual}。"
        f"当前仅支持 {SUPPORTED_REFERENCE_FORMAT}。"
    )


def ensure_supported_reference_format(payload: Any) -> None:
    actual = get_reference_format(payload)
    if actual != SUPPORTED_REFERENCE_FORMAT:
        raise ValueError(unsupported_reference_format_message(actual))


def is_dsl_compiled_reference(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    source = payload.get("source")
    return isinstance(source, dict) and source.get("type") == DSL_REFERENCE_SOURCE_TYPE
