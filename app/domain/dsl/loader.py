from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from app.domain.dsl.core import Circuit

_CACHE: dict[Path, tuple[int, dict[str, Any]]] = {}


def load_dsl_reference(path: Path) -> dict[str, Any]:
    """Execute a reference DSL file and return its compiled payload."""
    resolved = path.resolve()
    mtime_ns = resolved.stat().st_mtime_ns
    cached = _CACHE.get(resolved)
    if cached and cached[0] == mtime_ns:
        return _copy_payload(cached[1])

    module = _load_module(resolved)
    circuit = _extract_circuit(module, resolved)
    payload = circuit.to_logical_reference()
    source = dict(payload.get("source") or {})
    source.setdefault("type", "dsl_python_v1")
    source.setdefault("path", str(resolved))
    payload["source"] = source
    _CACHE[resolved] = (mtime_ns, _copy_payload(payload))
    return payload


def clear_dsl_reference_cache() -> None:
    _CACHE.clear()


def _load_module(path: Path) -> ModuleType:
    module_name = f"labguardian_reference_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载 DSL 参考电路: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _extract_circuit(module: ModuleType, path: Path) -> Circuit:
    candidate = getattr(module, "circuit", None)
    if candidate is None:
        candidate = getattr(module, "c", None)
    if not isinstance(candidate, Circuit):
        raise ValueError(f"{path.name} 必须定义 module-level circuit 或 c = Circuit(...)")
    return candidate


def _copy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(payload)
