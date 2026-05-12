from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.domain.reference_formats import (
    SUPPORTED_REFERENCE_FORMAT,
    get_reference_format,
    unsupported_reference_format_message,
)


_REFERENCE_DIR = PROJECT_ROOT / "knowledge" / "references"

# 安全的 reference_id：只允许字母、数字、下划线、连字符、点
_SAFE_REF_ID_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")


class ReferenceService:
    """后端统一维护参考电路 JSON 文件的服务。"""

    def __init__(self, reference_dir: Path | None = None):
        self.reference_dir = reference_dir or _REFERENCE_DIR

    def list_references(self) -> list[dict[str, Any]]:
        """扫描 knowledge/references/*.json，返回所有 logical_reference_v1 文件摘要。"""
        results: list[dict[str, Any]] = []
        if not self.reference_dir.exists():
            return results

        for path in sorted(self.reference_dir.glob("*.json")):
            try:
                payload = self._load_file(path)
            except Exception:
                continue
            if get_reference_format(payload) != SUPPORTED_REFERENCE_FORMAT:
                continue
            components = payload.get("components", [])
            nets = payload.get("nets", [])
            results.append(
                {
                    "reference_id": payload.get("reference_id", path.stem),
                    "name": payload.get("name", ""),
                    "description": payload.get("description", ""),
                    "format": SUPPORTED_REFERENCE_FORMAT,
                    "component_count": len(components),
                    "net_count": len(nets),
                }
            )
        return results

    def load_reference(self, reference_id: str) -> dict[str, Any]:
        """根据 reference_id 加载完整参考电路 JSON。"""
        if not _SAFE_REF_ID_RE.match(reference_id):
            raise ValueError(f"非法 reference_id: {reference_id}")

        path = self.reference_dir / f"{reference_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"参考电路未找到: {reference_id}")

        payload = self._load_file(path)
        self._validate_payload(payload, expected_reference_id=reference_id)
        return payload

    def validate_reference(self, payload: dict[str, Any]) -> None:
        """校验 logical_reference_v1 格式。"""
        self._validate_payload(payload)

    # ---- 内部方法 ----

    @staticmethod
    def _load_file(path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("参考电路 JSON 顶层必须是对象")
        return data

    @staticmethod
    def _validate_payload(
        payload: dict[str, Any], *, expected_reference_id: str | None = None
    ) -> None:
        actual_format = get_reference_format(payload)
        if actual_format != SUPPORTED_REFERENCE_FORMAT:
            raise ValueError(unsupported_reference_format_message(actual_format))

        if expected_reference_id is not None:
            actual = payload.get("reference_id")
            if actual != expected_reference_id:
                raise ValueError(
                    f"reference_id 不匹配: 文件名为 {expected_reference_id}，"
                    f"JSON 内为 {actual}"
                )

        components = payload.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError("components 必须是非空数组")

        nets = payload.get("nets", [])
        if not isinstance(nets, list):
            raise ValueError("nets 如果存在必须是数组")

        net_names: set[str] = set()
        for idx, net in enumerate(nets):
            if not isinstance(net, dict):
                raise ValueError(f"nets[{idx}] 必须是对象")
            net_name = str(net.get("net") or "").strip()
            if not net_name:
                raise ValueError(f"nets[{idx}] 必须包含 net")
            net_names.add(net_name)

        comp_ids: set[str] = set()
        for idx, comp in enumerate(components):
            if not isinstance(comp, dict):
                raise ValueError(f"components[{idx}] 必须是对象")
            ref_id = str(comp.get("ref_id") or "").strip()
            if not ref_id:
                raise ValueError(f"components[{idx}] 必须包含 ref_id")
            if ref_id in comp_ids:
                raise ValueError(f"存在重复 ref_id: {ref_id}")
            comp_ids.add(ref_id)

            comp_type = str(comp.get("type") or "").strip()
            if not comp_type:
                raise ValueError(f"components[{idx}] 必须包含 type")

            pins = comp.get("pins")
            if not isinstance(pins, list) or not pins:
                raise ValueError(f"components[{idx}] 的 pins 必须是非空数组")

            for pidx, pin in enumerate(pins):
                if not isinstance(pin, dict):
                    raise ValueError(f"components[{idx}].pins[{pidx}] 必须是对象")
                pin_name = str(pin.get("pin") or "").strip()
                if not pin_name:
                    raise ValueError(f"components[{idx}].pins[{pidx}] 必须包含 pin")
                if pin.get("nc") is True:
                    continue
                pin_net = str(pin.get("net") or "").strip()
                if not pin_net:
                    raise ValueError(f"components[{idx}].pins[{pidx}] 必须包含 net")
                if net_names and pin_net not in net_names:
                    raise ValueError(
                        f"components[{idx}] 的 pin '{pin_name}' 引用了未定义的 net '{pin_net}'"
                    )
