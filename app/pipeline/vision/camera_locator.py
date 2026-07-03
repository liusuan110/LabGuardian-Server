"""Linux V4L2 camera locator helpers.

Resolves stable USB port hints such as ``usb-0000:00:14.0-3`` to the actual
``/dev/videoN`` capture node by inspecting ``/sys/class/video4linux``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional


VIDEO4LINUX_CLASS_ROOT = Path("/sys/class/video4linux")


@dataclass(frozen=True)
class VideoDeviceInfo:
    dev_node: str
    device_index: int
    sysfs_name: str
    usb_device_path: str
    capture_index: Optional[int] = None


def list_video_devices(
    *, class_root: Path = VIDEO4LINUX_CLASS_ROOT,
) -> list[VideoDeviceInfo]:
    if not class_root.exists():
        return []

    devices: list[VideoDeviceInfo] = []
    for entry in sorted(class_root.glob("video*")):
        match = re.fullmatch(r"video(\d+)", entry.name)
        if match is None:
            continue
        device_index = int(match.group(1))
        resolved_device = (entry / "device").resolve(strict=False)
        capture_index = _read_optional_int(entry / "index")
        devices.append(
            VideoDeviceInfo(
                dev_node=f"/dev/{entry.name}",
                device_index=device_index,
                sysfs_name=entry.name,
                usb_device_path=str(resolved_device),
                capture_index=capture_index,
            )
        )
    return devices


def resolve_video_device_index_by_usb_hint(
    usb_hint: str,
    *,
    class_root: Path = VIDEO4LINUX_CLASS_ROOT,
) -> int:
    normalized_hint = _normalize_usb_hint(usb_hint)
    if not normalized_hint:
        raise ValueError(f"invalid usb hint: {usb_hint!r}")

    candidates = [
        item for item in list_video_devices(class_root=class_root)
        if _matches_usb_hint(item.usb_device_path, normalized_hint)
    ]
    if not candidates:
        raise ValueError(f"no /dev/video* matches usb hint {usb_hint!r}")

    candidates.sort(
        key=lambda item: (
            0 if item.capture_index == 0 else 1,
            item.capture_index if item.capture_index is not None else 99,
            item.device_index,
        )
    )
    return candidates[0].device_index


def _normalize_usb_hint(usb_hint: str) -> str:
    hint = str(usb_hint or "").strip()
    if not hint:
        return ""

    # Accept:
    # - usb-0000:00:14.0-3   -> 3
    # - usb-0000:00:14.0-3.2 -> 3.2
    # - 1-3 / 1-3.2          -> 3 / 3.2
    # - 3 / 3.2              -> 3 / 3.2
    if hint.startswith("usb-"):
        tail = hint.rsplit("-", 1)[-1]
        return tail.strip()
    if "-" in hint:
        return hint.rsplit("-", 1)[-1].strip()
    return hint


def _matches_usb_hint(usb_device_path: str, normalized_hint: str) -> bool:
    suffix = f"-{normalized_hint}"
    return any(part.endswith(suffix) for part in Path(usb_device_path).parts)


def _read_optional_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
