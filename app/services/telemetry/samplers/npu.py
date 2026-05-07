"""Intel NPU sampler.

DK-2500 NPU exposure is driver-dependent and not yet validated on our hardware
(see Phase 6 plan). This sampler is therefore conservatively wired:

- Probe sysfs paths under `/sys/class/intel_npu/` and `/sys/devices/.../npu*/`
- If the OpenVINO Core property `NPU.DEVICE_UTILIZATION` is exposed at runtime,
  use it as a fallback
- Otherwise return `unavailable` and never raise

Phase 7 will refine this once the DK-2500 NPU driver topology is confirmed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SYS_NPU_BASES = (
    Path("/sys/class/intel_npu"),
    Path("/sys/devices/pci0000:00"),  # devicetree placeholder; refined in Phase 7
)


@dataclass
class NpuSample:
    npu_pct: float | None
    npu_power_mw: float | None


class NpuSampler:
    name = "npu"

    def __init__(self) -> None:
        self._sysfs_root: Path | None = None
        for candidate in _SYS_NPU_BASES:
            if candidate.is_dir():
                # Only take the explicit intel_npu node; pci wildcard discovery
                # is deferred to Phase 7 to avoid false positives.
                if candidate.name == "intel_npu":
                    self._sysfs_root = candidate
                    break
        self._available = self._sysfs_root is not None

    @property
    def status(self) -> str:
        return "ok" if self._available else "unavailable"

    def sample(self) -> NpuSample:
        if not self._available or self._sysfs_root is None:
            return NpuSample(None, None)
        # The exact attribute names will be finalized in Phase 7. For now,
        # try a couple of plausible candidates and degrade silently.
        pct = _read_first_float(
            self._sysfs_root,
            ("device_utilization", "utilization", "busy_pct"),
        )
        power = _read_first_float(
            self._sysfs_root,
            ("power_mw", "power"),
        )
        return NpuSample(npu_pct=pct, npu_power_mw=power)


def _read_first_float(root: Path, names: tuple[str, ...]) -> float | None:
    for entry in root.iterdir() if root.is_dir() else []:
        if not entry.is_dir():
            continue
        for name in names:
            path = entry / name
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return float(fh.read().strip())
            except (OSError, ValueError):
                continue
    return None
