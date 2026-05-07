"""Intel iGPU sampler.

Strategy (best-effort, defensive):
1. Read `/sys/class/drm/card0/gt_cur_freq_mhz` for current frequency.
2. Try to read engine busyness from `/sys/class/drm/card0/engine/*/busy`
   (kernel exposes monotonic ns counters; we compute deltas across calls).
3. If sysfs is unavailable (macOS / dev laptops / containers without /sys
   mounted), report `unavailable` and return all-None samples — never raise.

We deliberately avoid spawning `intel_gpu_top` here: subprocess overhead and
permissions complicate the 5Hz loop. Sysfs is enough for an indicator.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns

logger = logging.getLogger(__name__)


_DRM_BASE = Path("/sys/class/drm")


@dataclass
class IgpuSample:
    igpu_pct: float | None
    igpu_freq_mhz: float | None


class IgpuSampler:
    name = "igpu"

    def __init__(self, card_index: int = 0) -> None:
        self._card_dir = _DRM_BASE / f"card{card_index}"
        self._freq_path = self._card_dir / "gt_cur_freq_mhz"
        self._engine_dir = self._card_dir / "engine"
        self._available = self._card_dir.is_dir() and (
            self._freq_path.is_file() or self._engine_dir.is_dir()
        )
        # Track previous busy counters (per-engine ns) for delta-based %.
        self._prev_busy_total_ns: int | None = None
        self._prev_wall_ns: int | None = None
        self._engine_busy_paths: list[Path] = []
        if self._engine_dir.is_dir():
            try:
                self._engine_busy_paths = [
                    p / "busy" for p in self._engine_dir.iterdir() if (p / "busy").is_file()
                ]
            except OSError:
                self._engine_busy_paths = []

    @property
    def status(self) -> str:
        return "ok" if self._available else "unavailable"

    def sample(self) -> IgpuSample:
        if not self._available:
            return IgpuSample(None, None)
        freq = _safe_read_float(self._freq_path)
        pct = self._sample_busy_pct()
        return IgpuSample(igpu_pct=pct, igpu_freq_mhz=freq)

    def _sample_busy_pct(self) -> float | None:
        if not self._engine_busy_paths:
            return None
        try:
            current_busy_ns = 0
            for path in self._engine_busy_paths:
                value = _safe_read_int(path)
                if value is None:
                    continue
                current_busy_ns += value
        except OSError:
            return None

        wall_ns = monotonic_ns()
        prev_busy = self._prev_busy_total_ns
        prev_wall = self._prev_wall_ns
        self._prev_busy_total_ns = current_busy_ns
        self._prev_wall_ns = wall_ns

        if prev_busy is None or prev_wall is None:
            return None
        wall_delta = wall_ns - prev_wall
        if wall_delta <= 0:
            return None
        busy_delta = max(0, current_busy_ns - prev_busy)
        pct = 100.0 * busy_delta / wall_delta
        # Across multiple engines we may exceed 100% — clamp for UI sanity.
        return min(100.0 * len(self._engine_busy_paths), max(0.0, pct))


def _safe_read_float(path: Path) -> float | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return float(fh.read().strip())
    except (OSError, ValueError):
        return None


def _safe_read_int(path: Path) -> int | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


# Used by tests to override the discovery root.
def _set_drm_base_for_tests(path: os.PathLike[str] | str) -> None:
    global _DRM_BASE
    _DRM_BASE = Path(path)
