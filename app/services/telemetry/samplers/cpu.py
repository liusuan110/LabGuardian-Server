"""CPU + memory sampler backed by `psutil`."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


@dataclass
class CpuSample:
    cpu_pct: float | None
    mem_used_mb: float | None
    mem_total_mb: float | None


class CpuSampler:
    name = "cpu"

    def __init__(self) -> None:
        self._available = psutil is not None
        if self._available:
            # Prime psutil's first non-blocking call (it returns 0.0 on first hit).
            try:
                psutil.cpu_percent(interval=None)
            except Exception:  # pragma: no cover
                self._available = False

    @property
    def status(self) -> str:
        return "ok" if self._available else "unavailable"

    def sample(self) -> CpuSample:
        if not self._available:
            return CpuSample(None, None, None)
        try:
            cpu_pct = float(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            return CpuSample(
                cpu_pct=cpu_pct,
                mem_used_mb=float(mem.used) / (1024.0 * 1024.0),
                mem_total_mb=float(mem.total) / (1024.0 * 1024.0),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("CPU sampler failed: %s", exc)
            return CpuSample(None, None, None)
