"""Singleton TelemetryService: periodic sampling + ring buffer + pub/sub fanout.

Lifecycle:
- Started inside FastAPI lifespan when `TELEMETRY_ENABLED=true`
- Owns one asyncio task that samples at `TELEMETRY_HZ` (default 5 Hz)
- Each subscriber gets its own bounded `asyncio.Queue`; slow consumers drop
  oldest frames instead of stalling the sampler
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from app.services.telemetry.samplers import CpuSampler, IgpuSampler, NpuSampler
from app.services.telemetry.schema import SamplerStatus, TelemetryFrame

logger = logging.getLogger(__name__)


_DEFAULT_RING_SECONDS = 120
_PUBSUB_QUEUE_MAXSIZE = 16  # ~3s of buffered frames at 5Hz; old frames dropped on overflow


@dataclass
class _Subscriber:
    queue: asyncio.Queue[TelemetryFrame]
    label: str = ""


@dataclass
class TelemetryConfig:
    enabled: bool = True
    sample_hz: float = 5.0
    ring_seconds: int = _DEFAULT_RING_SECONDS


class TelemetryService:
    """Periodic sampler + pub/sub for hardware metrics."""

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        self._config = config or TelemetryConfig()
        self._cpu = CpuSampler()
        self._igpu = IgpuSampler()
        self._npu = NpuSampler()
        ring_capacity = max(10, int(self._config.ring_seconds * max(1.0, self._config.sample_hz)))
        self._ring: deque[TelemetryFrame] = deque(maxlen=ring_capacity)
        self._subscribers: list[_Subscriber] = []
        self._task: asyncio.Task | None = None
        self._running = False
        self._pipeline_stage = ""
        self._lock = asyncio.Lock()

    # ---- lifecycle ----

    @property
    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    async def start(self) -> None:
        if not self._config.enabled:
            logger.info("Telemetry service disabled by config; skipping start")
            return
        if self.is_running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="telemetry-loop")
        logger.info("Telemetry service started (hz=%s)", self._config.sample_hz)

    async def stop(self) -> None:
        self._running = False
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover
                logger.warning("Telemetry loop exit error: %s", exc)
        async with self._lock:
            self._subscribers.clear()
        logger.info("Telemetry service stopped")

    # ---- pub/sub ----

    async def subscribe(self, label: str = "") -> asyncio.Queue[TelemetryFrame]:
        queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=_PUBSUB_QUEUE_MAXSIZE)
        async with self._lock:
            self._subscribers.append(_Subscriber(queue=queue, label=label))
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[TelemetryFrame]) -> None:
        async with self._lock:
            self._subscribers = [s for s in self._subscribers if s.queue is not queue]

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    # ---- pipeline correlation ----

    def mark_stage(self, stage_name: str) -> None:
        """Cheap, non-async hook for pipeline orchestrator boundary marks."""
        self._pipeline_stage = str(stage_name or "")

    # ---- introspection ----

    def latest_frame(self) -> Optional[TelemetryFrame]:
        return self._ring[-1] if self._ring else None

    def ring_snapshot(self) -> list[TelemetryFrame]:
        return list(self._ring)

    # ---- internals ----

    async def _run_loop(self) -> None:
        period = 1.0 / max(0.1, self._config.sample_hz)
        next_tick = asyncio.get_event_loop().time()
        while self._running:
            try:
                frame = self._sample_once()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Telemetry sample error: %s", exc)
                frame = TelemetryFrame(ts=time.time(), pipeline_stage=self._pipeline_stage)
            self._ring.append(frame)
            await self._fanout(frame)

            next_tick += period
            now = asyncio.get_event_loop().time()
            sleep = next_tick - now
            if sleep < 0:
                # We fell behind; reset the schedule.
                next_tick = now
                sleep = 0
            try:
                await asyncio.sleep(sleep)
            except asyncio.CancelledError:
                raise

    def _sample_once(self) -> TelemetryFrame:
        cpu = self._cpu.sample()
        igpu = self._igpu.sample()
        npu = self._npu.sample()
        sampler_status: dict[str, SamplerStatus] = {
            "cpu": _to_status(self._cpu.status),
            "igpu": _to_status(self._igpu.status),
            "npu": _to_status(self._npu.status),
        }
        return TelemetryFrame(
            ts=time.time(),
            cpu_pct=cpu.cpu_pct,
            mem_used_mb=cpu.mem_used_mb,
            mem_total_mb=cpu.mem_total_mb,
            igpu_pct=igpu.igpu_pct,
            igpu_freq_mhz=igpu.igpu_freq_mhz,
            npu_pct=npu.npu_pct,
            npu_power_mw=npu.npu_power_mw,
            pipeline_stage=self._pipeline_stage,
            sampler_status=sampler_status,
        )

    async def _fanout(self, frame: TelemetryFrame) -> None:
        async with self._lock:
            subscribers = list(self._subscribers)
        for sub in subscribers:
            queue = sub.queue
            if queue.full():
                # Drop oldest to keep the queue current; never block the sampler.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover
                    pass
            try:
                queue.put_nowait(frame)
            except asyncio.QueueFull:  # pragma: no cover
                pass


def _to_status(raw: str) -> SamplerStatus:
    if raw in {"ok", "degraded", "unavailable", "error"}:
        return raw  # type: ignore[return-value]
    return "error"


# ---- module-level singleton ----

_singleton: TelemetryService | None = None


def get_telemetry_service() -> TelemetryService:
    global _singleton
    if _singleton is None:
        from app.core.config import settings

        config = TelemetryConfig(
            enabled=getattr(settings, "TELEMETRY_ENABLED", True),
            sample_hz=getattr(settings, "TELEMETRY_HZ", 5.0),
            ring_seconds=getattr(settings, "TELEMETRY_RING_SECONDS", _DEFAULT_RING_SECONDS),
        )
        _singleton = TelemetryService(config=config)
    return _singleton


def reset_telemetry_service_for_tests() -> None:
    """Drop the cached singleton; subsequent `get_telemetry_service()` rebuilds."""
    global _singleton
    _singleton = None
