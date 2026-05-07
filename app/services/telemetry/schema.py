"""Pydantic schema for one telemetry frame pushed to subscribers.

Versioned via `frame_version="telemetry_frame_v1"` so the front-end can
contract-test against a stable shape before each release.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SamplerStatus = Literal["ok", "degraded", "unavailable", "error"]


class TelemetryFrame(BaseModel):
    """One sample of system telemetry.

    Any field marked Optional may be `None` when its sampler is unavailable
    (e.g. NPU on a dev laptop) — the consumer must tolerate `None` rather than
    crash. `sampler_status` exposes per-sampler health so a UI can dim
    unavailable lanes instead of plotting zero.
    """

    frame_version: Literal["telemetry_frame_v1"] = "telemetry_frame_v1"
    ts: float = Field(..., description="Unix epoch seconds at sample time")

    # CPU / memory
    cpu_pct: float | None = Field(default=None, description="CPU utilization %")
    mem_used_mb: float | None = Field(default=None, description="Used RAM in MiB")
    mem_total_mb: float | None = Field(default=None, description="Total RAM in MiB")

    # iGPU (Intel Xe / Arc)
    igpu_pct: float | None = Field(default=None, description="iGPU render-engine busy %")
    igpu_freq_mhz: float | None = Field(default=None, description="iGPU current frequency in MHz")

    # NPU
    npu_pct: float | None = Field(default=None, description="NPU utilization %")
    npu_power_mw: float | None = Field(default=None, description="NPU power draw in milliwatts")

    # Pipeline correlation: latest stage marked via `mark_stage()`.
    pipeline_stage: str = ""

    # Per-sampler health (one entry per sampler in the registry).
    sampler_status: dict[str, SamplerStatus] = Field(default_factory=dict)
