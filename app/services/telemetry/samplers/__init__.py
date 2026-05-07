"""Per-device samplers. Each sampler MUST be defensive: never raise upward."""

from app.services.telemetry.samplers.cpu import CpuSampler
from app.services.telemetry.samplers.igpu import IgpuSampler
from app.services.telemetry.samplers.npu import NpuSampler

__all__ = ["CpuSampler", "IgpuSampler", "NpuSampler"]
