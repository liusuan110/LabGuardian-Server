"""Hardware telemetry service for the DK-2500 edge device.

Phase 5 ships a backend-only WebSocket pusher: 5Hz heterogeneous sampling
(CPU / memory / iGPU / NPU) with a ring buffer and pub/sub fanout.

Front-end is intentionally out of scope; consumers connect to
`/ws/telemetry/system` and receive Pydantic-validated JSON frames.
"""

from app.services.telemetry.schema import TelemetryFrame, SamplerStatus
from app.services.telemetry.service import (
    TelemetryService,
    get_telemetry_service,
    reset_telemetry_service_for_tests,
)

__all__ = [
    "TelemetryFrame",
    "SamplerStatus",
    "TelemetryService",
    "get_telemetry_service",
    "reset_telemetry_service_for_tests",
]
