"""Phase 5 telemetry service + WebSocket integration tests.

Covers:
- TelemetryService start/stop lifecycle and ring buffer fill
- pub/sub: subscriber receives non-empty frames
- macOS / dev laptop degrade: iGPU + NPU samplers report `unavailable` cleanly
- mark_stage propagates to subsequent frames
- TELEMETRY_ENABLED=false short-circuits start
- WebSocket endpoint sends valid TelemetryFrame JSON via FastAPI TestClient
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.services.telemetry import (
    TelemetryFrame,
    get_telemetry_service,
    reset_telemetry_service_for_tests,
)
from app.services.telemetry.service import TelemetryConfig, TelemetryService


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_telemetry_service_for_tests()
    yield
    reset_telemetry_service_for_tests()


@pytest.mark.asyncio
async def test_service_starts_and_fills_ring_buffer() -> None:
    service = TelemetryService(TelemetryConfig(enabled=True, sample_hz=20.0, ring_seconds=5))
    await service.start()
    try:
        await asyncio.sleep(0.25)  # ~5 samples at 20Hz
        snapshot = service.ring_snapshot()
        assert len(snapshot) >= 2
        assert all(isinstance(f, TelemetryFrame) for f in snapshot)
        assert all(f.frame_version == "telemetry_frame_v1" for f in snapshot)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_service_disabled_does_not_start_loop() -> None:
    service = TelemetryService(TelemetryConfig(enabled=False))
    await service.start()
    try:
        assert service.is_running is False
        assert service.latest_frame() is None
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_subscriber_receives_frames() -> None:
    service = TelemetryService(TelemetryConfig(enabled=True, sample_hz=20.0, ring_seconds=5))
    await service.start()
    try:
        queue = await service.subscribe(label="t")
        frame = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert isinstance(frame, TelemetryFrame)
        assert frame.frame_version == "telemetry_frame_v1"
        assert "cpu" in frame.sampler_status
        assert "igpu" in frame.sampler_status
        assert "npu" in frame.sampler_status
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_mark_stage_propagates_to_next_frame() -> None:
    service = TelemetryService(TelemetryConfig(enabled=True, sample_hz=20.0, ring_seconds=5))
    await service.start()
    try:
        service.mark_stage("S2")
        await asyncio.sleep(0.2)
        latest = service.latest_frame()
        assert latest is not None
        assert latest.pipeline_stage == "S2"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_dev_laptop_degrades_gracefully() -> None:
    """On macOS (where /sys is absent) the service must still emit frames
    rather than crashing. iGPU and NPU samplers should report unavailable.
    """
    service = TelemetryService(TelemetryConfig(enabled=True, sample_hz=20.0, ring_seconds=5))
    await service.start()
    try:
        await asyncio.sleep(0.15)
        latest = service.latest_frame()
        assert latest is not None
        # Either the field is None or the sampler is unavailable; never raise.
        if latest.sampler_status["igpu"] == "unavailable":
            assert latest.igpu_pct is None
            assert latest.igpu_freq_mhz is None
        if latest.sampler_status["npu"] == "unavailable":
            assert latest.npu_pct is None
            assert latest.npu_power_mw is None
    finally:
        await service.stop()


def test_websocket_endpoint_streams_frames(monkeypatch) -> None:
    """End-to-end: connect via TestClient, receive at least one frame."""
    from app.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/ws/telemetry/system") as ws:
            text = ws.receive_text()
            payload = json.loads(text)
            # Either a real frame or the initial heartbeat.
            assert payload.get("frame_version") == "telemetry_frame_v1"


def test_rest_latest_endpoint_returns_frame() -> None:
    from app.main import app

    with TestClient(app) as client:
        # Trigger lifespan + give the loop a moment to sample once.
        for _ in range(10):
            response = client.get("/api/v1/telemetry/latest")
            if response.status_code == 200:
                break
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["frame_version"] == "telemetry_frame_v1"
        assert "sampler_status" in data
