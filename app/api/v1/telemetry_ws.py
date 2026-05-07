"""WebSocket endpoint that streams `TelemetryFrame` JSON at `TELEMETRY_HZ`.

Phase 5 — backend only. Front-end consumers (UI dashboards, demo HTML, paper
charts) connect via:

    ws://<host>:8000/ws/telemetry/system

Each client gets its own bounded queue; the sampler never blocks on slow
consumers (oldest frames are dropped instead).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from app.services.telemetry import get_telemetry_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["telemetry"])


@router.websocket("/ws/telemetry/system")
async def telemetry_stream(websocket: WebSocket) -> None:
    service = get_telemetry_service()
    if not service.is_running:
        # Cleanly reject when telemetry is disabled or not started.
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="telemetry_disabled")
        return

    await websocket.accept()
    queue = await service.subscribe(label=str(websocket.client))
    logger.info("[Telemetry] subscriber connected (count=%d)", service.subscriber_count)

    try:
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                # No frame in 10s typically means the sampler is wedged; send
                # a heartbeat so the client can still detect liveness.
                await websocket.send_json({"frame_version": "telemetry_frame_v1", "heartbeat": True})
                continue
            await websocket.send_json(frame.model_dump())
    except WebSocketDisconnect:
        logger.info("[Telemetry] subscriber disconnected")
    finally:
        await service.unsubscribe(queue)


@router.get("/api/v1/telemetry/latest")
async def latest_frame() -> dict:
    """REST helper for quick smoke tests (not the primary push channel)."""
    service = get_telemetry_service()
    if not service.is_running:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="telemetry service is not running",
        )
    frame = service.latest_frame()
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no frame yet",
        )
    return frame.model_dump()
