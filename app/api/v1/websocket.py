"""
WebSocket 端点

学生端 WebSocket 连接, 教师发送指导时通过此通道实时推送
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.core.deps import get_classroom
from app.services.classroom_state import ClassroomState

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/station/{station_id}")
async def ws_station(websocket: WebSocket, station_id: str):
    """学生端 WebSocket 连接"""
    classroom = get_classroom()
    await websocket.accept()
    classroom.register_websocket(station_id, websocket)
    logger.info(f"[WS] 工位 {station_id} 已连接")

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        logger.info(f"[WS] 工位 {station_id} 已断开")
    finally:
        classroom.unregister_websocket(station_id)
