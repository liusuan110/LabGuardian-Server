"""
教室 REST API (← teacher/server.py)

拆分为 APIRouter, 符合 full-stack-fastapi-template 模式
"""

from __future__ import annotations

import threading
from typing import Dict

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.core.deps import get_classroom
from app.schemas.classroom import (
    BroadcastMessage,
    GuidanceMessage,
    StationHeartbeat,
)
from app.services.classroom_state import ClassroomState

router = APIRouter(prefix="/classroom", tags=["classroom"])

# 最新帧缓存
_frame_lock = threading.Lock()
_frame_cache: Dict[str, str] = {}


# ---- 学生端上报 ----


@router.post("/heartbeat")
async def receive_heartbeat(
    heartbeat: StationHeartbeat,
    classroom: ClassroomState = Depends(get_classroom),
):
    """接收学生工位心跳 (每 2 秒)"""
    data = heartbeat.model_dump()
    new_alerts = classroom.update_station(data)

    thumb = data.get("thumbnail_b64", "")
    if thumb:
        with _frame_lock:
            _frame_cache[heartbeat.station_id] = thumb

    return {"status": "ok", "new_alerts": len(new_alerts)}


# ---- 教师端查询 ----


@router.get("/stations")
async def get_all_stations(classroom: ClassroomState = Depends(get_classroom)):
    """获取全班工位状态"""
    return classroom.get_all_stations()


@router.get("/ranking")
async def get_ranking(classroom: ClassroomState = Depends(get_classroom)):
    """进度排行榜"""
    return classroom.get_ranking()


@router.get("/alerts")
async def get_alerts(classroom: ClassroomState = Depends(get_classroom)):
    """活跃风险警报"""
    return classroom.get_alerts()


@router.get("/stats")
async def get_stats(classroom: ClassroomState = Depends(get_classroom)):
    """班级聚合统计"""
    return classroom.get_stats()


@router.get("/station/{station_id}")
async def get_station(
    station_id: str,
    classroom: ClassroomState = Depends(get_classroom),
):
    """获取单个工位详情"""
    stations = classroom.get_all_stations()
    if station_id in stations:
        return stations[station_id]
    return JSONResponse(status_code=404, content={"error": "station not found"})


@router.get("/station/{station_id}/thumbnail")
async def get_thumbnail(station_id: str):
    """获取工位最新缩略图 (base64)"""
    with _frame_lock:
        thumb = _frame_cache.get(station_id, "")
    if thumb:
        return {"thumbnail_b64": thumb}
    return JSONResponse(status_code=404, content={"error": "no thumbnail"})


# ---- 教师指导推送 ----


@router.post("/station/{station_id}/guidance")
async def send_guidance(
    station_id: str,
    msg: GuidanceMessage,
    classroom: ClassroomState = Depends(get_classroom),
):
    """教师 → 单个学生发送指导消息"""
    guidance_data = msg.model_dump()
    classroom.add_guidance_record(station_id, guidance_data)

    ws = classroom.get_websocket(station_id)
    if ws:
        try:
            await ws.send_json(guidance_data)
            return {"status": "delivered"}
        except Exception:
            return {"status": "queued", "reason": "ws_send_failed"}

    return {"status": "queued", "reason": "ws_not_connected"}


@router.post("/broadcast")
async def broadcast(
    msg: BroadcastMessage,
    classroom: ClassroomState = Depends(get_classroom),
):
    """教师 → 全班广播"""
    data = msg.model_dump()
    websockets = classroom.get_all_websockets()
    sent = 0
    for ws in websockets:
        try:
            await ws.send_json(data)
            sent += 1
        except Exception:
            pass
    return {"status": "ok", "sent": sent, "total": len(websockets)}


# ---- 参考电路 ----


@router.post("/reference")
async def set_reference(
    body: dict,
    classroom: ClassroomState = Depends(get_classroom),
):
    """设置本节课的参考电路"""
    classroom.set_reference(body)
    return {"status": "ok"}


@router.get("/reference")
async def get_reference(classroom: ClassroomState = Depends(get_classroom)):
    """获取参考电路"""
    ref = classroom.get_reference()
    if ref:
        return ref
    return JSONResponse(status_code=404, content={"error": "no reference set"})


# ---- 会话管理 ----


@router.post("/reset")
async def reset_session(classroom: ClassroomState = Depends(get_classroom)):
    """重置课堂会话"""
    classroom.reset()
    return {"status": "ok"}
