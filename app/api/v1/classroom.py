"""
教室 REST API (← teacher/server.py)

拆分为 APIRouter, 符合 full-stack-fastapi-template 模式
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.core.deps import get_classroom, get_guidance_service
from app.schemas.classroom import (
    BroadcastMessage,
    GuidanceMessage,
    StationHeartbeat,
)
from app.services.classroom_state import ClassroomState
from app.services.guidance_service import GuidanceService

router = APIRouter(prefix="/classroom", tags=["classroom"])


# ---- 学生端上报 ----


@router.post("/heartbeat")
async def receive_heartbeat(
    heartbeat: StationHeartbeat,
    classroom: ClassroomState = Depends(get_classroom),
    guidance_service: GuidanceService = Depends(get_guidance_service),
):
    """接收学生工位心跳 (每 2 秒)"""
    data = heartbeat.model_dump()
    new_alerts = classroom.update_station(data)

    thumb = data.get("thumbnail_b64", "")
    guidance_service.cache_thumbnail(heartbeat.station_id, thumb)

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
async def get_thumbnail(
    station_id: str,
    guidance_service: GuidanceService = Depends(get_guidance_service),
):
    """获取工位最新缩略图 (base64)"""
    thumb = guidance_service.get_thumbnail(station_id)
    if thumb:
        return {"thumbnail_b64": thumb}
    return JSONResponse(status_code=404, content={"error": "no thumbnail"})


# ---- 教师指导推送 ----


@router.post("/station/{station_id}/guidance")
async def send_guidance(
    station_id: str,
    msg: GuidanceMessage,
    classroom: ClassroomState = Depends(get_classroom),
    guidance_service: GuidanceService = Depends(get_guidance_service),
):
    """教师 → 单个学生发送指导消息"""
    return await guidance_service.send_guidance(classroom, station_id, msg.model_dump())


@router.post("/broadcast")
async def broadcast(
    msg: BroadcastMessage,
    classroom: ClassroomState = Depends(get_classroom),
    guidance_service: GuidanceService = Depends(get_guidance_service),
):
    """教师 → 全班广播"""
    return await guidance_service.broadcast(classroom, msg.model_dump())


@router.get("/guidance/audit")
async def get_guidance_audit(
    station_id: str | None = None,
    limit: int = 100,
    guidance_service: GuidanceService = Depends(get_guidance_service),
):
    """查询指导审计记录."""
    return guidance_service.list_audit_records(station_id=station_id, limit=limit)


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
