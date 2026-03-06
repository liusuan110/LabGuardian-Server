"""
FastAPI 依赖注入

参考: fastapi/full-stack-fastapi-template 的 deps.py 模式
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, settings
from app.services.classroom_state import ClassroomState


def get_settings() -> Settings:
    return settings


# 课堂状态单例
_classroom: ClassroomState | None = None


def get_classroom() -> ClassroomState:
    global _classroom
    if _classroom is None:
        _classroom = ClassroomState(
            online_timeout=settings.STATION_ONLINE_TIMEOUT,
        )
    return _classroom
