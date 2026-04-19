"""
LabGuardian Server — FastAPI 入口

参考: fastapi/full-stack-fastapi-template
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.deps import get_version_service
from app.api.v1 import angnt, classroom, kb, pipeline, websocket
from app.schemas.version import VersionInfoResponse
from app.services.version_service import VersionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期: 启动/关闭时执行的逻辑"""
    # startup
    yield
    # shutdown


app = FastAPI(
    title="LabGuardian Server",
    description="检测→映射→拓扑→检错 统一后端",
    version=settings.CODE_VERSION,
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url="/docs",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由
app.include_router(classroom.router, prefix=settings.API_V1_PREFIX)
app.include_router(pipeline.router, prefix=settings.API_V1_PREFIX)
app.include_router(angnt.router, prefix=settings.API_V1_PREFIX)
app.include_router(kb.router, prefix=settings.API_V1_PREFIX)
app.include_router(websocket.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/version", response_model=VersionInfoResponse)
async def get_version(
    version_service: VersionService = Depends(get_version_service),
):
    return version_service.get_version_info()
