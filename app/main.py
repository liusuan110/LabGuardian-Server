"""
LabGuardian Server — FastAPI 入口

参考: fastapi/full-stack-fastapi-template
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1 import classroom, pipeline, websocket, aoi


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期: 启动/关闭时执行的逻辑"""
    # startup
    yield
    # shutdown


app = FastAPI(
    title="LabGuardian Server",
    description="检测→映射→拓扑→检错 统一后端",
    version="0.1.0",
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
app.include_router(aoi.router, prefix=settings.API_V1_PREFIX)
app.include_router(websocket.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
