"""
Celery 应用实例

参考: GregaVrbancic/fastapi-celery
四阶段 AI Pipeline 作为异步 job 执行
"""

from __future__ import annotations

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "labguardian",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    # 任务结果过期 (1 小时)
    result_expires=3600,
    # Worker 预取数量 (CV 任务较重, 避免抢占)
    worker_prefetch_multiplier=1,
    # 任务路由
    task_routes={
        "app.worker.tasks.run_pipeline": {"queue": "pipeline"},
        "app.worker.tasks.run_detect": {"queue": "pipeline"},
    },
)

celery_app.autodiscover_tasks(["app.worker"])
