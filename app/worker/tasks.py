"""
Celery Worker Tasks

定义异步任务：run_pipeline_task
通过 task.update_state 推送阶段进度，供 API 轮询。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from celery import current_task

from app.core.celery_app import celery_app
from app.pipeline.orchestrator import run_pipeline

logger = logging.getLogger(__name__)


@celery_app.task(
    name="pipeline.run",
    bind=True,
    acks_late=True,
    max_retries=0,
)
def run_pipeline_task(
    self,
    images_b64: List[str],
    reference_path: Dict[str, Any] | str | None = None,
    rail_assignments: Dict[str, str] | None = None,
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
) -> Dict[str, Any]:
    """Celery 异步执行完整 4 阶段流水线

    前端通过 GET /api/v1/pipeline/status/{task_id} 轮询状态：
      - state="PROGRESS", meta={"stage": "detect", "progress": 0.5}
      - state="SUCCESS", result={...}
    """

    def _progress_cb(stage: str, progress: float) -> None:
        self.update_state(
            state="PROGRESS",
            meta={"current_stage": stage, "stage": stage, "progress": progress},
        )

    ref_path = reference_path if isinstance(reference_path, str) else None

    try:
        result = run_pipeline(
            images_b64=images_b64,
            reference_path=ref_path,
            rail_assignments=rail_assignments,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            progress_cb=_progress_cb,
        )
        return result
    except Exception as exc:
        logger.exception("Pipeline task failed")
        raise
