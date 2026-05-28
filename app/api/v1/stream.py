"""视频流控制 HTTP 端点（task #132 阶段 2 - 前端 / 外部进程能控）。

设计要点:

1. **默认 off**: lifespan 不自动启 stream — 前端显式 ``POST /stream/start`` 才启。
   理由：(a) 板上 USB camera 可能未插；(b) NPU 持续推理会增 +4.1W 功耗，不演示时关掉。
2. **幂等**: 重复 ``/start`` 返回当前 singleton；``/stop`` 不在跑也返回 200。
3. **不阻塞**: ``/start`` 返回后 model compile 仍在后台跑，前端轮询 ``/status``
   看 ``yolo.model_loaded=true`` 才能拿真检测。
4. **图像端点用 FileResponse**: 避免 base64 编码开销，直接 serve JPG bytes。

典型前端联调流程::

    curl -X POST http://board:8000/api/v1/stream/start
    # → {"started":true,"runner_pid":...,"consumer_status":"loading"}

    # 等 2-3s 模型 compile
    curl http://board:8000/api/v1/stream/status
    # → {"runner":{...},"yolo":{...,"model_loaded":true}}

    curl -o latest.jpg http://board:8000/api/v1/stream/keyframe/annotated/latest
    curl http://board:8000/api/v1/stream/detection/latest
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.pipeline.vision.stability_detector import (
    StabilityConfig,
    get_stability_detector,
    start_stability_detector,
    stop_stability_detector,
)
from app.pipeline.vision.stream_runner import (
    StreamConfig,
    get_stream_runner,
    start_stream_runner,
    stop_stream_runner,
)
from app.pipeline.vision.yolo_stream_consumer import (
    YoloConsumerConfig,
    get_yolo_consumer,
    start_yolo_consumer,
    stop_yolo_consumer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["stream"])


# -----------------------------------------------------------------------------
# Request / Response schemas
# -----------------------------------------------------------------------------


class StreamStartRequest(BaseModel):
    """启动视频流的可选覆盖参数。全 None 则用默认配置。"""

    device_index: Optional[int] = Field(default=None, description="/dev/videoN 中的 N")
    fps_target: Optional[int] = Field(default=None, ge=1, le=30)
    frame_diff_threshold: Optional[float] = Field(default=None, ge=0.0)
    resolution: Optional[tuple[int, int]] = Field(
        default=None, description="(宽, 高) 像素"
    )
    yolo_device: Optional[str] = Field(
        default=None,
        description='OpenVINO device: "intel:npu" / "intel:gpu" / "cpu"',
    )
    yolo_model_dir: Optional[str] = Field(
        default=None, description="ultralytics OpenVINO 模型目录绝对路径"
    )
    enable_yolo: bool = Field(
        default=True,
        description="False 时只抓帧不推理（debug 摄像头时用）",
    )
    enable_stability_trigger: bool = Field(
        default=True,
        description="True 时启 StabilityDetector：画面稳定后自动跑完整 pipeline + 写 station state",
    )
    stable_duration_s: Optional[float] = Field(
        default=None, ge=0.5, le=20.0, description="判稳时长（默认 3s）"
    )
    station_id: Optional[str] = Field(
        default=None, description="trigger pipeline 时用的 station_id"
    )
    reference_id: Optional[str] = Field(
        default=None,
        description=(
            "学生手动覆盖 scene。None = GNN-A 自动分类（推荐）"
        ),
    )


class StreamStartResponse(BaseModel):
    started: bool
    runner_already_running: bool
    consumer_already_running: bool
    runner_stats: dict[str, Any]
    consumer_stats: Optional[dict[str, Any]] = None
    stability_stats: Optional[dict[str, Any]] = None


# -----------------------------------------------------------------------------
# 启停 + status
# -----------------------------------------------------------------------------


@router.post(
    "/start",
    response_model=StreamStartResponse,
    summary="启动视频流 + YOLO 推理消费器",
)
def start_stream(payload: StreamStartRequest | None = None) -> StreamStartResponse:
    payload = payload or StreamStartRequest()

    # 先看现状
    existing_runner = get_stream_runner()
    runner_already = (
        existing_runner is not None
        and existing_runner._thread is not None
        and existing_runner._thread.is_alive()
    )

    # 1. 启 YOLO consumer（如果要的话）
    consumer_already = False
    consumer_stats: Optional[dict[str, Any]] = None
    if payload.enable_yolo:
        existing_consumer = get_yolo_consumer()
        consumer_already = (
            existing_consumer is not None
            and existing_consumer._thread is not None
            and existing_consumer._thread.is_alive()
        )
        if not consumer_already:
            consumer_config = YoloConsumerConfig()
            if payload.yolo_device:
                consumer_config.device = payload.yolo_device
            if payload.yolo_model_dir:
                consumer_config.model_path = Path(payload.yolo_model_dir)
            consumer = start_yolo_consumer(consumer_config)
            logger.info("Started YOLO consumer on %s", consumer_config.device)
        else:
            consumer = existing_consumer
        consumer_stats = consumer.stats() if consumer else None

    # 2. 启 stream runner
    if runner_already:
        runner = existing_runner
    else:
        stream_config = StreamConfig()
        if payload.device_index is not None:
            stream_config.device_index = payload.device_index
        if payload.fps_target is not None:
            stream_config.fps_target = payload.fps_target
        if payload.frame_diff_threshold is not None:
            stream_config.frame_diff_threshold = payload.frame_diff_threshold
        if payload.resolution is not None:
            stream_config.resolution = payload.resolution

        runner = start_stream_runner(stream_config)
        logger.info(
            "Started stream runner: device=%s fps=%s",
            stream_config.device_index,
            stream_config.fps_target,
        )

        # 把 consumer 的 enqueue 接到 runner（关键拼接）
        if payload.enable_yolo:
            consumer = get_yolo_consumer()
            if consumer is not None:
                runner.on_keyframe = consumer.enqueue

    # 3. 启 StabilityDetector（如果要的话）
    stability_stats: Optional[dict[str, Any]] = None
    if payload.enable_stability_trigger and payload.enable_yolo:
        consumer = get_yolo_consumer()
        if consumer is not None:
            stab_config = StabilityConfig()
            if payload.stable_duration_s is not None:
                stab_config.stable_duration_s = payload.stable_duration_s
            if payload.station_id is not None:
                stab_config.station_id = payload.station_id
            if payload.reference_id is not None:
                stab_config.reference_id = payload.reference_id
            detector = start_stability_detector(
                stream_runner=runner,
                yolo_consumer=consumer,
                config=stab_config,
            )
            stability_stats = _stability_stats(detector)
            logger.info(
                "Started StabilityDetector: station_id=%s ref=%s window=%ss",
                stab_config.station_id,
                stab_config.reference_id,
                stab_config.stable_duration_s,
            )

    return StreamStartResponse(
        started=True,
        runner_already_running=runner_already,
        consumer_already_running=consumer_already,
        runner_stats=runner.stats(),
        consumer_stats=consumer_stats,
        stability_stats=stability_stats,
    )


def _stability_stats(detector: Any) -> dict[str, Any]:
    snapshot = detector.snapshot()
    return {
        "state": snapshot.state.value,
        "stable_since_ts": snapshot.stable_since_ts,
        "last_trigger_ts": snapshot.last_trigger_ts,
        "last_trigger_keyframe": snapshot.last_trigger_keyframe,
        "last_trigger_outcome": snapshot.last_trigger_outcome,
        "trigger_count": snapshot.trigger_count,
        "poll_count": snapshot.poll_count,
        "history_size": snapshot.history_size,
        "station_id": detector.config.station_id,
        "reference_id": detector.config.reference_id,
        "error": snapshot.error,
    }


@router.post("/stop", summary="停止视频流 + 释放摄像头与 NPU")
def stop_stream() -> dict[str, Any]:
    runner = get_stream_runner()
    consumer = get_yolo_consumer()
    detector = get_stability_detector()
    if runner is None and consumer is None and detector is None:
        return {"stopped": False, "reason": "not running"}
    stop_stability_detector()
    stop_stream_runner()
    stop_yolo_consumer()
    return {"stopped": True}


@router.get("/status", summary="实时状态（运行中 + 处理统计）")
def get_status() -> dict[str, Any]:
    runner = get_stream_runner()
    consumer = get_yolo_consumer()
    detector = get_stability_detector()
    return {
        "runner": runner.stats() if runner else None,
        "yolo": consumer.stats() if consumer else None,
        "stability": _stability_stats(detector) if detector else None,
    }


# -----------------------------------------------------------------------------
# 关键帧 / annotated 图 / detection JSON
# -----------------------------------------------------------------------------


@router.post(
    "/force-trigger",
    summary="手动触发一次完整 pipeline（不等画面自动稳定，调试/演示用）",
)
def force_trigger() -> dict[str, Any]:
    """强制把 stream 当前最新关键帧喂给 pipeline_service.run_sync。

    - 不影响 stability 状态机
    - 同步阻塞直到 pipeline 跑完（几百 ms ~ 几秒）
    - 用途：失焦时验证 trigger 链路 / 演示按钮 / 集成测试
    """
    runner = get_stream_runner()
    if runner is None:
        raise HTTPException(404, "stream not running, call /stream/start first")
    keyframe = runner.latest_keyframe()
    if keyframe is None or not keyframe.path.exists():
        raise HTTPException(404, "no keyframe captured yet")

    from app.pipeline.vision.stability_detector import make_pipeline_trigger  # noqa: PLC0415

    detector = get_stability_detector()
    station_id = (
        detector.config.station_id if detector else "live_camera_default_force_trigger"
    )
    reference_id = detector.config.reference_id if detector else None

    trigger = make_pipeline_trigger(
        station_id=station_id, reference_id=reference_id, imgsz=640, conf=0.20
    )
    import time as _time  # noqa: PLC0415

    t0 = _time.time()
    try:
        trigger(keyframe.path)
        outcome = "ok"
    except Exception as exc:
        logger.exception("force-trigger failed: %s", exc)
        outcome = f"err: {type(exc).__name__}: {exc}"
    elapsed_ms = (_time.time() - t0) * 1000.0
    return {
        "outcome": outcome,
        "elapsed_ms": round(elapsed_ms, 1),
        "keyframe": str(keyframe.path),
        "station_id": station_id,
        "reference_id": reference_id,
    }


@router.get(
    "/keyframe/latest",
    summary="最新关键帧 JPG（原图，未画 bbox）",
    responses={
        200: {"content": {"image/jpeg": {}}},
        404: {"description": "No keyframe yet"},
    },
)
def latest_keyframe() -> FileResponse:
    runner = get_stream_runner()
    if runner is None:
        raise HTTPException(404, "stream not running")
    keyframe = runner.latest_keyframe()
    if keyframe is None:
        raise HTTPException(404, "no keyframe captured yet")
    if not keyframe.path.exists():
        raise HTTPException(404, f"keyframe file missing: {keyframe.path}")
    return FileResponse(
        path=str(keyframe.path),
        media_type="image/jpeg",
        headers={"X-Keyframe-Timestamp": str(keyframe.timestamp)},
    )


@router.get(
    "/keyframe/annotated/latest",
    summary="最新 annotated 图（含 YOLO bbox + 引脚 keypoint）",
    responses={
        200: {"content": {"image/jpeg": {}}},
        404: {"description": "No annotated frame yet"},
    },
)
def latest_annotated() -> FileResponse:
    consumer = get_yolo_consumer()
    if consumer is None:
        raise HTTPException(404, "yolo consumer not running")
    # 查 annotated 目录里时间戳最大的
    annotated_dir = consumer.config.annotated_image_dir
    if not annotated_dir.exists():
        raise HTTPException(404, "annotated dir not initialized")
    files = sorted(annotated_dir.glob("det_*.jpg"))
    if not files:
        raise HTTPException(404, "no annotated frame yet")
    latest = files[-1]
    return FileResponse(
        path=str(latest),
        media_type="image/jpeg",
        headers={"X-Annotated-File": latest.name},
    )


@router.get(
    "/detection/latest",
    summary="最新检测 JSON（元件类别 + bbox + 引脚关键点）",
)
def latest_detection() -> JSONResponse:
    consumer = get_yolo_consumer()
    if consumer is None:
        raise HTTPException(404, "yolo consumer not running")
    latest = consumer.latest()
    if latest is None:
        # consumer 在跑但还没出过结果（warmup 中或刚启动）
        return JSONResponse(
            {
                "ready": False,
                "reason": "no detection yet (consumer warming up?)",
                "stats": consumer.stats(),
            },
            status_code=202,
        )
    return JSONResponse(
        {
            "ready": True,
            "keyframe_path": str(latest.keyframe_path),
            "timestamp": latest.timestamp,
            "inference_ms": round(latest.inference_ms, 2),
            "components_count": len(latest.components),
            "components": [
                {
                    "cls_name": c.cls_name,
                    "cls_id": c.cls_id,
                    "conf": round(c.conf, 3),
                    "bbox_xyxy": [round(x, 1) for x in c.bbox_xyxy],
                    "keypoints_xy": (
                        [[round(p, 1) for p in pt] for pt in c.keypoints_xy]
                        if c.keypoints_xy
                        else None
                    ),
                }
                for c in latest.components
            ],
            "error": latest.error,
        }
    )
