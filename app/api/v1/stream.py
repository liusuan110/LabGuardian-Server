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

from app.pipeline.vision.camera_locator import (
    resolve_video_device_index_by_usb_hint,
)
from app.pipeline.vision.stability_detector import (
    StabilityConfig,
    get_stability_detector,
    start_stability_detector,
    stop_stability_detector,
)
from app.pipeline.vision.side_camera_gate import (
    SideGateConfig,
    get_side_camera_gate_detector,
    start_side_camera_gate_detector,
    stop_side_camera_gate_detector,
)
from app.pipeline.vision.stream_runner import (
    StreamConfig,
    StreamRunner,
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

_side_runners: dict[str, StreamRunner] = {}
DEFAULT_USB_PORT_HINT_BY_VIEW = {
    "top": "usb-0000:00:14.0-3",
    "left_front": "usb-0000:00:14.0-6",
    "right_front": "usb-0000:00:14.0-7",
}


# -----------------------------------------------------------------------------
# Request / Response schemas
# -----------------------------------------------------------------------------


class StreamStartRequest(BaseModel):
    """启动视频流的可选覆盖参数。全 None 则用默认配置。"""

    device_index: Optional[int] = Field(default=None, description="顶视主摄像头 /dev/videoN 中的 N")
    left_device_index: Optional[int] = Field(default=None, description="左侧摄像头 /dev/videoN 中的 N")
    right_device_index: Optional[int] = Field(default=None, description="右侧摄像头 /dev/videoN 中的 N")
    top_usb_port_hint: Optional[str] = Field(
        default=None,
        description='顶视主摄像头 USB 物理口 hint；未传时默认 "usb-0000:00:14.0-3"',
    )
    left_usb_port_hint: Optional[str] = Field(
        default=None,
        description='左侧摄像头 USB 物理口 hint；未传时默认 "usb-0000:00:14.0-6"',
    )
    right_usb_port_hint: Optional[str] = Field(
        default=None,
        description='右侧摄像头 USB 物理口 hint；未传时默认 "usb-0000:00:14.0-7"',
    )
    fps_target: Optional[int] = Field(default=None, ge=1, le=30)
    frame_diff_threshold: Optional[float] = Field(default=None, ge=0.0)
    resolution: Optional[tuple[int, int]] = Field(
        default=None, description="(宽, 高) 像素"
    )
    enable_side_gate: bool = Field(
        default=False,
        description="True 时先用左右侧摄像头判断'有板且静止'，通过后再启动顶视正式链路",
    )
    side_stable_duration_s: Optional[float] = Field(
        default=None, ge=0.5, le=20.0, description="两侧摄像头静止放行时长（默认 2.5s）"
    )
    side_presence_diff_threshold: Optional[float] = Field(
        default=None, ge=0.0, description="两侧摄像头相对背景的存在阈值"
    )
    side_motion_threshold: Optional[float] = Field(
        default=None, ge=0.0, description="两侧摄像头静止判定的最大运动分数"
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
    runner_stats: Optional[dict[str, Any]] = None
    consumer_stats: Optional[dict[str, Any]] = None
    stability_stats: Optional[dict[str, Any]] = None
    side_runner_stats: Optional[dict[str, dict[str, Any]]] = None
    side_gate_stats: Optional[dict[str, Any]] = None


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
    use_side_gate = bool(
        payload.enable_side_gate
        or payload.left_device_index is not None
        or payload.right_device_index is not None
        or payload.left_usb_port_hint
        or payload.right_usb_port_hint
    )
    if use_side_gate and (
        not _has_camera_target(
            payload.left_device_index,
            _effective_usb_port_hint("left_front", payload.left_usb_port_hint),
        )
        or not _has_camera_target(
            payload.right_device_index,
            _effective_usb_port_hint("right_front", payload.right_usb_port_hint),
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="side gate mode requires both left and right camera targets (device_index or usb hint)",
        )

    # 先看现状
    existing_runner = get_stream_runner()
    runner_already = (
        existing_runner is not None
        and existing_runner._thread is not None
        and existing_runner._thread.is_alive()
    )

    side_gate = get_side_camera_gate_detector()
    side_gate_already = (
        side_gate is not None
        and side_gate._thread is not None
        and side_gate._thread.is_alive()
    )

    consumer_already = False
    consumer_stats: Optional[dict[str, Any]] = None
    runner: Optional[StreamRunner] = existing_runner
    stability_stats: Optional[dict[str, Any]] = None
    side_runner_stats: Optional[dict[str, dict[str, Any]]] = None
    side_gate_stats: Optional[dict[str, Any]] = None

    if use_side_gate:
        side_runners = _ensure_side_runners(payload)
        side_runner_stats = {
            view_id: side_runner.stats()
            for view_id, side_runner in side_runners.items()
        }

        if not side_gate_already:
            gate_config = SideGateConfig()
            if payload.side_stable_duration_s is not None:
                gate_config.stable_duration_s = payload.side_stable_duration_s
            if payload.side_motion_threshold is not None:
                gate_config.max_motion_score = payload.side_motion_threshold
            if payload.side_presence_diff_threshold is not None:
                gate_config.min_presence_score = payload.side_presence_diff_threshold

            start_side_camera_gate_detector(
                runners=side_runners,
                config=gate_config,
                on_ready=lambda: _start_top_phase(payload),
            )
            logger.info(
                "Started side gate: views=%s stable=%ss",
                sorted(side_runners.keys()),
                gate_config.stable_duration_s,
            )
        side_gate_stats = _side_gate_stats(get_side_camera_gate_detector())
        runner = get_stream_runner()
    else:
        runner, consumer_already, consumer_stats, stability_stats = _start_top_phase(payload)

    consumer = get_yolo_consumer()
    if consumer is not None:
        consumer_stats = consumer.stats()

    detector = get_stability_detector()
    if detector is not None:
        stability_stats = _stability_stats(detector)

    if side_runner_stats is None and _side_runners:
        side_runner_stats = {
            view_id: side_runner.stats()
            for view_id, side_runner in _side_runners.items()
        }
    if side_gate_stats is None:
        side_gate_stats = _side_gate_stats(get_side_camera_gate_detector())

    return StreamStartResponse(
        started=True,
        runner_already_running=runner_already,
        consumer_already_running=consumer_already,
        runner_stats=runner.stats() if runner else None,
        consumer_stats=consumer_stats,
        stability_stats=stability_stats,
        side_runner_stats=side_runner_stats,
        side_gate_stats=side_gate_stats,
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


def _side_gate_stats(detector: Any) -> Optional[dict[str, Any]]:
    if detector is None:
        return None
    snapshot = detector.snapshot()
    return {
        "state": snapshot.state.value,
        "stable_since_ts": snapshot.stable_since_ts,
        "last_ready_ts": snapshot.last_ready_ts,
        "trigger_count": snapshot.trigger_count,
        "poll_count": snapshot.poll_count,
        "views": snapshot.views,
        "error": snapshot.error,
    }


def _build_stream_config(
    *,
    view_id: str,
    device_index: int,
    payload: StreamStartRequest,
) -> StreamConfig:
    stream_config = StreamConfig(view_id=view_id, device_index=device_index)
    if payload.fps_target is not None:
        stream_config.fps_target = payload.fps_target
    if payload.frame_diff_threshold is not None:
        stream_config.frame_diff_threshold = payload.frame_diff_threshold
    if payload.resolution is not None:
        stream_config.resolution = payload.resolution
    if payload.side_presence_diff_threshold is not None:
        stream_config.presence_diff_threshold = payload.side_presence_diff_threshold
    stream_config.keyframe_dir = stream_config.keyframe_dir / view_id
    return stream_config


def _ensure_side_runners(payload: StreamStartRequest) -> dict[str, StreamRunner]:
    desired = {
        "left_front": _resolve_requested_device_index(
            view_id="left_front",
            device_index=payload.left_device_index,
            usb_port_hint=payload.left_usb_port_hint,
        ),
        "right_front": _resolve_requested_device_index(
            view_id="right_front",
            device_index=payload.right_device_index,
            usb_port_hint=payload.right_usb_port_hint,
        ),
    }
    active: dict[str, StreamRunner] = {}
    for view_id, device_index in desired.items():
        if device_index is None:
            continue
        existing = _side_runners.get(view_id)
        if existing is not None and existing._thread is not None and existing._thread.is_alive():
            active[view_id] = existing
            continue
        runner = StreamRunner(
            _build_stream_config(
                view_id=view_id,
                device_index=device_index,
                payload=payload,
            )
        )
        runner.start()
        _side_runners[view_id] = runner
        active[view_id] = runner
        logger.info("Started side runner: view=%s device=%s", view_id, device_index)
    return active


def _start_top_phase(
    payload: StreamStartRequest,
) -> tuple[Optional[StreamRunner], bool, Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    existing_runner = get_stream_runner()
    runner_already = (
        existing_runner is not None
        and existing_runner._thread is not None
        and existing_runner._thread.is_alive()
    )

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

    if runner_already:
        runner = existing_runner
    else:
        top_device_index = _resolve_requested_device_index(
            view_id="top",
            device_index=payload.device_index,
            usb_port_hint=payload.top_usb_port_hint,
            default_index=0,
        )
        runner = start_stream_runner(
            _build_stream_config(
                view_id="top",
                device_index=top_device_index,
                payload=payload,
            )
        )
        logger.info(
            "Started top runner: device=%s fps=%s",
            top_device_index,
            runner.config.fps_target,
        )

    if payload.enable_yolo:
        consumer = get_yolo_consumer()
        if consumer is not None and runner is not None:
            runner.on_keyframe = consumer.enqueue

    stability_stats: Optional[dict[str, Any]] = None
    if payload.enable_stability_trigger and payload.enable_yolo and runner is not None:
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

    return runner, consumer_already, consumer_stats, stability_stats


def _has_camera_target(device_index: Optional[int], usb_port_hint: Optional[str]) -> bool:
    return device_index is not None or bool(str(usb_port_hint or "").strip())


def _effective_usb_port_hint(view_id: str, usb_port_hint: Optional[str]) -> str:
    hint = str(usb_port_hint or "").strip()
    if hint:
        return hint
    return DEFAULT_USB_PORT_HINT_BY_VIEW.get(view_id, "")


def _resolve_requested_device_index(
    *,
    view_id: str,
    device_index: Optional[int],
    usb_port_hint: Optional[str],
    default_index: Optional[int] = None,
) -> int:
    hint = _effective_usb_port_hint(view_id, usb_port_hint)
    if hint:
        try:
            resolved = resolve_video_device_index_by_usb_hint(hint)
            logger.info(
                "Resolved %s camera by usb hint %s -> /dev/video%s",
                view_id,
                hint,
                resolved,
            )
            return resolved
        except ValueError as exc:
            if device_index is None:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            logger.warning(
                "usb hint resolve failed for %s (%s), falling back to explicit /dev/video%s",
                view_id,
                exc,
                device_index,
            )
    if device_index is not None:
        return device_index
    if default_index is not None:
        return default_index
    raise HTTPException(
        status_code=400,
        detail=f"missing camera target for {view_id}: provide device_index or usb hint",
    )


@router.post("/stop", summary="停止视频流 + 释放摄像头与 NPU")
def stop_stream() -> dict[str, Any]:
    runner = get_stream_runner()
    consumer = get_yolo_consumer()
    detector = get_stability_detector()
    side_gate = get_side_camera_gate_detector()
    if runner is None and consumer is None and detector is None and side_gate is None and not _side_runners:
        return {"stopped": False, "reason": "not running"}
    stop_side_camera_gate_detector()
    stop_stability_detector()
    stop_stream_runner()
    stop_yolo_consumer()
    for side_runner in list(_side_runners.values()):
        side_runner.stop()
    _side_runners.clear()
    return {"stopped": True}


@router.get("/status", summary="实时状态（运行中 + 处理统计）")
def get_status() -> dict[str, Any]:
    runner = get_stream_runner()
    consumer = get_yolo_consumer()
    detector = get_stability_detector()
    side_gate = get_side_camera_gate_detector()
    return {
        "runner": runner.stats() if runner else None,
        "yolo": consumer.stats() if consumer else None,
        "stability": _stability_stats(detector) if detector else None,
        "side_gate": _side_gate_stats(side_gate),
        "side_runners": {
            view_id: side_runner.stats()
            for view_id, side_runner in _side_runners.items()
        } if _side_runners else None,
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
