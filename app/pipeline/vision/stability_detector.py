"""画面稳定检测 → 一次性触发完整 pipeline（task #132 阶段 3）。

设计思想:

- **NPU 一直忙** (yolo_stream_consumer 持续推理 ~30ms/帧)
- **CPU 偶尔忙** (完整 S1→S2→S3→S4 + 电路对比，仅画面稳定时跑一次)

依赖现有工程链路（**绝不重复造轮子**）::

    StabilityDetector
        ↓ on_stable 触发
    pipeline_service.run_sync(PipelineRequest, classroom, guidance_service)
        ↓ 内部自动
    sync_result_to_classroom() → classroom.update_station(topology_label="")
        ↓ 后续学生提问时
    agent.build_context_pack → scene_resolver(station) → 自动选 scene_id → evidence

也就是：本模块只负责"判稳 + 触发一次"。pipeline 结果如何走到 agent
是 WP-1/WP-3 已经修干净的事，**完全复用**。

判稳逻辑（3 个信号合体）:

1. **类别集合**: YOLO 当前帧的类别 set 与历史 K 帧一致
2. **bbox IoU**: 同类别 bbox 平均 IoU > 0.8（没挪动）
3. **时长**: 上述持续 ``stable_duration_s`` 秒

状态机::

    UNSTABLE ──┐
       ↑       │ 出现稳定候选
       │       ↓
       │   STABILIZING ──────┐
       │       ↑             │ 持续 N 秒
       │       │ 候选断裂     ↓
       │       │           STABLE
       │       │             │ 一次性 fire
       │       │             ↓
       │       │          TRIGGERED ──┐
       │       │                       │
       └───────┴──── 帧差/类集合变大 ──┘
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from app.pipeline.vision.stream_runner import StreamRunner
    from app.pipeline.vision.yolo_stream_consumer import (
        DetectedComponent,
        DetectionResult,
        YoloStreamConsumer,
    )

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# State enum + config
# -----------------------------------------------------------------------------


class StabilityState(str, Enum):
    UNSTABLE = "unstable"
    STABILIZING = "stabilizing"
    STABLE = "stable"
    TRIGGERED = "triggered"


@dataclass
class StabilityConfig:
    stable_duration_s: float = 3.0
    """连续多久无变化才算 STABLE。用户决策：3 秒。"""

    poll_interval_s: float = 0.5
    """轮询周期。0.5s 兼顾响应速度与开销。"""

    bbox_iou_threshold: float = 0.8
    """同类元件 bbox IoU 低于此阈值视为'挪动'。"""

    require_at_least_n_components: int = 1
    """至少检测到 N 个元件才考虑稳定（0 = 空画面也算稳定，不推荐）。"""

    station_id: str = "live_camera_default"
    """触发 pipeline 时用的 station_id。一台演示机一个固定 ID 即可。"""

    reference_id: Optional[str] = None
    """学生手动覆盖 scene。None = 不覆盖。"""

    trigger_imgsz: int = 640
    """触发 pipeline 用的 imgsz。必须与 YOLO 模型 export 时一致（板上 merged_det_v2 = 640）。"""

    debounce_after_trigger_s: float = 2.0
    """trigger 后强制冷却 N 秒，防 ms 级抖动重复触发。"""


@dataclass
class StabilitySnapshot:
    state: StabilityState
    stable_since_ts: Optional[float] = None
    last_trigger_ts: Optional[float] = None
    last_trigger_keyframe: Optional[str] = None
    last_trigger_outcome: Optional[str] = None  # ok / err: <reason>
    trigger_count: int = 0
    poll_count: int = 0
    history_size: int = 0
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# StabilityDetector
# -----------------------------------------------------------------------------


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """xyxy 像素 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _components_to_class_groups(
    components: list["DetectedComponent"],
) -> dict[str, list[list[float]]]:
    groups: dict[str, list[list[float]]] = {}
    for c in components:
        groups.setdefault(c.cls_name, []).append(c.bbox_xyxy)
    return groups


def _same_class_set(
    a: dict[str, list[list[float]]], b: dict[str, list[list[float]]]
) -> bool:
    return set(a.keys()) == set(b.keys())


def _matched_bbox_iou_avg(
    a: dict[str, list[list[float]]], b: dict[str, list[list[float]]]
) -> Optional[float]:
    """同类元件按贪心匹配后的平均 IoU。None = 类不一致或没有元件可比。"""
    if not _same_class_set(a, b):
        return None
    if not a:
        return None
    ious: list[float] = []
    for cls_name, boxes_a in a.items():
        boxes_b = list(b.get(cls_name, []))
        if not boxes_b:
            continue
        for box_a in boxes_a:
            if not boxes_b:
                ious.append(0.0)
                continue
            best = max(_bbox_iou(box_a, box_b) for box_b in boxes_b)
            ious.append(best)
    if not ious:
        return None
    return sum(ious) / len(ious)


class StabilityDetector:
    """轮询 yolo consumer 的 latest detection → 判稳 → fire trigger 一次。

    使用方式::

        detector = StabilityDetector(
            stream_runner=runner,
            yolo_consumer=consumer,
            config=StabilityConfig(),
            on_stable=lambda keyframe_path: trigger_pipeline(keyframe_path),
        )
        detector.start()
        ...
        detector.stop()
    """

    def __init__(
        self,
        *,
        stream_runner: "StreamRunner",
        yolo_consumer: "YoloStreamConsumer",
        config: StabilityConfig,
        on_stable: Callable[[Path], None],
    ) -> None:
        self._runner = stream_runner
        self._consumer = yolo_consumer
        self.config = config
        self._on_stable = on_stable

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._state: StabilityState = StabilityState.UNSTABLE
        self._stable_since_ts: Optional[float] = None
        self._last_trigger_ts: Optional[float] = None
        self._last_trigger_keyframe: Optional[str] = None
        self._last_trigger_outcome: Optional[str] = None
        self._trigger_count: int = 0
        self._poll_count: int = 0
        self._error: Optional[str] = None

        # 历史窗口：每条是 (timestamp, class_groups)
        n_history = max(2, int(config.stable_duration_s / config.poll_interval_s) + 2)
        self._history: deque[tuple[float, dict[str, list[list[float]]], str]] = deque(
            maxlen=n_history
        )

    # ---- 公共接口 ----

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="labguardian-stability"
        )
        self._thread.start()
        logger.info(
            "StabilityDetector started: window=%ss poll=%ss bbox_iou=%s station=%s",
            self.config.stable_duration_s,
            self.config.poll_interval_s,
            self.config.bbox_iou_threshold,
            self.config.station_id,
        )

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def snapshot(self) -> StabilitySnapshot:
        return StabilitySnapshot(
            state=self._state,
            stable_since_ts=self._stable_since_ts,
            last_trigger_ts=self._last_trigger_ts,
            last_trigger_keyframe=self._last_trigger_keyframe,
            last_trigger_outcome=self._last_trigger_outcome,
            trigger_count=self._trigger_count,
            poll_count=self._poll_count,
            history_size=len(self._history),
            error=self._error,
        )

    # ---- 内部 ----

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._poll_count += 1
            try:
                self._tick()
            except Exception as exc:  # pragma: no cover
                self._error = f"{type(exc).__name__}: {exc}"
                logger.exception("stability tick failed: %s", exc)
            time.sleep(self.config.poll_interval_s)

    def _tick(self) -> None:
        result: Optional["DetectionResult"] = self._consumer.latest()
        if result is None:
            # 还没出过 detection（warmup 中）
            return

        # 1. 提取当前帧的"类组"
        groups = _components_to_class_groups(result.components)
        now = time.time()
        self._history.append((now, groups, str(result.keyframe_path)))

        # 2. 一系列守门检查
        n_components = len(result.components)
        if n_components < self.config.require_at_least_n_components:
            # 空画面 / 元件太少 → 不进入稳定路径
            self._transition_to(StabilityState.UNSTABLE, reset_since=True)
            return

        if len(self._history) < 2:
            return

        # 3. 与上一帧比对
        _prev_ts, prev_groups, _prev_kf = self._history[-2]
        if not _same_class_set(prev_groups, groups):
            # 类别集合变 → 帧不稳定 → 状态机回退
            self._transition_to(StabilityState.UNSTABLE, reset_since=True)
            return

        iou_avg = _matched_bbox_iou_avg(prev_groups, groups)
        if iou_avg is None or iou_avg < self.config.bbox_iou_threshold:
            self._transition_to(StabilityState.UNSTABLE, reset_since=True)
            return

        # 4. 当前帧与上一帧稳定 → 进入 STABILIZING；如已 STABILIZING 检查时长
        if self._state == StabilityState.UNSTABLE:
            self._transition_to(StabilityState.STABILIZING, reset_since=False)
            self._stable_since_ts = now
            return

        if self._state == StabilityState.STABILIZING:
            assert self._stable_since_ts is not None
            elapsed = now - self._stable_since_ts
            if elapsed >= self.config.stable_duration_s:
                self._transition_to(StabilityState.STABLE, reset_since=False)
            return

        if self._state == StabilityState.STABLE:
            self._fire_trigger(result)
            self._transition_to(StabilityState.TRIGGERED, reset_since=False)
            return

        if self._state == StabilityState.TRIGGERED:
            # 已 trigger 过；只有 UNSTABLE 才会回来
            return

    def _transition_to(self, new_state: StabilityState, *, reset_since: bool) -> None:
        if new_state == self._state:
            return
        logger.info("stability %s → %s", self._state.value, new_state.value)
        self._state = new_state
        if reset_since:
            self._stable_since_ts = None

    def _fire_trigger(self, result: "DetectionResult") -> None:
        if (
            self._last_trigger_ts is not None
            and time.time() - self._last_trigger_ts < self.config.debounce_after_trigger_s
        ):
            return
        keyframe = result.keyframe_path
        logger.info("STABLE → firing trigger on %s", keyframe.name)
        try:
            self._on_stable(keyframe)
            self._last_trigger_outcome = "ok"
        except Exception as exc:
            logger.exception("trigger callback failed: %s", exc)
            self._last_trigger_outcome = f"err: {type(exc).__name__}: {exc}"
        self._last_trigger_ts = time.time()
        self._last_trigger_keyframe = str(keyframe)
        self._trigger_count += 1


# -----------------------------------------------------------------------------
# Trigger 实现：调现有 pipeline_service.run_sync()
# -----------------------------------------------------------------------------


def make_pipeline_trigger(
    *,
    station_id: str,
    reference_id: Optional[str] = None,
    imgsz: int = 640,
    conf: float = 0.20,
) -> Callable[[Path], None]:
    """工厂：返回一个 trigger 函数，把 keyframe 喂给现有 pipeline。

    复用 ``pipeline_service.run_sync`` 的副作用——它会自动:
    - 跑 pipeline → 写空 ``topology_label`` 到 station（不启用实验分类器）
    - 写 ``netlist_v2`` / ``components`` / ``diagnostics`` 到 station
    - 后续 agent.build_context_pack 经 scene_resolver 自动选 scene_id

    所以本 trigger 真的就只做"读图 + 调 service"。
    """

    def trigger(keyframe: Path) -> None:
        # 延迟 import 避免循环依赖
        from app.core.deps import (  # noqa: PLC0415
            get_classroom,
            get_guidance_service,
            get_pipeline_service,
        )
        from app.schemas.pipeline import PipelineRequest  # noqa: PLC0415

        img_bytes = keyframe.read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("ascii")

        request = PipelineRequest(
            station_id=station_id,
            images_b64=[img_b64],
            reference_id=reference_id,
            imgsz=imgsz,
            conf=conf,
        )
        pipeline_service = get_pipeline_service()
        classroom = get_classroom()
        guidance_service = get_guidance_service()

        t0 = time.time()
        result = pipeline_service.run_sync(
            request, classroom=classroom, guidance_service=guidance_service
        )
        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            "pipeline run_sync done: job=%s station=%s elapsed=%.0fms components=%s",
            result.job_id,
            result.station_id,
            elapsed_ms,
            result.component_count,
        )

    return trigger


# -----------------------------------------------------------------------------
# Global singleton（沿用 stream_runner / yolo_stream_consumer 模式）
# -----------------------------------------------------------------------------

_detector: Optional[StabilityDetector] = None
_lock = threading.Lock()


def get_stability_detector() -> Optional[StabilityDetector]:
    with _lock:
        return _detector


def start_stability_detector(
    *,
    stream_runner: "StreamRunner",
    yolo_consumer: "YoloStreamConsumer",
    config: Optional[StabilityConfig] = None,
    on_stable: Optional[Callable[[Path], None]] = None,
) -> StabilityDetector:
    global _detector
    with _lock:
        if _detector is not None and _detector._thread is not None and _detector._thread.is_alive():
            return _detector
        cfg = config or StabilityConfig()
        trigger = on_stable or make_pipeline_trigger(
            station_id=cfg.station_id,
            reference_id=cfg.reference_id,
            imgsz=cfg.trigger_imgsz,
        )
        _detector = StabilityDetector(
            stream_runner=stream_runner,
            yolo_consumer=yolo_consumer,
            config=cfg,
            on_stable=trigger,
        )
        _detector.start()
        return _detector


def stop_stability_detector() -> None:
    global _detector
    with _lock:
        if _detector is not None:
            _detector.stop()
            _detector = None
