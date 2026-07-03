"""两侧摄像头预备门控：有板且静止一段时间后，才允许顶视正式链路启动。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from app.pipeline.vision.stream_runner import StreamRunner

logger = logging.getLogger(__name__)


class SideGateState(str, Enum):
    WAITING = "waiting"
    STABILIZING = "stabilizing"
    READY = "ready"
    TRIGGERED = "triggered"


@dataclass
class SideGateConfig:
    stable_duration_s: float = 2.5
    """两侧同时静止多久才放行顶视主流程。"""

    poll_interval_s: float = 0.5
    """轮询周期。"""

    max_motion_score: float = 0.5
    """允许的最大运动分数，直接复用 StreamRunner 的 diff_score 语义。"""

    min_presence_score: float = 4.0
    """相对背景帧的最小存在分数，避免空画面误触发。"""

    debounce_after_trigger_s: float = 5.0
    """放行后的冷却时间，避免重复触发。"""


@dataclass
class SideGateSnapshot:
    state: SideGateState
    stable_since_ts: Optional[float] = None
    last_ready_ts: Optional[float] = None
    trigger_count: int = 0
    poll_count: int = 0
    error: Optional[str] = None
    views: dict[str, dict[str, Any]] | None = None


class SideCameraGateDetector:
    """监视两侧相机：检测到有板且稳定后，触发一次顶视主流程启动。"""

    def __init__(
        self,
        *,
        runners: dict[str, "StreamRunner"],
        config: SideGateConfig,
        on_ready: Callable[[], None],
    ) -> None:
        self._runners = dict(runners)
        self.config = config
        self._on_ready = on_ready

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._state: SideGateState = SideGateState.WAITING
        self._stable_since_ts: Optional[float] = None
        self._last_ready_ts: Optional[float] = None
        self._trigger_count: int = 0
        self._poll_count: int = 0
        self._error: Optional[str] = None
        self._latest_view_stats: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="labguardian-side-gate"
        )
        self._thread.start()
        logger.info(
            "SideCameraGateDetector started: views=%s window=%ss motion<=%.3f presence>=%.3f",
            sorted(self._runners.keys()),
            self.config.stable_duration_s,
            self.config.max_motion_score,
            self.config.min_presence_score,
        )

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def snapshot(self) -> SideGateSnapshot:
        return SideGateSnapshot(
            state=self._state,
            stable_since_ts=self._stable_since_ts,
            last_ready_ts=self._last_ready_ts,
            trigger_count=self._trigger_count,
            poll_count=self._poll_count,
            error=self._error,
            views=dict(self._latest_view_stats),
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._poll_count += 1
            try:
                self._tick()
            except Exception as exc:  # pragma: no cover
                self._error = f"{type(exc).__name__}: {exc}"
                logger.exception("side gate tick failed: %s", exc)
            time.sleep(self.config.poll_interval_s)

    def _tick(self) -> None:
        now = time.time()
        stats_by_view = {
            view_id: runner.stats() for view_id, runner in self._runners.items()
        }
        self._latest_view_stats = stats_by_view

        if not stats_by_view:
            self._transition_to(SideGateState.WAITING, reset_since=True)
            return

        if not all(bool(stats.get("running")) for stats in stats_by_view.values()):
            self._transition_to(SideGateState.WAITING, reset_since=True)
            return

        if not all(self._has_presence(stats) for stats in stats_by_view.values()):
            self._transition_to(SideGateState.WAITING, reset_since=True)
            return

        if not all(self._is_still(stats) for stats in stats_by_view.values()):
            self._transition_to(SideGateState.WAITING, reset_since=True)
            return

        if self._state == SideGateState.WAITING:
            self._transition_to(SideGateState.STABILIZING, reset_since=False)
            self._stable_since_ts = now
            return

        if self._state == SideGateState.STABILIZING:
            assert self._stable_since_ts is not None
            if now - self._stable_since_ts >= self.config.stable_duration_s:
                self._transition_to(SideGateState.READY, reset_since=False)
            return

        if self._state == SideGateState.READY:
            self._fire_ready(now)
            self._transition_to(SideGateState.TRIGGERED, reset_since=False)
            return

    def _has_presence(self, stats: dict[str, Any]) -> bool:
        presence_score = float(stats.get("latest_presence_score", 0.0) or 0.0)
        presence_detected = bool(stats.get("presence_detected"))
        return presence_detected or presence_score >= self.config.min_presence_score

    def _is_still(self, stats: dict[str, Any]) -> bool:
        diff_score = float(stats.get("latest_diff_score", 0.0) or 0.0)
        latest_frame_ts = stats.get("latest_frame_ts")
        if latest_frame_ts is None:
            return False
        return diff_score <= self.config.max_motion_score

    def _transition_to(self, new_state: SideGateState, *, reset_since: bool) -> None:
        if new_state == self._state:
            return
        logger.info("side gate %s -> %s", self._state.value, new_state.value)
        self._state = new_state
        if reset_since:
            self._stable_since_ts = None

    def _fire_ready(self, now: float) -> None:
        if (
            self._last_ready_ts is not None
            and now - self._last_ready_ts < self.config.debounce_after_trigger_s
        ):
            return
        logger.info("side gate ready -> starting top pipeline")
        try:
            self._on_ready()
        except Exception as exc:
            logger.exception("side gate callback failed: %s", exc)
            self._error = f"{type(exc).__name__}: {exc}"
            return
        self._last_ready_ts = now
        self._trigger_count += 1


_detector: Optional[SideCameraGateDetector] = None
_lock = threading.Lock()


def get_side_camera_gate_detector() -> Optional[SideCameraGateDetector]:
    with _lock:
        return _detector


def start_side_camera_gate_detector(
    *,
    runners: dict[str, "StreamRunner"],
    config: Optional[SideGateConfig] = None,
    on_ready: Optional[Callable[[], None]] = None,
) -> SideCameraGateDetector:
    global _detector
    with _lock:
        if _detector is not None and _detector._thread is not None and _detector._thread.is_alive():
            return _detector
        if on_ready is None:
            raise ValueError("on_ready callback is required")
        _detector = SideCameraGateDetector(
            runners=runners,
            config=config or SideGateConfig(),
            on_ready=on_ready,
        )
        _detector.start()
        return _detector


def stop_side_camera_gate_detector() -> None:
    global _detector
    with _lock:
        if _detector is not None:
            _detector.stop()
            _detector = None
