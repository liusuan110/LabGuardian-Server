"""单路 USB 摄像头视频流 + 帧差关键帧抽取（task #132 阶段 1）。

设计目标:
1. **NPU duty cycle 提升**: 从原"按拍照按钮"模式（NPU 0.04% busy）→ 持续 15fps 流 +
   帧差触发推理（NPU ~20% busy）。
2. **去拍照按钮**: 学生把元件放上面包板就自动识别，不用主动操作。
3. **不破坏现有上传链路**: 模块默认 off，仅 `STREAM_MODE=true` env 或显式
   `start_stream_runner()` 才启动；原有 `POST /api/v1/pipeline/run` 上传图链路完全保留。

后续扩展方向（task #132 阶段 2 / 阶段 3）:
- 阶段 2: 加 AI 引导补拍（检测到引脚遮挡 → LLM 提示学生调整角度）
- 阶段 3: 升级到 3 路同步流（device_index 0/1/2）+ projection.py 多视图融合

板上验证: `scripts/board/stream_smoke.py`（15s 跑通就算 PASS）。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 配置 + 关键帧事件结构
# -----------------------------------------------------------------------------


@dataclass
class StreamConfig:
    """单路 USB 摄像头流配置。

    全部字段都有合理默认。覆盖时按 env 注入更稳，避免硬编码。
    """

    device_index: int = 0
    """`/dev/videoN` 的 N。板上 `/dev/video0` 主流 / `/dev/video1` 通常是 metadata。"""

    fps_target: int = 15
    """目标抓帧速率。15 是 NPU duty cycle 与 CPU 编码开销的甜点（见 plan Day 13）。"""

    resolution: tuple[int, int] = (640, 480)
    """抓帧分辨率 (W, H)。USB camera 默认 640x480 兼容性最好，pipeline 后续会 resize。"""

    frame_diff_threshold: float = 0.5
    """帧差触发关键帧阈值（cv2.absdiff(prev, curr).mean()）。

    经验值：
    - 0.3-0.5: 任何明显变化（元件挪动 / 手伸入）都抓 → 推荐起步
    - 0.8-1.2: 只抓大动作，避免光线波动假阳性 → 演示时调高
    """

    min_keyframe_interval_s: float = 1.0
    """两次关键帧的最小间隔，防止 1s 内连发 N 张挤爆 NPU 队列。"""

    max_keyframe_interval_s: float = 5.0
    """无变化时的兜底抓帧间隔。防止画面长时间静止 → evidence 过期。"""

    keyframe_dir: Path = field(default_factory=lambda: Path("/tmp/labguardian_keyframes"))
    """关键帧落盘目录。pipeline 通过 path 消费，前端可 sftp 取调试。"""

    max_keyframes_on_disk: int = 50
    """超过这个数量后自动清理最旧的，避免 /tmp 写爆。"""


@dataclass
class KeyframeEvent:
    """一次关键帧事件。on_keyframe 回调收到的载荷。"""

    timestamp: float
    """触发时刻的 wall clock（time.time()）。"""

    frame_idx: int
    """从启动到现在的总帧序号。"""

    path: Path
    """落盘的 JPG 绝对路径。"""

    trigger_reason: str
    """``"frame_diff"`` / ``"timeout"`` / ``"manual"``。"""

    diff_score: float
    """触发时的帧差分数（0 = 完全相同）。"""

    resolution: tuple[int, int] = (0, 0)
    """抓到的实际分辨率（可能不等于 config，取决于 USB camera 驱动）。"""


# -----------------------------------------------------------------------------
# StreamRunner: 背景 thread 持续抓帧 + 关键帧判定
# -----------------------------------------------------------------------------


class StreamRunner:
    """单路 USB 摄像头持续抓帧 → 帧差判定 → 关键帧落盘 + 回调推送。

    使用方式（嵌入式）::

        runner = StreamRunner(StreamConfig(), on_keyframe=lambda ev: ...)
        runner.start()
        # ... 主程序做别的事
        runner.stop()

    线程安全: ``start/stop/stats/latest_keyframe`` 都可在主线程随时调用。
    """

    def __init__(
        self,
        config: StreamConfig,
        on_keyframe: Optional[Callable[[KeyframeEvent], None]] = None,
    ) -> None:
        self.config = config
        self.on_keyframe = on_keyframe
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_keyframe: Optional[KeyframeEvent] = None
        self._last_keyframe_time: float = 0.0
        self._frame_count: int = 0
        self._keyframe_count: int = 0
        self._error: Optional[str] = None

        self.config.keyframe_dir.mkdir(parents=True, exist_ok=True)

    # ---- 公共接口 ----

    def start(self) -> None:
        """启动背景抓帧 thread。重复调用幂等。"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("StreamRunner already running, ignored start()")
            return
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="labguardian-stream"
        )
        self._thread.start()
        logger.info(
            "StreamRunner started: device=%s fps=%s res=%s",
            self.config.device_index,
            self.config.fps_target,
            self.config.resolution,
        )

    def stop(self, timeout: float = 3.0) -> None:
        """通知 thread 停止并 join。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info(
            "StreamRunner stopped: frames=%d keyframes=%d",
            self._frame_count,
            self._keyframe_count,
        )

    def latest_keyframe(self) -> Optional[KeyframeEvent]:
        """返回最近一次关键帧事件（线程安全的快照读取）。"""
        return self._latest_keyframe

    def stats(self) -> dict[str, Any]:
        """运行时统计，供 health endpoint / 前端展示 NPU busy 指示灯。"""
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "frames_total": self._frame_count,
            "keyframes_total": self._keyframe_count,
            "device_index": self.config.device_index,
            "fps_target": self.config.fps_target,
            "latest_keyframe_path": (
                str(self._latest_keyframe.path) if self._latest_keyframe else None
            ),
            "latest_keyframe_ts": (
                self._latest_keyframe.timestamp if self._latest_keyframe else None
            ),
            "error": self._error,
        }

    def force_capture(self, reason: str = "manual") -> Optional[KeyframeEvent]:
        """强制把下一帧抓为关键帧（前端按钮兜底）。

        不能直接保存当前帧（thread 在抓），只能告诉 thread 下一帧无条件 save。
        实现简单粗暴：把 last_keyframe_time 调旧 → 触发 timeout 路径。
        """
        self._last_keyframe_time = 0.0
        return self._latest_keyframe

    # ---- 内部 thread 主循环 ----

    def _run(self) -> None:
        # 延迟 import cv2 — 避免本地 dev 环境（不装 opencv）导入失败
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            self._error = f"opencv not installed: {exc}"
            logger.exception(self._error)
            return

        cap = cv2.VideoCapture(self.config.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.config.fps_target)

        if not cap.isOpened():
            self._error = f"cannot open /dev/video{self.config.device_index}"
            logger.error(self._error)
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "USB camera ready: actual_res=%dx%d actual_fps=%.1f",
            actual_w,
            actual_h,
            actual_fps,
        )

        frame_interval = 1.0 / self.config.fps_target
        prev_frame: Optional[Any] = None

        try:
            while not self._stop_event.is_set():
                loop_start = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                self._frame_count += 1

                now = time.time()
                reason: Optional[str] = None
                diff_score: float = 0.0

                # 帧差判定（前后帧整体平均亮度差）
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, frame)
                    diff_score = float(diff.mean())
                    elapsed_since_last = now - self._last_keyframe_time
                    if (
                        diff_score > self.config.frame_diff_threshold
                        and elapsed_since_last >= self.config.min_keyframe_interval_s
                    ):
                        reason = "frame_diff"

                # 兜底: 太久没抓也要强制抓一张
                if reason is None:
                    elapsed_since_last = now - self._last_keyframe_time
                    if elapsed_since_last >= self.config.max_keyframe_interval_s:
                        reason = "timeout"

                if reason is not None:
                    self._save_keyframe(
                        cv2, frame, now, reason, diff_score, (actual_w, actual_h)
                    )

                prev_frame = frame

                # 节流到目标 fps（避免 CPU 跑满）
                consumed = time.time() - loop_start
                slack = frame_interval - consumed
                if slack > 0:
                    time.sleep(slack)
        finally:
            cap.release()

    def _save_keyframe(
        self,
        cv2_mod: Any,
        frame: Any,
        now: float,
        reason: str,
        diff_score: float,
        resolution: tuple[int, int],
    ) -> None:
        ts_ms = int(now * 1000)
        path = self.config.keyframe_dir / f"kf_{ts_ms}.jpg"
        try:
            cv2_mod.imwrite(str(path), frame)
        except Exception as exc:  # pragma: no cover - 防御性
            logger.warning("imwrite failed: %s", exc)
            return

        event = KeyframeEvent(
            timestamp=now,
            frame_idx=self._frame_count,
            path=path,
            trigger_reason=reason,
            diff_score=diff_score,
            resolution=resolution,
        )
        self._latest_keyframe = event
        self._last_keyframe_time = now
        self._keyframe_count += 1
        logger.info(
            "keyframe[%d] %s reason=%s diff=%.2f",
            self._keyframe_count,
            path.name,
            reason,
            diff_score,
        )

        if self.on_keyframe is not None:
            try:
                self.on_keyframe(event)
            except Exception:  # pragma: no cover
                logger.exception("on_keyframe callback raised")

        # 旧帧清理（保留最近 N 张）
        self._cleanup_old_keyframes()

    def _cleanup_old_keyframes(self) -> None:
        try:
            keyframes = sorted(self.config.keyframe_dir.glob("kf_*.jpg"))
        except OSError:
            return
        excess = len(keyframes) - self.config.max_keyframes_on_disk
        if excess <= 0:
            return
        for old in keyframes[:excess]:
            try:
                old.unlink()
            except OSError:
                pass


# -----------------------------------------------------------------------------
# 全局 singleton（供 FastAPI 端点共享同一个 runner 实例）
# -----------------------------------------------------------------------------

_runner: Optional[StreamRunner] = None
_runner_lock = threading.Lock()


def get_stream_runner() -> Optional[StreamRunner]:
    """返回当前 StreamRunner singleton，未启动返回 None。"""
    with _runner_lock:
        return _runner


def start_stream_runner(config: Optional[StreamConfig] = None) -> StreamRunner:
    """启动 singleton。已在跑则直接返回现有的。"""
    global _runner
    with _runner_lock:
        if _runner is not None and _runner._thread is not None and _runner._thread.is_alive():
            return _runner
        _runner = StreamRunner(config or StreamConfig())
        _runner.start()
        return _runner


def stop_stream_runner() -> None:
    """停止 singleton（如有）。"""
    global _runner
    with _runner_lock:
        if _runner is not None:
            _runner.stop()
            _runner = None
