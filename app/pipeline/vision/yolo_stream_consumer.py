"""把 stream_runner 的关键帧喂给 YOLO-pose（OpenVINO NPU）持续推理。

设计要点（task #132 阶段 1 - 视频流接入 YOLO 推理）:

1. **解耦消费**: 单独 thread 消费 keyframe 队列，capture 与 inference 互不阻塞。
   capture 线程跑 15fps，inference 线程能消费多快算多快。
2. **模型一次加载**: ``YOLO(<openvino_model_dir>, task="pose")`` 加载一次，
   反复推理几百帧。第二次起 NPU compile 缓存命中只需 ~17ms。
3. **背压保护**: queue 默认 maxsize=5，满了丢最旧（产能跟不上时优先看新画面，
   旧 evidence 没意义了）。``dropped_total`` 统计可见。
4. **结果两路输出**:
   - 内存 ``latest()`` 给后续 agent / pipeline 同进程消费
   - ``/tmp/labguardian_detections_latest.json`` 原子写，供前端/外部进程拉
   - 可选 annotated 图 ``/tmp/labguardian_annotated/det_<ts>.jpg`` 调试用

板上设备实测（best.xml in /home/bupt/models/yolo_pose_int8_openvino_model）::

    CPU       : 48ms/帧  20fps
    intel:gpu : 18ms/帧  56fps
    intel:npu : 19ms/帧  53fps   ← 推荐（不抢 LLM 的 iGPU 显存 + 功耗低）

后续 stages（不在本文件）:
- 阶段 2: 把 detection 写入 RuntimeEvidence.station_findings（agent 能消费）
- 阶段 3: 加 AI 引导补拍（检测漏检引脚 → LLM 提示）
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.pipeline.vision.stream_runner import KeyframeEvent

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config + Result dataclasses
# -----------------------------------------------------------------------------


@dataclass
class YoloConsumerConfig:
    """YOLO 推理消费器配置。"""

    model_path: Path = field(
        default_factory=lambda: Path(
            "/home/bupt/models/yolo_pose_int8_openvino_model"
        )
    )
    """OpenVINO IR 目录。ultralytics 要求目录名以 `_openvino_model` 结尾。"""

    device: str = "intel:npu"
    """OpenVINO 设备：``intel:npu`` / ``intel:gpu`` / ``cpu``。"""

    task: str = "pose"
    """YOLO 任务类型。yolov8s-pose 训了 7 类元件 + 引脚 keypoint。"""

    imgsz: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5

    max_queue_size: int = 5
    """队列容量。满则丢最旧，保最新画面优先。"""

    output_json: Path = field(
        default_factory=lambda: Path("/tmp/labguardian_detections_latest.json")
    )
    """最新检测结果落盘（原子写）。"""

    annotated_image_dir: Path = field(
        default_factory=lambda: Path("/tmp/labguardian_annotated")
    )
    """annotated 图调试目录。"""

    save_annotated: bool = True
    max_annotated_on_disk: int = 30


@dataclass
class DetectedComponent:
    """单个检测到的元件。"""

    cls_id: int
    cls_name: str
    conf: float
    bbox_xyxy: list[float]
    """[x1, y1, x2, y2] 像素坐标。"""

    keypoints_xy: Optional[list[list[float]]] = None
    """引脚关键点 [[x, y], ...]，每个 keypoint 一行。pose 任务才有。"""


@dataclass
class DetectionResult:
    """一帧的完整检测结果。"""

    keyframe_path: Path
    timestamp: float
    """完成推理时的 wall clock。"""

    inference_ms: float
    components: list[DetectedComponent] = field(default_factory=list)
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# YoloStreamConsumer
# -----------------------------------------------------------------------------


class YoloStreamConsumer:
    """背景 thread：消费 keyframe 队列 → YOLO 推理 → 结果落盘。

    嵌入式用法::

        consumer = YoloStreamConsumer(YoloConsumerConfig(device="intel:npu"))
        consumer.start()
        # 第一帧前 compile 要 1-2s，等 warmup 完
        time.sleep(2)

        # 把 consumer.enqueue 给 StreamRunner 作回调
        runner = StreamRunner(StreamConfig(), on_keyframe=consumer.enqueue)
        runner.start()
        # ... 主程做别的事 ...
        runner.stop()
        consumer.stop()
    """

    def __init__(self, config: YoloConsumerConfig) -> None:
        self.config = config
        self._queue: "queue.Queue[KeyframeEvent]" = queue.Queue(
            maxsize=config.max_queue_size
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._model: Any = None
        self._latest_result: Optional[DetectionResult] = None
        self._processed_count: int = 0
        self._dropped_count: int = 0
        self._error: Optional[str] = None
        self._model_loaded_event = threading.Event()

        config.annotated_image_dir.mkdir(parents=True, exist_ok=True)
        config.output_json.parent.mkdir(parents=True, exist_ok=True)

    # ---- 公共接口 ----

    def start(self) -> None:
        """启动消费 thread。模型加载在 thread 内做。"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("YoloStreamConsumer already running")
            return
        self._stop_event.clear()
        self._model_loaded_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="labguardian-yolo-consumer"
        )
        self._thread.start()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """阻塞直到模型加载完。返回 True=就绪，False=超时或失败。"""
        ok = self._model_loaded_event.wait(timeout=timeout)
        if not ok:
            return False
        return self._error is None

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info(
            "YoloStreamConsumer stopped: processed=%d dropped=%d",
            self._processed_count,
            self._dropped_count,
        )

    def enqueue(self, keyframe_event: "KeyframeEvent") -> None:
        """从 StreamRunner 接关键帧事件。队列满则丢最旧。"""
        try:
            self._queue.put_nowait(keyframe_event)
        except queue.Full:
            # 丢最旧 → 让最新画面进队
            try:
                self._queue.get_nowait()
                self._dropped_count += 1
                self._queue.put_nowait(keyframe_event)
            except queue.Empty:  # pragma: no cover - 并发场景
                pass

    def latest(self) -> Optional[DetectionResult]:
        return self._latest_result

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "model_loaded": self._model_loaded_event.is_set(),
            "device": self.config.device,
            "queue_size": self._queue.qsize(),
            "processed_total": self._processed_count,
            "dropped_total": self._dropped_count,
            "latest_inference_ms": (
                self._latest_result.inference_ms if self._latest_result else None
            ),
            "latest_components_count": (
                len(self._latest_result.components) if self._latest_result else 0
            ),
            "latest_keyframe": (
                str(self._latest_result.keyframe_path) if self._latest_result else None
            ),
            "error": self._error,
        }

    # ---- 内部 ----

    def _load_model(self) -> Any:
        """加载 ultralytics YOLO（含 OpenVINO 后端 compile 一次）。"""
        from ultralytics import YOLO  # noqa: PLC0415

        logger.info(
            "Loading YOLO model: %s task=%s device=%s",
            self.config.model_path,
            self.config.task,
            self.config.device,
        )
        model = YOLO(str(self.config.model_path), task=self.config.task)

        # Warmup: 喂一帧 dummy 触发 compile + cache
        import numpy as np  # noqa: PLC0415

        dummy = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
        t0 = time.time()
        _ = model(
            dummy,
            device=self.config.device,
            imgsz=self.config.imgsz,
            verbose=False,
        )
        logger.info("Warmup done in %.2fs", time.time() - t0)
        return model

    def _run(self) -> None:
        try:
            self._model = self._load_model()
        except Exception as exc:
            self._error = f"YOLO model load failed: {exc}"
            logger.exception(self._error)
            self._model_loaded_event.set()  # 让 wait_ready 解锁
            return
        self._model_loaded_event.set()

        # 延迟 import cv2
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            self._error = f"opencv not installed: {exc}"
            logger.exception(self._error)
            return

        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._process_one(cv2, event)

    def _process_one(self, cv2_mod: Any, event: "KeyframeEvent") -> None:
        img = cv2_mod.imread(str(event.path))
        if img is None:
            logger.warning("Cannot read keyframe %s", event.path)
            return

        t0 = time.time()
        try:
            results = self._model(
                img,
                device=self.config.device,
                imgsz=self.config.imgsz,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("YOLO inference failed: %s", exc)
            error_result = DetectionResult(
                keyframe_path=event.path,
                timestamp=time.time(),
                inference_ms=0.0,
                error=str(exc),
            )
            self._latest_result = error_result
            return

        inference_ms = (time.time() - t0) * 1000.0
        r = results[0]
        class_names = r.names

        components: list[DetectedComponent] = []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes
            kpts = r.keypoints
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                kpts_xy: Optional[list[list[float]]] = None
                if kpts is not None and i < len(kpts):
                    try:
                        kpts_xy = kpts.xy[i].cpu().numpy().tolist()
                    except (AttributeError, IndexError):
                        kpts_xy = None
                components.append(
                    DetectedComponent(
                        cls_id=cls_id,
                        cls_name=class_names.get(cls_id, str(cls_id)),
                        conf=conf,
                        bbox_xyxy=bbox,
                        keypoints_xy=kpts_xy,
                    )
                )

        result = DetectionResult(
            keyframe_path=event.path,
            timestamp=time.time(),
            inference_ms=inference_ms,
            components=components,
        )
        self._latest_result = result
        self._processed_count += 1

        logger.info(
            "yolo[%d] %s inf=%.1fms n_components=%d",
            self._processed_count,
            event.path.name,
            inference_ms,
            len(components),
        )

        # 落盘 JSON（原子写）
        self._write_result_json(result)

        # 落盘 annotated 图
        if self.config.save_annotated:
            self._save_annotated(cv2_mod, r, result)

    def _write_result_json(self, result: DetectionResult) -> None:
        payload = {
            "keyframe_path": str(result.keyframe_path),
            "timestamp": result.timestamp,
            "inference_ms": round(result.inference_ms, 2),
            "components_count": len(result.components),
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
                for c in result.components
            ],
            "error": result.error,
        }
        try:
            tmp_path = self.config.output_json.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            tmp_path.replace(self.config.output_json)
        except OSError as exc:
            logger.warning("Failed to write detection JSON: %s", exc)

    def _save_annotated(
        self, cv2_mod: Any, ultra_result: Any, result: DetectionResult
    ) -> None:
        try:
            annotated = ultra_result.plot()
            ts_ms = int(result.timestamp * 1000)
            out = self.config.annotated_image_dir / f"det_{ts_ms}.jpg"
            cv2_mod.imwrite(str(out), annotated)
        except Exception:  # pragma: no cover
            logger.exception("annotated save failed")
            return

        # 清理旧
        try:
            files = sorted(self.config.annotated_image_dir.glob("det_*.jpg"))
            excess = len(files) - self.config.max_annotated_on_disk
            for old in files[: max(0, excess)]:
                try:
                    old.unlink()
                except OSError:
                    pass
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Global singleton
# -----------------------------------------------------------------------------

_consumer: Optional[YoloStreamConsumer] = None
_consumer_lock = threading.Lock()


def get_yolo_consumer() -> Optional[YoloStreamConsumer]:
    with _consumer_lock:
        return _consumer


def start_yolo_consumer(
    config: Optional[YoloConsumerConfig] = None,
) -> YoloStreamConsumer:
    """启动 singleton。已启动则返回现有的。"""
    global _consumer
    with _consumer_lock:
        if (
            _consumer is not None
            and _consumer._thread is not None
            and _consumer._thread.is_alive()
        ):
            return _consumer
        _consumer = YoloStreamConsumer(config or YoloConsumerConfig())
        _consumer.start()
        return _consumer


def stop_yolo_consumer() -> None:
    global _consumer
    with _consumer_lock:
        if _consumer is not None:
            _consumer.stop()
            _consumer = None
