"""板上 smoke：摄像头视频流 → YOLO-pose NPU 推理 → 检测结果落盘（task #132 阶段 1）。

板上用法::

    cd ~/LabGuardian-Server
    /home/bupt/miniconda3/envs/labguardian/bin/python -m scripts.board.stream_yolo_smoke \\
        --device intel:npu --duration 25

预期输出（在镜头前放真实元件）::

    t= 1s capture=  15f/  1kf  yolo proc= 1 latest_n=0 inf=  2.3ms
    t= 5s capture=  74f/  3kf  yolo proc= 3 latest_n=2 inf= 19.8ms
                                              ^^^^^^^^
                                              检测到 2 个元件
    ...
    Final: processed=14 dropped=0 avg_inf=18.7ms classes_seen={'resistor', 'led'}

PASS 条件:
- consumer.stats().model_loaded == True
- processed_total > 2（NPU 真在推理）
- error 为空
- 至少有一帧 latest_components_count > 0（有人放元件时）
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.vision.stream_runner import (  # noqa: E402
    KeyframeEvent,
    StreamConfig,
    StreamRunner,
)
from app.pipeline.vision.yolo_stream_consumer import (  # noqa: E402
    YoloConsumerConfig,
    YoloStreamConsumer,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream → YOLO NPU smoke (task #132 阶段 1)"
    )
    parser.add_argument("--video-device", type=int, default=0, help="/dev/videoN")
    parser.add_argument(
        "--device",
        default="intel:npu",
        choices=["intel:npu", "intel:gpu", "cpu"],
        help="YOLO 推理设备",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/bupt/models/yolo_pose_int8_openvino_model"),
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=0.5,
        help="帧差关键帧阈值（mean absdiff）",
    )
    parser.add_argument("--no-annotated", action="store_true", help="不存 annotated 图")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    print("\n" + "=" * 60)
    print("Task #132 阶段 1: 视频流 → YOLO NPU 推理")
    print("=" * 60)
    print(f"  Camera   : /dev/video{args.video_device} @ {args.fps}fps")
    print(f"  Model    : {args.model_dir.name}")
    print(f"  Device   : {args.device}")
    print(f"  Duration : {args.duration}s")
    print()

    # 1. 启 YOLO consumer（先启，先 warmup 模型）
    consumer_config = YoloConsumerConfig(
        model_path=args.model_dir,
        device=args.device,
        save_annotated=not args.no_annotated,
    )
    consumer = YoloStreamConsumer(consumer_config)
    consumer.start()

    print("⏳ Loading YOLO model (compile + warmup)...")
    ready = consumer.wait_ready(timeout=60.0)
    if not ready:
        stats = consumer.stats()
        print(f"❌ FAIL: model not ready. stats={stats}", file=sys.stderr)
        consumer.stop()
        return 1
    print(f"✅ Model ready ({consumer.stats()['device']})")

    # 2. 启 StreamRunner，关键帧推给 consumer
    stream_config = StreamConfig(
        device_index=args.video_device,
        fps_target=args.fps,
        frame_diff_threshold=args.diff_threshold,
    )
    runner = StreamRunner(stream_config, on_keyframe=consumer.enqueue)
    runner.start()

    # 早期 camera 错误探活
    time.sleep(1.5)
    rstats = runner.stats()
    if rstats.get("error"):
        print(f"❌ FAIL camera: {rstats['error']}", file=sys.stderr)
        runner.stop()
        consumer.stop()
        return 1

    print()
    print("📷 请在镜头前放电阻 / LED / 三极管，看实时检测...")
    print()

    # 3. 每秒打点
    classes_seen: Counter[str] = Counter()
    inference_samples: list[float] = []
    try:
        for sec in range(int(args.duration)):
            time.sleep(1.0)
            cs = consumer.stats()
            rs = runner.stats()
            latest = consumer.latest()
            n_components = cs["latest_components_count"]
            if latest:
                for c in latest.components:
                    classes_seen[c.cls_name] += 1
                if latest.inference_ms > 0:
                    inference_samples.append(latest.inference_ms)
            print(
                f"  t={sec + 1:2d}s  capture={rs['frames_total']:4d}f/{rs['keyframes_total']:2d}kf  "
                f"yolo proc={cs['processed_total']:3d}/drop={cs['dropped_total']:2d}  "
                f"latest_n={n_components}  inf={cs['latest_inference_ms'] or 0:5.1f}ms"
            )
    except KeyboardInterrupt:
        print("\n(interrupted)")
    finally:
        runner.stop()
        consumer.stop()

    # 4. PASS 判定
    final = consumer.stats()
    rfinal = runner.stats()
    avg_inf = (
        sum(inference_samples) / len(inference_samples) if inference_samples else 0.0
    )

    print()
    print("=" * 60)
    print("Final stats:")
    print(f"  capture       : {rfinal['frames_total']} frames / {rfinal['keyframes_total']} keyframes")
    print(f"  yolo processed: {final['processed_total']}  dropped: {final['dropped_total']}")
    print(f"  avg inference : {avg_inf:.1f}ms")
    print(f"  classes seen  : {dict(classes_seen) if classes_seen else '(none — 镜头前可能没东西)'}")

    failures: list[str] = []
    if final.get("error"):
        failures.append(f"consumer error: {final['error']}")
    if not final["model_loaded"]:
        failures.append("model_loaded=False")
    if final["processed_total"] < 2:
        failures.append(
            f"processed_total={final['processed_total']} < 2 (NPU 没真推理)"
        )

    if failures:
        print("\n❌ FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\n✅ PASS: stream → YOLO NPU 通畅工作")
    if not classes_seen:
        print(
            "   提示：本次没看到任何检测类别。如果镜头前确实放了元件还是没出，"
            "去 /tmp/labguardian_annotated/ 看 annotated 图查原因。"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
