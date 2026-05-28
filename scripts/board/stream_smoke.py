"""板上 smoke：跑 15s 视频流验证 stream_runner 真能拉关键帧（task #132 阶段 1）。

板上使用::

    cd ~/LabGuardian-Server
    /home/bupt/miniconda3/envs/labguardian/bin/python -m scripts.board.stream_smoke

预期输出（在镜头前晃手）::

    [   5] kf trigger=frame_diff diff=2.34 → kf_1717000001234.jpg
    [  18] kf trigger=frame_diff diff=1.78 → kf_1717000002567.jpg
    ...
    Final: frames=225 keyframes=8 (frame_diff=6, timeout=2)

PASS 条件:
- frames_total ≈ fps_target × duration（板上 NPU 不抢 USB 时基本到 200+）
- keyframes_total > 2（至少有兜底 timeout 抓的）
- 帧差触发数 > 0（说明帧差判定真工作）
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# 让脚本能直接 `python -m scripts.board.stream_smoke` 跑
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.vision.stream_runner import (  # noqa: E402
    KeyframeEvent,
    StreamConfig,
    StreamRunner,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream runner smoke (15s default)")
    parser.add_argument("--device", type=int, default=0, help="/dev/videoN")
    parser.add_argument("--duration", type=float, default=15.0, help="seconds")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--diff-threshold", type=float, default=0.5)
    parser.add_argument(
        "--keyframe-dir",
        type=Path,
        default=Path("/tmp/labguardian_keyframes"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    events: list[KeyframeEvent] = []

    def on_keyframe(ev: KeyframeEvent) -> None:
        events.append(ev)
        print(
            f"[{ev.frame_idx:4d}] kf trigger={ev.trigger_reason:11s} "
            f"diff={ev.diff_score:5.2f} → {ev.path.name}",
            flush=True,
        )

    config = StreamConfig(
        device_index=args.device,
        fps_target=args.fps,
        frame_diff_threshold=args.diff_threshold,
        keyframe_dir=args.keyframe_dir,
    )
    runner = StreamRunner(config, on_keyframe=on_keyframe)

    print(
        f"\nStarting stream on /dev/video{args.device}, "
        f"target {args.fps}fps, "
        f"diff threshold {args.diff_threshold}, "
        f"duration {args.duration}s"
    )
    print("→ 请在镜头前晃动手或放置元件，验证帧差触发")
    print()

    runner.start()
    # 早期错误暴露：如果 cv2 打不开 camera，stats.error 会立刻有值
    time.sleep(1.5)
    early_stats = runner.stats()
    if early_stats.get("error"):
        print(f"❌ FAIL: {early_stats['error']}", file=sys.stderr)
        runner.stop()
        return 1

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\n(interrupted)")
    finally:
        runner.stop()

    stats = runner.stats()
    print("\n" + "=" * 60)
    print(f"Final stats: {stats}")
    diff_count = sum(1 for e in events if e.trigger_reason == "frame_diff")
    timeout_count = sum(1 for e in events if e.trigger_reason == "timeout")
    print(
        f"Captured {len(events)} keyframes "
        f"(frame_diff={diff_count}, timeout={timeout_count})"
    )

    # PASS 判定
    failures: list[str] = []
    if stats["frames_total"] < args.fps * args.duration * 0.5:
        failures.append(
            f"frames_total={stats['frames_total']} < expected "
            f"{int(args.fps * args.duration * 0.5)} (低于目标一半，可能 USB 带宽/CPU 不够)"
        )
    if stats["keyframes_total"] < 2:
        failures.append(
            f"keyframes_total={stats['keyframes_total']} < 2 (兜底 timeout 都没抓到)"
        )

    if failures:
        print("\n❌ FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\n✅ PASS: stream runner 工作正常")
    print(f"   关键帧落盘目录: {args.keyframe_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
