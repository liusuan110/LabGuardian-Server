"""
YOLO 端到端冒烟测试 (需要服务器运行)

向 POST /api/v1/pipeline/run 发送一张真实图片, 检查 detect 阶段返回的目标
类别和置信度。

测试图片来源 (按优先级):
  1. 环境变量 YOLO_TEST_IMAGES_DIR
  2. 项目内默认路径 tests/fixtures/yolo_smoke_images/

如果两个位置都没有 *.jpg/*.png 文件, 测试会优雅跳过 (退出码 0), 不视为失败。

运行方式:
  # 1. 启动服务
  uvicorn app.main:app --host 0.0.0.0 --port 8000

  # 2. 指定图片目录 (可选, 不设也行)
  $env:YOLO_TEST_IMAGES_DIR = "E:\path\to\images"

  # 3. 跑测试
  python tests/manual/smoke/test_yolo_e2e_smoke.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "yolo_smoke_images"
PIPELINE_URL = "http://localhost:8000/api/v1/pipeline/run"


def resolve_images_dir() -> Path | None:
    env_dir = os.environ.get("YOLO_TEST_IMAGES_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists() and p.is_dir():
            return p
        print(f"⚠ YOLO_TEST_IMAGES_DIR 指向不存在的目录: {env_dir}")
        return None
    if DEFAULT_IMAGES_DIR.exists() and DEFAULT_IMAGES_DIR.is_dir():
        return DEFAULT_IMAGES_DIR
    return None


def pick_first_image(dir_path: Path) -> Path | None:
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img in sorted(dir_path.glob(ext)):
            return img
    return None


def main() -> int:
    images_dir = resolve_images_dir()
    if images_dir is None:
        print(
            "SKIP: 没找到测试图片目录\n"
            f"  - 设置环境变量 YOLO_TEST_IMAGES_DIR 指向含 .jpg/.png 的目录\n"
            f"  - 或放图片到 {DEFAULT_IMAGES_DIR}"
        )
        return 0

    image = pick_first_image(images_dir)
    if image is None:
        print(f"SKIP: 目录 {images_dir} 下没有 .jpg/.jpeg/.png 文件")
        return 0

    print(f"图片: {image.name}")
    print(f"大小: {image.stat().st_size / 1024:.1f} KB")
    print(f"目录: {images_dir}\n")

    b64 = base64.b64encode(image.read_bytes()).decode()
    payload = json.dumps({
        "station_id": "yolo_e2e_smoke",
        "images_b64": [b64],
        "conf": 0.25,
        "iou": 0.5,
        "imgsz": 960,
    }).encode()

    print(f"POST {PIPELINE_URL} ...")
    req = urllib.request.Request(
        PIPELINE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"FAIL: 无法连接到 {PIPELINE_URL} — {e}")
        print("  请先启动服务: uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return 1

    stages = {s["stage"]: s["data"] for s in body.get("stages", [])}
    detect = stages.get("detect", {})
    dets = detect.get("detections", [])
    total_ms = body.get("total_duration_ms", 0)

    print(f"HTTP 200  |  总耗时: {total_ms:.0f} ms")
    print(f"检测到 {len(dets)} 个目标\n")

    if dets:
        dets_sorted = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        header = f"{'#':<3} {'component_type':<22} {'package_type':<22} {'pin_schema_id':<22} {'conf':>6}  bbox"
        print(header)
        print("-" * 100)
        for i, d in enumerate(dets_sorted, 1):
            bbox = d.get("bbox", [])
            print(
                f"{i:<3} {d['component_type']:<22} {d['package_type']:<22} "
                f"{d['pin_schema_id']:<22} {d['confidence']:>6.3f}  {bbox}"
            )

        print("\n类别统计:")
        for ct, n in sorted(Counter(d["component_type"] for d in dets).items()):
            print(f"  {ct}: {n}")
        return 0

    print("(无检测结果 — 可能图片不在模型训练类别内, 或置信度低于阈值)")
    contract = detect.get("detector_contract", {})
    print(f"\ndetector_backend: {detect.get('detector_backend')}")
    print(f"model loaded    : {contract.get('loaded')}")
    print(f"task            : {contract.get('task')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
