"""端到端检测测试：用真实图片调用 pipeline，展示检测结果"""
import base64
import json
import sys
import urllib.request
from pathlib import Path

IMAGE_PATH = Path(r"E:\inter\yolo8\test\images\IMG_2693_JPG.rf.aed631e00941ad4c8fc7cf6a41baa712.jpg")

with open(IMAGE_PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

print(f"图片: {IMAGE_PATH.name}")
print(f"大小: {IMAGE_PATH.stat().st_size / 1024:.1f} KB")
print()

payload = json.dumps({
    "station_id": "e2e_test",
    "images_b64": [b64],
    "conf": 0.25,
    "iou": 0.5,
    "imgsz": 960,
}).encode()

req = urllib.request.Request(
    "http://localhost:8000/api/v1/pipeline/run",
    data=payload,
    headers={"Content-Type": "application/json"},
)

print("POST /api/v1/pipeline/run ...")
with urllib.request.urlopen(req, timeout=120) as resp:
    body = json.loads(resp.read())

stages = {s["stage"]: s["data"] for s in body.get("stages", [])}
detect = stages.get("detect", {})
dets = detect.get("detections", [])
total_ms = body.get("total_duration_ms", 0)

print(f"HTTP 200  |  总耗时: {total_ms:.0f} ms")
print(f"检测到 {len(dets)} 个目标\n")

if dets:
    # 按置信度排序
    dets_sorted = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    print(f"{'#':<3} {'component_type':<22} {'package_type':<22} {'pin_schema_id':<22} {'conf':>6}  bbox")
    print("-" * 100)
    for i, d in enumerate(dets_sorted, 1):
        bbox = d.get("bbox", [])
        print(f"{i:<3} {d['component_type']:<22} {d['package_type']:<22} {d['pin_schema_id']:<22} {d['confidence']:>6.3f}  {bbox}")

    # 类别汇总
    print()
    from collections import Counter
    counts = Counter(d["component_type"] for d in dets)
    print("类别统计:")
    for ct, n in sorted(counts.items()):
        print(f"  {ct}: {n}")
else:
    print("(无检测结果 — 可能图片内容不在模型训练类别内，或置信度低于阈值)")
    print(f"\ndetector_backend: {detect.get('detector_backend')}")
    contract = detect.get("detector_contract", {})
    print(f"model loaded    : {contract.get('loaded')}")
    print(f"task            : {contract.get('task')}")
