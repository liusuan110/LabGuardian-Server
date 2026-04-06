"""
演示脚本: 跑 Pipeline + 模拟学生心跳 → 教师端看到数据
"""
import base64
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import requests

SERVER = "http://127.0.0.1:8000"
DEMO_IMG = r"D:\desktop\inter\LabGuardian\dataset\images\demo_self\valid\images\-_20260307102606_394_202_jpg.rf.50f22610ff8c8d5baf0027f46fa1eb98.jpg"

# ── 1. 读取演示照片 ──
with open(DEMO_IMG, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()
print(f"[1] 读取演示照片: {len(img_b64)} chars")

# ── 2. 调用同步 Pipeline ──
print("[2] 调用 POST /api/v1/pipeline/run ...")
t0 = time.time()
resp = requests.post(
    f"{SERVER}/api/v1/pipeline/run",
    json={
        "station_id": "S01",
        "images_b64": [img_b64],
    },
    timeout=120,
)
elapsed = time.time() - t0
print(f"    状态码: {resp.status_code}  耗时: {elapsed:.1f}s")

if resp.status_code != 200:
    print(f"    错误: {resp.text[:500]}")
    sys.exit(1)

pipeline_result = resp.json()
print(f"    total_duration_ms: {pipeline_result.get('total_duration_ms', 0):.0f}")
print(f"    component_count:   {pipeline_result.get('component_count', 0)}")
print(f"    net_count:         {pipeline_result.get('net_count', 0)}")
print(f"    risk_level:        {pipeline_result.get('risk_level', '?')}")
print(f"    diagnostics:       {pipeline_result.get('diagnostics', [])}")
print(f"    risk_reasons:      {pipeline_result.get('risk_reasons', [])}")

# 提取关键数据用于心跳
component_count = pipeline_result.get("component_count", 0)
net_count = pipeline_result.get("net_count", 0)
diagnostics = pipeline_result.get("diagnostics", [])
risk_level = pipeline_result.get("risk_level", "safe")
risk_reasons = pipeline_result.get("risk_reasons", [])
similarity = pipeline_result.get("similarity", 0.0)
progress = pipeline_result.get("progress", 0.0)

# 缩略图 (缩小原图)
import cv2
img_raw = cv2.imread(DEMO_IMG)
h, w = img_raw.shape[:2]
scale = 320 / max(h, w)
thumb = cv2.resize(img_raw, (int(w * scale), int(h * scale)))
_, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
thumb_b64 = base64.b64encode(buf).decode()

# 从 S3 获取电路快照
stages = pipeline_result.get("stages", [])
circuit_snapshot = ""
for s in stages:
    if s.get("stage") == "topology":
        circuit_snapshot = s.get("data", {}).get("circuit_description", "")
        break

# 提取 components 列表
components_info = []
for s in stages:
    if s.get("stage") == "detect":
        dets = s.get("data", {}).get("detections", [])
        for d in dets[:20]:
            components_info.append({
                "name": d.get("class_name", ""),
                "type": d.get("class_name", ""),
                "confidence": d.get("confidence", 0),
            })
        break

print(f"\n[3] Pipeline 完成! 开始模拟学生心跳...")
print(f"    thumbnail: {len(thumb_b64)} chars")
print(f"    circuit_snapshot: {len(circuit_snapshot)} chars")
print(f"    components: {len(components_info)} items")

# ── 3. 模拟两个学生工位持续心跳 ──
STATIONS = [
    {
        "station_id": "S01",
        "student_name": "张三",
        "progress": 0.85,
        "similarity": 0.78,
        "risk_level": risk_level,
    },
    {
        "station_id": "S02",
        "student_name": "李四",
        "progress": 0.45,
        "similarity": 0.30,
        "risk_level": "safe",
    },
]

heartbeat_count = 0
stop_event = threading.Event()

def send_heartbeats():
    global heartbeat_count
    while not stop_event.is_set():
        for station in STATIONS:
            hb = {
                "station_id": station["station_id"],
                "student_name": station["student_name"],
                "timestamp": time.time(),
                "component_count": component_count if station["station_id"] == "S01" else 3,
                "net_count": net_count if station["station_id"] == "S01" else 1,
                "components": components_info if station["station_id"] == "S01" else [],
                "progress": station["progress"],
                "similarity": station["similarity"],
                "match_level": "L2" if station["station_id"] == "S01" else "L0",
                "missing_components": [] if station["station_id"] == "S01" else ["LED", "电阻"],
                "diagnostics": diagnostics if station["station_id"] == "S01" else ["L0: 元件数量不足"],
                "risk_level": station["risk_level"],
                "risk_reasons": risk_reasons if station["station_id"] == "S01" else [],
                "circuit_snapshot": circuit_snapshot if station["station_id"] == "S01" else "",
                "fps": 15.0,
                "detector_ok": "ok",
                "thumbnail_b64": thumb_b64 if station["station_id"] == "S01" else "",
            }
            try:
                r = requests.post(f"{SERVER}/api/v1/classroom/heartbeat", json=hb, timeout=5)
                if r.status_code != 200:
                    print(f"    [心跳] {station['station_id']} 失败: {r.status_code} {r.text[:100]}")
            except Exception as e:
                print(f"    [心跳] {station['station_id']} 异常: {e}")

        heartbeat_count += 1
        if heartbeat_count <= 3 or heartbeat_count % 10 == 0:
            print(f"    [心跳] #{heartbeat_count} 已发送 (2 工位)")
        stop_event.wait(2.0)  # 每 2 秒一次

# 启动心跳线程
hb_thread = threading.Thread(target=send_heartbeats, daemon=True)
hb_thread.start()

# ── 4. 验证教师端 API ──
time.sleep(3)  # 等几秒让心跳生效
print("\n[4] 验证教师端 API ...")

resp = requests.get(f"{SERVER}/api/v1/classroom/stations", timeout=5)
stations = resp.json()
print(f"    /stations: {len(stations)} 工位")
for sid, data in stations.items():
    print(f"      {sid}: online={data.get('online')} progress={data.get('progress')} risk={data.get('risk_level')}")

resp = requests.get(f"{SERVER}/api/v1/classroom/ranking", timeout=5)
ranking = resp.json()
print(f"    /ranking: {len(ranking)} 条")
for r in ranking:
    print(f"      #{r['rank']} {r['station_id']} {r['student_name']} progress={r['progress']}")

resp = requests.get(f"{SERVER}/api/v1/classroom/stats", timeout=5)
stats = resp.json()
print(f"    /stats: online={stats['online_count']}/{stats['total_stations']} avg_progress={stats['avg_progress']:.2f}")

resp = requests.get(f"{SERVER}/api/v1/classroom/alerts", timeout=5)
alerts = resp.json()
print(f"    /alerts: {len(alerts)} 条")

resp = requests.get(f"{SERVER}/api/v1/classroom/station/S01", timeout=5)
print(f"    /station/S01: status={resp.status_code}")

resp = requests.get(f"{SERVER}/api/v1/classroom/station/S01/thumbnail", timeout=5)
print(f"    /station/S01/thumbnail: status={resp.status_code} has_thumb={'thumbnail_b64' in resp.json()}")

print("\n" + "=" * 60)
print("全部流程跑通! 教师端(Electron)应该已看到 2 个工位数据。")
print("心跳持续发送中... 按 Ctrl+C 停止。")
print("=" * 60)

# 保持运行
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stop_event.set()
    print("\n已停止心跳。")
