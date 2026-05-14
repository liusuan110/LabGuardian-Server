"""触发一次最小 pipeline 请求，让 YOLO 模型完成懒加载，确认日志"""
import base64
import json
import sys
import urllib.request

import numpy as np
import cv2

# 生成 100x100 黑色 JPEG
img = np.zeros((100, 100, 3), dtype=np.uint8)
_, buf = cv2.imencode(".jpg", img)
b64 = base64.b64encode(buf).decode()

payload = json.dumps({"station_id": "smoke_test", "images_b64": [b64]}).encode()

req = urllib.request.Request(
    "http://localhost:8000/api/v1/pipeline/run",
    data=payload,
    headers={"Content-Type": "application/json"},
)

print("发送请求 → POST /api/v1/pipeline/run ...")
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
        print(f"HTTP {resp.status}")
        print(f"stages 数量: {len(body.get('stages', []))}")
        for s in body.get("stages", []):
            print(f"  - {s['stage']}")
        print("请求成功 ✓")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP Error {e.code}: {body[:500]}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
