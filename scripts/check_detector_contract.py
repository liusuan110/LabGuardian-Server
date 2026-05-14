"""从 pipeline 响应里提取 detector_contract，确认 YOLO 加载成功"""
import base64
import json
import sys
import urllib.request

import numpy as np
import cv2

img = np.zeros((100, 100, 3), dtype=np.uint8)
_, buf = cv2.imencode(".jpg", img)
b64 = base64.b64encode(buf).decode()

payload = json.dumps({"station_id": "smoke_test", "images_b64": [b64]}).encode()
req = urllib.request.Request(
    "http://localhost:8000/api/v1/pipeline/run",
    data=payload,
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req, timeout=60) as resp:
    body = json.loads(resp.read())

stages = {s["stage"]: s["data"] for s in body.get("stages", [])}
detect_data = stages.get("detect", {})
contract = detect_data.get("detector_contract", {})
runtime = body.get("runtime_metadata", {})

print("=== detector_contract ===")
print(f"  exists  : {contract.get('exists')}")
print(f"  task    : {contract.get('task')}")
print(f"  loaded  : {contract.get('loaded')}")
print(f"  backend : {detect_data.get('detector_backend')}")
print(f"  names   : {contract.get('names')}")
print()
print("=== runtime_metadata ===")
print(f"  component_model_path : {runtime.get('component_model_path')}")
print(f"  yolo_device          : {runtime.get('yolo_device')}")
print()
print("=== pipeline summary ===")
print(f"  S1 detections : {len(detect_data.get('detections', []))}")
meta = body.get("runtime_metadata", {})
print(f"  total_duration_ms : {body.get('total_duration_ms', 0):.0f} ms")

loaded_ok = contract.get("loaded") is True and contract.get("task") == "detect"
print()
print("YOLO model loaded OK" if loaded_ok else "FAIL: YOLO model NOT loaded")
sys.exit(0 if loaded_ok else 1)
