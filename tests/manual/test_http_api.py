"""Test the HTTP API endpoint end-to-end."""
import base64
import requests

img_path = r"D:\desktop\inter\LabGuardian\dataset\images\demo_self\valid\images\-_20260307102606_394_202_jpg.rf.50f22610ff8c8d5baf0027f46fa1eb98.jpg"
with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:8000/api/v1/pipeline/run",
    json={
        "station_id": "S01",
        "images_b64": [img_b64],
        "rail_assignments": {
            "top_plus": "VCC",
            "top_minus": "GND",
            "bot_plus": "VCC",
            "bot_minus": "GND",
        },
    },
    timeout=120,
)

print(f"HTTP {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    # Find validate stage
    for stage in data.get("stages", []):
        name = stage.get("stage")
        sd = stage.get("data", {})
        if name == "detect":
            print(f"S1: {len(sd.get('detections', []))} detections ({stage['duration_ms']:.0f}ms)")
        elif name == "mapping":
            for c in sd.get("components", []):
                print(f"  {c['class_name']:12s} pin1={c.get('pin1_logic')} pin2={c.get('pin2_logic')}")
        elif name == "topology":
            print(f"S3: {sd.get('component_count', 0)} components ({stage['duration_ms']:.0f}ms)")
        elif name == "validate":
            print(f"S4: risk={sd.get('risk_level')} similarity={sd.get('similarity')}")
            print(f"  correct: {sd.get('is_correct')}")
            print(f"  diagnosis: {sd.get('diagnosis')}")
            print(f"  diagnostics: {sd.get('diagnostics')}")
            print(f"  progress: {sd.get('progress')}")
else:
    print(resp.text[:500])
