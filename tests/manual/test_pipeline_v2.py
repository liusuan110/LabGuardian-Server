"""Test the fixed pipeline with pin mapping, rail assignments, and reference."""
import sys
import base64
import importlib

sys.path.insert(0, ".")

# Force reload of orchestrator to pick up new .env settings
import app.pipeline.orchestrator as orch_mod
orch_mod._shared_ctx = None  # reset singleton so it reloads the model

img_path = r"D:\desktop\inter\LabGuardian\dataset\images\demo_self\valid\images\-_20260307102606_394_202_jpg.rf.50f22610ff8c8d5baf0027f46fa1eb98.jpg"
with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

from app.pipeline.orchestrator import run_pipeline

result = run_pipeline(
    images_b64=[img_b64],
    rail_assignments={
        "top_plus": "VCC",
        "top_minus": "GND",
        "bot_plus": "VCC",
        "bot_minus": "GND",
    },
)

stages = result["stages"]
print(f"Total: {result['total_duration_ms']:.0f}ms")

# S1
s1 = stages["detect"]
print(f"\nS1 detect: {len(s1['detections'])} detections ({s1['duration_ms']:.0f}ms)")
for d in s1["detections"]:
    print(
        f"  {d['class_name']:12s} conf={d['confidence']:.3f} "
        f"bbox={d['bbox']} pin1={d['pin1_pixel']} pin2={d['pin2_pixel']}"
    )

# S2
s2 = stages["mapping"]
print(f"\nS2 mapping: {len(s2['components'])} components ({s2['duration_ms']:.0f}ms)")
for c in s2["components"]:
    print(
        f"  {c['class_name']:12s} pin1_logic={c['pin1_logic']} "
        f"pin2_logic={c['pin2_logic']}"
    )

# S3
s3 = stages["topology"]
print(f"\nS3 topology: {s3['component_count']} components ({s3['duration_ms']:.0f}ms)")
desc = s3["circuit_description"]
print(f"  desc:\n{desc}")
netlist = s3["netlist"]
print(f"  netlist components: {len(netlist.get('components', []))}")
for comp in netlist.get("components", []):
    print(f"    {comp['name']} ({comp['type']}): pins={comp['pins']}")

# S4
s4 = stages["validate"]
print(f"\nS4 validate: risk={s4['risk_level']} ({s4['duration_ms']:.0f}ms)")
print(f"  diagnosis: {s4['diagnosis']}")
