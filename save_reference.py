"""Save the current pipeline result as the reference circuit for station S01."""
import sys
import base64
import json

sys.path.insert(0, ".")

import app.pipeline.orchestrator as orch_mod
orch_mod._shared_ctx = None

img_path = r"D:\desktop\inter\LabGuardian\dataset\images\demo_self\valid\images\-_20260307102606_394_202_jpg.rf.50f22610ff8c8d5baf0027f46fa1eb98.jpg"
with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

from app.pipeline.orchestrator import run_pipeline
from app.domain.circuit import CircuitAnalyzer, CircuitComponent, Polarity
from app.domain.validator import CircuitValidator

# Run pipeline
result = run_pipeline(
    images_b64=[img_b64],
    rail_assignments={
        "top_plus": "VCC",
        "top_minus": "GND",
        "bot_plus": "VCC",
        "bot_minus": "GND",
    },
)

# Build reference analyzer from S2 components
s2 = result["stages"]["mapping"]
analyzer = CircuitAnalyzer()
for comp in s2["components"]:
    pin1 = tuple(comp["pin1_logic"]) if comp.get("pin1_logic") else None
    pin2 = tuple(comp["pin2_logic"]) if comp.get("pin2_logic") else None
    if pin1 is None or pin2 is None:
        continue
    polarity = Polarity.UNKNOWN if comp["class_name"].lower() == "led" else Polarity.NONE
    cc = CircuitComponent(
        name="",
        type=comp["class_name"],
        pin1_loc=pin1,
        pin2_loc=pin2,
        polarity=polarity,
        confidence=comp.get("confidence", 1.0),
    )
    analyzer.add_component(cc)

# Save as reference
validator = CircuitValidator()
validator.set_reference(analyzer)
ref_path = r"D:\desktop\LabGuardian-Server-main\LabGuardian-Server-main\reference_S01.json"
validator.save_reference(ref_path)
print(f"Reference saved to {ref_path}")
print(f"Components: {len(analyzer.components)}")
for c in analyzer.components:
    print(f"  {c}")

# Verify by loading and comparing
validator2 = CircuitValidator()
validator2.load_reference(ref_path)
result2 = validator2.compare(analyzer)
print(f"\nSelf-compare: match={result2['is_match']}, similarity={result2['similarity']:.2f}")
