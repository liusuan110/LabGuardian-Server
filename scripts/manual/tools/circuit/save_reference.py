"""Generate a v4 reference circuit JSON from a single image pipeline run."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import app.pipeline.orchestrator as orch_mod

orch_mod._shared_ctx = None

from app.domain.validator import CircuitValidator
from app.pipeline.topology_input import build_analyzer_from_components
from app.pipeline.orchestrator import run_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", help="Path to the reference circuit image.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reference_S01.json"),
        help="Output reference JSON path.",
    )
    parser.add_argument(
        "--rails",
        default='{"top_plus":"VCC","top_minus":"GND","bot_plus":"VCC","bot_minus":"GND"}',
        help="JSON string for rail assignments.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    image_path = Path(args.image_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    rail_assignments = json.loads(args.rails)

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    result = run_pipeline(
        images_b64=[img_b64],
        rail_assignments=rail_assignments,
    )

    s2 = result["stages"]["mapping"]
    analyzer, normalized_components = build_analyzer_from_components(s2["components"])
    for track_id, label in rail_assignments.items():
        analyzer.set_rail_assignment(track_id, label)

    validator = CircuitValidator()
    validator.set_reference(analyzer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validator.save_reference(str(output_path))

    print(f"Reference saved to {output_path}")
    print(f"Normalized components: {len(normalized_components)}")
    print(f"Legacy components: {len(analyzer.components)}")
    print(f"V2 nets: {len(analyzer.export_netlist_v2()['nets'])}")

    validator2 = CircuitValidator()
    validator2.load_reference(str(output_path))
    result2 = validator2.compare(analyzer)
    print(
        "Self-compare:"
        f" match={result2['is_match']},"
        f" similarity={result2['similarity']:.2f},"
        f" pin_mismatches={len(result2.get('pin_mismatches', []))},"
        f" hole_mismatches={len(result2.get('hole_mismatches', []))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
