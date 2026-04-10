"""
S01 dual sample smoke test.

验证:
1) 正确样例可匹配 reference_S01.json
2) 故障样例会产生 node/hole 相关诊断
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.validator import CircuitValidator
from app.pipeline.topology_input import build_analyzer_from_components


FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "netlist_v2"
REFERENCE_PATH = PROJECT_ROOT / "reference_S01.json"


def _load_components(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_case(path: Path) -> dict:
    components = _load_components(path)
    analyzer, normalized_components = build_analyzer_from_components(components)
    validator = CircuitValidator()
    validator.load_reference(str(REFERENCE_PATH))
    result = validator.compare(analyzer)
    result["normalized_components"] = len(normalized_components)
    return result


def main() -> int:
    correct_path = FIXTURE_DIR / "mapped_components_s01_correct.json"
    faulty_path = FIXTURE_DIR / "mapped_components_s01_faulty.json"

    correct = _run_case(correct_path)
    print("[correct] is_match=", correct.get("is_match"))
    print("[correct] similarity=", f"{correct.get('similarity', 0.0):.2f}")
    print("[correct] normalized_components=", correct.get("normalized_components"))

    if not correct.get("is_match"):
        raise SystemExit("correct sample should match reference_S01.json")

    faulty = _run_case(faulty_path)
    report = faulty.get("report", {})
    node_errors = report.get("node_errors", [])
    hole_errors = report.get("hole_errors", [])

    print("[faulty] is_match=", faulty.get("is_match"))
    print("[faulty] similarity=", f"{faulty.get('similarity', 0.0):.2f}")
    print("[faulty] normalized_components=", faulty.get("normalized_components"))
    print("[faulty] node_errors=", len(node_errors))
    print("[faulty] hole_errors=", len(hole_errors))

    if faulty.get("is_match"):
        raise SystemExit("faulty sample should not match reference_S01.json")

    if not node_errors:
        raise SystemExit("faulty sample should produce node_errors")

    if not hole_errors:
        raise SystemExit("faulty sample should produce hole_errors")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
