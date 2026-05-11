"""
Reference v4 roundtrip smoke test.

验证最小 `netlist_v2` 参考文件可以被加载，并与对应的 S2 风格输入匹配。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.topology_input import build_analyzer_from_components
from app.domain.validator import CircuitValidator


def main() -> int:
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures" / "netlist_v2"
    reference_path = fixture_dir / "reference_simple_v4.json"
    mapped_path = fixture_dir / "mapped_components_simple.json"

    with open(mapped_path, "r", encoding="utf-8") as f:
        mapped_components = json.load(f)

    analyzer, normalized_components = build_analyzer_from_components(mapped_components)
    validator = CircuitValidator()
    validator.load_reference(str(reference_path))
    result = validator.compare(analyzer)

    print(f"normalized_components={len(normalized_components)}")
    print(f"is_match={result['is_match']}")
    print(f"similarity={result['similarity']:.2f}")
    print(f"pin_mismatches={result.get('pin_mismatches', [])}")
    print(f"hole_mismatches={result.get('hole_mismatches', [])}")

    if not result["is_match"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
