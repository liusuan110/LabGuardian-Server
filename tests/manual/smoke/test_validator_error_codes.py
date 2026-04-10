"""
Validator error code smoke tests.

为关键 error code 提供最小 fixture 回归。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.validator import CircuitValidator
from app.pipeline.topology_input import build_analyzer_from_components


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_analyzer(mapped_path: Path):
    mapped = _load_json(mapped_path)
    analyzer, _normalized = build_analyzer_from_components(mapped)
    return analyzer


def _assert_compare_code(reference_path: Path, mapped_path: Path, expected_code: str):
    analyzer = _build_analyzer(mapped_path)
    validator = CircuitValidator()
    validator.load_reference(str(reference_path))
    result = validator.compare(analyzer)
    items = result["report"]["items"]
    codes = [item["error_code"] for item in items]
    print(f"{mapped_path.name}: {codes}")
    if expected_code not in codes:
        raise SystemExit(1)
    matched_item = next(item for item in items if item["error_code"] == expected_code)
    if not matched_item.get("suggested_action"):
        raise SystemExit(1)
    if not matched_item.get("evidence_refs"):
        raise SystemExit(1)


def _assert_compare_code_without_reference(mapped_path: Path, expected_code: str):
    analyzer = _build_analyzer(mapped_path)
    validator = CircuitValidator()
    result = validator.compare(analyzer)
    items = result["report"]["items"]
    codes = [item["error_code"] for item in items]
    print(f"{mapped_path.name} (no reference): {codes}")
    if expected_code not in codes:
        raise SystemExit(1)
    matched_item = next(item for item in items if item["error_code"] == expected_code)
    if not matched_item.get("suggested_action"):
        raise SystemExit(1)
    if not matched_item.get("evidence_refs"):
        raise SystemExit(1)


def _assert_diagnose_code(mapped_path: Path, expected_code: str):
    analyzer = _build_analyzer(mapped_path)
    items = CircuitValidator.diagnose_items(analyzer)
    codes = [item["error_code"] for item in items]
    print(f"{mapped_path.name}: {codes}")
    if expected_code not in codes:
        raise SystemExit(1)
    matched_item = next(item for item in items if item["error_code"] == expected_code)
    if not matched_item.get("suggested_action"):
        raise SystemExit(1)
    if not matched_item.get("evidence_refs"):
        raise SystemExit(1)


def main() -> int:
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures" / "validator_error_codes"

    _assert_compare_code_without_reference(
        fixture_dir / "mapped_hole_mismatch.json",
        "REFERENCE_NOT_SET",
    )
    _assert_compare_code(
        fixture_dir / "reference_resistor_v4.json",
        fixture_dir / "mapped_empty.json",
        "COMPONENT_MISSING",
    )
    _assert_compare_code(
        fixture_dir / "reference_resistor_v4.json",
        fixture_dir / "mapped_empty.json",
        "TOPOLOGY_VALID_SUBSET",
    )
    _assert_compare_code(
        fixture_dir / "reference_resistor_v4.json",
        fixture_dir / "mapped_extra_resistor.json",
        "COMPONENT_EXTRA",
    )
    _assert_compare_code(
        fixture_dir / "reference_resistor_v4.json",
        fixture_dir / "mapped_wrong_type.json",
        "COMPONENT_INSTANCE_MISSING",
    )
    _assert_compare_code(
        fixture_dir / "reference_resistor_v4.json",
        fixture_dir / "mapped_resistor_onepin.json",
        "COMPONENT_SYMMETRY_GROUP_INCOMPLETE",
    )
    _assert_compare_code(
        fixture_dir / "reference_resistor_v4.json",
        fixture_dir / "mapped_hole_mismatch.json",
        "HOLE_MISMATCH",
    )
    _assert_compare_code(
        fixture_dir / "reference_resistor_v4.json",
        fixture_dir / "mapped_node_mismatch.json",
        "NODE_MISMATCH",
    )
    _assert_compare_code(
        fixture_dir / "reference_led_forward_v4.json",
        fixture_dir / "mapped_led_reversed.json",
        "POLARITY_REVERSED",
    )
    _assert_compare_code(
        fixture_dir / "reference_led_forward_v4.json",
        fixture_dir / "mapped_led_unknown_polarity.json",
        "POLARITY_UNKNOWN",
    )
    _assert_compare_code(
        fixture_dir / "reference_led_forward_v4.json",
        fixture_dir / "mapped_led_pin_missing.json",
        "PIN_MISSING",
    )
    _assert_compare_code(
        fixture_dir / "reference_led_forward_v4.json",
        fixture_dir / "mapped_led_pin_extra.json",
        "PIN_EXTRA",
    )

    _assert_diagnose_code(
        fixture_dir / "mapped_floating_pin.json",
        "FLOATING_PIN",
    )
    _assert_diagnose_code(
        fixture_dir / "mapped_component_shorted.json",
        "COMPONENT_SHORTED_SAME_NET",
    )
    _assert_diagnose_code(
        fixture_dir / "mapped_led_missing_resistor.json",
        "LED_SERIES_RESISTOR_MISSING",
    )
    _assert_diagnose_code(
        fixture_dir / "mapped_disconnected_subgraphs.json",
        "MULTIPLE_DISCONNECTED_SUBGRAPHS",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
