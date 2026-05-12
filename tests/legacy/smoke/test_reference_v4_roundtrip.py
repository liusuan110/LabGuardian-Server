"""
Retired reference v4 smoke check.

`labguardian_ref_v4` is intentionally rejected by the current validation path.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.validator import CircuitValidator


def main() -> int:
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures" / "netlist_v2"
    reference_path = fixture_dir / "reference_simple_v4.json"

    validator = CircuitValidator()
    try:
        validator.load_reference(str(reference_path))
    except ValueError as exc:
        print(f"legacy reference v4 rejected: {exc}")
        return 0
    else:
        print("legacy reference v4 unexpectedly loaded")
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
