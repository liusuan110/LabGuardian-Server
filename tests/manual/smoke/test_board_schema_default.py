"""
Smoke test for the default breadboard schema expansion.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.board_schema import BoardSchema


def main() -> int:
    schema = BoardSchema.default_breadboard()
    checks = {
        "A1": schema.resolve_hole_to_node("A1"),
        "E63": schema.resolve_hole_to_node("E63"),
        "F1": schema.resolve_hole_to_node("F1"),
        "J63": schema.resolve_hole_to_node("J63"),
        "LP1": schema.resolve_hole_to_node("LP1"),
        "LP32": schema.resolve_hole_to_node("LP32"),
        "RN63": schema.resolve_hole_to_node("RN63"),
        "legacy rail top+": schema.logic_loc_to_hole_id(("19", "rail_top+")),
        "legacy rail bot-": schema.logic_loc_to_hole_id(("32", "rail_bot-")),
    }
    for key, value in checks.items():
        print(f"{key}: {value}")

    expected_hole_count = (5 * 63) + (5 * 63) + (4 * 63) + 2
    actual_hole_count = len(schema.holes)
    print(f"hole_count={actual_hole_count}")
    if checks["LP1"] != "TRACK_LP_SEG1":
        raise SystemExit(1)
    if checks["LP32"] != "TRACK_LP_SEG2":
        raise SystemExit(1)
    if checks["RN63"] != "TRACK_RN_SEG2":
        raise SystemExit(1)
    if actual_hole_count != expected_hole_count:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
