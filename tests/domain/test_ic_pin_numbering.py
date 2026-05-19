from __future__ import annotations

from app.domain.ic_models import build_dip8_pin_locs


def test_dip8_left_notch_pin1_starts_at_lower_left_and_counts_counterclockwise() -> None:
    locs = build_dip8_pin_locs(pin1=("17", "f"), pin2=("18", "f"))

    assert locs == [
        ("17", "f"),
        ("18", "f"),
        ("19", "f"),
        ("20", "f"),
        ("20", "e"),
        ("19", "e"),
        ("18", "e"),
        ("17", "e"),
    ]
