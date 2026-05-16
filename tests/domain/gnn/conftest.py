"""Shared fixtures for ``app.domain.gnn`` P0 unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures"
REFERENCES_DIR = FIXTURE_ROOT / "references"
NETLIST_V2_DIR = FIXTURE_ROOT / "netlist_v2"


@pytest.fixture
def rc_reference_payload() -> dict:
    """Two-component RC circuit (R1 + C1, three nets)."""

    return json.loads((REFERENCES_DIR / "test_rc_v1.json").read_text())


@pytest.fixture
def led_reference_payload() -> dict:
    """All-signal fixture containing R1 + LED1, useful for polarity tests."""

    return json.loads((REFERENCES_DIR / "test_all_signal_v1.json").read_text())


@pytest.fixture
def all_reference_payloads() -> dict[str, dict]:
    """Every JSON under tests/fixtures/references/ keyed by stem."""

    return {
        path.stem: json.loads(path.read_text())
        for path in sorted(REFERENCES_DIR.glob("*.json"))
    }


@pytest.fixture
def simple_netlist_v2() -> dict:
    """A minimal NetlistV2 dict (extracted from reference_simple_v4.json)."""

    bundle = json.loads((NETLIST_V2_DIR / "reference_simple_v4.json").read_text())
    return bundle["netlist_v2"]
