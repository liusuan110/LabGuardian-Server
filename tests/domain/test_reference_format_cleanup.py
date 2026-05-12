from __future__ import annotations

import json

import pytest

from app.domain.validator import CircuitValidator


def test_circuit_validator_rejects_legacy_ref_v4_file(tmp_path) -> None:
    path = tmp_path / "legacy_ref.json"
    path.write_text(
        json.dumps(
            {
                "meta": {"format": "labguardian_ref_v4"},
                "netlist_v2": {"components": [], "nets": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="不支持旧参考电路格式"):
        CircuitValidator().load_reference(str(path))


def test_circuit_validator_rejects_direct_netlist_v2_payload() -> None:
    payload = {
        "components": [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [],
            }
        ],
        "nets": [],
    }

    with pytest.raises(ValueError, match="不支持旧参考电路格式"):
        CircuitValidator().load_reference_payload(payload)
