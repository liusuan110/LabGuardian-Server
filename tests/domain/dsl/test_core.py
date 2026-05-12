from __future__ import annotations

import pytest

from app.domain.dsl import CapacitorCeramic, Circuit, Resistor, Transistor


def test_compiles_skidl_style_rc_reference() -> None:
    c = Circuit(reference_id="rc_first_order_v1", name="RC")
    vin = c.input("VIN")
    vout = c.output("VOUT")
    gnd = c.ground()
    vlp = c.net("VLP")

    r1 = Resistor("R1")
    r2 = Resistor("R2")
    c1 = CapacitorCeramic("C1")
    c2 = CapacitorCeramic("C2")

    r1[1, 2] += vin, vlp
    c1[1, 2] += vlp, gnd
    c2[1, 2] += vlp, vout
    r2[1, 2] += vout, gnd

    payload = c.to_logical_reference()

    assert payload["format"] == "logical_reference_v1"
    assert payload["source"]["type"] == "dsl_python_v1"
    assert {item["ref_id"] for item in payload["components"]} == {"R1", "R2", "C1", "C2"}
    assert {item["net"] for item in payload["nets"]} == {"VIN", "VOUT", "GND", "VLP"}
    assert next(item for item in payload["nets"] if item["net"] == "VIN")["role"] == "input"


def test_explicit_symmetry_group_compiles() -> None:
    c = Circuit(reference_id="diff")
    vin_p = c.input("VIN+")
    vin_n = c.input("VIN-")
    c.symmetry(vin_p, vin_n)

    payload = c.to_logical_reference()

    assert payload["symmetry_groups"] == [
        {"mode": "swap_allowed", "nets": [["VIN+", "VIN-"]]}
    ]


def test_transistor_pin_names_are_semantic() -> None:
    c = Circuit(reference_id="bjt")
    collector = c.output("OUT")
    base = c.input("IN")
    emitter = c.ground()
    q1 = Transistor("Q1")

    q1["collector", "base", "emitter"] += collector, base, emitter

    pins = c.to_logical_reference()["components"][0]["pins"]
    assert [pin["pin"] for pin in pins] == ["collector", "base", "emitter"]


def test_pin_count_mismatch_is_rejected() -> None:
    c = Circuit(reference_id="bad")
    net = c.net("N1")
    r1 = Resistor("R1")

    with pytest.raises(ValueError, match="selected 2 pins"):
        r1[1, 2] += (net,)
