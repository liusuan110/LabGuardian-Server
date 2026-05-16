"""P0.5 · IC + Potentiometer port semantics tests.

Covers:
- New PortType members (INVERTING_INPUT / NON_INVERTING_INPUT / OUTPUT /
  V_PLUS / V_MINUS / OFFSET_NULL / NC)
- IC_PIN_MAPS for UA741 — alignment with ic_models.UA741_PIN_ROLES
- normalize_port_type's subtype-aware + alias-aware paths
- Potentiometer wiper is polarity-sensitive (terminal_a/b are not)
- Op-amp ports carry polarity_sensitive=True
- UA741 buffer fixture roundtrips with parallel-pin (pin2↔pin6 on VOUT)
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from app.domain.gnn import (
    PORT_FEAT_DIM,
    PortType,
    build_from_logical_reference,
    build_hetero_circuit_graph,
)
from app.domain.gnn.graph_schema import (
    IC_PIN_MAPS,
    POLARITY_SENSITIVE_PORT_TYPES,
    POWER_PORT_TYPES,
    normalize_port_type,
)

FIXTURE_OPAMP = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_opamp_buffer_v1.json"
)


# ---------------------------------------------------------------------------
# PortType / dim coherence after P0.5 expansion
# ---------------------------------------------------------------------------


def test_port_type_count_is_23() -> None:
    assert len(list(PortType)) == 23


def test_port_type_includes_opamp_roles() -> None:
    for name in (
        "INVERTING_INPUT",
        "NON_INVERTING_INPUT",
        "OUTPUT",
        "OFFSET_NULL",
        "NC",
        "V_PLUS",
        "V_MINUS",
    ):
        assert hasattr(PortType, name), f"PortType missing {name}"


def test_port_feat_dim_recomputed_to_50() -> None:
    # P0.5: 44 = 23 (port_type) + 16 (parent_ctype) + 5 flags
    # P0.6: 50 = + 3 connection_policy_one_hot + 1 has_pin_number
    #            + 1 pin_number_log + 1 symmetry_class_size_inverse
    assert PORT_FEAT_DIM == 50


# ---------------------------------------------------------------------------
# UA741 pin map alignment with ic_models
# ---------------------------------------------------------------------------


def test_ic_pin_maps_has_ua741() -> None:
    assert "UA741" in IC_PIN_MAPS


def test_ic_pin_maps_ua741_covers_all_eight_pins() -> None:
    pinmap = IC_PIN_MAPS["UA741"]
    # Both "N" and "pinN" forms should resolve for each of pin 1..8.
    for n in range(1, 9):
        assert str(n) in pinmap, f"missing pin number '{n}'"
        assert f"pin{n}" in pinmap, f"missing pin name 'pin{n}'"
        assert pinmap[str(n)] == pinmap[f"pin{n}"], f"pin{n} inconsistent"


def test_ic_pin_maps_values_are_known_port_types() -> None:
    known = {pt.value for pt in PortType}
    for subtype, pinmap in IC_PIN_MAPS.items():
        for pin, port_type in pinmap.items():
            assert port_type in known, f"{subtype}.{pin} → unknown port_type {port_type}"


def test_ic_pin_maps_ua741_matches_ic_models_roles() -> None:
    from app.domain.ic_models import UA741_PIN_ROLES

    # Quick spot checks (more than just length): pin 2 = inverting, pin 3 =
    # non-inverting, pin 6 = output, pin 7 = V+, pin 4 = V-, pin 8 = NC,
    # pin 1 + 5 = offset_null.
    pinmap = IC_PIN_MAPS["UA741"]
    assert pinmap["2"] == PortType.INVERTING_INPUT.value
    assert pinmap["3"] == PortType.NON_INVERTING_INPUT.value
    assert pinmap["6"] == PortType.OUTPUT.value
    assert pinmap["7"] == PortType.V_PLUS.value
    assert pinmap["4"] == PortType.V_MINUS.value
    assert pinmap["8"] == PortType.NC.value
    assert pinmap["1"] == PortType.OFFSET_NULL.value
    assert pinmap["5"] == PortType.OFFSET_NULL.value
    # ic_models 顺序 = pin1..pin8
    assert UA741_PIN_ROLES[1] == "inverting_input"
    assert UA741_PIN_ROLES[5] == "output"


# ---------------------------------------------------------------------------
# normalize_port_type — IC pin map path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pin_raw", "expected"),
    [
        ("1", PortType.OFFSET_NULL.value),
        ("2", PortType.INVERTING_INPUT.value),
        ("3", PortType.NON_INVERTING_INPUT.value),
        ("4", PortType.V_MINUS.value),
        ("5", PortType.OFFSET_NULL.value),
        ("6", PortType.OUTPUT.value),
        ("7", PortType.V_PLUS.value),
        ("8", PortType.NC.value),
        ("pin3", PortType.NON_INVERTING_INPUT.value),
    ],
)
def test_normalize_port_type_ua741_by_pin(pin_raw: str, expected: str) -> None:
    # pin_role normalization on IC just returns the pin name (lower-cased) —
    # so pass it through verbatim to match the real upstream flow.
    pin_role = pin_raw.lower()
    got = normalize_port_type(
        pin_role,
        "IC",
        part_subtype="UA741",
        pin_raw=pin_raw,
    )
    assert got == expected


def test_normalize_port_type_subtype_case_insensitive() -> None:
    assert (
        normalize_port_type("2", "IC", part_subtype="ua741", pin_raw="2")
        == PortType.INVERTING_INPUT.value
    )


def test_normalize_port_type_unknown_ic_subtype_falls_through() -> None:
    # No map registered → numeric pin yields pin_n_generic via the digit
    # fallback (still nicer than crashing).
    got = normalize_port_type("3", "IC", part_subtype="LM386", pin_raw="3")
    assert got == PortType.PIN_N_GENERIC.value


# ---------------------------------------------------------------------------
# normalize_port_type — op-amp alias path (no subtype known)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pin_role", "expected"),
    [
        ("in-", PortType.INVERTING_INPUT.value),
        ("-in", PortType.INVERTING_INPUT.value),
        ("inv", PortType.INVERTING_INPUT.value),
        ("inverting_input", PortType.INVERTING_INPUT.value),
        ("in+", PortType.NON_INVERTING_INPUT.value),
        ("non_inverting", PortType.NON_INVERTING_INPUT.value),
        ("out", PortType.OUTPUT.value),
        ("v+", PortType.V_PLUS.value),
        ("v-", PortType.V_MINUS.value),
        ("vee", PortType.V_MINUS.value),
    ],
)
def test_normalize_port_type_opamp_aliases(pin_role: str, expected: str) -> None:
    assert normalize_port_type(pin_role, "OpAmp") == expected


def test_normalize_port_type_opamp_alias_ignored_for_non_ic_ctype() -> None:
    # "in-" against a Resistor must NOT shortcut into INVERTING_INPUT.
    got = normalize_port_type("in-", "Resistor")
    assert got == PortType.GENERIC.value


# ---------------------------------------------------------------------------
# Polarity sensitivity widening
# ---------------------------------------------------------------------------


def test_polarity_sensitive_set_includes_wiper_and_opamp_roles() -> None:
    for pt in (
        PortType.WIPER,
        PortType.INVERTING_INPUT,
        PortType.NON_INVERTING_INPUT,
        PortType.OUTPUT,
        PortType.V_PLUS,
        PortType.V_MINUS,
    ):
        assert pt.value in POLARITY_SENSITIVE_PORT_TYPES, pt


def test_terminal_a_b_remain_swappable() -> None:
    # Pot terminal_a / terminal_b are swappable in a linear pot — they must
    # NOT be polarity-sensitive (only the wiper is).
    assert PortType.TERMINAL_A.value not in POLARITY_SENSITIVE_PORT_TYPES
    assert PortType.TERMINAL_B.value not in POLARITY_SENSITIVE_PORT_TYPES


def test_power_port_set_includes_opamp_supplies() -> None:
    assert PortType.V_PLUS.value in POWER_PORT_TYPES
    assert PortType.V_MINUS.value in POWER_PORT_TYPES


# ---------------------------------------------------------------------------
# UA741 buffer fixture end-to-end
# ---------------------------------------------------------------------------


def test_opamp_buffer_fixture_resolves_all_connected_pins() -> None:
    payload = json.loads(FIXTURE_OPAMP.read_text())
    hcg = build_from_logical_reference(payload)
    summary = hcg.summary()
    # P0.6: all 8 package pins materialized (5 connected + 3 floating); only
    # 5 (port, net) edges because pins 1/5/8 are NC in DSL.
    assert summary == {
        "n_components": 1,
        "n_ports": 8,
        "n_nets": 4,
        "n_edges": 5,
    }, summary

    by_key = {p.port_key: p for p in hcg.ports.values()}
    assert by_key["2"].port_type == PortType.INVERTING_INPUT.value
    assert by_key["3"].port_type == PortType.NON_INVERTING_INPUT.value
    assert by_key["4"].port_type == PortType.V_MINUS.value
    assert by_key["6"].port_type == PortType.OUTPUT.value
    assert by_key["7"].port_type == PortType.V_PLUS.value

    # All five connected pins must be polarity_sensitive
    for key in ("2", "3", "4", "6", "7"):
        assert by_key[key].polarity_sensitive, by_key[key]
        assert by_key[key].is_floating is False, key

    # V+/V- pins flagged as power_port
    assert by_key["7"].is_power_port is True
    assert by_key["4"].is_power_port is True

    # P0.6: pins 1 / 5 / 8 are now materialized as floating
    for key in ("1", "5", "8"):
        assert by_key[key].is_floating is True, key


def test_opamp_buffer_preserves_parallel_feedback_pins() -> None:
    """pin 2 and pin 6 both wire to VOUT (unity-gain feedback). Earlier the
    raw nx.Graph collapsed them into a single edge, losing pin 2 entirely.
    """

    payload = json.loads(FIXTURE_OPAMP.read_text())
    hcg = build_from_logical_reference(payload)
    vout_edges = [e for e in hcg.edges if e.dst_net_id.endswith(":VOUT")]
    src_ports = {e.src_port_id for e in vout_edges}
    assert {"ref_port:U1.2", "ref_port:U1.6"}.issubset(src_ports)


# ---------------------------------------------------------------------------
# Potentiometer — wiper polarity-sensitive, terminals are not
# ---------------------------------------------------------------------------


def _build_pot_payload() -> dict:
    """A 10k pot with wiper on VOUT, terminals on VIN / GND."""

    return {
        "format": "logical_reference_v1",
        "reference_id": "test_pot_v1",
        "name": "Potentiometer divider",
        "components": [
            {
                "ref_id": "RV1",
                "type": "Potentiometer",
                "pins": [
                    {"pin": "wiper", "net": "VOUT"},
                    {"pin": "terminal_a", "net": "VIN"},
                    {"pin": "terminal_b", "net": "GND"},
                ],
            }
        ],
        "nets": [
            {"net": "VIN", "role": "input"},
            {"net": "VOUT", "role": "output"},
            {"net": "GND", "role": "ground"},
        ],
    }


def test_potentiometer_wiper_is_polarity_sensitive() -> None:
    hcg = build_from_logical_reference(_build_pot_payload())
    by_key = {p.port_key: p for p in hcg.ports.values()}
    assert by_key["wiper"].port_type == PortType.WIPER.value
    assert by_key["wiper"].polarity_sensitive is True


def test_potentiometer_terminals_are_not_polarity_sensitive() -> None:
    hcg = build_from_logical_reference(_build_pot_payload())
    by_key = {p.port_key: p for p in hcg.ports.values()}
    assert by_key["terminal_a"].port_type == PortType.TERMINAL_A.value
    assert by_key["terminal_b"].port_type == PortType.TERMINAL_B.value
    assert by_key["terminal_a"].polarity_sensitive is False
    assert by_key["terminal_b"].polarity_sensitive is False


# ---------------------------------------------------------------------------
# Hand-built nx graph with op-amp alias — the subtype kwarg path
# ---------------------------------------------------------------------------


def test_build_hetero_with_subtype_kwarg_for_handcrafted_graph() -> None:
    g = nx.Graph()
    g.add_node("ref_comp:U1", kind="comp", ctype="IC", source_id="U1")
    g.add_node("ref_net:N_IN", kind="net", role="input", source_id="N_IN")
    g.add_node("ref_net:N_OUT", kind="net", role="output", source_id="N_OUT")
    # Hand-built nx graph — single edge per pin is OK (no parallel pins here).
    g.add_edge(
        "ref_comp:U1",
        "ref_net:N_IN",
        pin="3",
        pin_role="3",
        comp_type="IC",
    )
    g.add_edge(
        "ref_comp:U1",
        "ref_net:N_OUT",
        pin="6",
        pin_role="6",
        comp_type="IC",
    )

    hcg = build_hetero_circuit_graph(
        g,
        side="ref",
        subtype_by_source_id={"U1": "UA741"},
    )
    by_key = {p.port_key: p for p in hcg.ports.values()}
    assert by_key["3"].port_type == PortType.NON_INVERTING_INPUT.value
    assert by_key["6"].port_type == PortType.OUTPUT.value
