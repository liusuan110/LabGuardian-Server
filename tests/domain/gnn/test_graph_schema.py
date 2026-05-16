"""Schema sanity tests — guarantee enums / dim constants stay aligned with
``app.domain.circuit`` and the plan §三 contract.
"""

from __future__ import annotations

import inspect

import pytest

from app.domain.circuit import (
    NON_POLAR_TYPES,
    POLARIZED_TYPES,
    THREE_PIN_TYPES,
    PinRole,
    norm_component_type,
)
from app.domain.gnn.graph_schema import (
    COMPONENT_FEAT_DIM,
    COMPONENT_FEAT_LAYOUT,
    DRNL_LABEL_DIM,
    IC_PIN_POLICIES,
    IC_PIN_SYMMETRY,
    NET_FEAT_DIM,
    NET_FEAT_LAYOUT,
    PACKAGE_PIN_SPECS,
    POLARITY_CLASS_OF,
    PORT_FEAT_DIM,
    PORT_FEAT_LAYOUT,
    PORT_NET_EDGE_FEAT_DIM,
    PORT_NET_EDGE_FEAT_LAYOUT,
    ComponentType,
    ConnectionPolicy,
    NetRole,
    PinSpec,
    PolarityClass,
    PortType,
    SourceType,
    get_expected_pin_specs,
    make_ic_pin_specs,
    normalize_port_type,
)

# ---------------------------------------------------------------------------
# Enum coverage
# ---------------------------------------------------------------------------


def _norm_component_type_outputs() -> set[str]:
    """Inspect ``norm_component_type`` source to enumerate every literal it may
    return (the function is a series of conditionals returning literal strs).
    """

    src = inspect.getsource(norm_component_type)
    # Crude but sufficient: pull every quoted literal that follows ``return``.
    import re

    return set(re.findall(r'return\s+"([^"]+)"', src))


def test_component_type_enum_covers_normalize_outputs() -> None:
    expected_subset = _norm_component_type_outputs()
    enum_values = {c.value for c in ComponentType}
    missing = expected_subset - enum_values
    assert not missing, (
        f"ComponentType is missing values produced by norm_component_type: {missing}"
    )


def test_port_type_includes_all_pin_roles() -> None:
    pin_role_values = {role.value for role in PinRole}
    port_type_values = {pt.value for pt in PortType}
    missing = pin_role_values - port_type_values
    assert not missing, f"PortType missing PinRole entries: {missing}"


def test_net_role_includes_normalize_net_role_outputs() -> None:
    # normalize_net_role returns one of these literals; PORT/NET schema must
    # support each.
    expected = {"input", "output", "power", "ground", "signal"}
    enum_values = {nr.value for nr in NetRole}
    assert expected.issubset(enum_values), enum_values


def test_polarity_class_table_covers_all_component_types() -> None:
    enum_values = {c.value for c in ComponentType}
    table_values = set(POLARITY_CLASS_OF.keys())
    assert enum_values == table_values, (
        f"POLARITY_CLASS_OF must cover every ComponentType. "
        f"missing={enum_values - table_values}, extra={table_values - enum_values}"
    )


# ---------------------------------------------------------------------------
# Dimension consistency
# ---------------------------------------------------------------------------


def test_component_layout_width_sums_to_dim() -> None:
    assert sum(w for _, w in COMPONENT_FEAT_LAYOUT) == COMPONENT_FEAT_DIM


def test_port_layout_width_sums_to_dim() -> None:
    assert sum(w for _, w in PORT_FEAT_LAYOUT) == PORT_FEAT_DIM


def test_net_layout_width_sums_to_dim() -> None:
    assert sum(w for _, w in NET_FEAT_LAYOUT) == NET_FEAT_DIM


def test_port_net_edge_layout_width_sums_to_dim() -> None:
    assert sum(w for _, w in PORT_NET_EDGE_FEAT_LAYOUT) == PORT_NET_EDGE_FEAT_DIM


def test_drnl_dim_matches_plan() -> None:
    # 0..15 (16 buckets) + 1 overflow → 17 (plan §三.6)
    assert DRNL_LABEL_DIM == 17


# ---------------------------------------------------------------------------
# Polarity table semantics (mirrors circuit.py)
# ---------------------------------------------------------------------------


def test_polarity_two_polar_matches_polarized_types() -> None:
    two_polar = {
        v for v, pc in POLARITY_CLASS_OF.items() if pc == PolarityClass.TWO_POLAR
    }
    assert two_polar == set(POLARIZED_TYPES)


def test_polarity_multi_asymmetric_includes_three_pin_types() -> None:
    multi = {
        v
        for v, pc in POLARITY_CLASS_OF.items()
        if pc == PolarityClass.MULTI_ASYMMETRIC
    }
    assert set(THREE_PIN_TYPES).issubset(multi), multi
    # IC + OpAmp 也必须归入 multi
    assert ComponentType.IC.value in multi
    assert ComponentType.OPAMP.value in multi


def test_resistor_and_wire_are_non_polar() -> None:
    for v in ("Resistor", "Wire", "Capacitor", "CapacitorCeramic"):
        assert v in NON_POLAR_TYPES  # sanity
        assert POLARITY_CLASS_OF[v] == PolarityClass.NONE


# ---------------------------------------------------------------------------
# normalize_port_type
# ---------------------------------------------------------------------------


def test_normalize_port_type_known_pin_roles() -> None:
    assert normalize_port_type("anode") == PortType.ANODE.value
    assert normalize_port_type("cathode") == PortType.CATHODE.value
    assert normalize_port_type("base") == PortType.BASE.value
    assert normalize_port_type("collector") == PortType.COLLECTOR.value
    assert normalize_port_type("emitter") == PortType.EMITTER.value
    assert normalize_port_type("pin1") == PortType.PIN1.value
    assert normalize_port_type("pin2") == PortType.PIN2.value


def test_normalize_port_type_numeric_pin_falls_back_to_pin_n_generic() -> None:
    assert normalize_port_type("3") == PortType.PIN_N_GENERIC.value
    assert normalize_port_type("12") == PortType.PIN_N_GENERIC.value


def test_normalize_port_type_unknown_falls_back_to_generic() -> None:
    assert normalize_port_type("") == PortType.GENERIC.value
    assert normalize_port_type(None) == PortType.GENERIC.value
    assert normalize_port_type("weird_pin_name") == PortType.GENERIC.value


# ---------------------------------------------------------------------------
# SourceType is a closed set
# ---------------------------------------------------------------------------


def test_source_type_values() -> None:
    assert {s.value for s in SourceType} == {"dsl", "vision", "inferred"}


# ---------------------------------------------------------------------------
# ConnectionPolicy + PACKAGE_PIN_SPECS (P0.6)
# ---------------------------------------------------------------------------


def test_connection_policy_values() -> None:
    assert {p.value for p in ConnectionPolicy} == {"required", "optional", "forbidden"}


def test_package_pin_specs_covers_core_components() -> None:
    # The 9 non-IC ctypes that have explicit specs in P0.6.
    required = {
        ComponentType.RESISTOR.value,
        ComponentType.CAPACITOR.value,
        ComponentType.CAPACITOR_CERAMIC.value,
        ComponentType.CAPACITOR_ELECTROLYTIC.value,
        ComponentType.WIRE.value,
        ComponentType.LED.value,
        ComponentType.DIODE.value,
        ComponentType.TRANSISTOR.value,
        ComponentType.POTENTIOMETER.value,
    }
    assert required.issubset(set(PACKAGE_PIN_SPECS.keys())), (
        set(PACKAGE_PIN_SPECS.keys())
    )


def test_package_pin_specs_pin_keys_are_unique_per_component() -> None:
    for ctype, specs in PACKAGE_PIN_SPECS.items():
        keys = [s.pin_key for s in specs]
        assert len(keys) == len(set(keys)), f"{ctype}: duplicate pin_key in spec"


def test_package_pin_specs_all_port_types_are_known() -> None:
    known = {pt.value for pt in PortType}
    for ctype, specs in PACKAGE_PIN_SPECS.items():
        for s in specs:
            assert s.port_type in known, f"{ctype}.{s.pin_key}: {s.port_type}"


def test_package_pin_specs_all_policies_valid() -> None:
    valid = {p.value for p in ConnectionPolicy}
    for ctype, specs in PACKAGE_PIN_SPECS.items():
        for s in specs:
            assert s.connection_policy in valid, (
                f"{ctype}.{s.pin_key}: {s.connection_policy}"
            )


@pytest.fixture(scope="module")
def resistor_spec() -> list[PinSpec]:
    return PACKAGE_PIN_SPECS[ComponentType.RESISTOR.value]


def test_resistor_two_pins_share_symmetry_class(resistor_spec: list[PinSpec]) -> None:
    assert len(resistor_spec) == 2
    assert resistor_spec[0].symmetry_class == resistor_spec[1].symmetry_class


def test_led_anode_cathode_distinct_symmetry_classes() -> None:
    specs = PACKAGE_PIN_SPECS[ComponentType.LED.value]
    assert specs[0].symmetry_class != specs[1].symmetry_class


def test_diode_matches_led_symmetry_topology() -> None:
    led = PACKAGE_PIN_SPECS[ComponentType.LED.value]
    diode = PACKAGE_PIN_SPECS[ComponentType.DIODE.value]
    assert [s.pin_key for s in led] == [s.pin_key for s in diode]
    assert [s.port_type for s in led] == [s.port_type for s in diode]


def test_potentiometer_terminals_share_class_wiper_alone() -> None:
    specs = PACKAGE_PIN_SPECS[ComponentType.POTENTIOMETER.value]
    by_key = {s.pin_key: s for s in specs}
    assert by_key["terminal_a"].symmetry_class == by_key["terminal_b"].symmetry_class
    assert by_key["wiper"].symmetry_class != by_key["terminal_a"].symmetry_class


def test_transistor_three_distinct_classes() -> None:
    specs = PACKAGE_PIN_SPECS[ComponentType.TRANSISTOR.value]
    classes = {s.symmetry_class for s in specs}
    assert len(classes) == 3


def test_capacitor_electrolytic_distinguishes_polarity() -> None:
    specs = PACKAGE_PIN_SPECS[ComponentType.CAPACITOR_ELECTROLYTIC.value]
    by_key = {s.pin_key: s for s in specs}
    assert by_key["positive"].port_type == PortType.POSITIVE.value
    assert by_key["negative"].port_type == PortType.NEGATIVE.value
    assert by_key["positive"].symmetry_class != by_key["negative"].symmetry_class


# ---------------------------------------------------------------------------
# UA741 IC_PIN_POLICIES / IC_PIN_SYMMETRY (P0.6)
# ---------------------------------------------------------------------------


def test_ic_pin_policies_has_ua741() -> None:
    assert "UA741" in IC_PIN_POLICIES


def test_ua741_pin_8_is_forbidden() -> None:
    assert IC_PIN_POLICIES["UA741"]["8"] == ConnectionPolicy.FORBIDDEN


def test_ua741_offset_null_pins_are_optional() -> None:
    assert IC_PIN_POLICIES["UA741"]["1"] == ConnectionPolicy.OPTIONAL
    assert IC_PIN_POLICIES["UA741"]["5"] == ConnectionPolicy.OPTIONAL


def test_ua741_signal_pins_inherit_required_default() -> None:
    # Pins 2/3/4/6/7 not in the policy overlay → default REQUIRED.
    overlay = IC_PIN_POLICIES["UA741"]
    for pk in ("2", "3", "4", "6", "7"):
        assert pk not in overlay, pk


def test_ic_pin_symmetry_ua741_groups_offset_null_pair() -> None:
    groups = IC_PIN_SYMMETRY["UA741"]
    assert any(set(g) == {"1", "5"} for g in groups), groups


# ---------------------------------------------------------------------------
# make_ic_pin_specs + get_expected_pin_specs (composed table)
# ---------------------------------------------------------------------------


def test_make_ic_pin_specs_ua741_full_set() -> None:
    specs = make_ic_pin_specs("UA741")
    assert specs is not None
    assert [s.pin_key for s in specs] == ["1", "2", "3", "4", "5", "6", "7", "8"]
    assert [s.pin_number for s in specs] == list(range(1, 9))


def test_make_ic_pin_specs_subtype_case_insensitive() -> None:
    a = make_ic_pin_specs("ua741")
    b = make_ic_pin_specs("UA741")
    assert a == b


def test_make_ic_pin_specs_offset_null_pair_shares_symmetry_class() -> None:
    specs = make_ic_pin_specs("UA741")
    by_key = {s.pin_key: s for s in specs}
    assert by_key["1"].symmetry_class == by_key["5"].symmetry_class
    # All other pins each in their own class
    others = {by_key[k].symmetry_class for k in ("2", "3", "4", "6", "7", "8")}
    assert by_key["1"].symmetry_class not in others
    assert len(others) == 6


def test_make_ic_pin_specs_symmetry_class_ids_are_zero_indexed_contiguous() -> None:
    specs = make_ic_pin_specs("UA741")
    classes = sorted({s.symmetry_class for s in specs})
    # 7 distinct classes (offset_null pair + 6 unique pins), 0-indexed.
    assert classes == list(range(len(classes)))


def test_make_ic_pin_specs_returns_none_for_unknown_subtype() -> None:
    assert make_ic_pin_specs(None) is None
    assert make_ic_pin_specs("") is None
    assert make_ic_pin_specs("LM386") is None  # not in IC_PIN_MAPS yet


def test_get_expected_pin_specs_routes_ic_via_subtype() -> None:
    via_ic = get_expected_pin_specs("IC", "UA741")
    via_opamp = get_expected_pin_specs("OpAmp", "UA741")
    assert via_ic == via_opamp
    assert via_ic is not None
    assert len(via_ic) == 8


def test_get_expected_pin_specs_routes_non_ic_via_table() -> None:
    direct = PACKAGE_PIN_SPECS["Resistor"]
    via = get_expected_pin_specs("Resistor")
    assert via == direct


def test_get_expected_pin_specs_returns_none_for_unknown_components() -> None:
    assert get_expected_pin_specs("Sensor") is None
    assert get_expected_pin_specs("UNKNOWN") is None
    assert get_expected_pin_specs("IC") is None  # no subtype → no spec
