"""Schema sanity tests — guarantee enums / dim constants stay aligned with
``app.domain.circuit`` and the plan §三 contract.
"""

from __future__ import annotations

import inspect

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
    NET_FEAT_DIM,
    NET_FEAT_LAYOUT,
    POLARITY_CLASS_OF,
    PORT_FEAT_DIM,
    PORT_FEAT_LAYOUT,
    PORT_NET_EDGE_FEAT_DIM,
    PORT_NET_EDGE_FEAT_LAYOUT,
    ComponentType,
    NetRole,
    PolarityClass,
    PortType,
    SourceType,
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
