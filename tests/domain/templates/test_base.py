"""Unit tests for TopologyTemplate dataclass + validate()."""

from __future__ import annotations

import pytest

from app.domain.templates.base import (
    ComponentSlot,
    EdgeSpec,
    NetSlot,
    ParametricInvariant,
    TopologyTemplate,
    TopologyVariant,
)


def _trivial_template(**overrides) -> TopologyTemplate:
    """Minimal valid template — for smoke / negative tests."""
    kwargs = dict(
        template_id="trivial_v1",
        name="Trivial",
        topology_label="trivial",
        reference_id=None,
        required_components=(
            ComponentSlot(role="R1", component_type="Resistor"),
        ),
        required_nets=(
            NetSlot(role="input", canonical_name="VIN"),
            NetSlot(role="ground", canonical_name="GND"),
        ),
        required_edges=(
            EdgeSpec(component_role="R1", pin="pin1", net_role="VIN"),
            EdgeSpec(component_role="R1", pin="pin2", net_role="GND"),
        ),
    )
    kwargs.update(overrides)
    return TopologyTemplate(**kwargs)


class TestTemplateValidate:
    def test_trivial_template_validates(self) -> None:
        tpl = _trivial_template()
        assert tpl.validate() == []

    def test_unknown_component_role_in_edge_fails(self) -> None:
        tpl = _trivial_template(
            required_edges=(
                EdgeSpec(component_role="NONEXISTENT", pin="pin1", net_role="VIN"),
            ),
        )
        errors = tpl.validate()
        assert any("unknown component role" in e for e in errors)

    def test_unknown_net_role_in_edge_fails(self) -> None:
        tpl = _trivial_template(
            required_edges=(
                EdgeSpec(component_role="R1", pin="pin1", net_role="NONEXISTENT_NET"),
            ),
        )
        errors = tpl.validate()
        assert any("unknown net" in e for e in errors)

    def test_canonical_name_resolves_in_edge(self) -> None:
        # Two signal nets — must reference by canonical_name, not role.
        tpl = _trivial_template(
            required_nets=(
                NetSlot(role="signal", canonical_name="A"),
                NetSlot(role="signal", canonical_name="B"),
            ),
            required_edges=(
                EdgeSpec(component_role="R1", pin="pin1", net_role="A"),
                EdgeSpec(component_role="R1", pin="pin2", net_role="B"),
            ),
        )
        assert tpl.validate() == []

    def test_ambiguous_role_in_edge_fails(self) -> None:
        # Two signal nets but edge uses role "signal" instead of canonical.
        tpl = _trivial_template(
            required_nets=(
                NetSlot(role="signal", canonical_name="A"),
                NetSlot(role="signal", canonical_name="B"),
            ),
            required_edges=(
                EdgeSpec(component_role="R1", pin="pin1", net_role="signal"),
            ),
        )
        errors = tpl.validate()
        assert any("ambiguous net role" in e for e in errors)

    def test_unique_role_via_role_fallback_validates(self) -> None:
        # Single ground net — referencing by role "ground" should be ok.
        tpl = _trivial_template(
            required_edges=(
                EdgeSpec(component_role="R1", pin="pin1", net_role="input"),
                EdgeSpec(component_role="R1", pin="pin2", net_role="ground"),
            ),
        )
        assert tpl.validate() == []

    def test_duplicate_canonical_name_fails(self) -> None:
        tpl = _trivial_template(
            required_nets=(
                NetSlot(role="input", canonical_name="DUP"),
                NetSlot(role="signal", canonical_name="DUP"),
            ),
            required_edges=(),
        )
        errors = tpl.validate()
        assert any("duplicate net canonical_name" in e for e in errors)

    def test_duplicate_variant_id_fails(self) -> None:
        tpl = _trivial_template(
            variants=(
                TopologyVariant(variant_id="v1", description="x"),
                TopologyVariant(variant_id="v1", description="y"),
            ),
        )
        errors = tpl.validate()
        assert any("variant ids must be unique" in e for e in errors)


class TestFrozenDataclass:
    def test_topology_template_is_frozen(self) -> None:
        tpl = _trivial_template()
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            tpl.template_id = "mutated"  # type: ignore[misc]

    def test_component_slot_is_frozen(self) -> None:
        slot = ComponentSlot(role="R", component_type="Resistor")
        with pytest.raises(Exception):
            slot.role = "mutated"  # type: ignore[misc]


class TestParametricInvariant:
    def test_field_round_trip(self) -> None:
        inv = ParametricInvariant(
            name="x",
            formula="R1.value > 0",
            severity="warning",
            requires_values=True,
            violation_msg="msg",
        )
        assert inv.name == "x"
        assert inv.severity == "warning"
        assert inv.requires_values
