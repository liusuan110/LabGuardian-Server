"""UA741 反相加法器模板（输入数 2-5 可变）。

Canonical structure (matches ``ua741_inverting_summing_amp_v1`` reference):

    VIN1 --[R_in1]--+
    VIN2 --[R_in2]--+--- SUM ---[R_f]--- VOUT
    (VIN3 --[R_in3]+)         |
                              +-- IC.pin2 (inverting input)

    GND --[R_p]-- VREF -- IC.pin3 (non-inverting input)

The base template encodes 2 input resistors; the variants extend to 3 / 4 / 5.
Phase 0 matches them as separate variants; Phase 1 will collapse into a single
multiplicity-aware match.
"""

from __future__ import annotations

from app.domain.templates.base import (
    ComponentSlot,
    EdgeSpec,
    NetSlot,
    ParametricInvariant,
    TopologyTemplate,
    TopologyVariant,
)


TEMPLATE = TopologyTemplate(
    template_id="summing_amp_ua741_v1",
    name="UA741 反相加法器",
    topology_label="summing_amp_ua741",
    reference_id="ua741_inverting_summing_amp_v1",
    description=(
        "基于 UA741 的反相加法电路：VOUT = -(R_f/R_in1 · VIN1 + R_f/R_in2 · VIN2 + ...)。"
        "输入路数可在 2-5 之间灵活变化。"
    ),
    required_components=(
        ComponentSlot(role="opamp", component_type="IC", component_subtype="UA741"),
        ComponentSlot(role="R_in1", component_type="Resistor"),
        ComponentSlot(role="R_in2", component_type="Resistor"),
        ComponentSlot(role="R_f", component_type="Resistor"),
    ),
    optional_components=(
        ComponentSlot(role="R_p", component_type="Resistor", is_required=False),
    ),
    required_nets=(
        NetSlot(role="input", canonical_name="VIN1", role_label="UI1"),
        NetSlot(role="input", canonical_name="VIN2", role_label="UI2"),
        NetSlot(role="output", canonical_name="VOUT", role_label="UO1"),
        NetSlot(role="signal", canonical_name="SUM"),
        NetSlot(role="signal", canonical_name="VREF"),
        NetSlot(role="power", canonical_name="VCC", role_label="VCC"),
        NetSlot(role="power", canonical_name="VEE", role_label="VEE"),
        NetSlot(role="ground", canonical_name="GND", role_label="GND"),
    ),
    required_edges=(
        # R_in1: VIN1 -> SUM
        EdgeSpec(component_role="R_in1", pin="pin1", net_role="VIN1"),
        EdgeSpec(component_role="R_in1", pin="pin2", net_role="SUM"),
        # R_in2: VIN2 -> SUM
        EdgeSpec(component_role="R_in2", pin="pin1", net_role="VIN2"),
        EdgeSpec(component_role="R_in2", pin="pin2", net_role="SUM"),
        # R_f: SUM -> VOUT
        EdgeSpec(component_role="R_f", pin="pin1", net_role="SUM"),
        EdgeSpec(component_role="R_f", pin="pin2", net_role="VOUT"),
        # UA741
        EdgeSpec(component_role="opamp", pin="pin2", net_role="SUM"),
        EdgeSpec(component_role="opamp", pin="pin6", net_role="VOUT"),
        EdgeSpec(component_role="opamp", pin="pin4", net_role="VEE"),
        EdgeSpec(component_role="opamp", pin="pin7", net_role="VCC"),
    ),
    variants=(
        TopologyVariant(
            variant_id="2_inputs",
            description="2 路输入（最简加法器）",
        ),
        TopologyVariant(
            variant_id="3_inputs",
            description="3 路输入",
            additional_components=(
                ComponentSlot(role="R_in3", component_type="Resistor"),
            ),
            # NOTE: the 3rd input net would need its own NetSlot; Phase 0
            # leaves this as a known limitation — the 2-input base will
            # still match if the student board has 3+ inputs (subgraph iso
            # only requires a subset to match).
        ),
    ),
    parametric_invariants=(
        ParametricInvariant(
            name="input_resistors_balanced",
            formula="max(R_in1.value, R_in2.value) / min(R_in1.value, R_in2.value) <= 10",
            severity="warning",
            requires_values=True,
            violation_msg=(
                "各路输入电阻差异过大（>10×），加法器各路权重严重不均；"
                "如果是有意设计的加权加法器请忽略此提示"
            ),
        ),
    ),
)
