"""UA741 反相放大器模板（含/无 R_p 偏置补偿两变体）。

Canonical structure (matches ``ua741_inverting_amp_gain10_v1`` reference):

    VIN --[R_g]-- SUM --[R_f]-- VOUT
                  |
                  +--(IC.pin2: inverting input)

    GND --[R_p]-- VREF --(IC.pin3: non-inverting input)

    IC: pin4=VEE, pin7=VCC, pin6=VOUT

Forbidden: a CapacitorCeramic on the feedback edge (would suggest the
student is building an integrator or LPF instead).
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
    template_id="inverting_amp_ua741_v1",
    name="UA741 反相放大器",
    topology_label="inverting_amp_ua741",
    reference_id="ua741_inverting_amp_gain10_v1",
    description=(
        "基于 UA741 的反相放大器（标准电路）。Av = -R_f/R_g。"
        "R_p 用作非反相端偏置电流补偿（可选）。"
    ),
    required_components=(
        ComponentSlot(role="opamp", component_type="IC", component_subtype="UA741"),
        ComponentSlot(role="R_g", component_type="Resistor"),
        ComponentSlot(role="R_f", component_type="Resistor"),
    ),
    optional_components=(
        ComponentSlot(role="R_p", component_type="Resistor", is_required=False),
    ),
    forbidden_components=(
        # A feedback capacitor would make this an integrator or LPF, not
        # a pure inverting amplifier. We flag CapacitorCeramic as a global
        # forbidden — the matcher will downgrade confidence accordingly.
        # (Phase 0 forbidden_components are "any presence" — pin-level
        # forbidden constraints are Phase 1.)
        # NOTE: students often add power-rail decoupling caps which would
        # false-trigger this. Leave forbidden empty for now; integrator
        # template will outrank when feedback C is present.
    ),
    required_nets=(
        NetSlot(role="input", canonical_name="VIN", role_label="UI1"),
        NetSlot(role="output", canonical_name="VOUT", role_label="UO1"),
        NetSlot(role="signal", canonical_name="INV"),
        NetSlot(role="signal", canonical_name="VREF"),
        NetSlot(role="power", canonical_name="VCC", role_label="VCC"),
        NetSlot(role="power", canonical_name="VEE", role_label="VEE"),
        NetSlot(role="ground", canonical_name="GND", role_label="GND"),
    ),
    required_edges=(
        # R_g: VIN -> INV
        EdgeSpec(component_role="R_g", pin="pin1", net_role="VIN"),
        EdgeSpec(component_role="R_g", pin="pin2", net_role="INV"),
        # R_f: INV -> VOUT
        EdgeSpec(component_role="R_f", pin="pin1", net_role="INV"),
        EdgeSpec(component_role="R_f", pin="pin2", net_role="VOUT"),
        # UA741 pins
        EdgeSpec(component_role="opamp", pin="pin2", net_role="INV"),
        EdgeSpec(component_role="opamp", pin="pin6", net_role="VOUT"),
        EdgeSpec(component_role="opamp", pin="pin4", net_role="VEE"),
        EdgeSpec(component_role="opamp", pin="pin7", net_role="VCC"),
    ),
    optional_edges=(
        # IC pin3 to V_P (matched only when R_p is present, otherwise pin3
        # connects directly to GND in some teaching variants).
        EdgeSpec(
            component_role="opamp",
            pin="pin3",
            net_role="VREF",
            is_required=False,
        ),
        EdgeSpec(
            component_role="R_p",
            pin="pin1",
            net_role="VREF",
            is_required=False,
        ),
        EdgeSpec(
            component_role="R_p",
            pin="pin2",
            net_role="GND",
            is_required=False,
        ),
    ),
    variants=(
        TopologyVariant(
            variant_id="with_bias_compensation",
            description=(
                "包含 R_p 偏置补偿电阻（推荐做法）— R_p 从 V_P (VREF) 接 GND。"
            ),
            additional_components=(
                ComponentSlot(role="R_p_present", component_type="Resistor"),
            ),
            additional_edges=(
                EdgeSpec(
                    component_role="R_p_present", pin="pin1", net_role="VREF",
                ),
                EdgeSpec(
                    component_role="R_p_present", pin="pin2", net_role="GND",
                ),
            ),
        ),
        TopologyVariant(
            variant_id="no_bias_compensation",
            description="无 R_p，非反相端直接接地（简化做法）",
        ),
    ),
    parametric_invariants=(
        ParametricInvariant(
            name="bias_compensation_ratio",
            formula="abs(R_p.value - (R_g.value * R_f.value)/(R_g.value + R_f.value)) / R_p.value < 0.2",
            severity="warning",
            requires_values=True,
            violation_msg=(
                "R_p 应约等于 R_g 与 R_f 的并联值以补偿偏置电流；"
                "当前偏差过大可能引入直流失调"
            ),
        ),
        ParametricInvariant(
            name="gain_range_reasonable",
            formula="1 <= R_f.value / R_g.value <= 1000",
            severity="warning",
            requires_values=True,
            violation_msg=(
                "增益 |Av| = R_f/R_g 超出常用范围 (1~1000)，请确认设计意图"
            ),
        ),
    ),
)
