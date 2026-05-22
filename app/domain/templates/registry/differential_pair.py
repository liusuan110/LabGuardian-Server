"""BJT 差分对放大器模板（尾电阻 / 恒流源两变体）。

Canonical structure (matches ``diff_pair_current_source_ref_split_potentiometer`` reference):

           VCC
            |
            +--[RC1]--UO1     UO2--[RC2]--+
                       |       |          |
                       VT1.C   VT2.C      |
    UI1 ----VT1.B  VT2.B---- UI2          |
                       VT1.E   VT2.E      |
                       |       |          |
                       +--+----+          |
                          |               |
                         TAIL             |
                          |               |
                  [尾电阻 RE_TAIL]  或    [VT3 恒流源]
                          |               |
                         VEE             VEE

差分对的"对称性"是核心特征：RC1 与 RC2 阻值相等、VT1 与 VT2 同型号。
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
    template_id="differential_pair_v1",
    name="BJT 差分放大器",
    topology_label="differential_pair",
    reference_id="diff_pair_current_source_ref_split_potentiometer",
    description=(
        "BJT 差分对放大器。两个三极管 VT1/VT2 共发射极接尾电流源，"
        "集电极通过 RC1/RC2 接 VCC。尾部可用尾电阻或恒流源（变体）。"
    ),
    required_components=(
        ComponentSlot(role="VT1", component_type="Transistor"),
        ComponentSlot(role="VT2", component_type="Transistor"),
        ComponentSlot(role="RC1", component_type="Resistor"),
        ComponentSlot(role="RC2", component_type="Resistor"),
    ),
    required_nets=(
        NetSlot(role="input", canonical_name="UI1", role_label="UI1"),
        NetSlot(role="input", canonical_name="UI2", role_label="UI2"),
        NetSlot(role="output", canonical_name="UO1", role_label="UO1"),
        NetSlot(role="output", canonical_name="UO2", role_label="UO2"),
        # E1 / E2 are the two transistor emitters; in the canonical
        # reference they are bridged by a balancing potentiometer (split
        # into RP_LEFT / RP_RIGHT around a TAIL wiper node). In the simpler
        # tail-resistor variant E1 == E2 == TAIL (one shared net).
        NetSlot(role="signal", canonical_name="E1"),
        NetSlot(role="signal", canonical_name="E2"),
        NetSlot(role="signal", canonical_name="TAIL"),
        NetSlot(role="power", canonical_name="VCC", role_label="VCC"),
        NetSlot(role="power", canonical_name="VEE", role_label="VEE"),
    ),
    required_edges=(
        # VT1: collector -> UO1, base -> UI1, emitter -> E1
        EdgeSpec(component_role="VT1", pin="collector", net_role="UO1"),
        EdgeSpec(component_role="VT1", pin="base", net_role="UI1"),
        EdgeSpec(component_role="VT1", pin="emitter", net_role="E1"),
        # VT2: collector -> UO2, base -> UI2, emitter -> E2
        EdgeSpec(component_role="VT2", pin="collector", net_role="UO2"),
        EdgeSpec(component_role="VT2", pin="base", net_role="UI2"),
        EdgeSpec(component_role="VT2", pin="emitter", net_role="E2"),
        # RC1: VCC -> UO1, RC2: VCC -> UO2
        EdgeSpec(component_role="RC1", pin="pin1", net_role="VCC"),
        EdgeSpec(component_role="RC1", pin="pin2", net_role="UO1"),
        EdgeSpec(component_role="RC2", pin="pin1", net_role="VCC"),
        EdgeSpec(component_role="RC2", pin="pin2", net_role="UO2"),
    ),
    variants=(
        TopologyVariant(
            variant_id="split_potentiometer",
            description=(
                "RP 电位器拆分变体：E1 --[RP_LEFT]-- TAIL --[RP_RIGHT]-- E2，"
                "TAIL 接尾电阻或恒流源（匹配 reference DSL）"
            ),
            additional_components=(
                ComponentSlot(role="RP_LEFT", component_type="Potentiometer"),
                ComponentSlot(role="RP_RIGHT", component_type="Potentiometer"),
            ),
            # Pin labels for Potentiometer use "terminal_a"/"terminal_b".
            additional_edges=(
                EdgeSpec(component_role="RP_LEFT", pin="terminal_a", net_role="E1"),
                EdgeSpec(component_role="RP_LEFT", pin="terminal_b", net_role="TAIL"),
                EdgeSpec(component_role="RP_RIGHT", pin="terminal_a", net_role="TAIL"),
                EdgeSpec(component_role="RP_RIGHT", pin="terminal_b", net_role="E2"),
            ),
        ),
        TopologyVariant(
            variant_id="shared_emitter_node",
            description=(
                "两个发射极直接短接（E1 == E2 == TAIL），尾电阻或恒流源"
                "接 TAIL → VEE（最简差分对教学版本）"
            ),
        ),
    ),
    parametric_invariants=(
        ParametricInvariant(
            name="collector_resistor_symmetry",
            formula="abs(RC1.value - RC2.value) / RC1.value < 0.1",
            severity="error",
            requires_values=True,
            violation_msg=(
                "差分对的集电极电阻 RC1/RC2 应严格对称（误差<10%），"
                "否则会有显著直流输出失调"
            ),
        ),
    ),
)
