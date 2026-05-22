"""UA741 反相积分器模板（含/无 R_leak 漏放电阻两变体）。

Canonical structure:

    VIN --[R_in]-- INV --[C_f]-- VOUT
                          (parallel [R_leak])
                   |
                   +-- IC.pin2 (inverting input)

    GND -- VREF -- IC.pin3 (non-inverting input)

**关键区别**：反馈支路是电容 C_f（与反相放大器的电阻 R_f 区分）。

Reference netlist: ``ua741_integrator_v1``（待用户提供 — Phase 0 计划中
用户写后我审核；如果用户尚未写，此模板的 reference_id 暂时为 None，
不影响模板匹配本身）。
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
    template_id="integrator_ua741_v1",
    name="UA741 反相积分器",
    topology_label="integrator_ua741",
    # NOTE: reference_id will be set once the user provides
    # ``knowledge/references/ua741_integrator_v1.py``. Until then we leave
    # it as the planned id — load_reference will 404 gracefully and the
    # template still matches.
    reference_id="ua741_integrator_v1",
    description=(
        "基于 UA741 的反相积分器：VOUT = -(1/(R_in·C_f))∫VIN dt。"
        "可选 R_leak 与 C_f 并联防止直流漂移。"
    ),
    required_components=(
        ComponentSlot(role="opamp", component_type="IC", component_subtype="UA741"),
        ComponentSlot(role="R_in", component_type="Resistor"),
        ComponentSlot(role="C_f", component_type="CapacitorCeramic"),
    ),
    optional_components=(
        ComponentSlot(role="R_leak", component_type="Resistor", is_required=False),
        ComponentSlot(role="R_p", component_type="Resistor", is_required=False),
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
        # R_in: VIN -> INV
        EdgeSpec(component_role="R_in", pin="pin1", net_role="VIN"),
        EdgeSpec(component_role="R_in", pin="pin2", net_role="INV"),
        # ★ C_f feedback: INV -> VOUT (key discriminator from inverting_amp)
        EdgeSpec(component_role="C_f", pin="pin1", net_role="INV"),
        EdgeSpec(component_role="C_f", pin="pin2", net_role="VOUT"),
        # UA741
        EdgeSpec(component_role="opamp", pin="pin2", net_role="INV"),
        EdgeSpec(component_role="opamp", pin="pin6", net_role="VOUT"),
        EdgeSpec(component_role="opamp", pin="pin4", net_role="VEE"),
        EdgeSpec(component_role="opamp", pin="pin7", net_role="VCC"),
    ),
    variants=(
        TopologyVariant(
            variant_id="with_leak_resistor",
            description=(
                "C_f 并联 R_leak 防直流漂移（推荐做法）— 匹配用户 reference "
                "ua741_integrator_v1。R_leak 在 INV → VOUT 反馈支路上。"
            ),
            additional_components=(
                ComponentSlot(role="R_leak_present", component_type="Resistor"),
            ),
            # ★ Edges必填：R_leak 与 C_f 并联（同样跨 INV → VOUT），
            # 没有这两条边，变体退化为仅声明组件存在而不约束位置，
            # coverage 得分不变 → 与 base 变体打平 → 无法击败 inverting_amp。
            additional_edges=(
                EdgeSpec(component_role="R_leak_present", pin="pin1", net_role="INV"),
                EdgeSpec(component_role="R_leak_present", pin="pin2", net_role="VOUT"),
            ),
        ),
        TopologyVariant(
            variant_id="without_leak_resistor",
            description="无 R_leak（理想积分器，实际易直流饱和）",
        ),
    ),
    parametric_invariants=(
        ParametricInvariant(
            name="integration_time_reasonable",
            formula="0.001 <= R_in.value * C_f.value <= 10",
            severity="warning",
            requires_values=True,
            violation_msg=(
                "积分时间常数 τ=R·C 超出常用范围 (1ms ~ 10s)"
            ),
        ),
        ParametricInvariant(
            name="leak_resistor_ratio",
            formula="R_leak.value / R_in.value >= 10",
            severity="warning",
            requires_values=True,
            violation_msg=(
                "R_leak/R_in < 10 会引入显著直流增益误差，请增大 R_leak"
            ),
        ),
    ),
)
