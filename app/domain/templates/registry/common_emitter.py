"""共射放大器模板（多变体覆盖发射极配置 + 偏置类型）。

Canonical structure (most permissive — collector amplification + I/O coupling):

           VCC
            |
            +--[R_C]----+
            |           |
            |          COLLECTOR
            |           |
        [bias R(s)]    VT.C
            |           |
            BASE ---- VT.B            +---[C_out]--- VOUT
            |           |             |
       [C_in]          VT.E         (VOUT --[R_L]-- GND optional)
            |           |
           VIN         <emitter, see variants>

The matcher tries the base spec plus each variant; the variant with the
highest confidence wins. Phase 0 variants:

* ``direct_grounded_emitter`` — VT emitter wired directly to GND (no R_E).
  Simplest demo wiring, matches the user-authored
  ``ce_amp_fixed_bias_v1`` reference.
* ``emitter_resistor_no_bypass`` — VT.E → EMITTER → R_E → GND. Better
  thermal stability, no AC bypass.
* ``emitter_resistor_with_bypass`` — same plus C_E in parallel with R_E
  (the textbook "high-gain" configuration).

Variants add components but Phase 0 keeps the spec edges scoped to base
+ emitter; bias wiring is intentionally NOT enforced structurally so the
matcher tolerates the many valid teaching arrangements (single series R,
voltage divider, current-source bias, etc.). Bias-specific checks will
land in Phase 1 once we have richer per-variant edge constraints.

Reference netlist: ``ce_amp_fixed_bias_v1`` — matches the user-provided
图 1 (8050 + R_P + R fixed-series bias + emitter direct to GND + R_L load).
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
    template_id="common_emitter_v1",
    name="共射放大电路",
    topology_label="common_emitter",
    reference_id="ce_amp_fixed_bias_v1",
    description=(
        "BJT 共射放大器。变体覆盖 [发射极直接接地 | 含 R_E 无 C_E | 含 R_E + C_E] "
        "三种发射极配置。偏置结构（分压/固定/恒流）不在 Phase 0 强制约束以保持灵活度。"
    ),
    # ---------- REQUIRED components ----------
    # R_E moved to optional (variants decide whether emitter has resistor).
    required_components=(
        ComponentSlot(role="VT", component_type="Transistor"),
        ComponentSlot(role="R_C", component_type="Resistor"),
        ComponentSlot(role="C_in", component_type="Capacitor"),
        ComponentSlot(role="C_out", component_type="Capacitor"),
    ),
    # ---------- OPTIONAL components ----------
    optional_components=(
        # R_L: external load (present in 用户 reference, absent in many textbook variants).
        ComponentSlot(role="R_L", component_type="Resistor", is_required=False),
        # Bias resistors — any of these patterns is accepted.
        ComponentSlot(role="R_B_fixed", component_type="Resistor", is_required=False),
        ComponentSlot(role="R_B_fixed_2", component_type="Resistor", is_required=False),
        ComponentSlot(role="R_B1", component_type="Resistor", is_required=False),
        ComponentSlot(role="R_B2", component_type="Resistor", is_required=False),
    ),
    # ---------- NETS ----------
    required_nets=(
        NetSlot(role="input", canonical_name="VIN", role_label="UI1"),
        NetSlot(role="output", canonical_name="VOUT", role_label="UO1"),
        NetSlot(role="signal", canonical_name="BASE"),
        NetSlot(role="signal", canonical_name="COLLECTOR"),
        NetSlot(role="power", canonical_name="VCC", role_label="VCC"),
        NetSlot(role="ground", canonical_name="GND", role_label="GND"),
    ),
    # EMITTER net is only meaningful in R_E variants; declare it as
    # optional so variant specs can reference it without polluting the
    # base spec graph.
    optional_nets=(
        NetSlot(role="signal", canonical_name="EMITTER"),
    ),
    # ---------- REQUIRED edges ----------
    # Notice: NO VT.emitter edge here — each variant supplies its own,
    # because the emitter target net is variant-specific (GND vs EMITTER).
    required_edges=(
        EdgeSpec(component_role="VT", pin="collector", net_role="COLLECTOR"),
        EdgeSpec(component_role="VT", pin="base", net_role="BASE"),
        EdgeSpec(component_role="R_C", pin="pin1", net_role="VCC"),
        EdgeSpec(component_role="R_C", pin="pin2", net_role="COLLECTOR"),
        EdgeSpec(component_role="C_in", pin="pin1", net_role="VIN"),
        EdgeSpec(component_role="C_in", pin="pin2", net_role="BASE"),
        EdgeSpec(component_role="C_out", pin="pin1", net_role="COLLECTOR"),
        EdgeSpec(component_role="C_out", pin="pin2", net_role="VOUT"),
    ),
    optional_edges=(
        # Output load is common in teaching circuits but optional.
        EdgeSpec(component_role="R_L", pin="pin1", net_role="VOUT", is_required=False),
        EdgeSpec(component_role="R_L", pin="pin2", net_role="GND", is_required=False),
    ),
    variants=(
        TopologyVariant(
            variant_id="direct_grounded_emitter",
            description=(
                "发射极直接接地（无 R_E 与 C_E）— 最简教学版本，匹配用户 "
                "reference ce_amp_fixed_bias_v1。"
            ),
            additional_edges=(
                EdgeSpec(component_role="VT", pin="emitter", net_role="GND"),
            ),
        ),
        TopologyVariant(
            variant_id="emitter_resistor_no_bypass",
            description="含 R_E 发射极电阻，无 C_E 旁路（稳定但增益较低）",
            additional_components=(
                ComponentSlot(role="R_E", component_type="Resistor"),
            ),
            additional_edges=(
                EdgeSpec(component_role="VT", pin="emitter", net_role="EMITTER"),
                EdgeSpec(component_role="R_E", pin="pin1", net_role="EMITTER"),
                EdgeSpec(component_role="R_E", pin="pin2", net_role="GND"),
            ),
        ),
        TopologyVariant(
            variant_id="emitter_resistor_with_bypass",
            description="含 R_E + C_E 旁路电容（增益最高，温度稳定性好）",
            additional_components=(
                ComponentSlot(role="R_E", component_type="Resistor"),
                ComponentSlot(role="C_E", component_type="Capacitor"),
            ),
            additional_edges=(
                EdgeSpec(component_role="VT", pin="emitter", net_role="EMITTER"),
                EdgeSpec(component_role="R_E", pin="pin1", net_role="EMITTER"),
                EdgeSpec(component_role="R_E", pin="pin2", net_role="GND"),
                EdgeSpec(component_role="C_E", pin="pin1", net_role="EMITTER"),
                EdgeSpec(component_role="C_E", pin="pin2", net_role="GND"),
            ),
        ),
    ),
    parametric_invariants=(
        ParametricInvariant(
            name="emitter_collector_ratio",
            formula="0.05 <= R_E.value / R_C.value <= 0.5",
            severity="warning",
            requires_values=True,
            violation_msg=(
                "R_E/R_C 比例异常（仅在 R_E 存在时检查），请确认是否影响增益和稳定性"
            ),
        ),
        ParametricInvariant(
            name="voltage_divider_ratio_reasonable",
            formula="0.1 <= R_B2.value / (R_B1.value + R_B2.value) <= 0.3",
            severity="warning",
            requires_values=True,
            violation_msg=(
                "分压偏置比例不在常用范围 (V_B ≈ 0.1-0.3 VCC)；"
                "可能导致 BJT 工作点偏离放大区"
            ),
        ),
    ),
)
