"""First-order RC filter template (LPF / HPF variants).

Canonical structure:
    VIN --[R1]-- VC --[C1]-- GND
                 ^^^^^
                 VOUT (low-pass) or
    VIN --[C1]-- VC --[R1]-- GND
                 ^^^^^
                 VOUT (high-pass)

The template captures the **topological** distinction between LPF and HPF
via two variants. The Phase 0 matcher does not yet enforce which net is
"output" (relies on student net-role inference / port_annotation), so both
variants will match for typical student boards; downstream UI shows which
variant was preferred.

Reference netlist: ``rc_first_order_v1`` (bandpass variant already in
``knowledge/references/``). HPF / LPF variants share the same set of
components; only the role attached to each net differs.
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
    template_id="rc_first_order_v1",
    name="一阶 RC 滤波器",
    topology_label="rc_first_order",
    reference_id="rc_first_order_v1",
    description=(
        "一阶 RC 低通 / 高通 / 带通滤波器的最简实现。"
        "通过变体区分 LPF（输出取在 C 上）与 HPF（输出取在 R 上）。"
    ),
    required_components=(
        ComponentSlot(role="R1", component_type="Resistor"),
        ComponentSlot(role="C1", component_type="CapacitorCeramic"),
    ),
    required_nets=(
        NetSlot(role="input", canonical_name="VIN", role_label="UI1"),
        NetSlot(role="signal", canonical_name="VC"),
        NetSlot(role="ground", canonical_name="GND", role_label="GND"),
    ),
    optional_nets=(
        NetSlot(role="output", canonical_name="VOUT", role_label="UO1"),
    ),
    required_edges=(
        # Base shape: R1 between VIN and VC, C1 between VC and GND (LPF default).
        EdgeSpec(component_role="R1", pin="pin1", net_role="VIN"),
        EdgeSpec(component_role="R1", pin="pin2", net_role="VC"),
        EdgeSpec(component_role="C1", pin="pin1", net_role="VC"),
        EdgeSpec(component_role="C1", pin="pin2", net_role="GND"),
    ),
    variants=(
        TopologyVariant(
            variant_id="lowpass",
            description="低通：输出取自电容两端电压（信号节点 VC）",
        ),
        TopologyVariant(
            variant_id="highpass",
            description="高通：R 与 C 位置互换，输出取自电阻两端电压",
            # No structural additions; HPF is recognized in Phase 1 when
            # net-role inference / port_annotation propagates output role
            # to the R-side node.
        ),
    ),
    parametric_invariants=(
        ParametricInvariant(
            name="time_constant_in_audio_range",
            formula="0.0001 <= R1.value * C1.value <= 1.0",
            severity="warning",
            requires_values=True,
            violation_msg=(
                "时间常数 τ=RC 不在音频可用范围 (0.1ms ~ 1s)，"
                "请确认 R 与 C 的取值是否正确"
            ),
        ),
    ),
)
