"""共射放大器（固定偏置 + 发射极直接接地）逻辑参考电路。

来源：用户上传图 1（教学示意图）。

拓扑特征：
- 8050 NPN 三极管，集电极接 R_C 上拉到 VCC、发射极**直接接地**（无 R_E 也无 C_E）
- 偏置：R_P (原 1MΩ 电位器，演示版本用普通电阻) 与 R (510kΩ) 串联，
  从 VCC 经 R_P → RB_MID → R → BASE，**单支固定偏置**（不是分压偏置）
- 信号路径：VIN → C_B (33μF 输入耦合) → BASE → 8050.B → 8050.C → R_C 上拉，
  → COLLECTOR → C_C (33μF 输出耦合) → VOUT → R_L (2kΩ) → GND

注意事项：
- 缺 R_E 与 C_E 是教学最简版本，温度稳定性差但电压增益较高
- R_P 在物理板上若是电位器，对比 reference 时只需关注其作为电阻的连接拓扑
"""

from app.domain.dsl import Circuit, Capacitor, Resistor, Transistor

circuit = Circuit(
    reference_id="ce_amp_fixed_bias_v1",
    name="共射放大器（固定偏置 · 8050）逻辑参考电路",
    description=(
        "基于 8050 NPN 三极管的共射放大电路。固定偏置（R_P + R 串联），"
        "发射极直接接地（无 R_E 与 C_E），输入/输出 33μF 耦合，2kΩ 负载。"
    ),
    created_at="2026-05-22T00:00:00",
    source={
        "type": "schematic_image",
        "note": "用户上传图 1。演示版本：R_P 由 1MΩ 电位器替换为电阻。",
    },
    compare_options={
        "ignore_hole_id": True,
        "ignore_component_id": True,
        "ignore_passive_pin_order": True,
        "allow_extra_wires": True,
        "allow_equivalent_layout": True,
    },
    design_goal="common_emitter_voltage_amplification",
    bias_type="fixed_series",
    emitter_config="direct_to_ground",
)

# Nets
VCC = circuit.power("VCC", label="VCC", description="+12V 正电源 (Ucc)")
GND = circuit.ground(description="0V 参考地")
UI1 = circuit.input("UI1", label="UI1", description="输入信号（小信号电压源）")
UO1 = circuit.output("UO1", label="UO1", description="放大输出（接 R_L 负载）")
BASE = circuit.net("BASE", description="VT 基极 / C_B 输出 / 偏置串支汇合点")
COLLECTOR = circuit.net(
    "COLLECTOR",
    description="VT 集电极 / R_C 下端 / C_C 输入",
)
RB_MID = circuit.net(
    "RB_MID",
    description="R_P 与 R 串联中间节点（基极偏置串支）",
)

# Components
VT = Transistor("VT", subtype="NPN/BJT", description="8050 NPN 放大管")
R_P = Resistor(
    "R_P",
    value="1M",
    description="基极偏置串支上段（原为 1MΩ 电位器，演示用电阻替代）",
)
R = Resistor("R", value="510k", description="基极偏置串支下段")
R_C = Resistor("R_C", value="2k", description="集电极负载电阻")
R_L = Resistor("R_L", value="2k", description="输出端外接负载电阻")
C_B = Capacitor("C_B", value="33uF", description="输入耦合电容（隔直）")
C_C = Capacitor("C_C", value="33uF", description="输出耦合电容（隔直）")

# Wiring
C_B[1, 2] += UI1, BASE
R_P[1, 2] += VCC, RB_MID
R[1, 2] += RB_MID, BASE
R_C[1, 2] += VCC, COLLECTOR
C_C[1, 2] += COLLECTOR, UO1
R_L[1, 2] += UO1, GND
VT["collector", "base", "emitter"] += COLLECTOR, BASE, GND
