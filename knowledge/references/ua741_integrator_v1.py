"""UA741 反相积分器（含 R_f 漏放电阻）逻辑参考电路。

来源：用户上传图 3（Multisim 仿真图）。

拓扑特征：
- UA741 反相积分器：C1 提供积分反馈，R_f 与 C1 **并联**作直流漏放电阻
  （防输出在直流下饱和漂移）。理想积分器需 R_f → ∞，实际教学电路必须有 R_f
- 输入：XFG1 函数发生器 → R1 → INV（反相输入 pin2）
- 输出：pin6 → VOUT
- 偏置：非反相端 pin3 → VREF → R_p → GND（电流补偿）
- 双电源：V1 = +12V (pin7 VCC)，V2 = -12V (pin4 VEE)

注意事项：
- XFG1 函数发生器不作为元件建模，使用 UI1 节点代表输入信号
- 严格意义上这是"含漏放的反相积分器"或"反相低通滤波器"，但教学场景
  通常归类为"积分器"（设计意图）。模板匹配会同时报告 integrator 与
  inverting_amp 两个假设的高置信，由 UI 多假设展示
"""

from app.domain.dsl import Circuit, Capacitor, IC, Resistor

circuit = Circuit(
    reference_id="ua741_integrator_v1",
    name="UA741 反相积分器（含 R_f 漏放）逻辑参考电路",
    description=(
        "基于 UA741 的反相积分器：R1 输入电阻 + C1 反馈电容，"
        "R_f 与 C1 并联作直流漏放防饱和，R_p 非反相端偏置补偿，±12V 双电源。"
    ),
    created_at="2026-05-22T00:00:00",
    source={
        "type": "schematic_image",
        "note": "用户上传图 3 (Multisim)。XFG1 函数发生器未建模，用 UI1 节点代表。",
    },
    compare_options={
        "ignore_hole_id": True,
        "ignore_component_id": True,
        "ignore_passive_pin_order": True,
        "allow_extra_wires": True,
        "allow_equivalent_layout": True,
    },
    design_goal="inverting_integrator_with_dc_leak",
    transfer_function="VOUT(s) = -1/(R1·C1·s + R1/R_f)",
)

# Nets
VCC = circuit.power("VCC", label="VCC", description="+12V 正电源 (V1, UA741 pin7)")
VEE = circuit.power("VEE", label="VEE", description="-12V 负电源 (V2, UA741 pin4)")
GND = circuit.ground(description="0V 参考地")
UI1 = circuit.input("UI1", label="UI1", description="输入信号 (XFG1 函数发生器输出)")
UO1 = circuit.output("UO1", label="UO1", description="积分输出 (UA741 pin6)")
INV = circuit.net(
    "INV",
    description="UA741 反相输入 (pin2) / R1 输出端 / R_f 左端 / C1 左端 汇合点",
)
VREF = circuit.net(
    "VREF",
    description="UA741 同相输入 (pin3) / R_p 上端 偏置补偿节点",
)

# Components
R1 = Resistor("R1", description="输入电阻：VIN → 反相输入")
R_f = Resistor(
    "R_f",
    description="反馈漏放电阻（与 C1 并联，防直流积分饱和）",
)
C1 = Capacitor("C1", description="反馈积分电容（与 R_f 并联）")
R_p = Resistor("R_p", description="同相端偏置电流补偿电阻：VREF → GND")
U1 = IC(
    "U1",
    subtype="UA741",
    pins=("pin2", "pin3", "pin4", "pin6", "pin7"),
)

# Wiring
R1[1, 2] += UI1, INV
R_f[1, 2] += INV, UO1
C1[1, 2] += INV, UO1
R_p[1, 2] += VREF, GND
U1["pin2", "pin3", "pin4", "pin6", "pin7"] += INV, VREF, VEE, UO1, VCC
