from app.domain.dsl import Circuit, IC, Resistor

circuit = Circuit(
    reference_id="ua741_inverting_summing_amp_v1",
    name="uA741 反相加法器逻辑参考电路",
    description="由 uA741、两路输入电阻、反馈电阻、VCC 到 GND 的 UI1 分压支路和同相端接地偏置电阻构成的反相加法器逻辑参考电路。",
    created_at="2026-05-15T00:00:00",
    source={
        "type": "schematic_image",
        "note": "只表达连接拓扑；函数发生器和双电源不作为元件建模，Vi1 分压支路按 VCC-R11-UI1-R12-GND 表达。",
    },
    compare_options={
        "ignore_hole_id": True,
        "ignore_component_id": True,
        "ignore_passive_pin_order": True,
        "allow_extra_wires": True,
        "allow_equivalent_layout": True,
    },
    design_goal="inverting_summing_amplifier",
    transfer_function="UO1 = -(2 * UI1 + UI2)",
    ratio_hint="Rf/R1 = 2, Rf/R2 = 1",
    divider_hint="With VCC=12V, use R11:R12 = 11:1 to obtain UI1 approximately 1V.",
)

VCC = circuit.power("VCC", label="VCC", description="+12V 正电源 / UI1 分压上端")
VEE = circuit.power("VEE", label="VEE", description="-12V 负电源")
GND = circuit.ground(description="0V 参考地 / UI1 分压下端")
UI1 = circuit.input("UI1", label="UI1", description="分压得到的直流输入 Vi1")
UI2 = circuit.input("UI2", label="UI2", description="交流输入 Vi2")
UO1 = circuit.output("UO1", label="UO1", description="反相加法输出")
SUM = circuit.net("SUM", description="uA741 反相输入 / 两路输入和反馈求和节点")
VREF = circuit.net("VREF", description="uA741 同相输入偏置节点")

R11 = Resistor("R11", value="11 * R12", description="UI1 分压上臂：连接 VCC 与 UI1")
R12 = Resistor("R12", description="UI1 分压下臂：连接 UI1 与 GND")
R1 = Resistor("R1", description="Vi1 输入电阻：连接 UI1 与求和节点，Rf/R1=2")
R2 = Resistor("R2", description="Vi2 输入电阻：连接 UI2 与求和节点，Rf/R2=1")
Rf = Resistor("Rf", description="反馈电阻：连接输出与求和节点")
Rp = Resistor("Rp", description="同相输入偏置电阻：连接 VREF 与 GND")
U1 = IC("U1", subtype="UA741", pins=("pin2", "pin3", "pin4", "pin6", "pin7"))

R11[1, 2] += VCC, UI1
R12[1, 2] += UI1, GND
R1[1, 2] += UI1, SUM
R2[1, 2] += UI2, SUM
Rf[1, 2] += SUM, UO1
Rp[1, 2] += VREF, GND
U1["pin2", "pin3", "pin4", "pin6", "pin7"] += SUM, VREF, VEE, UO1, VCC
