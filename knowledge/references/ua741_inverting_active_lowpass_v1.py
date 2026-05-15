from app.domain.dsl import CapacitorCeramic, Circuit, IC, Resistor

circuit = Circuit(
    reference_id="ua741_inverting_active_lowpass_v1",
    name="uA741 反相有源低通滤波器逻辑参考电路",
    description="由 uA741、输入电阻 R1、反馈电阻 Rf、并联反馈电容 C1 和同相端接地偏置电阻 Rp 构成的反相有源低通逻辑参考电路。",
    created_at="2026-05-15T00:00:00",
    source={
        "type": "schematic_image",
        "note": "只表达连接拓扑；函数发生器和双电源不作为元件建模，使用 UI1/VCC/VEE/GND 节点表示。",
    },
    compare_options={
        "ignore_hole_id": True,
        "ignore_component_id": True,
        "ignore_passive_pin_order": True,
        "allow_extra_wires": True,
        "allow_equivalent_layout": True,
    },
    design_goal="inverting_active_lowpass",
    feedback_network="Rf and C1 in parallel between SUM and UO1",
)

VCC = circuit.power("VCC", label="VCC", description="+12V 正电源")
VEE = circuit.power("VEE", label="VEE", description="-12V 负电源")
GND = circuit.ground(description="0V 参考地")
UI1 = circuit.input("UI1", label="UI1", description="输入信号")
UO1 = circuit.output("UO1", label="UO1", description="低通滤波后的反相输出")
SUM = circuit.net("SUM", description="uA741 反相输入 / 反馈求和节点")
VREF = circuit.net("VREF", description="uA741 同相输入偏置节点")

R1 = Resistor("R1", description="输入电阻：连接 UI1 与反相输入求和节点")
Rf = Resistor("Rf", description="反馈电阻：连接输出与反相输入")
C1 = CapacitorCeramic("C1", description="反馈电容：与 Rf 并联形成有源低通反馈网络")
Rp = Resistor("Rp", description="同相输入偏置电阻：连接 VREF 与 GND")
U1 = IC("U1", subtype="UA741", pins=("pin2", "pin3", "pin4", "pin6", "pin7"))

R1[1, 2] += UI1, SUM
Rf[1, 2] += SUM, UO1
C1[1, 2] += SUM, UO1
Rp[1, 2] += VREF, GND
U1["pin2", "pin3", "pin4", "pin6", "pin7"] += SUM, VREF, VEE, UO1, VCC
