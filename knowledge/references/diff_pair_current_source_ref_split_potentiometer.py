from app.domain.dsl import Circuit, Resistor, Transistor

circuit = Circuit(
    reference_id="diff_pair_current_source_ref_split_potentiometer",
    name="差分放大器 + VT3 恒流源参考电路",
    description="由图中 VT1/VT2 差分对、RC1/RC2 集电极负载、RP 发射极平衡电位器等效拆分为 RP_LEFT/RP_RIGHT 两个电阻、VT3+R1/R2/RE 尾电流源构成的逻辑参考电路。",
    created_at="2026-05-10T16:27:32",
    source={
        "type": "schematic_image",
        "note": "只表达原理图连接拓扑，不包含具体面包板 hole_id。 RP 电位器已按 wiper 节点 TAIL 拆分为两个两端电阻。",
    },
)

VCC = circuit.power("VCC", label="VCC", description="+UCC 正电源")
VEE = circuit.power("VEE", label="VEE", description="-UEE 负电源")
GND = circuit.ground(description="0V 参考地")
UI1 = circuit.input("UI1", label="UI1", description="ui1 差分输入 +")
UI2 = circuit.input("UI2", label="UI2", description="ui2 差分输入 -")
UO1 = circuit.output("UO1", label="UO1", description="uo1 差分输出 +")
UO2 = circuit.output("UO2", label="UO2", description="uo2 差分输出 -")
VT1_EMITTER = circuit.net("VT1_EMITTER", description="VT1 发射极 / RP_LEFT 左端")
VT2_EMITTER = circuit.net("VT2_EMITTER", description="VT2 发射极 / RP_RIGHT 右端")
TAIL = circuit.net("TAIL", description="RP_LEFT-RP_RIGHT 中点 / VT3 集电极")
VT3_BASE_BIAS = circuit.net("VT3_BASE_BIAS", description="R1-R2 分压 / VT3 基极")
VT3_EMITTER = circuit.net("VT3_EMITTER", description="VT3 发射极 / RE 顶端")

RC1 = Resistor("RC1", value="", description="VT1 collector load")
RC2 = Resistor("RC2", value="", description="VT2 collector load")
VT1 = Transistor("VT1", subtype="NPN/BJT", description="left transistor of differential pair")
VT2 = Transistor("VT2", subtype="NPN/BJT", description="right transistor of differential pair")
RP_LEFT = Resistor(
    "RP_LEFT",
    value="",
    description="left equivalent segment of RP balancing potentiometer, from VT1 emitter to wiper/tail node",
)
RP_RIGHT = Resistor(
    "RP_RIGHT",
    value="",
    description="right equivalent segment of RP balancing potentiometer, from wiper/tail node to VT2 emitter",
)
VT3 = Transistor("VT3", subtype="NPN/BJT", description="tail current source transistor")
R1 = Resistor("R1", description="upper/base-bias resistor to ground")
R2 = Resistor("R2", description="lower/base-bias resistor to negative supply")
RE = Resistor("RE", description="VT3 emitter resistor")

RC1[1, 2] += VCC, UO1
RC2[1, 2] += VCC, UO2
VT1["collector", "base", "emitter"] += UO1, UI1, VT1_EMITTER
VT2["collector", "base", "emitter"] += UO2, UI2, VT2_EMITTER
RP_LEFT[1, 2] += VT1_EMITTER, TAIL
RP_RIGHT[1, 2] += TAIL, VT2_EMITTER
VT3["collector", "base", "emitter"] += TAIL, VT3_BASE_BIAS, VT3_EMITTER
R1[1, 2] += GND, VT3_BASE_BIAS
R2[1, 2] += VT3_BASE_BIAS, VEE
RE[1, 2] += VT3_EMITTER, VEE
