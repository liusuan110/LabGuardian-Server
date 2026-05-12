from app.domain.dsl import CapacitorCeramic, Circuit, Resistor

circuit = Circuit(
    reference_id="rc_first_order_v1",
    name="一阶 RC 带通滤波器逻辑参考电路",
    description="由串联电阻 R1、对地电容 C1 构成低通级，再由输入耦合电容 C2、下拉电阻 R2 构成高通级级联的一阶 RC 带通滤波器逻辑参考电路。",
    created_at="2026-05-11T00:00:00",
    source={
        "type": "logical_reference_json",
        "note": "按“低通级 R1+C1 → 高通级 C2+R2”的级联结构修订；只表达电路连接拓扑，不包含具体面包板 hole_id。",
    },
    compare_options={
        "ignore_hole_id": True,
        "ignore_component_id": True,
        "ignore_passive_pin_order": True,
        "allow_extra_wires": True,
        "allow_equivalent_layout": True,
    },
)

VIN = circuit.input("VIN", label="input signal / R1 input")
VLP = circuit.net("VLP", label="low-pass output node / C2 input node")
VOUT = circuit.output("VOUT", label="band-pass filtered output signal")
GND = circuit.ground(label="0V")

R1 = Resistor("R1", value="", description="series resistor of the low-pass stage; connects VIN to VLP")
C1 = CapacitorCeramic("C1", value="", description="shunt capacitor of the low-pass stage from VLP to ground")
C2 = CapacitorCeramic("C2", value="", description="input coupling capacitor of the high-pass stage; connects VLP to VOUT")
R2 = Resistor("R2", value="", description="high-pass stage pull-down resistor from VOUT to ground")

R1[1, 2] += VIN, VLP
C1[1, 2] += VLP, GND
C2[1, 2] += VLP, VOUT
R2[1, 2] += VOUT, GND
