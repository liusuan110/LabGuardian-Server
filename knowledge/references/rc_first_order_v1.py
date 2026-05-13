from app.domain.dsl import CapacitorCeramic, Circuit, Resistor

circuit = Circuit(
    reference_id="rc_first_order_v1",
    name="一阶 RC 带通滤波器逻辑参考电路",
    description="由串联电阻 R1、对地电容 C1 构成低通级，再由输入耦合电容 C2、下拉电阻 R2 构成高通级级联的一阶 RC 带通滤波器逻辑参考电路。",
    created_at="2026-05-11T00:00:00",
    source={
        "type": "logical_reference_json",
        "note": "按低通级 R1+C1 → 高通级 C2+R2 的级联结构修订；只表达电路连接拓扑，不包含具体面包板 hole_id。",
    },
    compare_options={
        "ignore_hole_id": True,
        "ignore_component_id": True,
        "ignore_passive_pin_order": True,
        "allow_extra_wires": True,
        "allow_equivalent_layout": True,
    },
)

VIN = circuit.input("VIN", label="UI1", description="输入信号 / R1 输入端")
VLP = circuit.net("VLP", description="低通级输出 / C2 输入端 (内部节点)")
VOUT = circuit.output("VOUT", label="UO1", description="带通滤波后的输出信号")
GND = circuit.ground(description="0V 参考地")

R1 = Resistor("R1", description="低通级串联电阻：连接 VIN 与 VLP")
C1 = CapacitorCeramic("C1", description="低通级对地电容：从 VLP 到地")
C2 = CapacitorCeramic("C2", description="高通级输入耦合电容：连接 VLP 与 VOUT")
R2 = Resistor("R2", description="高通级下拉电阻：从 VOUT 到地")

R1[1, 2] += VIN, VLP
C1[1, 2] += VLP, GND
C2[1, 2] += VLP, VOUT
R2[1, 2] += VOUT, GND
