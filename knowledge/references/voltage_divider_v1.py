from app.domain.dsl import Circuit, Resistor

circuit = Circuit(
    reference_id="voltage_divider_v1",
    name="电阻分压电路",
    description="两个电阻串联分压实验参考电路",
    compare_options={
        "ignore_hole_id": True,
        "ignore_component_id": True,
        "ignore_polarity": True,
        "allow_extra_wires": True,
        "allow_equivalent_layout": True,
    },
)

VIN = circuit.input("VIN")
VOUT = circuit.output("VOUT")
GND = circuit.ground()

R1 = Resistor("R1")
R2 = Resistor("R2")

R1[1, 2] += VIN, VOUT
R2[1, 2] += VOUT, GND
