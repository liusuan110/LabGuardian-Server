from app.domain.dsl import CapacitorCeramic, Circuit, Resistor

circuit = Circuit(
    reference_id="rc_lowpass_v1",
    name="RC 低通滤波器",
    description="一阶 RC 低通滤波器，允许低频信号通过，衰减高频信号",
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
C1 = CapacitorCeramic("C1")

R1[1, 2] += VIN, VOUT
C1[1, 2] += VOUT, GND
