from app.domain.dsl import Circuit, LED, Resistor

circuit = Circuit(
    reference_id="basic_series_resistor_v1",
    name="LED 串联电阻电路",
    description="LED 与限流电阻串联基础实验参考电路",
    compare_options={
        "ignore_hole_id": True,
        "ignore_component_id": True,
        "ignore_polarity": True,
        "allow_extra_wires": True,
        "allow_equivalent_layout": True,
    },
)

VCC = circuit.power("VCC")
LED_NET = circuit.net("LED_NET")
GND = circuit.ground()

R1 = Resistor("R1")
LED1 = LED("LED1")

R1[1, 2] += VCC, LED_NET
LED1["anode", "cathode"] += LED_NET, GND
