from app.domain.dsl import Circuit, Resistor

circuit = Circuit(reference_id="test_rc_v1")

ROW_12_L = circuit.net("ROW_12_L")
ROW_12_R = circuit.net("ROW_12_R")

R1 = Resistor("R1")
R1[1, 2] += ROW_12_L, ROW_12_R
