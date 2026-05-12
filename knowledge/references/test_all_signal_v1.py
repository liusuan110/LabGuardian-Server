from app.domain.dsl import Circuit, Resistor

circuit = Circuit(reference_id="test_all_signal_v1")

NET_A = circuit.net("NET_A")
NET_B = circuit.net("NET_B")

R1 = Resistor("R1")
R1[1, 2] += NET_A, NET_B
