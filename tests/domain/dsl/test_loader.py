from __future__ import annotations

import json
from pathlib import Path

from app.domain.dsl.loader import clear_dsl_reference_cache, load_dsl_reference
from app.services.reference_service import ReferenceService


def test_load_dsl_reference_from_c_variable(tmp_path: Path) -> None:
    path = tmp_path / "example_ref.py"
    path.write_text(
        "\n".join(
            [
                "from app.domain.dsl import Circuit, Resistor",
                "c = Circuit(reference_id='example_ref', name='Example')",
                "VIN = c.input('VIN')",
                "VOUT = c.output('VOUT')",
                "R1 = Resistor('R1')",
                "R1[1, 2] += VIN, VOUT",
            ]
        ),
        encoding="utf-8",
    )

    payload = load_dsl_reference(path)

    assert payload["reference_id"] == "example_ref"
    assert payload["source"]["type"] == "dsl_python_v1"
    assert len(payload["components"]) == 1


def test_reference_service_prefers_python_dsl_over_json(tmp_path: Path) -> None:
    (tmp_path / "same_ref.json").write_text(
        json.dumps(
            {
                "format": "logical_reference_v1",
                "reference_id": "same_ref",
                "name": "JSON",
                "components": [
                    {"ref_id": "R_JSON", "type": "Resistor", "pins": [{"pin": "pin1", "net": "A"}]}
                ],
                "nets": [{"net": "A"}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "same_ref.py").write_text(
        "\n".join(
            [
                "from app.domain.dsl import Circuit, Resistor",
                "circuit = Circuit(reference_id='same_ref', name='DSL')",
                "A = circuit.net('A')",
                "B = circuit.net('B')",
                "R1 = Resistor('R_DSL')",
                "R1[1, 2] += A, B",
            ]
        ),
        encoding="utf-8",
    )
    clear_dsl_reference_cache()

    svc = ReferenceService(reference_dir=tmp_path)
    payload = svc.load_reference("same_ref")
    refs = svc.list_references()

    assert payload["name"] == "DSL"
    assert payload["components"][0]["ref_id"] == "R_DSL"
    assert len([item for item in refs if item["reference_id"] == "same_ref"]) == 1
    assert refs[0]["source_type"] == "dsl_python_v1"


def test_production_references_are_dsl_only() -> None:
    reference_dir = Path("knowledge/references")
    assert not list(reference_dir.glob("*.json"))

    svc = ReferenceService(reference_dir=reference_dir)
    refs = svc.list_references()

    assert {item["reference_id"] for item in refs} == {
        "basic_series_resistor_v1",
        "ce_amp_fixed_bias_v1",  # CADx Phase 0 — 共射放大器 (user 图 1)
        "diff_pair_current_source_ref_split_potentiometer",
        "rc_first_order_v1",
        "rc_highpass_v1",
        "rc_lowpass_v1",
        "test_all_signal_v1",
        "test_rc_v1",
        "ua741_integrator_v1",  # CADx Phase 0 — UA741 反相积分器 (user 图 3)
        "ua741_inverting_active_lowpass_v1",
        "ua741_inverting_amp_gain10_v1",
        "ua741_inverting_summing_amp_v1",
        "voltage_divider_v1",
    }
    assert all(item["source_type"] == "dsl_python_v1" for item in refs)


def test_migrated_reference_dsl_topology_signatures() -> None:
    expected_counts = {
        "basic_series_resistor_v1": (2, 3),
        # CADx Phase 0 (user 图 1): 8050 + R_P + R + R_C + R_L + C_B + C_C = 7 comps
        # nets: VCC + GND + UI1 + UO1 + BASE + COLLECTOR + RB_MID = 7
        "ce_amp_fixed_bias_v1": (7, 7),
        "diff_pair_current_source_ref_split_potentiometer": (10, 12),
        "rc_first_order_v1": (4, 4),
        "rc_highpass_v1": (2, 3),
        "rc_lowpass_v1": (2, 3),
        "test_all_signal_v1": (1, 2),
        "test_rc_v1": (1, 2),
        # CADx Phase 0 (user 图 3): UA741 + R1 + R_f + C1 + R_p = 5 comps
        # nets: VCC + VEE + GND + UI1 + UO1 + INV + VREF = 7
        "ua741_integrator_v1": (5, 7),
        "ua741_inverting_active_lowpass_v1": (5, 7),
        "ua741_inverting_amp_gain10_v1": (4, 7),
        "ua741_inverting_summing_amp_v1": (7, 8),
        "voltage_divider_v1": (2, 3),
    }

    for reference_id, (component_count, net_count) in expected_counts.items():
        payload = ReferenceService().load_reference(reference_id)
        assert len(payload["components"]) == component_count
        assert len(payload["nets"]) == net_count
        assert payload["source"]["type"] == "dsl_python_v1"
