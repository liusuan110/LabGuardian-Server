from __future__ import annotations

from app.domain.net_normalization import normalize_current_netlist
from app.pipeline.stages.s5_semantic_analysis import run_semantic_analysis


def test_manual_merge_preserves_source_ids_and_canonical_name() -> None:
    netlist = {
        "components": [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "NET_001"},
                    {"pin_name": "pin2", "electrical_net_id": "NET_002"},
                ],
            },
            {
                "component_id": "C1",
                "component_type": "CapacitorCeramic",
                "pins": [
                    {"pin_name": "pin1", "electrical_net_id": "NET_003"},
                    {"pin_name": "pin2", "electrical_net_id": "NET_004"},
                ],
            },
        ],
        "nets": [
            {"electrical_net_id": "NET_001"},
            {"electrical_net_id": "NET_002"},
            {"electrical_net_id": "NET_003"},
            {"electrical_net_id": "NET_004"},
        ],
    }

    result = normalize_current_netlist(
        netlist,
        net_merge_assignments=[
            {"source_net_ids": ["NET_002", "NET_003"], "target_canonical_name": "VLP"}
        ],
    )

    assert result["applied_merges"][0]["kept_net_id"] == "NET_002"
    assert len(netlist["nets"]) == 3
    assert netlist["components"][1]["pins"][0]["electrical_net_id"] == "NET_002"
    merged = next(net for net in result["logical_nets"] if net["source_id"] == "NET_002")
    assert merged["canonical_name"] == "VLP"
    assert merged["source_id"] == "NET_002"


def test_wrong_merge_surfaces_erc_power_ground_short_with_canonical_name() -> None:
    netlist = {
        "components": [
            {
                "component_id": "U1",
                "component_type": "IC",
                "pins": [
                    {"pin_name": "vcc", "electrical_net_id": "NET_VCC"},
                    {"pin_name": "gnd", "electrical_net_id": "NET_GND"},
                ],
            }
        ],
        "nets": [
            {"electrical_net_id": "NET_VCC", "role": "power", "role_label": "VCC", "power_role": "VCC"},
            {"electrical_net_id": "NET_GND", "role": "ground", "role_label": "GND", "power_role": "GND"},
        ],
    }

    normalize_current_netlist(
        netlist,
        net_merge_assignments=[
            {"source_net_ids": ["NET_VCC", "NET_GND"], "target_canonical_name": "VCC_GND"}
        ],
    )
    s5 = run_semantic_analysis(netlist)

    errors = s5["wiring_errors"]
    assert any(item["error_code"] == "POWER_GND_SHORT" for item in errors)
    short = next(item for item in errors if item["error_code"] == "POWER_GND_SHORT")
    assert short["source_net_id"] == "NET_VCC"
    assert short["net_id"] == "VCC_GND"
