"""Probe: which REAL diff_report error_code does each fault FAMILY emit?

Why: the fault_case KB ``related_error_codes`` + ErrorTagService._CODE_TO_TAG
are tagged with the s5/ERC vocabulary (NODE_MISMATCH / FLOATING_PIN /
COMPONENT_SHORTED_SAME_NET / POLARITY_REVERSED …), but the production
``comparison_report`` comes from ``diff_report`` (reference compare), whose
vocabulary is disjoint except for COMPONENT_MISSING. This probe runs the
*real* comparator on canonical perturbations so we learn the true
fault-family → diff-code map before re-tagging the KB.

Run locally:
    .venv/bin/python -m scripts.kb.probe_diff_codes
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from app.domain.graph_compare import compare_logical_graphs
from app.domain.logical_reference import (
    current_netlist_v2_to_graph,
    logical_reference_to_graph,
)


# ----------------------------------------------------------------------------
# reference payloads (ref format: pins carry "net")
# ----------------------------------------------------------------------------
RC_REF: dict[str, Any] = {
    "format": "logical_reference_v1",
    "reference_id": "rc_first_order_v1",
    "name": "一阶 RC",
    "components": [
        {"ref_id": "R1", "type": "Resistor",
         "pins": [{"pin": "pin1", "net": "VIN"}, {"pin": "pin2", "net": "VC"}]},
        {"ref_id": "C1", "type": "CapacitorCeramic",
         "pins": [{"pin": "pin1", "net": "VC"}, {"pin": "pin2", "net": "GND"}]},
    ],
    "nets": [
        {"net": "VIN", "role": "input"},
        {"net": "VC", "role": "signal"},
        {"net": "GND", "role": "ground"},
    ],
}

POT_REF: dict[str, Any] = {
    "format": "logical_reference_v1",
    "reference_id": "pot_divider_v1",
    "name": "电位器分压",
    "components": [
        {"ref_id": "POT1", "type": "Potentiometer",
         "pins": [{"pin": "terminal_a", "net": "VIN"},
                  {"pin": "wiper", "net": "VOUT"},
                  {"pin": "terminal_b", "net": "GND"}]},
    ],
    "nets": [
        {"net": "VIN", "role": "input"},
        {"net": "VOUT", "role": "output"},
        {"net": "GND", "role": "ground"},
    ],
}


def _load_fixture_ref(name: str) -> dict[str, Any] | None:
    p = Path("tests/fixtures/references") / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# ref → equivalent current netlist_v2 (cur format: pins carry electrical_net_id)
# ----------------------------------------------------------------------------
def ref_to_current(ref: dict[str, Any]) -> dict[str, Any]:
    comps = []
    for c in ref.get("components", []):
        pins = []
        for p in c.get("pins", []):
            if p.get("nc") is True or not p.get("net"):
                continue
            pins.append({
                "pin_name": str(p.get("pin")),
                "electrical_net_id": p["net"],
                "hole_id": f"H_{c['ref_id']}_{p.get('pin')}",
            })
        comp = {"component_id": c["ref_id"],
                "component_type": c.get("type"),
                "pins": pins}
        if c.get("subtype"):
            comp["subtype"] = c["subtype"]
        comps.append(comp)
    nets = []
    for n in ref.get("nets", []):
        net = {"electrical_net_id": n["net"], "role": n.get("role", "signal")}
        if n.get("role_label"):
            net["role_label"] = n["role_label"]
        nets.append(net)
    return {"components": comps, "nets": nets}


def run(ref: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    ref_graph = logical_reference_to_graph(ref)
    cur_graph = current_netlist_v2_to_graph(cur)
    result = compare_logical_graphs(ref_graph, cur_graph, ref_payload=ref, cur_netlist_v2=cur)
    items = result.get("items", []) or result.get("report", {}).get("items", [])
    codes = sorted({str(i.get("error_code")) for i in items if i.get("error_code")})
    return {"logic_correct": result.get("logic_correct"),
            "match_type": result.get("details", {}).get("match_type"),
            "codes": codes}


# ----------------------------------------------------------------------------
# perturbations (operate on a current netlist copy)
# ----------------------------------------------------------------------------
def drop_component(cur, cid):
    cur = copy.deepcopy(cur)
    cur["components"] = [c for c in cur["components"] if c["component_id"] != cid]
    return cur


def add_extra_resistor(cur):
    cur = copy.deepcopy(cur)
    cur["components"].append({"component_id": "X_EXTRA", "component_type": "Resistor",
        "pins": [{"pin_name": "pin1", "electrical_net_id": cur["nets"][0]["electrical_net_id"], "hole_id": "HX1"},
                 {"pin_name": "pin2", "electrical_net_id": "NET_EXTRA", "hole_id": "HX2"}]})
    cur["nets"].append({"electrical_net_id": "NET_EXTRA", "role": "signal"})
    return cur


def repoint_pin(cur, cid, pin_name, new_net):
    cur = copy.deepcopy(cur)
    for c in cur["components"]:
        if c["component_id"] == cid:
            for p in c["pins"]:
                if p["pin_name"] == pin_name:
                    p["electrical_net_id"] = new_net
    if new_net not in {n["electrical_net_id"] for n in cur["nets"]}:
        cur["nets"].append({"electrical_net_id": new_net, "role": "signal"})
    return cur


def merge_nets(cur, keep, drop):
    """Merge net `drop` into `keep` (short)."""
    cur = copy.deepcopy(cur)
    for c in cur["components"]:
        for p in c["pins"]:
            if p["electrical_net_id"] == drop:
                p["electrical_net_id"] = keep
    cur["nets"] = [n for n in cur["nets"] if n["electrical_net_id"] != drop]
    return cur


def set_net_role(cur, net, role):
    cur = copy.deepcopy(cur)
    for n in cur["nets"]:
        if n["electrical_net_id"] == net:
            n["role"] = role
            n.pop("role_label", None)
    return cur


def swap_pins(cur, cid, pin_a, pin_b):
    cur = copy.deepcopy(cur)
    for c in cur["components"]:
        if c["component_id"] == cid:
            pa = next(p for p in c["pins"] if p["pin_name"] == pin_a)
            pb = next(p for p in c["pins"] if p["pin_name"] == pin_b)
            pa["electrical_net_id"], pb["electrical_net_id"] = pb["electrical_net_id"], pa["electrical_net_id"]
    return cur


def line(label, res):
    lc = "OK" if res["logic_correct"] else "ERR"
    print(f"  {label:42s} -> [{lc}] match={str(res['match_type']):28s} codes={res['codes']}")


def main() -> None:
    print("=" * 100)
    print("RC (passive families)")
    base = ref_to_current(RC_REF)
    line("baseline (correct)", run(RC_REF, base))
    line("missing component (drop C1)", run(RC_REF, drop_component(base, "C1")))
    line("extra component (+R)", run(RC_REF, add_extra_resistor(base)))
    line("wrong conn (R1.pin2 -> GND)", run(RC_REF, repoint_pin(base, "R1", "pin2", "GND")))
    line("open (split VC: C1.pin1 -> NET_NEW)", run(RC_REF, repoint_pin(base, "C1", "pin1", "NET_NEW")))
    line("node-role (GND role->input)", run(RC_REF, set_net_role(base, "GND", "input")))

    print("=" * 100)
    print("POT (input+output present → short/critical)")
    pbase = ref_to_current(POT_REF)
    line("baseline (correct)", run(POT_REF, pbase))
    line("short input+output (merge VOUT->VIN)", run(POT_REF, merge_nets(pbase, "VIN", "VOUT")))
    line("wiper swapped (wiper<->terminal_a)", run(POT_REF, swap_pins(pbase, "POT1", "wiper", "terminal_a")))

    op = _load_fixture_ref("test_opamp_inverting_v1.json")
    if op:
        print("=" * 100)
        print("OPAMP inverting (IC pin faults — the 'polarity/pins swapped' family)")
        obase = ref_to_current(op)
        line("baseline (correct)", run(op, obase))
        line("input pins swapped (U1.2<->U1.3)", run(op, swap_pins(obase, "U1", "2", "3")))
        line("Rf missing (feedback open)", run(op, drop_component(obase, "R_f")))
        line("VEE/GND pin (U1.4 -> floats NET_NEW)", run(op, repoint_pin(obase, "U1", "4", "NET_FLOAT")))

    bjt = _load_fixture_ref("test_bjt_diff_amp_v1.json") or _load_fixture_ref("test_npn_switch_v1.json")
    if bjt:
        print("=" * 100)
        print(f"TRANSISTOR (strict-pin-role) — {bjt.get('reference_id')}")
        bbase = ref_to_current(bjt)
        line("baseline (correct)", run(bjt, bbase))
        # swap first transistor's first two connected pins
        tr = next((c for c in bjt["components"] if c.get("type") == "Transistor"), None)
        if tr:
            conn = [p["pin"] for p in tr["pins"] if p.get("net") and not p.get("nc")]
            if len(conn) >= 2:
                line(f"{tr['ref_id']} pins swapped ({conn[0]}<->{conn[1]})",
                     run(bjt, swap_pins(bbase, tr["ref_id"], str(conn[0]), str(conn[1]))))
    print("=" * 100)


if __name__ == "__main__":
    main()
