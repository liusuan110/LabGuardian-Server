"""Self-compare smoke test: every shipped DSL reference must full-isomorphism-match
itself when compiled to a synthetic netlist_v2 representation.

This locks in the DSL → graph_compare flow so refactoring the DSL semantics
cannot silently break end-to-end matching.
"""

from __future__ import annotations

import pytest

from app.domain.compare import compare_logical_graphs
from app.domain.logical_reference import (
    current_netlist_v2_to_graph,
    logical_reference_to_graph,
)
from app.services.reference_service import ReferenceService


def _payload_to_netlist_v2(payload: dict) -> dict:
    """Convert a logical_reference_v1 payload into an equivalent netlist_v2 dict.

    Treats the reference itself as a successfully-detected current circuit.
    """
    nets = []
    for net in payload.get("nets", []) or []:
        net_id = str(net.get("net") or "")
        if not net_id:
            continue
        role = str(net.get("role") or "signal")
        role_label = str(net.get("role_label") or net.get("label") or "")
        entry: dict = {
            "electrical_net_id": net_id,
            "role": role,
            "role_label": role_label,
            "canonical_name": role_label or net_id,
            "aliases": [],
            "member_node_ids": [f"NODE_{net_id}"],
            "member_hole_ids": [],
            "manual_role": role if role != "signal" else None,
            "role_source": "manual_role" if role != "signal" else "default_signal",
        }
        if role == "power":
            entry["power_role"] = role_label if role_label in {"VCC", "VEE", "VDD", "VSS"} else "VCC"
        elif role == "ground":
            entry["power_role"] = "GND"
        else:
            entry["power_role"] = ""
        nets.append(entry)

    components = []
    for comp in payload.get("components", []) or []:
        ref_id = str(comp.get("ref_id") or "")
        ctype = str(comp.get("type") or "")
        pins = []
        for pin in comp.get("pins", []) or []:
            if pin.get("nc") is True:
                continue
            pins.append(
                {
                    "pin_name": str(pin.get("pin") or ""),
                    "electrical_net_id": str(pin.get("net") or ""),
                    "hole_id": "",
                }
            )
        if not pins:
            continue
        components.append(
            {
                "component_id": ref_id,
                "component_type": ctype,
                "polarity": "none",
                "pins": pins,
            }
        )

    return {
        "scene_id": "self_compare",
        "board_schema_id": "synthetic",
        "components": components,
        "nets": nets,
    }


@pytest.fixture(scope="module")
def all_references() -> list[dict]:
    svc = ReferenceService()
    refs = []
    for summary in svc.list_references():
        ref_id = summary["reference_id"]
        refs.append(svc.load_reference(ref_id))
    assert refs, "no references discovered — knowledge/references is empty?"
    return refs


def test_all_dsl_references_self_compare_match(all_references):
    """Each shipped DSL reference matches its own synthetic netlist as is_correct=True."""
    failures = []
    for payload in all_references:
        ref_id = payload["reference_id"]
        ref_graph = logical_reference_to_graph(payload)
        cur_netlist = _payload_to_netlist_v2(payload)
        cur_graph = current_netlist_v2_to_graph(cur_netlist)
        result = compare_logical_graphs(
            ref_graph,
            cur_graph,
            ref_payload=payload,
            cur_netlist_v2=cur_netlist,
        )
        if not result.get("is_correct"):
            failures.append(
                f"{ref_id}: is_correct={result.get('is_correct')}, "
                f"match_type={result.get('details', {}).get('match_type')}, "
                f"similarity={result.get('similarity')}"
            )
    assert not failures, "DSL self-compare regressed:\n" + "\n".join(failures)


def test_all_dsl_references_compile_to_logical_reference_v1(all_references):
    """Every DSL reference compiles to a payload with required fields."""
    for payload in all_references:
        ref_id = payload.get("reference_id")
        assert payload.get("format") == "logical_reference_v1", f"{ref_id}: bad format"
        assert payload.get("source", {}).get("type") == "dsl_python_v1", f"{ref_id}: not DSL-sourced"
        assert isinstance(payload.get("components"), list) and payload["components"], f"{ref_id}: no components"
        # every pin must reference a declared net (or be NC)
        net_names = {n["net"] for n in payload.get("nets", [])}
        for comp in payload["components"]:
            for pin in comp.get("pins", []):
                if pin.get("nc"):
                    continue
                assert pin.get("net") in net_names, (
                    f"{ref_id}: {comp['ref_id']}.{pin['pin']} references "
                    f"undeclared net '{pin.get('net')}'"
                )


def test_dsl_default_signal_nets_omit_role_field(all_references):
    """Internal signal nets (no explicit role) should NOT carry a role field —
    the matcher treats them as default_signal and uses inference."""
    for payload in all_references:
        ref_id = payload["reference_id"]
        for net in payload.get("nets", []):
            # If a net has no role_label and no role, it's an internal signal
            # and the field should simply be absent (or 'signal').
            role = net.get("role")
            if role is not None:
                assert role in {"signal", "input", "output", "power", "ground"}, (
                    f"{ref_id}: net {net['net']} has unexpected role={role!r}"
                )


def test_dsl_description_field_does_not_leak_into_role_label(all_references):
    """Long prose in description= must NOT end up as role_label (regression for
    rc_first_order_v1 where label= used to carry prose like '...input signal...')."""
    for payload in all_references:
        ref_id = payload["reference_id"]
        for net in payload.get("nets", []):
            role_label = net.get("role_label", "")
            if not role_label:
                continue
            # Critical role labels must be short canonical tokens, not sentences.
            assert len(role_label) <= 8, (
                f"{ref_id}: net {net['net']} has role_label={role_label!r} that "
                f"looks like prose; move it to description= instead."
            )
            assert " " not in role_label, (
                f"{ref_id}: net {net['net']} has whitespace in role_label={role_label!r}"
            )
