"""Shared fixtures for ``app.domain.gnn`` P0 unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures"
REFERENCES_DIR = FIXTURE_ROOT / "references"
NETLIST_V2_DIR = FIXTURE_ROOT / "netlist_v2"


def hcg_to_cur_nx(ref_hcg, *, perturbations: list | None = None):
    """Convert a ref-side HeteroCircuitGraph back to a cur-side nx.Graph
    (kind/source_id annotated), optionally apply simple perturbations:

    - ``("drop_edge", port_source_id, net_source_id)``: drop a specific edge
    - ``("add_edge", comp_source_id, port_key, net_source_id, pin_role)``: add
    - ``("swap_pins", comp_source_id, key_a, key_b)``: swap which net each
      pin connects to (legal sym swap for resistors)
    - ``("rename_component", old_source_id, new_source_id)``: rename
    - ``("rename_net", old_source_id, new_source_id)``: rename
    - ``("remove_component", source_id)``: remove component + all its edges
    """

    g = nx.Graph()
    perturbations = perturbations or []
    comp_rename: dict[str, str] = {}
    net_rename: dict[str, str] = {}
    removed_comps: set[str] = set()
    dropped_edges: set[tuple[str, str]] = set()
    # (comp_sid, pkey, net_sid, pin_role, ctype)
    extra_edges: list[tuple[str, str, str, str, str]] = []
    # (comp_sid, key_a, key_b)
    swap_ops: list[tuple[str, str, str]] = []

    for p in perturbations:
        op = p[0]
        if op == "drop_edge":
            dropped_edges.add((p[1], p[2]))
        elif op == "add_edge":
            extra_edges.append((p[1], p[2], p[3], p[4], p[5] if len(p) > 5 else "IC"))
        elif op == "swap_pins":
            swap_ops.append((p[1], p[2], p[3]))
        elif op == "rename_component":
            comp_rename[p[1]] = p[2]
        elif op == "rename_net":
            net_rename[p[1]] = p[2]
        elif op == "remove_component":
            removed_comps.add(p[1])

    def cur_comp(sid: str) -> str:
        return comp_rename.get(sid, sid)

    def cur_net(sid: str) -> str:
        return net_rename.get(sid, sid)

    # Nodes
    for cid, cnode in ref_hcg.components.items():
        if cnode.source_id in removed_comps:
            continue
        g.add_node(
            f"cur_comp:{cur_comp(cnode.source_id)}",
            kind="comp",
            ctype=cnode.ctype,
            source_id=cur_comp(cnode.source_id),
        )
    for nid, nnode in ref_hcg.nets.items():
        g.add_node(
            f"cur_net:{cur_net(nnode.source_id)}",
            kind="net",
            role=nnode.role,
            source_id=cur_net(nnode.source_id),
        )

    # Edges (with pin_swap)
    # Build per-component pin → net dict, then apply swaps in-place.
    pin_to_net: dict[str, dict[str, tuple[str, str]]] = {}
    for e in ref_hcg.edges:
        rp = ref_hcg.ports[e.src_port_id]
        rn = ref_hcg.nets[e.dst_net_id]
        comp = ref_hcg.components[rp.parent_component_id]
        if comp.source_id in removed_comps:
            continue
        if (rp.port_key, rn.source_id) in dropped_edges:
            continue
        pin_to_net.setdefault(comp.source_id, {})[rp.port_key] = (
            rn.source_id,
            rp.port_key,
        )

    for comp_sid, key_a, key_b in swap_ops:
        cm = pin_to_net.get(comp_sid, {})
        if key_a in cm and key_b in cm:
            cm[key_a], cm[key_b] = cm[key_b], cm[key_a]
        elif key_a in cm:
            cm[key_b] = cm.pop(key_a)
        elif key_b in cm:
            cm[key_a] = cm.pop(key_b)

    for comp_sid, port_map in pin_to_net.items():
        for pkey, (net_sid, pin_role) in port_map.items():
            g.add_edge(
                f"cur_comp:{cur_comp(comp_sid)}",
                f"cur_net:{cur_net(net_sid)}",
                pin=pkey,
                pin_role=pin_role,
                comp_type="Resistor",  # overridden by builder if needed
            )

    # Extra (FORBIDDEN_VIOLATED / wrong_observed insertion) — bypass normal mapping
    for comp_sid, pkey, net_sid, pin_role, ctype in extra_edges:
        # Ensure component / net nodes exist
        cnode_id = f"cur_comp:{cur_comp(comp_sid)}"
        nnode_id = f"cur_net:{cur_net(net_sid)}"
        if cnode_id not in g.nodes:
            g.add_node(cnode_id, kind="comp", ctype=ctype, source_id=cur_comp(comp_sid))
        if nnode_id not in g.nodes:
            g.add_node(nnode_id, kind="net", role="signal", source_id=cur_net(net_sid))
        g.add_edge(cnode_id, nnode_id, pin=pkey, pin_role=pin_role, comp_type=ctype)

    return g


@pytest.fixture
def rc_reference_payload() -> dict:
    """Two-component RC circuit (R1 + C1, three nets)."""

    return json.loads((REFERENCES_DIR / "test_rc_v1.json").read_text(encoding="utf-8"))


@pytest.fixture
def led_reference_payload() -> dict:
    """All-signal fixture containing R1 + LED1, useful for polarity tests."""

    return json.loads((REFERENCES_DIR / "test_all_signal_v1.json").read_text(encoding="utf-8"))


@pytest.fixture
def all_reference_payloads() -> dict[str, dict]:
    """Every JSON under tests/fixtures/references/ keyed by stem."""

    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(REFERENCES_DIR.glob("*.json"))
    }


@pytest.fixture
def simple_netlist_v2() -> dict:
    """A minimal NetlistV2 dict (extracted from reference_simple_v4.json)."""

    bundle = json.loads((NETLIST_V2_DIR / "reference_simple_v4.json").read_text(encoding="utf-8"))
    return bundle["netlist_v2"]
