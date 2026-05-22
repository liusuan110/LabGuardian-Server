"""Topology-preserving graph perturbations for GNN-A dataset generation.

## Why graph-level (not netlist-level) perturbations

Both ``logical_reference_to_graph`` and ``current_netlist_v2_to_graph``
emit the same bipartite ``nx.Graph`` shape (comp/net nodes + pin-labeled
edges). Working at the graph level means:

  * A single perturbation implementation covers both reference DSL seeds
    and synthetic netlists without format conversion.
  * Perturbations compose: a chain ``[rename_nodes → swap_pins → add_wire]``
    is just function composition over the graph.
  * Adding a perturbation later only requires implementing
    ``perturb(g: nx.Graph) -> nx.Graph``.

## Topology-preserving contract

Every perturbation in this module must satisfy::

    label_of(perturb(g)) == label_of(g)

That is, the canonical topology label of the perturbed graph is **the
same** as the seed. Perturbations that change the topology (e.g. swapping
a feedback resistor for a capacitor) belong in a different module —
those are "negative-mining" / "error-injection" perturbations and are
explicitly out of scope for Phase 1 (GNN-A is trained to recognize
intent, not to detect specific errors).

## Determinism

All perturbations accept a ``rng: random.Random`` so dataset builds are
deterministic given a seed. Tests rely on this for reproducibility.
"""

from __future__ import annotations

import copy
import random
import string
from typing import Callable

import networkx as nx


# Component types that participate in the bipartite spec graph.
_PASSIVE_TWO_PIN_TYPES = {"Resistor", "Capacitor", "CapacitorCeramic"}
_WIRE_TYPE = "Wire"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_id(rng: random.Random, prefix: str, length: int = 6) -> str:
    """Generate a short random alphanumeric id with a prefix.

    Used by ``rename_components`` and ``rename_nets`` to avoid collisions
    with the seed graph's existing ids.
    """
    suffix = "".join(rng.choices(string.ascii_uppercase + string.digits, k=length))
    return f"{prefix}_{suffix}"


def _enumerate_comp_nodes(g: nx.Graph) -> list[str]:
    return [n for n, d in g.nodes(data=True) if d.get("kind") == "comp"]


def _enumerate_net_nodes(g: nx.Graph) -> list[str]:
    return [n for n, d in g.nodes(data=True) if d.get("kind") == "net"]


# ---------------------------------------------------------------------------
# Perturbation: rename components
# ---------------------------------------------------------------------------


def rename_components(g: nx.Graph, rng: random.Random) -> nx.Graph:
    """Replace every component node id with a random ``cur_comp:R_XXXXXX``.

    The graph topology, node attributes, and edge attributes are all
    preserved — only the string ids change. This simulates the fact that
    real student boards use arbitrary component identifiers.
    """
    mapping = {
        node: f"cur_comp:{_random_id(rng, 'R')}" for node in _enumerate_comp_nodes(g)
    }
    return nx.relabel_nodes(g, mapping, copy=True)


# ---------------------------------------------------------------------------
# Perturbation: rename nets
# ---------------------------------------------------------------------------


def rename_nets(g: nx.Graph, rng: random.Random) -> nx.Graph:
    """Replace every net node id with a random ``cur_net:NET_XXXXXX``.

    The net's ``role`` / ``role_label`` / ``canonical_name`` attributes
    are preserved — the matcher and downstream code rely on these for
    semantic recognition, not on the id itself.
    """
    mapping = {
        node: f"cur_net:{_random_id(rng, 'NET')}" for node in _enumerate_net_nodes(g)
    }
    return nx.relabel_nodes(g, mapping, copy=True)


# ---------------------------------------------------------------------------
# Perturbation: swap pin1/pin2 on passive 2-pin components
# ---------------------------------------------------------------------------


def swap_passive_pin_assignments(g: nx.Graph, rng: random.Random, p: float = 0.5) -> nx.Graph:
    """For each passive 2-pin component, with probability ``p`` swap its
    two edges' ``pin`` attributes.

    Passive resistors and non-polar capacitors are pin-order agnostic,
    so swapping pin1/pin2 yields an electrically equivalent topology.
    The matcher's ``PASSIVE_TWO_PIN_TYPES`` short-circuits edge_match
    for these components, so this perturbation is invisible to template
    matching but adds noise to the GNN's input.
    """
    g = g.copy()
    for comp_node in _enumerate_comp_nodes(g):
        ctype = g.nodes[comp_node].get("ctype")
        if ctype not in _PASSIVE_TWO_PIN_TYPES:
            continue
        if rng.random() > p:
            continue
        edges = list(g.edges(comp_node, data=True))
        if len(edges) != 2:
            continue  # not a typical 2-pin component
        pin_a = edges[0][2].get("pin")
        pin_b = edges[1][2].get("pin")
        pin_role_a = edges[0][2].get("pin_role")
        pin_role_b = edges[1][2].get("pin_role")
        # Reassign the pin attrs.
        # Edge endpoints (comp_node, net_node) are unchanged.
        net_a = edges[0][1]
        net_b = edges[1][1]
        g[comp_node][net_a]["pin"] = pin_b
        g[comp_node][net_a]["pin_role"] = pin_role_b
        g[comp_node][net_b]["pin"] = pin_a
        g[comp_node][net_b]["pin_role"] = pin_role_a
    return g


# ---------------------------------------------------------------------------
# Perturbation: add a wire jumper on an existing net
# ---------------------------------------------------------------------------


def add_decoration_wire(g: nx.Graph, rng: random.Random) -> nx.Graph:
    """Insert a redundant ``Wire`` component bridging two pins on the
    *same* net.

    A wire that stays inside one net is electrically a no-op — the net is
    still the same equipotential node. But it shows up in the graph as
    an extra ``Wire`` node with 2 edges into the same net, which is
    exactly how messy student boards look when they "double up" jumpers.

    Topology preservation: trivially holds because no net is split or merged.
    """
    g = g.copy()
    net_nodes = _enumerate_net_nodes(g)
    if not net_nodes:
        return g
    # Pick a net that has at least one pin attached — otherwise the wire
    # would dangle.
    candidate_nets = [n for n in net_nodes if g.degree(n) > 0]
    if not candidate_nets:
        return g
    target_net = rng.choice(candidate_nets)
    wire_id = f"cur_comp:W_{_random_id(rng, '')[2:]}"  # short, looks like real student board ids
    g.add_node(wire_id, kind="comp", ctype=_WIRE_TYPE, subtype=None)
    g.add_edge(
        wire_id,
        target_net,
        pin="pin1",
        pin_role="pin1",
        comp_type=_WIRE_TYPE,
    )
    g.add_edge(
        wire_id,
        target_net,
        pin="pin2",
        pin_role="pin2",
        comp_type=_WIRE_TYPE,
    )
    return g


# ---------------------------------------------------------------------------
# Perturbation: add multiple decoration wires
# ---------------------------------------------------------------------------


def add_multiple_decoration_wires(
    g: nx.Graph,
    rng: random.Random,
    min_count: int = 1,
    max_count: int = 3,
) -> nx.Graph:
    """Convenience wrapper: apply :func:`add_decoration_wire` 1-3 times."""
    n = rng.randint(min_count, max_count)
    for _ in range(n):
        g = add_decoration_wire(g, rng)
    return g


# ---------------------------------------------------------------------------
# Perturbation chain
# ---------------------------------------------------------------------------


# Registry of all available perturbations. Used by
# :func:`apply_random_chain` to sample chains and by
# :func:`scripts.cadx.build_topology_dataset` to compose dataset variants.
PERTURBATIONS: dict[str, Callable[[nx.Graph, random.Random], nx.Graph]] = {
    "rename_components": rename_components,
    "rename_nets": rename_nets,
    "swap_passive_pin_assignments": swap_passive_pin_assignments,
    "add_decoration_wire": add_decoration_wire,
    "add_multiple_decoration_wires": add_multiple_decoration_wires,
}


def apply_random_chain(
    g: nx.Graph,
    rng: random.Random,
    chain_length: int = 3,
    perturbation_names: list[str] | None = None,
) -> tuple[nx.Graph, list[str]]:
    """Sample and apply a chain of ``chain_length`` perturbations.

    Args:
        g: Seed graph to perturb.
        rng: Random source for sampling.
        chain_length: Number of perturbations to compose. Each is sampled
            with replacement from ``perturbation_names``.
        perturbation_names: Pool of perturbation names to sample from.
            Defaults to **all** registered perturbations.

    Returns:
        Tuple of ``(perturbed_graph, names_applied_in_order)``. The names
        list is stored in dataset metadata for traceability.
    """
    if perturbation_names is None:
        perturbation_names = list(PERTURBATIONS.keys())
    out = g
    applied: list[str] = []
    for _ in range(chain_length):
        name = rng.choice(perturbation_names)
        fn = PERTURBATIONS[name]
        out = fn(out, rng)
        applied.append(name)
    return out, applied


__all__ = [
    "PERTURBATIONS",
    "add_decoration_wire",
    "add_multiple_decoration_wires",
    "apply_random_chain",
    "rename_components",
    "rename_nets",
    "swap_passive_pin_assignments",
]


# Silence unused-import guard from copy (kept available for future
# perturbations that need deep copies of complex attributes).
_ = copy
