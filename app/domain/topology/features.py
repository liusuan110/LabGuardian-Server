"""Graph → PyG tensor encoder for GNN-A.

## Why a custom encoder (not PyG's built-in transforms)

PyG transforms like ``OneHotDegree`` are generic; we know the **specific
categorical universe** of our bipartite circuit graph (8 component types,
6 net roles, 4 IC subtypes for our demo set), so a hand-crafted encoder
is shorter, faster, and easier to keep in sync with
:mod:`app.domain.topology.labels`.

## Feature layout (per node)

Every node — comp or net — gets the same fixed-width vector. Slots that
don't apply to a node type are zeroed.

::

    [0..1]   kind one-hot          (comp / net)
    [2..9]   component_type        (Resistor, Capacitor, CapacitorCeramic,
                                    IC, Transistor, Diode, LED,
                                    Potentiometer)
    [10..13] component_subtype     (UA741, NPN/BJT, PNP/BJT, other)
    [14..19] net_role              (input, output, power, ground,
                                    signal, unknown)
    [20]     degree (normalized)   used by the model as a structural hint;
                                    purely topological so it's stable
                                    across perturbations.
    [21]     num_resistor_neighbors  (net nodes only — counts adjacent
                                     Resistor components; clipped at 10)
    [22]     num_capacitor_neighbors (net nodes only — counts adjacent
                                     Capacitor/CapacitorCeramic; clipped at 10)

Total: **23 dims**. The two new neighbor-count features (v2) exist to
make the **feedback-element-type discriminator** explicit at the node
level — without them, distinguishing inverting_amp vs integrator
relies entirely on 2-layer message passing extracting "the INV node's
neighbor is a Capacitor vs a Resistor", which proved noisy on real
student boards.

## Version compatibility

* **v1 (21 dims)**: original baseline. ckpts in ``checkpoints/gnn_a_v1/``.
* **v2 (23 dims)**: adds neighbor-count features. ckpts in
  ``checkpoints/gnn_a_v2/``. Old ckpts WILL fail to load (in_dim
  mismatch). Service layer chooses by ckpt directory.

## Determinism

The encoder is stateless and deterministic. Same graph → same tensors.
This matters because training/eval must produce reproducible class ids,
and OpenVINO IR export needs a stable input schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import networkx as nx
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Categorical universes
# ---------------------------------------------------------------------------

_KIND_ORDER: Final[tuple[str, ...]] = ("comp", "net")

_COMP_TYPE_ORDER: Final[tuple[str, ...]] = (
    "Resistor",
    "Capacitor",
    "CapacitorCeramic",
    "IC",
    "Transistor",
    "Diode",
    "LED",
    "Potentiometer",
)

# Keep subtype categories small — Phase 1 demo set uses only these.
# Anything else maps to ``other``. Add new IC families before ``other``
# in Phase 2 (mirrors the append-only contract of TOPOLOGY_LABELS).
_SUBTYPE_ORDER: Final[tuple[str, ...]] = (
    "UA741",
    "NPN/BJT",
    "PNP/BJT",
    "other",
)

_NET_ROLE_ORDER: Final[tuple[str, ...]] = (
    "input",
    "output",
    "power",
    "ground",
    "signal",
    "unknown",
)

FEATURE_DIM: Final[int] = (
    len(_KIND_ORDER)
    + len(_COMP_TYPE_ORDER)
    + len(_SUBTYPE_ORDER)
    + len(_NET_ROLE_ORDER)
    + 1   # degree
    + 2   # v2: num_R_neighbors + num_C_neighbors (net nodes only)
)
assert FEATURE_DIM == 23, "FEATURE_DIM constant out of sync with category tuples"

# Cap neighbor counts so a pathological mega-net doesn't dominate the
# normalized feature. 10 is well above realistic teaching-circuit
# net degrees (most are 2-4).
_NEIGHBOR_COUNT_CLIP: Final[int] = 10


# ---------------------------------------------------------------------------
# Reusable index helpers
# ---------------------------------------------------------------------------


def _index_or_default(value: str | None, order: tuple[str, ...]) -> int:
    """Return the index of ``value`` in ``order``, or ``-1`` if absent."""
    if value is None:
        return -1
    if value in order:
        return order.index(value)
    return -1


def _normalize_subtype(raw: str | None) -> str:
    """Map free-form subtype string to one of :data:`_SUBTYPE_ORDER`."""
    if not raw:
        return "other"
    raw_upper = raw.upper()
    for canonical in _SUBTYPE_ORDER[:-1]:  # skip "other"
        if canonical.upper() == raw_upper or canonical.upper() in raw_upper:
            return canonical
    return "other"


# ---------------------------------------------------------------------------
# Node feature encoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodedGraph:
    """A graph encoded as PyG-ready tensors.

    Attributes:
        x: ``(num_nodes, FEATURE_DIM)`` float feature matrix.
        edge_index: ``(2, num_edges * 2)`` long tensor of edges
            (undirected, both directions stored as PyG convention).
        node_order: List of original node ids in the same order as rows
            of ``x``. Useful for debugging / inverse mapping.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    node_order: list[str]


def encode_node(
    data: dict,
    degree: int,
    max_degree: int,
    num_resistor_neighbors: int = 0,
    num_capacitor_neighbors: int = 0,
) -> list[float]:
    """Build the 23-dim feature vector for one node.

    See module docstring for the slot layout. The two new v2 args
    (``num_resistor_neighbors`` + ``num_capacitor_neighbors``) are
    populated only for net nodes; pass 0 for comp nodes.
    """
    vec = [0.0] * FEATURE_DIM
    cursor = 0

    # [0..1] kind one-hot
    kind = data.get("kind", "")
    kind_idx = _index_or_default(kind, _KIND_ORDER)
    if kind_idx >= 0:
        vec[cursor + kind_idx] = 1.0
    cursor += len(_KIND_ORDER)

    # [2..9] component_type one-hot (only meaningful when kind == "comp")
    if kind == "comp":
        ctype = data.get("ctype")
        ctype_idx = _index_or_default(ctype, _COMP_TYPE_ORDER)
        if ctype_idx >= 0:
            vec[cursor + ctype_idx] = 1.0
    cursor += len(_COMP_TYPE_ORDER)

    # [10..13] component_subtype one-hot
    if kind == "comp":
        subtype = _normalize_subtype(data.get("subtype"))
        subtype_idx = _SUBTYPE_ORDER.index(subtype)
        vec[cursor + subtype_idx] = 1.0
    cursor += len(_SUBTYPE_ORDER)

    # [14..19] net_role one-hot (only meaningful when kind == "net")
    if kind == "net":
        role = data.get("role")
        role_idx = _index_or_default(role, _NET_ROLE_ORDER)
        if role_idx >= 0:
            vec[cursor + role_idx] = 1.0
        else:
            # Fall back to ``unknown`` so the model never sees an all-zero
            # role region (gradient stalls).
            vec[cursor + _NET_ROLE_ORDER.index("unknown")] = 1.0
    cursor += len(_NET_ROLE_ORDER)

    # [20] normalized degree (clipped so max_degree dominates the scaling).
    # Normalize by max_degree (per-graph) so the feature stays in [0, 1]
    # regardless of graph size.
    vec[cursor] = degree / max_degree if max_degree > 0 else 0.0
    cursor += 1

    # [21..22] v2 — neighbor-type counts at net nodes. THIS is the key
    # signal for distinguishing the UA741 three-tribe (inverting / summing
    # / integrator), because the INV node's R-vs-C neighbors directly
    # encode the feedback path's element type.
    if kind == "net":
        r_norm = min(num_resistor_neighbors, _NEIGHBOR_COUNT_CLIP) / _NEIGHBOR_COUNT_CLIP
        c_norm = min(num_capacitor_neighbors, _NEIGHBOR_COUNT_CLIP) / _NEIGHBOR_COUNT_CLIP
        vec[cursor] = r_norm
        vec[cursor + 1] = c_norm

    return vec


def _count_neighbor_ctypes(graph: "nx.Graph", node: str) -> tuple[int, int]:
    """Return ``(num_R_neighbors, num_C_neighbors)`` for a net node.

    Counts Resistor / Capacitor / CapacitorCeramic / CapacitorElectrolytic
    components adjacent to ``node``. Returns ``(0, 0)`` for non-net or
    isolated nodes.
    """
    if graph.nodes[node].get("kind") != "net":
        return (0, 0)
    n_r = 0
    n_c = 0
    for nbr in graph.neighbors(node):
        nbr_data = graph.nodes[nbr]
        if nbr_data.get("kind") != "comp":
            continue
        ctype = nbr_data.get("ctype")
        if ctype == "Resistor":
            n_r += 1
        elif ctype in {"Capacitor", "CapacitorCeramic", "CapacitorElectrolytic"}:
            n_c += 1
    return (n_r, n_c)


def encode_graph(g: nx.Graph) -> EncodedGraph:
    """Convert an ``nx.Graph`` (bipartite student/reference graph) into
    PyG-ready tensors.

    Args:
        g: Graph with the standard ``kind``/``ctype``/``role`` attribute
            schema (as produced by ``logical_reference_to_graph`` or
            ``current_netlist_v2_to_graph``).

    Returns:
        :class:`EncodedGraph` with ``x`` (node features),
        ``edge_index`` (PyG-style coo), and ``node_order``.
    """
    node_order = list(g.nodes())
    node_index = {n: i for i, n in enumerate(node_order)}

    max_degree = max((d for _, d in g.degree()), default=1)

    x_rows: list[list[float]] = []
    for node in node_order:
        data = g.nodes[node]
        degree = g.degree(node)
        n_r, n_c = _count_neighbor_ctypes(g, node)
        x_rows.append(
            encode_node(
                data, degree, max_degree,
                num_resistor_neighbors=n_r,
                num_capacitor_neighbors=n_c,
            )
        )

    if x_rows:
        x = torch.tensor(x_rows, dtype=torch.float32)
    else:
        # Empty graph — produce shape ``(0, FEATURE_DIM)`` so downstream
        # PyG batching and model forward passes don't choke on rank-1
        # tensors. ``torch.tensor([])`` would give shape ``(0,)``.
        x = torch.empty((0, FEATURE_DIM), dtype=torch.float32)

    edges_src: list[int] = []
    edges_dst: list[int] = []
    for u, v in g.edges():
        src_idx = node_index[u]
        dst_idx = node_index[v]
        edges_src.append(src_idx)
        edges_dst.append(dst_idx)
        # Add reverse direction — PyG convention for undirected graphs.
        edges_src.append(dst_idx)
        edges_dst.append(src_idx)

    if edges_src:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return EncodedGraph(x=x, edge_index=edge_index, node_order=node_order)


def encoded_to_pyg_data(encoded: EncodedGraph, label_index: int | None = None) -> Data:
    """Wrap an :class:`EncodedGraph` in a PyG ``Data`` object.

    Args:
        encoded: Result of :func:`encode_graph`.
        label_index: Integer class id; stored in ``data.y``. ``None`` for
            inference-time graphs.
    """
    kwargs = {"x": encoded.x, "edge_index": encoded.edge_index}
    if label_index is not None:
        kwargs["y"] = torch.tensor([label_index], dtype=torch.long)
    return Data(**kwargs)


__all__ = [
    "EncodedGraph",
    "FEATURE_DIM",
    "encode_graph",
    "encode_node",
    "encoded_to_pyg_data",
]
