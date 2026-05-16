# `app.domain.gnn` · GNN-assisted Graph Comparator

**Status: Phase P0 (Schema only).**
Full plan: `~/.claude/plans/labguardian-server-glowing-galaxy.md`.

## What this module is

A **learning-guided** layer that augments the existing rule-based comparator
in `app/domain/compare/`. It will eventually answer two questions for every
`(port, net)` connection in a student's circuit:

1. `P(edge_correct)` — "should this wire be here?"
2. `suggested_target` — if not, "where should this pin be wired instead?"

The module **never decides pass / fail**. The deterministic comparator
remains the source of truth; GNN output flows into `validator_report_v2.gnn`
as advisory hints. See plan §一 (architecture) and §六 (orchestrator
integration) for the contract.

## What ships in P0

- `graph_schema.py` — node / edge feature dimensions, type enums, polarity
  metadata. Pure Python. No `torch` dependency.
- `hetero_circuit.py` — `HeteroCircuitGraph` middle structure: dataclasses
  for `ComponentNode` / `PortNode` / `NetNode` / `PortConnectsNetEdge`.
- `port_graph.py` — `build_hetero_circuit_graph(nx_graph, side)` and
  convenience wrappers `build_from_logical_reference` /
  `build_from_netlist_v2`. Lifts the existing component-net bipartite graphs
  produced by `app/domain/logical_reference.py` into a port-level
  three-class heterograph.

## Relationship to existing code

```
                              app/domain/logical_reference.py
                                       │
            ┌──────────────────────────┴───────────────────────────┐
            ▼                                                      ▼
  logical_reference_to_graph(payload)            current_netlist_v2_to_graph(netlist_v2)
            │                                                      │
            └────────────────── nx.Graph (comp-net bipartite) ─────┘
                                       │
                                       ▼            (this module)
                          app/domain/gnn/port_graph.py
                                       │
                                       ▼
                          HeteroCircuitGraph (comp + port + net)
                                       │
                                       ▼
                                 (P2) PyG HeteroData
```

The lifting is **lossless and 1-to-1**: every `(comp, net)` edge in the
NetworkX graph becomes one new `port` node plus one `(port, net)` edge.
Polarity / pin-role metadata that previously lived on edges is hoisted onto
the new port node — this is the core GNN-ACLP insight (paper: arxiv
2504.10240v5) that downstream link-prediction needs.

## What is **out of scope** for P0

- ❌ `torch` / `torch_geometric` imports (declared as `[gnn]` extras only)
- ❌ Feature vectorisation, padding, tensor packing → P2
- ❌ SEAL enclosing subgraph extraction & DRNL labelling → P0.7
- ❌ Model definition, training, inference → P3 / P4
- ❌ Any modification of `app/domain/compare/` → P4

## Running tests

```bash
pytest tests/domain/gnn/                       # P0 unit tests
pytest tests/domain/test_graph_compare.py      # rule comparator (zero regression expected)
ruff check app/domain/gnn/ tests/domain/gnn/
mypy app/domain/gnn/
```

Installing the GNN extras (only needed once P2+ lands):

```bash
pip install -e ".[gnn]"
```
