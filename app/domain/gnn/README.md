# `app.domain.gnn` · GNN-assisted Graph Comparator

**Status: P0.6 complete (schema + package port materialization).**
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

## Phase progress

### ✅ P0 — Schema foundation
- `graph_schema.py` — node / edge feature dimensions, type enums, polarity
  metadata. Pure Python. No `torch` dependency.
- `hetero_circuit.py` — `HeteroCircuitGraph` middle structure: dataclasses
  for `ComponentNode` / `PortNode` / `NetNode` / `PortConnectsNetEdge`.
- `port_graph.py` — `build_hetero_circuit_graph(nx_graph, side)` and
  convenience wrappers `build_from_logical_reference` /
  `build_from_netlist_v2`. Lifts the existing component-net bipartite graphs
  produced by `app/domain/logical_reference.py` into a port-level
  three-class heterograph.

### ✅ P0.5 — IC + Potentiometer port semantics
- `PortType` expanded **16 → 23** with op-amp / IC roles:
  `INVERTING_INPUT` / `NON_INVERTING_INPUT` / `OUTPUT` / `OFFSET_NULL` /
  `NC` / `V_PLUS` / `V_MINUS`.
- `IC_PIN_MAPS["UA741"]` registry sourced from `app/domain/ic_models`
  (single source of truth for IC pin layouts).
- `normalize_port_type` now subtype-aware: UA741 pin "3" → `non_inverting_input`.
- 22 op-amp pin-name aliases (`in-` / `v+` / `vee` / …) → correct PortType.
- `raw_pin_edges` bypass in `build_from_*` preserves parallel pin edges
  (e.g., UA741 pin2↔pin6 both wired to VOUT in a unity-gain buffer; the
  underlying `nx.Graph` would otherwise collapse them).

### ✅ P0.6 — Package port materialization + symmetry / connection policy
- `ConnectionPolicy` enum: `REQUIRED` / `OPTIONAL` / `FORBIDDEN`.
- `PinSpec` NamedTuple + `PACKAGE_PIN_SPECS` table for 9 component types +
  `IC_PIN_POLICIES` / `IC_PIN_SYMMETRY` overlay for UA741.
- `PortNode` gains `pin_number`, `connection_policy`, `symmetry_class_id`.
- `ComponentNode` gains `pin_symmetry_groups`.
- `port_graph.build_*` now runs a **materialize phase**: for every component
  whose spec is known, any unconnected expected pin (NC pin in DSL, missing
  pin in netlist_v2, `electrical_net_id=None` on cur side) becomes an
  `is_floating=True` PortNode. This is the blocker that had to land
  *before* SEAL — without it, candidate-edge generation can't ask
  "should pin 1 actually be connected somewhere?".
- `PORT_FEAT_DIM` now **50** (was 44): adds 3 connection_policy_one_hot,
  1 has_pin_number, 1 pin_number_log, 1 symmetry_class_size_inverse.

## Port lifecycle (P0.6)

```
                     ┌──────────────────────────────┐
                     │ get_expected_pin_specs(...)  │
                     │   PACKAGE_PIN_SPECS / IC_*    │
                     └──────────────┬───────────────┘
                                    │
        ┌───────────────────────────┴────────────────────────────┐
        ▼                                                        ▼
  Spec has pin                                            Spec missing pin
        │                                                        │
        ▼                                                        ▼
  Pin observed in     ┐                                  ┌  Pin observed in DSL /
  DSL / netlist_v2?   │                                  │  netlist_v2?
        │             │                                  │       │
   ┌────┴────┐        │                                  │  ┌────┴────┐
   ▼         ▼        │                                  │  ▼         ▼
  yes       no        │                                  │ yes       no
   │         │        │                                  │  │         │
   ▼         ▼        │                                  │  ▼         ▼
 connected  floating  │                                  │ connected  *not built*
 PortNode   PortNode  │                                  │ PortNode  (Sensor /
 (edge to   (no edge, │                                  │ (no spec   未知 ctype)
  net)      is_       │                                  │  fields   skipped
            floating  │                                  │  default)
            =True,    │                                  │
            policy    │                                  │
            from spec)│                                  │
                      │                                  │
   ConnectionPolicy ∈ {REQUIRED, OPTIONAL, FORBIDDEN}    │
   matters for missing_connection / wrong_connection     │
                                                          ▼
                                                  best-effort handling
                                                  (legacy compatibility)
```

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
                                       │  (+ raw_pin_edges bypass + materialize phase)
                                       ▼
                          HeteroCircuitGraph (comp + port + net)
                                       │
                                       ▼
                                 (P2) PyG HeteroData
```

## What is **out of scope** for P0 / P0.5 / P0.6

- ❌ `torch` / `torch_geometric` imports (declared as `[gnn]` extras only)
- ❌ Feature vectorisation, padding, tensor packing → P2
- ❌ SEAL enclosing subgraph extraction & DRNL labelling → P0.7
- ❌ Model definition, training, inference → P3 / P4
- ❌ Any modification of `app/domain/compare/` → P4
- ❌ `NetNode.swappable_with` (DSL top-level net swap groups) → P0.7+

## Running tests

```bash
pytest tests/domain/gnn/                       # P0 + P0.5 + P0.6 unit tests
pytest tests/domain/test_graph_compare.py      # rule comparator (zero regression expected)
ruff check app/domain/gnn/ tests/domain/gnn/
mypy app/domain/gnn/
```

Installing the GNN extras (only needed once P2+ lands):

```bash
pip install -e ".[gnn]"
```
