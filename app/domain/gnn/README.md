# `app.domain.gnn` · GNN-assisted Graph Comparator

**Status: P0.8 complete (label builder + ref↔cur alignment).**
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

### ✅ P0.8 — Label builder + ref↔cur alignment
- `alignment.py` — `ComponentAlignment` carries ref↔cur `source_id` maps for
  both components and nets. Constructors: `identity_alignment` (same-name
  auto-match) and `alignment_from_dicts` (explicit; for perturbations that
  rename). `to_dict()` / `alignment_from_dict_payload()` round-trip via JSON.
- `label_builder.py` — `build_seal_samples(ref, cur, alignment, ...)` runs
  a deterministic 6-step pipeline:
  1. **Ref-driven WRONG_EDGE positives** + symmetric sibling expansion
     (R.pin1↔pin2 swap → both labeled 1)
  2. **MISSING_EDGE groups** for REQUIRED ports that are `floating` OR
     `wrong_redirect` (correct net is mandatory candidate)
  3. **WRONG_OBSERVED strong negatives** — 100% coverage of cur edges that
     aren't ref-correct; not delegated to random sampling
  4. **FORBIDDEN_VIOLATED** — every actual edge on a FORBIDDEN pin
  5. **FORBIDDEN_NEGATIVE** — N synthetic non-edges per FORBIDDEN pin
     (default N=4)
  6. **NEGATIVE_RANDOM** — fills the `negatives_per_positive` budget,
     avoiding sym-equivalent positives + already-emitted pairs
- Returns `LabelBuildResult(samples, groups, stats)`. `LabelStats` records
  per-source / per-task counts + 5 skip reasons for silent-drift monitoring.
- `assert_observed_edges_covered()` provides the WRONG_OBSERVED 100%
  coverage invariant for P1 dataset sanity checks.
- `serialize_label_build_result()` / `deserialize_label_build_result()`
  define the on-disk JSON schema (v1.0) — P1 `dataset_builder` writes,
  P2 `pyg_converter` reads.
- Performance: full UA741 buffer pipeline < 80 ms (CPU). All pure Python.

### ✅ P0.7 — SEAL enclosing subgraph + DRNL labeling
- `seal_subgraph.py` — GNN-ACLP-style SEAL pipeline. For any candidate
  `(port, net)` edge, extract its h-hop enclosing subgraph on the bipartite
  port↔net graph (h=2 by default), drop the candidate edge from BFS and
  output edge list (SEAL convention), and label every node with its
  Double-Radius Node Labeling.
  - `SealSubgraph` dataclass (immutable; deterministic node order)
  - `extract_seal_subgraph(hcg, port_id, net_id)` — single edge
  - `extract_subgraphs_for_observed_edges(hcg)` — feed wrong-edge head
  - `extract_subgraphs_for_floating_ports(hcg, candidate_nets=None, *, policies=frozenset({REQUIRED}), include_same_component_edges=False)` — feed
    suggested-target head. **Default policies = `{REQUIRED}` only**
    (OPTIONAL pins like UA741 offset_null legitimately stay floating, so
    including them would inject systemic P1 label noise; opt-in via
    `policies=frozenset({REQUIRED, OPTIONAL})`). FORBIDDEN pins (UA741
    pin 8 NC) only appear when explicitly listed in `policies`.
  - `SealSubgraph.same_component_edges` — schema slot reserved for "same
    IC / same BJT pin pair" structural edges; default empty. Enabled via
    `include_same_component_edges=True` on either extractor. **Does not
    affect DRNL distances** (DRNL stays bipartite-only); P2 / P3 decides
    whether to consume.
- DRNL formula matches Zhang & Chen 2018: `1 + min(d_u, d_v) + d_half * (d_half + (d % 2) - 1)` with `d = d_u + d_v` and `d_half = d // 2`. Anchors → label 1; unreachable → label 0.
- Performance budget (`plan §三.6` / DoD): 50 candidate edges < 30 ms on CPU. Measured: < 1 ms (synthetic 50-edge chain).
- **Still no `torch`** — output is plain Python dataclasses; tensor packing belongs to P2 PyG converter.

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
- ❌ Model definition, training, inference → P3 / P4
- ❌ Any modification of `app/domain/compare/` → P4
- ❌ `NetNode.swappable_with` (DSL top-level net swap groups) → P0.7+

## Running tests

```bash
pytest tests/domain/gnn/                       # P0 + P0.5 + P0.6 + P0.7 unit tests
pytest tests/domain/test_graph_compare.py      # rule comparator (zero regression expected)
ruff check app/domain/gnn/ tests/domain/gnn/
mypy app/domain/gnn/
```

Installing the GNN extras (only needed once P2+ lands):

```bash
pip install -e ".[gnn]"
```
