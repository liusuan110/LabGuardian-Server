# `app.domain.gnn` · GNN-assisted Graph Comparator

**Status: P1 Phase C in progress (splits + resume + CLI driver landed; 12 operators × 4 fixtures = 600 sample / 30 s smoke run green).**
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

### ✅ P1 Phase A — Perturbation framework + dataset_builder
- `perturbation.py` — 4 deterministic perturbation operators built on a
  pluggable `Perturbation` base class + `PERTURBATION_REGISTRY`:
  - `IdentityPerturbation` — perfect cur copy; baseline REF_PRESENT positives
  - `PinSwapSymmetricPerturbation` — swaps two pins within a `symmetry_class_id`
    (e.g., R.pin1↔pin2); produces REF_SYMMETRIC_SWAP positives (label_builder
    sibling expansion validation)
  - `WrongConnectionPerturbation` — reroutes an edge to a different net;
    produces WRONG_OBSERVED strong negatives + MISSING_EDGE wrong_redirect group
  - `PinReversedPerturbation` — anode↔cathode swap on polarized devices
    (LED / Diode / electrolytic Cap)
  - **Mutation pipeline: raw_pin_edges (not nx.Graph).** All 4 operators
    pull `hcg_to_raw_pin_edges(ref)` → mutate the 7-tuple edge list →
    `_rebuild_cur_from_raw` calls `build_hetero_circuit_graph(...,
    raw_pin_edges=...)`. This preserves parallel pins (UA741 pin2 + pin6
    both wired to VOUT) and lets `WrongConnection` collide a new edge
    with a pre-existing one without silent collapse.
  - `hcg_to_nx(hcg)` is still exposed as a back-compat helper but is **no
    longer the perturbation hot path** — it folds parallel edges and is
    kept only for callers (debug tooling, label-builder inspection) that
    explicitly want a flat bipartite view.
- `dataset_builder.py` — orchestration layer enforcing the P0.8 contract:
  - `RefSpec` / `PerturbationPlan` / `DatasetSpec` dataclasses
  - `validate_dataset_spec(spec)` runs **first** in `generate_dataset` and
    raises :class:`DatasetSpecError` for config errors (bad ref path,
    unknown perturbation name, duplicate ref_id, empty refs/plan, negative
    count) — fail-fast **before** any directory creation or sample work
  - `generate_dataset(spec)` runs every (ref, perturbation, idx) combo,
    calls `build_seal_samples_with_coverage_check`, serializes labels JSON
    to `<output>/labels/<ref_id>/<sample_id>.json`, accumulates a
    `LabelManifest`, optional `assert_manifest_healthy` at the end (the
    manifest is always written before the health check so violation
    snapshots survive for diagnosis)
  - `RefSpec.subtype_by_source_id` overrides apply on **both** ref and cur
    build paths; payload-embedded `subtype` is also auto-stashed onto
    `ref_hcg.metadata["subtype_by_source_id"]` so perturbation cur rebuilds
    inherit UA741-style pin specs without the caller threading it manually
  - Per-sample seed via SHA-256 over `(base_seed, ref_id, sample_id)` for
    reproducibility across runs
  - `try/except CoverageError` per sample → failures recorded in manifest,
    pipeline never aborts mid-way
- New fixture: `tests/fixtures/references/test_voltage_divider_v1.json`
  (R1+R2 divider) to exercise multi-component perturbations.

### ✅ P1 Phase B — 8 additional perturbation operators
Same raw_pin_edges pipeline, same identity_alignment + label_builder contract.
`_rebuild_cur_from_raw` extended with `extra_components` / `extra_nets` /
`dropped_components` / `dropped_nets` kwargs so the new ops can add or
remove nodes without bypassing P0.6 materialization.

| Operator | `expected_outcome` | What it does |
|---|---|---|
| `missing_component` | `missing_required` | Drop one low-fanout component from cur. Ref edges from it land in `n_skipped_missing_component`. Falls back to identity on single-component circuits (UA741 buffer). |
| `extra_component` | `wrong_observed` | Inject a parasitic R/C/Wire with both pins on random existing nets → WRONG_OBSERVED on each pin. |
| `floating_net` | `missing_required` | Pick a net of ref-degree ≥ 2, keep one edge, delete the rest → REF_ABSENT_REQUIRED + wrong_redirect groups on affected ports. |
| `short_circuit` | `wrong_observed` | Re-route every edge on net N2 onto N1; N2 stays as isolated cur node so alignment maps cleanly. |
| `power_swapped` | `wrong_observed` | Swap all edges between `role=power` net and `role=ground` net. Fall back to identity if either is absent. |
| `input_output_swapped` | `wrong_observed` | Swap all edges between `role=input` and `role=output` nets. |
| `extra_wire_bridge` | `wrong_observed` | Add a Wire component bridging two nets that share no ref component. Preserves all ref edges + parasitic path. |
| `chained` | most severe of links | Compose 2–3 randomly-chosen Phase B operators in sequence. Each link's cur becomes the next's "ref". |

**Registry now ships 12 operators (4 Phase A + 8 Phase B).** Each has a
deterministic per-seed output and a safe identity fallback when its
preconditions aren't met (e.g. `power_swapped` on a circuit with no VCC).

### ✅ P1 Phase C — Splits + resume + CLI generation driver
- `splits.py` — ref-disjoint train/val/test partitioning per plan §五.
  - `SplitSpec` carries `test_ref_ids` (held-out **whole** ref circuits,
    never sample-level), `val_fraction`, `seed`.
  - `DatasetSplits` stores `("<ref_id>/<sample_id>", ...)` tuples sorted
    deterministically.
  - `discover_samples(labels_dir)` scans the disk layout that
    `generate_dataset` wrote (`<labels>/<ref_id>/<sample_id>.json`).
  - `build_splits(samples_by_ref, spec)` guarantees: train ∩ val ∩ test
    pairwise disjoint, every test sample comes from a `test_ref_id`,
    no train ref ever loses all train samples even at high
    `val_fraction`.
  - `write_splits` / `load_splits` JSON round-trip into
    `<output>/splits/{train,val,test,stats}.json`.
- `dataset_builder.generate_dataset(..., resume=True)` —
  if a `<ref_id>/<sample_id>.json` already exists, deserialize it into
  the manifest and skip regen; corrupted / unreadable files fall through
  to regenerate (logged). Without `resume`, behavior is unchanged
  (truncate + overwrite). Long runs become **idempotent + crash-safe**.
- `dataset_builder.generate_dataset(..., workers=N)` — parallel
  execution via `ProcessPoolExecutor`. Workers receive a picklable
  `_WorkerTask` (with the ref_hcg + spec params), do
  perturbation → label_builder → write the label JSON file → return
  only the `LabelStats` dataclass back to the main process (so the
  inter-process payload stays tiny). Main process aggregates via the
  new `LabelManifest.add_stats(sample_id, stats)` method. Resume runs
  in the main process before any worker dispatch (cheap json replay),
  so `workers=N + resume=True` compose correctly.
  **Cross-process determinism contract**: with the same `(spec,
  base_seed)`, `workers=1` and `workers=N` produce **byte-identical
  label JSON files** for every sample. The contract is enforced by
  `test_workers_gt1_writes_identical_label_payloads_as_serial`.
- `scripts/gnn_generate_dataset.py` — CLI driver that ties everything
  together: spec validation → labels generation (resume-aware) →
  manifest health gate → splits write. Flags:
  - `--output-dir DIR` (required)
  - `--config FILE` (optional; defaults to a built-in MVP config covering
    all 4 fixtures × 12 operators ≈ 600 samples)
  - `--base-seed N`, `--resume`, `--workers N`, `--skip-splits`,
    `--no-healthy`, `--progress`, `--verbose`
  - Exit codes: 0 = ok, 2 = spec validation failed, 3 = manifest health
    gate failed (manifest still on disk for diagnosis), 4 = splits error
  - Smoke-tested end-to-end: 600 samples / 0 failures / pos_neg_ratio ≈
    1.07, train=405 / val=45 / test=150 (opamp held out).

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
  P2 `pyg_converter` reads. File I/O round-trip + 4 edge cases tested.
- `build_seal_samples_with_coverage_check()` — atomic wrapper, raises
  `CoverageError` (subclass of `AssertionError`) when cur edges lack the
  WRONG_OBSERVED 100% coverage invariant. **P1 dataset_builder MUST use
  this wrapper** (not raw `build_seal_samples`) and drop any sample whose
  build raises CoverageError — never write coverage-broken JSON to disk.
- `LabelManifest` (in `label_manifest.py`) — cross-sample running counters,
  per-source / per-task ratios, failure tracking, periodic checkpoints
  (`checkpoint(every=100)`). `assert_manifest_healthy()` guards against
  silent dataset drift before training (e.g., WRONG_OBSERVED dropping to
  zero across an entire run). Ships with a documented dataset_builder
  usage pattern in its docstring.
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
