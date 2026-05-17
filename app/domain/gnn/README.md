# `app.domain.gnn` · GNN-assisted Graph Comparator

**Status: P3.1 ✅ — L1 HeteroConv backbone module + ablation harness landed; baseline 15-epoch run hit val F1 = 0.923, top-3 = 1.000. P3.2 = L1↔SEAL integration + aux heads.**
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

### ✅ P1 acceptance — 4 × 600 = 2400 sample dataset

Generated via `scripts/gnn_generate_dataset.py --workers 4 --base-seed 0`:

| Metric | Value |
|---|---|
| Samples processed | 2400 |
| Failures | 0 |
| pos_neg_ratio | 1.08 |
| Total labeled rows | 33518 |
| Avg rows per sample build | 14.0 |
| Required LabelSources covered | 6/6 |
| Splits | train=1620 / val=180 / test=600 (opamp_buffer held out) |
| Wall time (4 workers) | 0.78 s |

Reproducibility: same `--base-seed` always emits byte-identical label JSON
files (cross-process contract enforced by Phase C parity tests).
Acceptance test (`tests/domain/gnn/test_p1_acceptance.py`) runs a
scaled-down version of the same plan in CI.

### ✅ P2 — PyG converter + dataset loader (plan §二 / §三 / §三.6)
- `pyg_converter.py`
  - `to_hetero_data(hcg)` — HCG → `HeteroData` with three node types
    (`component` / `port` / `net`) and two edge types
    (`(component, has_port, port)` and `(port, connects, net)`). Feature
    dims follow §三 (30 / 50 / 11 / 5). `T.ToUndirected()` adds the
    reverse edges so it feeds a SAGE-style `HeteroConv` directly.
  - `seal_subgraph_to_pyg_data(sg, cur_hcg, label=..., ...)` — one
    `SealSubgraph` → one `Data` with node feature layout
    `DRNL[17] ⊕ port-or-net feat ⊕ target_flag[1]` (rectangular tensor
    padded to `max(PORT_FEAT_DIM, NET_FEAT_DIM)`). Undirected edges
    in the subgraph slice. Anchors at `target_port_idx` /
    `target_net_idx`. Optional string fields (`label_source` /
    `task_type` / `group_id`) always set with `""` sentinels so PyG
    `DataLoader` collation never crashes on heterogeneous batches.
- `pyg_dataset.py`
  - `RefRegistry` — ref_id → payload path + subtype dict with cached
    ref_hcg.
  - `reconstruct_cur_hcg(ref_hcg, cur_metadata, subtypes)` — replays the
    perturbation from the `seed + perturbation_chain[0]` recorded in the
    label JSON. Same seed → byte-identical cur_hcg.
  - `FlatSealDataset(labels_dir, refs, split_entries)` — PyG `Dataset`
    subclass; `__getitem__` returns one PyG `Data` per `SealSample` row.
    Index built from disk JSON headers (cheap). cur_hcg replay cached
    via `functools.lru_cache(maxsize=64)` for hot batching.
  - Direct compatibility with `torch_geometric.loader.DataLoader`:
    `DataLoader(ds, batch_size=16)` yields concatenated batches with
    per-graph `batch` / `ptr` vectors ready to feed the SEAL DGCNN head.

Pyproject extras: `[gnn]` declares `torch>=2.2` + `torch-geometric>=2.5`.
`app.domain.gnn.__init__` guards the PyG imports behind a `try/except
ImportError` so the schema / label_builder / dataset_builder layers stay
importable on boxes without torch installed.

### ✅ P2.5 — SpiceNetlist self-supervised SEAL pretraining
- `spicenetlist_loader.py` — parses GNN-ACLP's JSON dump
  (`<id>.json`, 155 circuits in the public release) into
  `HeteroCircuitGraph` with `side="ref"`. Component-type mapping:
  MOSFET → Transistor (Drain→Collector / Gate→Base / Source→Emitter),
  Inductor → UNKNOWN, BJT / Diode / Voltage / Current / IC → faithful.
  Net "0" is tagged as `role="gnd"` per SPICE convention. Two-pin
  passives (R / C / Ind / Wire) get the same `symmetry_class_id` on
  both pins; polar / multi-pin components get distinct classes.
- `pretrain_dataset.py:SpiceNetlistPretrainDataset` — for each circuit,
  emit one positive SealSubgraph per observed `(port, net)` edge plus
  `negatives_per_positive × N_pos` randomly sampled non-edges. Each
  positive subgraph is extracted with `edge_present=True` so the SEAL
  extractor excludes the anchor edge from the message-passing
  neighbourhood (plan §三.6 SEAL contract). `max_pairs_per_circuit` caps
  per-circuit contribution so dense circuits (237 edges max) don't
  dominate. Returned as plain `torch.utils.data.Dataset`; PyG
  `DataLoader` handles collation.
- `seal_dgcnn.py:SealDGCNN` — plan §四 L2 reference architecture:
  `[in_channels] → GCN(hidden) → GCN(hidden) → GCN(1) tanh + concat`
  per-node descriptor → `global_sort_pool(k=30)` → `Conv1d(kernel=2)` →
  MLP head → scalar logit. Default `hidden=32` keeps the CPU footprint
  small; `predict_prob(model, batch)` applies sigmoid.
- `scripts/gnn_pretrain_seal.py` — 5-fold-by-circuit CV training driver
  with manual `roc_auc` (no sklearn dep). CLI flags:
  `--spicenetlist-json DIR --output-dir DIR --epochs --folds --batch-size
  --lr --hidden --sort-k --num-hops --max-pairs-per-circuit
  --max-circuits --min-auc --cpu --verbose`. Per-fold history + summary
  JSON written to disk. Exit code 3 if mean AUC < `--min-auc`.

**Measured (full 155 circuits, 5-fold CV, 5 epochs / fold, CPU-only)**:

| Fold | val_auc (best) |
|---|---|
| 1 | 1.0000 |
| 2 | 1.0000 |
| 3 | 1.0000 |
| 4 | 1.0000 |
| 5 | 1.0000 |
| **mean** | **1.0000** (≥ plan §九 gate 0.95) |

Wall time: **10.8 s** on a single Mac CPU. Smoke test
(`test_one_epoch_training_decreases_loss`) gates regressions on
gradient flow.

### ✅ P3 MVP — CircuitMatchNet end-to-end training

- `model.py:CircuitMatchNet` — multi-task wrapper around the P2.5
  `SealDGCNN` main head. Returns a dict (`{"seal_logits": [B]}`) so
  later heads can be added without breaking the public interface.
  Two checkpoint methods:
  - `from_pretrained_backbone(p2_5_ckpt, strict=False, override_in_channels=…)`
    loads the SpiceNetlist-pretrained SEAL weights into `seal_head`.
    Rekey: P2.5's flat `SealDGCNN` state_dict → `seal_head.<param>` here.
  - `save(path) / load(path)` for P3 fine-tuned checkpoints; carries
    constructor `config` so the load side reconstructs the right model.
- `scripts/gnn_train_full.py` — end-to-end driver:
  - Loads P1 dataset via `FlatSealDataset(labels_dir, refs, splits)`.
  - Optional `--pretrain-ckpt checkpoints/pretrain_v1/backbone.pt`.
  - Multi-task: trains BCE on **every** sample (both `WRONG_EDGE` and
    `MISSING_EDGE`). At eval time, splits results by `task_type`:
    - **WRONG_EDGE**: F1@0.5 / precision / recall / accuracy / AUC.
    - **MISSING_EDGE**: groups rows by `group_id`, sorts by predicted
      probability, computes top-1 / top-3 / top-5 accuracy.
  - Saves `best_f1.pt` whenever val F1 improves; writes per-epoch
    history + final summary to `summary.json`.
  - Exit code 3 if both val gates fail (`--min-f1`, `--min-top3`).

**Measured (15 epochs, 4 refs × 600 P1 samples, backbone loaded, CPU)**:

| Metric | Train | Val (best) | Test (opamp held out) | Plan §九 gate |
|---|---|---|---|---|
| **WRONG_EDGE F1** | — | **0.923** | 0.619 | ≥ 0.88 ✅ |
| WRONG_EDGE AUC | — | 0.975 | 0.711 | — |
| **MISSING_EDGE top-3** | — | **1.000** | 0.400 | ≥ 0.85 ✅ |
| MISSING_EDGE top-1 | — | 0.818 (peak) | 0.400 | — |
| BCE train loss | 0.29 (ep 14) | 0.24 | 0.88 | — |
| Wall time | ~3 min (CPU) | — | — | — |

Both plan §九 P3 gates met on val. Test (OOD UA741) shows expected
generalisation gap — addressable via plan §十 mitigations (more
training fixtures, domain adaptation, aux heads). Reproduce with:

```sh
python -m scripts.gnn_train_full \
    --dataset-dir datasets/circuit_compare \
    --pretrain-ckpt checkpoints/pretrain_v1/backbone.pt \
    --output-dir checkpoints/p3_v1 \
    --epochs 15 --batch-size 128 --lr 1e-3 \
    --min-f1 0.88 --min-top3 0.85 -v
```

**P3 follow-up scope** (not in MVP):
- L1 shared `HeteroConv` backbone (replaces raw port/net features with
  128-dim embeddings before SEAL DGCNN, per plan §四 L1). **P3.1 ships
  the module** (`backbone.py`) ready for integration; SEAL head wiring
  deferred (changes SEAL input dim 68→146, breaks P2.5 transfer, needs
  sample-level dataloader).
- L4 auxiliary heads: `graph_similarity` / `error_type` / `hotspot` /
  `progress_score`. Requires extending `label_builder` to emit
  graph-level GT.
- Cross-topology test improvement (add more training-time fixtures so
  the UA741 OOD gap closes).

### ✅ P3.1 — L1 backbone module + ablation harness

- `backbone.py:HeteroNodeEncoder` — per-type Linear → tanh →
  `hidden_dim` (default 128), one head each for component / port / net.
- `backbone.py:HeteroSAGEBackbone` — plan §四 L1 reference: 3 stacked
  `HeteroConv(SAGEConv)` layers with residual + per-type LayerNorm +
  ReLU + dropout. Consumes the `HeteroData` from `to_hetero_data` after
  `ToUndirected()` adds reverse edges. Returns `z_comp / z_port / z_net`
  ready to be sliced via `embeddings_for_subgraph(z, port_ids, net_ids,
  hetero_data_node_ids)` for each `SealSubgraph`.
- `--no-drnl` flag on `scripts/gnn_train_full.py` (and matching
  `drop_drnl` kwarg on `seal_subgraph_to_pyg_data` /
  `FlatSealDataset`) — zeros the DRNL one-hot slice so the model
  trains on identical input dims but without DRNL structural signal.
- `scripts/gnn_ablation.py` — orchestrator that drives 3 configs in
  sequence:
  - `baseline` — P2.5 backbone loaded, DRNL on
  - `no_pretrain` — random-init `SealDGCNN`
  - `no_drnl` — P2.5 backbone loaded, DRNL slice zeroed
  - (`no_port` deferred — requires schema rewrite to component-net
    bipartite)
  Each config runs `train_full.main` and the runner aggregates per-config
  val/test metrics into `ablation_report.md` with verdicts:
  `pretrain ≥ +5%` and `DRNL ≥ +3%` against plan §九 expectations.

**Reproduce**:

```sh
python -m scripts.gnn_ablation \
    --dataset-dir datasets/circuit_compare \
    --pretrain-ckpt checkpoints/pretrain_v1/backbone.pt \
    --output-dir checkpoints/p3_ablation \
    --epochs 10 -v
```

Ablation results live in `checkpoints/p3_ablation/ablation_report.md`
after the run completes.

**P3.2 still pending**:
- Wire L1 backbone into a sample-level dataloader → SEAL head
  (replaces raw port/net dims with 128-d L1 embeddings).
- Implement `no_port` ablation (component-net bipartite).
- L4 auxiliary heads (graph_similarity / error_type / hotspot).

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
