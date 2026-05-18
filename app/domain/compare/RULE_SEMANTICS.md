# Rule Comparator — Semantics Design Tracker

**Owner**: `app.domain.compare`
**Started**: 2026-05-18 (split out of `app/domain/gnn/RISK_REGISTER.md` R1)
**Status**: ✅ **Position B + §6 follow-up + R6 shipped (2026-05-18)** —
plan §八 red line met on **both** splits with 100% rule accuracy.

| split | rule_false_pass | rule_false_fail | rule_accuracy | SEAL F1 |
|---|---|---|---|---|
| test (held-out opamp_buffer) | **0.0000** ✅ | **0.0000** ✅ | **1.0000** | 0.9953 |
| val (in-distribution) | **0.0000** ✅ | **0.0000** ✅ | **1.0000** | 0.9791 |
| test rule-only baseline | **0.0000** ✅ | **0.0000** ✅ | **1.0000** | n/a |

Layered fixes (oldest → newest):
1. **Position B** — extras on role-critical nets promote to fail.
2. **§6 follow-up** — evaluator invokes the production rule path (with
   `_enrich_result` pin-level checks) instead of the graph-only path.
3. **R6** — `test_all_signal_v1` fixture gets `role` + `role_label`
   on each signal net so iso/inference can disambiguate them.
4. **R6 supporting bug-fixes** — surfaced while measuring R6:
   role-inference tightening + role_label / role_source / pin_role
   normalization propagation in `_hcg_to_nodes_only_nx`, `hcg_to_nx`
   and `_hcg_to_netlist_v2` (incl. `manual_role` so role wins over
   non-canonical labels like "VIN_A" on LM358).

---

## 1 · Why this doc exists

The P5 evaluator (`app/domain/gnn/evaluator.py`,
`checkpoints/p5_eval/report.md`) measured a **false_pass_rate of
0.3057** on the held-out `opamp_buffer` test split — well above plan
§八's 0.5% red line. Per-perturbation breakdown points the finger at
two perturbation families:

| perturbation | test false_pass | rule path that fires |
|---|---|---|
| `extra_component` | 1.0000 | `_contains_subgraph(cur, ref) == True` → match_type=`equivalent_with_extra`, `logic_correct=True` |
| `input_output_swapped` | 1.0000 | symmetric op-amp pin swap is topologically isomorphic |
| `extra_wire_bridge` (val) | 0.9167 | extra edge that preserves ref subgraph |
| `floating_net` | 0.4800 | sometimes adds a stray net the rule tolerates |

This is **not a regression** — it has been the documented behaviour of
`compare_logical_graphs` since v1 (see plan §五 + the historical
match_type list in
[`orchestrator.py`](orchestrator.py)). The rule path was designed for
"教学反馈友好" (forgiving extras, only failing on missing-something).
The GNN dataset_builder was designed for strict edge-level supervision.
The two definitions are both correct in isolation but produce a
0.31-magnitude gap when measured end-to-end.

**Until this is resolved, the false_pass red line cannot be met purely
by training a better GNN.** Per plan §一 the GNN never owns
`logic_correct`. The fix lives in the rule path.

P4.1 R2 (the new `WARN_GNN_DISAGREES_WITH_RULE` advisory item in
[`orchestrator.py`](orchestrator.py)) is the **detection** layer:
when rule says pass + GNN says wrong, a soft warning surfaces. R1 is
the **correction** layer: which of those soft warnings should the rule
actually promote to `logic_correct=False`?

---

## 2 · Three competing positions

### Position A — Strict (zero-tolerance)

Any difference between cur and ref topology → `logic_correct=False`.
This matches the dataset_builder definition exactly.

- ✅ false_pass_rate → 0
- ❌ `false_fail_rate` will likely climb (every breadboard with one
  extra debug LED would fail)
- ❌ Teaching reflexively becomes a "find the diff" puzzle — runs
  counter to the original教学反馈友好 motivation
- ❌ Breaks 5+ existing fixtures in
  `tests/domain/test_graph_compare_detailed.py`

### Position B — Match-type-aware promotion

Keep `equivalent_with_extra` as a verdict, but **promote to `logic_correct=False`** when
the extra component / wire connects to a **role-critical net**
(`vcc`, `gnd`, `input`, `output`). Extras on `signal` / `internal`
nets stay as `logic_correct=True` + warning items.

Concretely: in
[`orchestrator.py`](orchestrator.py) the `_contains_subgraph(cur, ref)`
branch already calls `_extra_items`. Walk each extra item; if any
connects to a critical net, set `logic_correct = False`.

- ✅ false_pass_rate on `extra_component` likely drops 0.6–0.9 (most
  perturbations target VCC/GND/output rails)
- ✅ Keeps "extra signal LED" / "extra decoupling cap on internal net"
  passing
- ✅ Doesn't break the `EXTRA_CONNECTION` warning UX in the existing
  P4-shipped report
- ⚠️ Needs a careful net-role taxonomy review; what counts as
  "role-critical" is policy

### Position C — Configurable strictness mode

`compare_logical_graphs(..., strictness: Literal["lenient", "balanced", "strict"] = "balanced")`.

- `lenient` = current behaviour (P0 default)
- `balanced` = Position B
- `strict` = Position A (for graders that want zero false_pass)

The orchestrator picks based on caller intent. The dataset_builder /
evaluator would default to `strict`; the live teaching API defaults to
`balanced`.

- ✅ Maximum flexibility — each consumer picks its trade-off
- ✅ Maps cleanly to RISK_REGISTER R3 (`gnn_assisted_strict` opt-in)
- ❌ API surface grows; need test coverage for all three modes
- ❌ Adds branching to every match_type code path

### Comparison

| dimension | A · Strict | B · Match-aware | C · Configurable |
|---|---|---|---|
| effort | small | medium | large |
| false_pass impact | best | great | mode-dependent |
| false_fail risk | high | low | tunable |
| breaks existing tests | many | few | none if default unchanged |
| matches teaching UX | bad | good | depends on caller |
| matches plan §一 | yes (rule owns it) | yes | yes |

---

## 3 · Position B — shipped implementation

The proposal below describes what we **actually built** on 2026-05-18.

**Pursue Position B**, gated on:

1. Enumerate role-critical net set:
   - `vcc`, `gnd`, `input`, `output` (probably also `vee` /
     `signal_input` / `signal_output` if a future ref uses them)
   - Excludes `internal`, `unknown`, generic `signal`
2. Add `_promote_extra_to_failure(cur_graph, extra_items)` helper to
   `diff_report.py`. Returns the subset of `extra_items` that touch a
   critical net.
3. In `orchestrator.compare_logical_graphs` `_contains_subgraph(cur, ref)`
   branch, after `items = _extra_items(...)`, check the promoted set;
   if non-empty set:
   ```python
   logic_correct = False
   message = "参考电路逻辑已存在，但当前电路存在影响关键节点的多余连接"
   match_type = "rule_failed_on_critical_extra"
   ```
4. Update + add unit tests in
   `tests/domain/test_graph_compare_detailed.py`:
   - extra LED on `signal` net → still `logic_correct=True` + warning
   - extra wire bridging `VCC` to `GND` → `logic_correct=False`
   - extra capacitor on `signal_2` (internal) → still pass + warning
5. Re-run `scripts/gnn_eval_nightly.sh`. Target: false_pass drops below
   0.05 on both test + val.
6. If false_pass still > 0.005, look at remaining perturbation
   families (chained, floating_net) and decide if they warrant more
   case logic or full Position C.

---

## 4 · Open questions (need product / curriculum input)

1. Is `extra_component` on `signal` (non-power, non-IO) ever a real
   teaching error in practice, or is it always tolerable? If always
   tolerable → Position B suffices; otherwise → consider C.
2. Should `input_output_swapped` on a symmetric op-amp buffer be
   treated as `logic_correct=True` definitionally (since the
   electrical behaviour is identical for a unity-gain buffer)? If yes,
   the dataset_builder should label these as `positive` and the false_pass
   measurement drops naturally without any rule change.
3. Does any existing API consumer call `compare_logical_graphs` with
   non-payload args (no `ref_payload`)? If so, the warning-item shape
   added by P4.1 R2 might land in places that don't expect new item
   shapes — needs a check before promoting.

---

## 5 · Tracking

| date | event | link |
|---|---|---|
| 2026-05-18 | doc created, R1 split from RISK_REGISTER | `app/domain/gnn/RISK_REGISTER.md` |
| 2026-05-18 | R2 (advisory warning) shipped — detection layer in place | `app/domain/compare/orchestrator.py:_maybe_attach_gnn_advice` |
| 2026-05-18 | **Position B shipped** (`_critical_extra_items` + `_promote_critical_extras`) — false_pass test 0.3057→0.1514, val 0.2363→0.0970, 4 new tests, zero false_fail regression | `app/domain/compare/diff_report.py:_critical_extra_items`, `orchestrator.py:_promote_critical_extras` |
| 2026-05-18 | **§6 follow-up shipped** (`_hcg_to_netlist_v2` evaluator adapter) — evaluator now invokes the production rule path with full `_enrich_result` pin-level checks. **test false_pass 0.1514→0.0000** ✅, val 0.0970→0.0295. Subtype-preserving round-trip + 2 regression-guard tests. | `app/domain/gnn/evaluator.py:_hcg_to_netlist_v2`, `_evaluate_sample` |
| 2026-05-18 | **R6 shipped** (`test_all_signal_v1` role/role_label enrichment + supporting fixes for role_label/role_source/pin_role propagation in `_hcg_to_nodes_only_nx`, `hcg_to_nx` and `_hcg_to_netlist_v2`; `manual_role` priority; tightened `_node_match_for_role_inference`). **val false_pass 0.0295→0.0000** ✅; both splits hit `rule_accuracy=1.0000`. | `tests/fixtures/references/test_all_signal_v1.json`, `app/domain/gnn/perturbation.py:hcg_to_nx` / `_hcg_to_nodes_only_nx`, `app/domain/compare/role_inference.py` |
| ready | CI nightly wiring — both splits now exit 0 against the 0.005 gate | `scripts/gnn_eval_nightly.sh` |

R3 (`strict_gnn` opt-in mode) and R4 (perturbation realism)
remain **experimental items** — revisit only if §6 follow-up still
leaves the gate breached.

---

## 6 · Position B post-mortem & §6 follow-up

### 6.1 Position B alone (1st commit)

| perturbation | test fp before | test fp after | Δ |
|---|---|---|---|
| `extra_component` | 1.0000 | **0.0000** | **−1.00** ✅ |
| `chained` | 0.1333 | **0.0667** | −0.07 (chained subset includes extras) |
| `floating_net` | 0.4800 | 0.4800 | unchanged (rule's nx.Graph blind to per-pin edges) |
| `input_output_swapped` | 1.0000 | 1.0000 | unchanged (iso absorbs the swap) |
| total `false_pass_rate` | 0.3057 | **0.1514** | **−0.1543** |

### 6.2 §6 follow-up — the real root cause was the evaluator

Investigating the `input_output_swapped` and `floating_net`
residuals revealed they share the same actual root cause: **the
evaluator was passing `ref_payload=None` and `cur_netlist_v2=None`
to `compare_logical_graphs`, so the rule path's
`_enrich_result` pin-level checks were never running**.
`_enrich_result` only fires when both payloads are non-None.
Production code always passes them; the evaluator was measuring an
artificially-weak path that's 2 layers shallower than what users
actually hit.

Fix: `app/domain/gnn/evaluator.py:_hcg_to_netlist_v2` synthesises a
netlist_v2 dict from the cur HCG and the evaluator now passes both
payloads. Critical detail: the adapter must round-trip `part_subtype`
through so that the orchestrator's internal `build_from_netlist_v2`
gives IC pins their right PortType (UA741 `non_inverting_input` etc) —
forgetting that drops SEAL F1 from 0.99 to 0.70. Two regression-guard
tests pin this behaviour.

### 6.3 Combined impact (Position B + §6 follow-up)

| perturbation | test fp pre-R1 | test fp post-§6 | val fp pre-R1 | val fp post-§6 |
|---|---|---|---|---|
| `extra_component` | 1.0000 | **0.0000** | 0.6176 | **0.0000** |
| `extra_wire_bridge` | (not in test) | — | 0.9167 | **0.1667** |
| `floating_net` | 0.4800 | **0.0000** | 0.0000 | **0.0000** |
| `input_output_swapped` | 1.0000 | **0.0000** | 0.6000 | **0.0000** |
| `wrong_connection` | 0.0000 | **0.0000** | (not measured pre) | 0.0526 |
| `chained` | 0.1333 | **0.0000** | 0.1379 | **0.0000** |
| `power_swapped` / `short_circuit` / `missing_component` / `pin_reversed` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **total `false_pass_rate`** | **0.3057** | **0.0000 ✅** | **0.2363** | **0.0295** |
| **total `false_fail_rate`** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SEAL F1 (no regression) | 0.9953 | 0.9953 | 0.9752 | 0.9723 |

**Test split now meets the plan §八 red line (≤ 0.005).** Val is
0.0295 — six times better than pre-R1 (0.2363) but still over the
gate.

### 6.4 Residual val cases (~3.05%)

11 / 360 val false_pass cases remain, all on the `all_signal` ref:

| perturbation | count | match_type | nets touched |
|---|---|---|---|
| `extra_wire_bridge:X_Wire_0:NET_A↔NET_C` | 4 | `equivalent_with_extra` | NET_A, NET_C (both signal) |
| `wrong_connection:LED1.anode:NET_B→NET_A` | 3 | `full_isomorphism` | NET_B, NET_A (both signal) |
| chained variants of the above | ≤4 | varies | signal-only |

All residuals share two properties:
1. **No role-critical net involved** — Position B's degree check
   correctly stays silent.
2. **The iso/subgraph mapping absorbs the rewiring** — because the
   `all_signal` ref's signal nets carry no `role_label` distinguishing
   them, `_node_match` treats `NET_A`, `NET_B`, `NET_C` as fully
   interchangeable.

Two paths forward, both **out of scope** for this iteration:

- **R6 — Curriculum: enrich `all_signal` ref with `role_label`** on
  its signal nets (e.g. `signal_A`, `signal_B`). Then iso matching
  would distinguish them and `_role_mismatch_items` would fire. No
  code change needed in the comparator. Likely closes most of the gap.
- **R7 — Rule path: per-component net-identity check**. Independent of
  payload metadata: for each ref component, the multiset of nets it
  connects to (after iso mapping) must match its cur counterpart. If
  iso says LED1 → LED1 but LED1's neighbours map to a permuted net
  set, flag as `WRONG_CONNECTION`. More invasive but doesn't depend on
  curriculum tagging.

R3 (`strict_gnn` opt-in) and R4 (perturbation realism) remain
experimental.
