# Rule Comparator — Semantics Design Tracker

**Owner**: `app.domain.compare`
**Started**: 2026-05-18 (split out of `app/domain/gnn/RISK_REGISTER.md` R1)
**Status**: 🟡 Design — no code change yet. Discussion + acceptance
criteria below.

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

## 3 · Proposed plan (recommendation, NOT a decision yet)

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
| TBD | curriculum input gathered on §4 open questions | — |
| TBD | Position B prototype + tests | — |
| TBD | nightly re-run; verify false_pass ≤ 0.005 | — |

R3 (`strict_gnn` opt-in mode) and R4 (perturbation realism)
remain **experimental items** — revisit only if Position B falls short.
