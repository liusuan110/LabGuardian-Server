# LabGuardian-Server GNN — P5 risk register (plan §九 P5 / §十)

**Reporting date**: 2026-05-18
**Snapshot**: `checkpoints/p3_followup_v2/best_f1.pt` (val F1 0.950, test
F1 0.993 on observed-edge SEAL head)
**Evaluator**: `app.domain.gnn.evaluator.evaluate_split`
(scripts/gnn_eval.py)

Data sources for the numbers below:
- `checkpoints/p5_eval/report.md` — held-out test ref (`opamp_buffer`), 600 samples
- `checkpoints/p5_eval_val/report.md` — in-distribution val (~360 samples)
- `checkpoints/p5_eval_rule_only/report.md` — rule-only baseline (no advisor)

---

## 1 · Headline finding

**Rule comparator's `logic_correct` is more permissive than the
dataset-builder's `is_correct` notion. The mismatch surfaces as a
false_pass_rate of ~0.31 on the held-out test split and ~0.24 on the
val split — both far above plan §八's 0.5% red line.**

| split | n | rule false_pass | rule false_fail | rule accuracy | SEAL F1 (advisory) |
|---|---|---|---|---|---|
| **test** (opamp_buffer, held-out) | 600 | **0.3057** ⚠️ | 0.0000 ✅ | 0.8217 | 0.9953 ✅ |
| **val** (in-distribution) | 360 | **0.2363** ⚠️ | 0.0000 ✅ | 0.7361 | 0.9752 ✅ |

Per plan §一, GNN never overrides `logic_correct`, so
combined_false_pass == rule_false_pass (advisory cannot "save" the gate).

**Update 2026-05-18 — Post R1 (Position B + §6 follow-up + R6)**

After shipping (a) `_critical_extra_items` + `_promote_critical_extras`
in the rule path, (b) `_hcg_to_netlist_v2` adapter so the evaluator
invokes the full production rule path (with `_enrich_result` pin-level
checks), and (c) R6 — `test_all_signal_v1` enrichment + supporting
role_label / role_source / pin_role normalization propagation:

| split | rule false_pass | rule false_fail | rule accuracy | R2 warnings | SEAL F1 |
|---|---|---|---|---|---|
| test | **0.0000 ✅** | **0.0000 ✅** | **1.0000** | 0 | 0.9953 |
| val | **0.0000 ✅** | **0.0000 ✅** | **1.0000** | 0 | 0.9791 |
| test rule-only baseline | **0.0000 ✅** | **0.0000 ✅** | **1.0000** | 0 | n/a |

**Both splits now meet the plan §八 red line (`false_pass ≤ 0.005`)**
with **zero false_fail regression**. Nightly script (`bash
scripts/gnn_eval_nightly.sh`) exits 0, ready for CI wiring.

---

## 2 · Root cause — per-perturbation breakdown

The false_pass rate is **not uniform**. It concentrates on three
perturbation families that the rule path treats as `equivalent_with_extra`
or as topologically isomorphic to ref:

| perturbation | test fp_rate | val fp_rate | reason (rule path verdict) |
|---|---|---|---|
| `extra_component` | **1.0000** | 0.6176 | `_contains_subgraph(cur, ref)` true → match_type=`equivalent_with_extra`, `logic_correct=True` by design |
| `extra_wire_bridge` | — (not in test) | **0.9167** | Same: extra edge between two existing nets still preserves ref subgraph |
| `input_output_swapped` | **1.0000** | 0.6000 | Op-amp pin 2 (out) ↔ pin 6 (out) both connect to VOUT — swap is topologically identical for a non-feedback buffer |
| `floating_net` | 0.4800 | 0.0000 | Test fixture has 4-net cur that retains isomorphism after a non-critical floating net is added |
| `chained` | 0.1333 | 0.1379 | Inherits ~the dominant chained op's behaviour |
| `wrong_connection` | 0.0000 ✅ | — | Rule's missing-edge path catches these |
| `power_swapped` | 0.0000 ✅ | 0.0000 ✅ | Net role check catches VCC↔GND |
| `short_circuit` | 0.0000 ✅ | 0.0000 ✅ | Rule's structural check catches collapsed nets |
| `missing_component` | — | 0.0000 ✅ | Rule's `_missing_items` fires |
| `pin_reversed` | — | 0.0000 ✅ | Polarity check fires |

**Reading**: the rule comparator is **strict on missing-something, lenient
on having-extra-stuff** — exactly the design from
`compare_logical_graphs` (see `equivalent_with_extra` branch). The
dataset, by contrast, labels every deviation as a negative.

This is a **definitional mismatch**, not a regression. P0 / P3 weren't
expected to surface it — only end-to-end evaluation does.

---

## 3 · SEAL head is healthy and would catch these

GNN-advisor SEAL head on the same observed cur edges:

| split | observed edges | AUC | F1 | precision | recall |
|---|---|---|---|---|---|
| test | 3085 | 0.9993 ✅ | 0.9953 ✅ | 0.9907 | 1.0000 |
| val | 2014 | 0.9923 ✅ | 0.9752 ✅ | 0.9521 | 0.9991 |

Total **rule↔GNN disagreements on test**: 113 / 600
  - **79** rule_pass + GNN flags wrong edge (the case that drives
    false_pass → almost all `extra_component` + `input_output_swapped`)
  - **34** rule_fail + GNN sees no wrong edge

The advisor is correctly identifying the cases the rule misses. The
gate is not GNN quality; it is the policy that GNN advice cannot
promote `logic_correct=False`.

**Post-R2 update (2026-05-18)**: those 79 false_pass cases now each
emit a `WARN_GNN_DISAGREES_WITH_RULE` advisory item in
`validator_report_v2.items[*]`. Per-perturbation breakdown shows R2
fires on **100%** of `extra_component` (50/50) and **100%** of
`input_output_swapped` (25/25) — perfect recall on the two
false_pass-driving families. `floating_net` shows 0% R2-warn because
the floating perturbation removes edges rather than adding wrong ones,
which the SEAL head doesn't model. R1 (rule semantics fix) remains
the path to actually closing the red line.

---

## 4 · Runtime (CPU, M-series MacBook)

| stage | mean (ms) | p95 (ms) |
|---|---|---|
| rule comparator (per sample) | 0.19 | 0.35 |
| GNN advise (per sample) | 1.01 | 1.25 |

Plan §十 risk "推理延迟 < 100 ms" is comfortable — single-digit ms with
torch on CPU + ~50 nodes.

---

## 5 · Recommendations (input to P4.1 / P6 plan revision)

| # | Action | Owner | Priority | Status |
|---|---|---|---|---|
| **R1** | Re-examine `equivalent_with_extra` semantics in `compare_logical_graphs` — promote to `logic_correct=False` when extra connects to a role-critical net (vcc/gnd/input/output). Three positions enumerated in [`app/domain/compare/RULE_SEMANTICS.md`](../compare/RULE_SEMANTICS.md). | Rule path | **P0 (red-line breach)** | ✅ Position B shipped (2026-05-18): `_critical_extra_items` + `_promote_critical_extras` in `app/domain/compare/`. **false_pass test 0.3057 → 0.1514, val 0.2363 → 0.0970** (zero false_fail regression). Residual `input_output_swapped` + `floating_net` tracked as Position B follow-up in RULE_SEMANTICS §6. |
| **R2** | When GNN advice has ≥ 1 edge with `p_correct < 0.3` **and** rule says pass, set `report.summary.gnn.disagreement_with_rule=True` + emit a `WARN_GNN_DISAGREES_WITH_RULE` advisory item (does NOT flip `logic_correct`). | Orchestrator (P4.1 §六) | **P0** | ✅ shipped — `_maybe_attach_gnn_advice` (`orchestrator.py`); 3 new tests in `test_p4_inference.py` (20 total); R2 column now in evaluator `report.md` |
| **R3** | Add `gnn_assisted_strict` opt-in mode: `logic_correct = rule AND (no edge below threshold)` — for downstream graders that want zero false_pass at the cost of false_fail. | Orchestrator | P2 | 🧪 experimental — revisit only if R1 (Position B) falls short |
| **R4** | Tighten `input_output_swapped` perturbation to only apply on non-symmetric op-amp roles (or relabel as `expected_outcome="positive"` when symmetric). | Dataset builder | P2 | 🧪 experimental — see RULE_SEMANTICS §4 Q2 |
| **R5** | Add `scripts/gnn_eval_nightly.sh` to CI nightly so any regression in false_pass surfaces immediately. | CI / build | P1 | 🟢 **ready to wire** — both splits now exit 0 (`rule_accuracy=1.0000`); CI integration unblocked. |

---

## 6 · DoD vs plan §九 P5

| plan §九 deliverable | status |
|---|---|
| 离线 evaluator (`evaluator.py`) | ✅ — `app/domain/gnn/evaluator.py` (461 LOC) |
| Ablation table | ✅ — already in `checkpoints/p3_followup_ablation/ablation_report.md` (P3 deliverable) |
| 风险报告 | ✅ — this document |
| `false_pass ≤ 0.5%` gate | **test ✅ 0.0000** + **val ✅ 0.0000** after R1 (Position B + §6 follow-up + R6 + supporting bug-fixes). Both splits exit 0 in the nightly script. |
| `GraphMatcher runtime −50%` | ❌ not measured — requires P4.1 seed-mapping integration (`_find_isomorphism(..., seed_node_mapping=advice.top1_component_mapping)` does not yet pass the seed). Tracked as P4.1 follow-up |

The red-line breach on `false_pass_rate` is **a measurement
artifact of two design choices in tension** (rule's `equivalent_with_extra`
vs dataset's strict outcome labelling), not a regression of the GNN
model. The model itself meets every plan §八 main target on observed
edges (AUC 0.999, F1 0.995). The unblocking work is in the rule
comparator + orchestrator layers (R1 + R2).
