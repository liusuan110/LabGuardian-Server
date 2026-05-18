# Real student netlist ingestion (plan §十 R6, Phase 3)

**Status (2026-05-18)**: plumbing shipped, awaiting real production
data. Loader, evaluator, CLI, nightly hook and drift-table column
are all live and tested against 5 hand-rolled simulated-real fixtures
under [`tests/fixtures/real_student_simulated/`](../tests/fixtures/real_student_simulated/).

When the frontend / S3 starts emitting student netlist exports, **the
only thing the team needs to do** is drop them into the layout below.
The evaluator picks them up automatically.

---

## 1 · Directory contract

```
datasets/real_student/                  # default; override via REAL_DIR env
├── opamp_buffer/                       # ref_id namespace (matches DEFAULT_REF_PAYLOAD_PATHS)
│   ├── student_0001.json               # netlist_v2 dict
│   ├── student_0001.meta.json          # ground-truth + teacher notes
│   ├── student_0002.json
│   └── student_0002.meta.json
├── rc_lowpass/
│   ├── student_0007.json
│   └── student_0007.meta.json
└── manifest.json                       # optional, free-form
```

A sample is a **`(netlist.json, netlist.meta.json)` pair**. Both
must be present or the loader skips the sample (with a warning, not
a crash — see §5).

Supported `ref_id`s are whatever `DEFAULT_REF_PAYLOAD_PATHS` in
[`app/domain/gnn/evaluator.py`](../app/domain/gnn/evaluator.py)
registers (currently 7: rc_lowpass / divider / all_signal /
opamp_buffer / opamp_inverting / npn_switch / lm358_dual_buffer).
If you need a new ref, add it to that dict first.

---

## 2 · `<sample>.json` — netlist_v2 schema

Mirrors the production `netlist_v2` shape from
[`app/domain/netlist_models.py`](../app/domain/netlist_models.py).

**Required fields**:

```json
{
  "components": [
    {
      "component_id": "R_a",                       // unique within sample
      "component_type": "Resistor",                // ComponentType enum value
      "pins": [
        {"pin_name": "pin1", "electrical_net_id": "n_in"},
        {"pin_name": "pin2", "electrical_net_id": "n_mid"}
      ]
    }
  ],
  "nets": [
    {"electrical_net_id": "n_in"}
  ]
}
```

**Optional / recommended fields** (the rule path consumes them when
present):

| field | shape | impact when missing |
|---|---|---|
| `components[*].part_subtype` | `"UA741"` / `"LM358"` / `""` | IC pin types fall back to generic — see RULE_SEMANTICS §6 |
| `components[*].package_type` | `"DIP8"` / `"axial"` / `""` | informational only |
| `components[*].polarity` | `"none"` / `"two_polar"` / ... | informational only |
| `components[*].confidence` | float in [0, 1] | rule path ignores this; advisory only |
| `pins[*].pin_id` | int | informational only |
| `pins[*].hole_id` | str like `"B12"` | informational only |
| `pins[*].confidence` | float | rule path ignores |
| `nets[*].manual_role` | `"input"` / `"output"` / `"power"` / `"ground"` / `"signal"` | net role inference may misclassify |
| `nets[*].role_label` | `"VIN"` / `"UI1"` / `"VCC"` / ... | role-inference confusion (see RULE_SEMANTICS §6.4) |
| `nets[*].canonical_name` | human label | only used for the warning UI |
| `nets[*].power_role` | `"VCC"` / `"GND"` | filled-in when role_label missing |
| `scene_id`, `board_schema_id` | str | informational only |

Vendor-specific keys (anything not in the table) pass through and are
ignored. Extra fields never cause loader rejection.

---

## 3 · `<sample>.meta.json` — annotation sidecar

Carries the ground truth for the sample. Without this file the
evaluator can't score the sample (it has no idea what the student
*should* have built).

**Required**:

```json
{
  "sample_id": "student_0001",
  "ref_id": "opamp_buffer",
  "expected_outcome": "positive"
}
```

`expected_outcome` is one of:
- `"positive"` — student's circuit is electrically equivalent to ref
- `"wrong_observed"` — student's circuit has a wrong (port, net) connection
- `"missing_required"` — student's circuit is missing a required ref edge

**Optional**:

```json
{
  "annotation_source": "teacher"     | "self_report" | "auto",
  "perturbation_chain": ["extra_wire_bridge:VIN→GND"],   // free-form audit trail
  "notes": "Student forgot the feedback path on op-amp pin 6."
}
```

---

## 4 · Running the evaluator

### One-off CLI run

```bash
python -m scripts.gnn_eval \
    --real-dir datasets/real_student \
    --ckpt   checkpoints/p3_followup_v2/best_f1.pt \
    --output checkpoints/p5_eval_real \
    --false-pass-gate 0.005
```

Produces:
- `checkpoints/p5_eval_real/metrics.json` — full report (per-sample + aggregates)
- `checkpoints/p5_eval_real/report.md` — plan §八 markdown table

Exit codes match `scripts/gnn_eval_nightly.sh`:
- 0 = ok, 3 = false_pass over gate, 2 = crash, 4 = N/A (no real corpus discovered)

### Wiring into the nightly

`scripts/gnn_eval_nightly.sh` **already** runs a real-corpus pass
when `REAL_DIR` points at a directory containing at least one
`*.meta.json` file. Default `REAL_DIR=datasets/real_student`.
Layout the data, and the next nightly run picks it up. Override
with:

```bash
REAL_DIR=/mnt/teacher_uploads/2026_spring bash scripts/gnn_eval_nightly.sh
```

The drift table (`docs/SIM_TO_REAL.md`,
`scripts/gnn_sim2real_drift.py`) automatically adds a "real (Phase 3)"
column when `checkpoints/p5_eval_real/metrics.json` is present.

---

## 5 · Skip behaviour & observability

The loader is **fail-soft** — it never crashes on a malformed sample;
it skips and tallies in `LoadStats`. The counters surface in the
evaluator's log output. Skip reasons:

| reason | meaning |
|---|---|
| `n_skipped_no_meta` | netlist.json without sidecar .meta.json |
| `n_skipped_bad_outcome` | meta.expected_outcome ∉ allowed enum, or meta.ref_id empty |
| `n_skipped_invalid_schema` | netlist_v2 missing required fields (component_id / electrical_net_id) |
| `n_skipped_other` | I/O error reading netlist file |

`evaluate_real_samples` raises `ValueError` only if **zero** samples
load (so you don't silently get an empty report). The error message
includes the LoadStats breakdown so you know which fixture to fix.

---

## 6 · What Phase 3 does NOT include yet

- **Per-edge SEAL labels.** Real samples don't ship with the
  fine-grained edge-correctness annotations the synthetic pipeline
  produces, so `seal_edge_f1` is reported as `None` for real
  samples. The GNN advisor still runs and surfaces its scores
  through the orchestrator's `report.summary.gnn` block; we just
  can't grade it against ground truth at the edge level. A follow-up
  could let teachers annotate specific edges as right/wrong.
- **Automatic real-corpus diffing across snapshots.** Each nightly
  run overwrites `checkpoints/p5_eval_real/`. If you want to track
  drift over time, archive the metrics.json after each run.
- **Privacy / PII review.** Real student exports may include
  identifying metadata (scene_id with student ID, hole IDs that
  encode a session). Add a redaction step before persisting.

---

## 7 · Smoke test

```bash
# Verify the plumbing with the simulated fixtures committed to the repo
python -m scripts.gnn_eval \
    --real-dir tests/fixtures/real_student_simulated \
    --output /tmp/p5_eval_real_smoke
cat /tmp/p5_eval_real_smoke/metrics.json | python -c '
import json, sys
d = json.load(sys.stdin)
print(f"n_samples={d[\"n_samples\"]}  rule_accuracy={d[\"rule_accuracy\"]}")
'
```

Expected output: `n_samples=5  rule_accuracy=1.0` — all 5 simulated
samples land on the right verdict.
