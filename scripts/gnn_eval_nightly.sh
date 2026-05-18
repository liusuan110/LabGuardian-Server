#!/usr/bin/env bash
# scripts/gnn_eval_nightly.sh — repeatable CI / cron / manual entry point
# for the P5 evaluator. Wraps `python -m scripts.gnn_eval` with the
# project defaults the team agreed on (plan §八 false_pass red line,
# current best ckpt, dual-split eval).
#
# Usage:
#   bash scripts/gnn_eval_nightly.sh                  # default ckpt + both splits
#   bash scripts/gnn_eval_nightly.sh path/to/ckpt.pt  # alt ckpt
#
# Environment overrides:
#   LABEL_DIR           dataset labels root (default: datasets/circuit_compare/labels)
#   FALSE_PASS_GATE     threshold for exit-3 (default: 0.005, plan §八 red line)
#   PYTHON              python interpreter (default: python3)
#   SKIP_IF_MISSING_DATA  "1" (default) → exit 4 when dataset or ckpt absent;
#                       "0" → exit 2 (treat as hard failure). CI defaults to
#                       SKIP_IF_MISSING_DATA=1 so fresh checkouts don't fail.
#
# Exit codes:
#   0   both splits passed gate
#   2   eval crashed / bad args (hard failure)
#   3   at least one split exceeded false_pass_gate (caller should alert)
#   4   skipped: required data / checkpoint missing (soft skip — not a regression)
#
# Wires into:
#   - checkpoints/p5_eval/             (held-out test split, with GNN)
#   - checkpoints/p5_eval_val/         (in-distribution val split, with GNN)
#   - checkpoints/p5_eval_rule_only/   (rule-only baseline for diffs)
#
# See docs/CI_NIGHTLY.md for wiring this into CI (GitHub Actions / cron).
# See RISK_REGISTER.md §5 R5 for the rationale.

set -euo pipefail

CKPT="${1:-checkpoints/p3_followup_v2/best_f1.pt}"
LABEL_DIR="${LABEL_DIR:-datasets/circuit_compare/labels}"
SPLITS_DIR="${SPLITS_DIR:-datasets/circuit_compare/splits}"
FALSE_PASS_GATE="${FALSE_PASS_GATE:-0.005}"
PYTHON="${PYTHON:-python3}"
SKIP_IF_MISSING_DATA="${SKIP_IF_MISSING_DATA:-1}"

# ---- Pre-flight: graceful skip when artifacts are missing ----------------
# CI on a fresh checkout has neither the generated dataset nor the trained
# checkpoint. We want exit 4 ("skip, not failure") in that case so the
# nightly job stays green until somebody seeds those artifacts.

missing=()
[ -d "${LABEL_DIR}" ] || missing+=("LABEL_DIR=${LABEL_DIR}")
[ -f "${SPLITS_DIR}/test.json" ] || missing+=("test split=${SPLITS_DIR}/test.json")
[ -f "${SPLITS_DIR}/val.json" ] || missing+=("val split=${SPLITS_DIR}/val.json")
[ -f "${CKPT}" ] || missing+=("ckpt=${CKPT}")

if [ ${#missing[@]} -gt 0 ]; then
    echo "⏭  gnn_eval_nightly: skipping — missing required artifact(s):"
    for m in "${missing[@]}"; do echo "     - ${m}"; done
    echo
    echo "   To seed locally:"
    echo "     python -m scripts.gnn_generate_dataset    # builds the dataset + splits"
    echo "     python -m scripts.gnn_train_full          # produces best_f1.pt"
    echo "     bash scripts/gnn_eval_nightly.sh          # re-run this"
    if [ "${SKIP_IF_MISSING_DATA}" = "1" ]; then
        exit 4
    else
        echo "   SKIP_IF_MISSING_DATA=0 → treating as hard failure"
        exit 2
    fi
fi

mkdir -p checkpoints/p5_eval checkpoints/p5_eval_val checkpoints/p5_eval_rule_only

worst_exit=0

run_split () {
    local name="$1"; local split="$2"; local out="$3"; shift 3
    echo
    echo "=== [${name}] ${split} -> ${out} ==="
    set +e
    "${PYTHON}" -m scripts.gnn_eval \
        --label-dir "${LABEL_DIR}" \
        --split "${split}" \
        --output "${out}" \
        --false-pass-gate "${FALSE_PASS_GATE}" \
        "$@"
    rc=$?
    set -e
    if [ ${rc} -ne 0 ] && [ ${rc} -ne 3 ]; then
        echo "FATAL: ${name} crashed (exit ${rc})" >&2
        exit 2
    fi
    if [ ${rc} -eq 3 ] && [ ${worst_exit} -lt 3 ]; then
        worst_exit=3
    fi
}

run_split "test (held-out opamp_buffer) + GNN" \
    "${SPLITS_DIR}/test.json" \
    "checkpoints/p5_eval" \
    --ckpt "${CKPT}"

run_split "val (in-distribution refs) + GNN" \
    "${SPLITS_DIR}/val.json" \
    "checkpoints/p5_eval_val" \
    --ckpt "${CKPT}"

run_split "test rule-only baseline" \
    "${SPLITS_DIR}/test.json" \
    "checkpoints/p5_eval_rule_only"

# Phase 3 (plan §十 R6) — optional real student corpus eval. Runs only
# when REAL_DIR points at an existing directory; otherwise silently
# skips so existing CI doesn't fail-loud before any real data arrives.
REAL_DIR="${REAL_DIR:-datasets/real_student}"
if [ -d "${REAL_DIR}" ] && [ -n "$(find "${REAL_DIR}" -name '*.meta.json' -print -quit 2>/dev/null)" ]; then
    echo
    echo "=== [real student corpus + GNN] ${REAL_DIR} -> checkpoints/p5_eval_real ==="
    mkdir -p checkpoints/p5_eval_real
    set +e
    "${PYTHON}" -m scripts.gnn_eval \
        --real-dir "${REAL_DIR}" \
        --ckpt "${CKPT}" \
        --output "checkpoints/p5_eval_real" \
        --false-pass-gate "${FALSE_PASS_GATE}"
    rc=$?
    set -e
    if [ ${rc} -ne 0 ] && [ ${rc} -ne 3 ]; then
        echo "FATAL: real eval crashed (exit ${rc})" >&2
        exit 2
    fi
    if [ ${rc} -eq 3 ] && [ ${worst_exit} -lt 3 ]; then
        worst_exit=3
    fi
fi

echo
echo "=== Summary ==="
SUMMARY_DIRS=(
    checkpoints/p5_eval
    checkpoints/p5_eval_val
    checkpoints/p5_eval_rule_only
    checkpoints/p5_eval_real
)
for d in "${SUMMARY_DIRS[@]}"; do
    if [ -f "${d}/metrics.json" ]; then
        false_pass=$("${PYTHON}" -c "import json,sys; d=json.load(open('${d}/metrics.json')); print(f\"{d['rule_false_pass_rate']:.4f}\")")
        seal=$("${PYTHON}" -c "import json,sys; d=json.load(open('${d}/metrics.json')); v=d.get('seal_edge_f1'); print(f\"{v:.4f}\" if v is not None else 'n/a')")
        r2=$("${PYTHON}" -c "import json,sys; d=json.load(open('${d}/metrics.json')); print(d.get('n_r2_warnings', 0))")
        echo "  ${d}: rule_false_pass=${false_pass} seal_f1=${seal} n_r2_warnings=${r2}"
    fi
done

if [ ${worst_exit} -ne 0 ]; then
    echo
    echo "❌ at least one split exceeded false_pass_gate=${FALSE_PASS_GATE}"
    echo "   see RISK_REGISTER.md §5 R1 (rule semantics) for remediation"
fi
exit ${worst_exit}
