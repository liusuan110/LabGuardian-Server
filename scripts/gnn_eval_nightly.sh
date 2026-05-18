#!/usr/bin/env bash
# scripts/gnn_eval_nightly.sh — repeatable manual / cron entry point for
# the P5 evaluator. Wraps `python -m scripts.gnn_eval` with the project
# defaults the team agreed on (plan §八 false_pass red line, current
# best ckpt, dual-split eval).
#
# Usage:
#   bash scripts/gnn_eval_nightly.sh                  # default ckpt + both splits
#   bash scripts/gnn_eval_nightly.sh path/to/ckpt.pt  # alt ckpt
#
# Exit codes:
#   0   both splits passed gate
#   3   at least one split exceeded false_pass_gate (caller should alert)
#   2   eval crashed / bad args
#
# Wires into:
#   - checkpoints/p5_eval/             (held-out test split)
#   - checkpoints/p5_eval_val/         (in-distribution val split)
#   - checkpoints/p5_eval_rule_only/   (rule-only baseline for diffs)
#
# Per RISK_REGISTER §5 R5, this is the cron-runnable surface. Wire it
# into CI nightly once the false_pass red line is repaired (R1).

set -euo pipefail

CKPT="${1:-checkpoints/p3_followup_v2/best_f1.pt}"
LABEL_DIR="${LABEL_DIR:-datasets/circuit_compare/labels}"
FALSE_PASS_GATE="${FALSE_PASS_GATE:-0.005}"
PYTHON="${PYTHON:-python3}"

if [ ! -d "${LABEL_DIR}" ]; then
    echo "ERROR: label-dir not found: ${LABEL_DIR}" >&2
    exit 2
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
    "datasets/circuit_compare/splits/test.json" \
    "checkpoints/p5_eval" \
    --ckpt "${CKPT}"

run_split "val (in-distribution refs) + GNN" \
    "datasets/circuit_compare/splits/val.json" \
    "checkpoints/p5_eval_val" \
    --ckpt "${CKPT}"

run_split "test rule-only baseline" \
    "datasets/circuit_compare/splits/test.json" \
    "checkpoints/p5_eval_rule_only"

echo
echo "=== Summary ==="
for d in checkpoints/p5_eval checkpoints/p5_eval_val checkpoints/p5_eval_rule_only; do
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
