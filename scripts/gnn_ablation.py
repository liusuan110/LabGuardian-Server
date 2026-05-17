"""P3.1 · Ablation harness (plan §九 "ablation: 去预训练 / 去 DRNL / 去 port").

Drives :mod:`scripts.gnn_train_full` once per ablation configuration and
aggregates per-configuration val/test metrics into a markdown table.

**This session ships 3 of 4 configs**:

1. ``baseline``       — full P3 MVP (P2.5 backbone loaded, DRNL on)
2. ``no_pretrain``    — random-init SealDGCNN (no ``--pretrain-ckpt``)
3. ``no_drnl``        — DRNL one-hot zeroed via ``--no-drnl``
4. ``no_port``        — **deferred** (requires component-net bipartite
   schema rewrite; documented as P3.2 follow-up)

Plan §九 expects ``pretrain ≥ +5%`` F1 and ``DRNL ≥ +3%`` F1 over
``no_*`` baselines. If those deltas don't show up, the model isn't
benefiting from those ingredients and we should investigate the
backbone transfer / DRNL encoding before scaling training.

Usage::

    python -m scripts.gnn_ablation \\
        --dataset-dir datasets/circuit_compare \\
        --pretrain-ckpt checkpoints/pretrain_v1/backbone.pt \\
        --output-dir checkpoints/p3_ablation \\
        --epochs 10

Each ablation lands in ``<output-dir>/<config_name>/`` with the same
artefacts ``gnn_train_full`` writes (history, best_f1.pt, summary.json).
The aggregate table is written to ``<output-dir>/ablation_report.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import scripts.gnn_train_full as train_full

log = logging.getLogger("gnn.ablation")


# ---------------------------------------------------------------------------
# Configuration matrix
# ---------------------------------------------------------------------------


def build_ablation_argv(
    base_argv: list[str],
    ablation: str,
    output_subdir: Path,
    pretrain_ckpt: Path | None,
) -> list[str]:
    """Compose the argv to pass to ``train_full.main`` for one ablation."""

    argv = list(base_argv) + ["--output-dir", str(output_subdir)]
    if ablation == "baseline":
        if pretrain_ckpt is not None:
            argv += ["--pretrain-ckpt", str(pretrain_ckpt)]
    elif ablation == "no_pretrain":
        # Just don't pass --pretrain-ckpt — model inits fresh
        pass
    elif ablation == "no_drnl":
        if pretrain_ckpt is not None:
            argv += ["--pretrain-ckpt", str(pretrain_ckpt)]
        argv += ["--no-drnl"]
    else:
        raise ValueError(f"unknown ablation: {ablation!r}")
    return argv


ABLATIONS = ("baseline", "no_pretrain", "no_drnl")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _safe_float(d: dict, *keys: str) -> float | None:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return float(cur) if isinstance(cur, (int, float)) else None


def render_report(
    results: dict[str, dict[str, Any]],
    *,
    target_f1_gate: float,
    target_top3_gate: float,
) -> str:
    """Build a markdown table comparing each ablation's val F1, val
    top-3, val AUC, test F1 + the delta vs baseline."""

    rows: list[str] = []
    rows.append("# P3.1 ablation report")
    rows.append("")
    rows.append("**Plan §九 expectations**: `no_pretrain` should hurt F1 by "
                f"≥ 5 pts, `no_drnl` by ≥ 3 pts. Gates: val F1 ≥ "
                f"{target_f1_gate}, val top-3 ≥ {target_top3_gate}.")
    rows.append("")
    rows.append("| Config | val F1 | Δ F1 vs base | val top-3 | val AUC | test F1 | test top-3 |")
    rows.append("|---|---|---|---|---|---|---|")

    baseline = results.get("baseline", {})
    base_f1 = _safe_float(baseline, "best_val_f1") or 0.0

    for cfg in ABLATIONS:
        r = results.get(cfg, {})
        val_f1 = _safe_float(r, "best_val_f1") or 0.0
        val_top3 = _safe_float(r, "best_val_top3") or 0.0
        val_auc = _safe_float(r, "history", "-1", "val", "wrong_edge", "auc")
        # history is a list of dicts — pull the final epoch's val AUC
        hist = r.get("history") or []
        if hist:
            final_we = (hist[-1].get("val") or {}).get("wrong_edge") or {}
            val_auc = float(final_we.get("auc", 0.0))
        else:
            val_auc = 0.0
        test_metrics = r.get("test_metrics") or {}
        test_f1 = _safe_float(test_metrics, "wrong_edge", "f1") or 0.0
        test_top3 = _safe_float(test_metrics, "missing_edge", "top3") or 0.0
        delta = val_f1 - base_f1 if cfg != "baseline" else 0.0
        delta_str = "—" if cfg == "baseline" else f"{delta:+.3f}"
        rows.append(
            f"| `{cfg}` | {val_f1:.3f} | {delta_str} | {val_top3:.3f} | "
            f"{val_auc:.3f} | {test_f1:.3f} | {test_top3:.3f} |"
        )

    rows.append("")
    rows.append("## Verdicts")
    no_pre = _safe_float(results.get("no_pretrain", {}), "best_val_f1") or 0.0
    no_drnl = _safe_float(results.get("no_drnl", {}), "best_val_f1") or 0.0
    verdict_pre = (
        "✅ pretraining helps"
        if base_f1 - no_pre >= 0.05
        else "⚠️ no clear benefit on this dataset"
    )
    verdict_drnl = (
        "✅ DRNL helps"
        if base_f1 - no_drnl >= 0.03
        else "⚠️ no clear benefit on this dataset"
    )
    rows.append(
        f"- **去预训练**: val F1 drops by `{base_f1 - no_pre:+.3f}` "
        f"(plan target ≥ +0.05). Verdict: {verdict_pre}"
    )
    rows.append(
        f"- **去 DRNL**: val F1 drops by `{base_f1 - no_drnl:+.3f}` "
        f"(plan target ≥ +0.03). Verdict: {verdict_drnl}"
    )
    rows.append(
        "- **去 port**: deferred to P3.2 (requires schema rewrite to "
        "component-net bipartite). See README."
    )
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pretrain-ckpt", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--ablations", nargs="+", default=list(ABLATIONS),
        choices=ABLATIONS,
        help="subset of ablations to run (default: all)",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_argv = [
        "--dataset-dir", str(args.dataset_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
    ]
    if args.cpu:
        base_argv.append("--cpu")
    if args.verbose:
        base_argv.append("-v")

    results: dict[str, dict[str, Any]] = {}
    for cfg in args.ablations:
        log.info("===== ablation: %s =====", cfg)
        cfg_dir = args.output_dir / cfg
        cfg_argv = build_ablation_argv(
            base_argv, cfg, cfg_dir, args.pretrain_ckpt
        )
        rc = train_full.main(cfg_argv)
        if rc not in (0, 3):
            log.error("ablation %s failed with exit code %d", cfg, rc)
            return rc
        summary_path = cfg_dir / "summary.json"
        if summary_path.is_file():
            results[cfg] = json.loads(summary_path.read_text())
        else:
            log.warning("no summary.json for %s — skipping in report", cfg)

    report = render_report(
        results, target_f1_gate=0.88, target_top3_gate=0.85
    )
    report_path = args.output_dir / "ablation_report.md"
    report_path.write_text(report)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"[ablation] wrote {report_path}")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
