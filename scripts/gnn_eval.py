"""P5 offline evaluator CLI (plan §九 P5 + plan §八 metric table).

Usage:
    python -m scripts.gnn_eval \
        --label-dir datasets/circuit_compare/labels \
        --split datasets/circuit_compare/splits/test.json \
        --ckpt checkpoints/p3_followup_v2/best_f1.pt \
        --output checkpoints/p5_eval

What it writes:
    <output>/metrics.json   — full :class:`EvaluationReport` (incl. per-sample)
    <output>/report.md      — plan §八 markdown table + per-perturbation breakdown

Exit codes:
    0 = ok
    2 = bad args
    3 = false_pass_rate exceeded plan §八 red line (≤ 0.5%)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.domain.gnn.evaluator import evaluate_split

log = logging.getLogger("gnn.eval.cli")


def _parse_split(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LabGuardian-Server GNN P5 evaluator")
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("datasets/circuit_compare/labels"),
        help="root directory of label JSON files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="datasets/circuit_compare/splits/test.json",
        help="splits JSON file (list of '<ref_id>/<sample_id>' keys); "
        "pass --split=ALL or --split='' to walk the whole label-dir",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="GNN checkpoint to load; omit to run rule-only baseline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/p5_eval"),
        help="output dir for metrics.json + report.md",
    )
    parser.add_argument(
        "--seal-threshold", type=float, default=0.5,
        help="SEAL head decision threshold (default 0.5)",
    )
    parser.add_argument(
        "--netlist-dir", type=Path, default=None,
        help=(
            "**Sim → real Phase 1+2** (plan §十 R6). Path to a "
            "directory of pre-baked netlist_v2 JSON files, layout "
            "<dir>/<ref_id>/<sample_id>.json. When set the evaluator "
            "uses these as the cur side instead of synthesising one "
            "from the label's perturbation chain. Generate one via "
            "`python -m scripts.gnn_export_pseudo_real`."
        ),
    )
    parser.add_argument(
        "--real-dir", type=Path, default=None,
        help=(
            "**Sim → real Phase 3** (plan §十 R6). Path to a "
            "directory of real student netlist exports, layout "
            "<dir>/<ref_id>/<sample_id>.json + "
            "<sample_id>.meta.json. No synthetic label_dir needed. "
            "Mutually exclusive with --netlist-dir / --split. See "
            "docs/REAL_STUDENT_NETLIST.md for the schema contract."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap on samples (smoke test)",
    )
    parser.add_argument(
        "--false-pass-gate", type=float, default=0.005,
        help="exit 3 if rule_false_pass_rate exceeds this (default 0.005 = plan §八 red line)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    real_mode = args.real_dir is not None
    if real_mode:
        if args.netlist_dir is not None:
            parser.error("--real-dir is mutually exclusive with --netlist-dir")
            return 2
        if not args.real_dir.is_dir():
            parser.error(f"--real-dir not found: {args.real_dir}")
            return 2
        split_path: Path | None = None
        split_ids: list[str] | None = None
    else:
        if not args.label_dir.is_dir():
            parser.error(f"--label-dir not found: {args.label_dir}")
            return 2

        # `--split=` (empty) or `--split=ALL` ⇒ walk the whole label-dir.
        # Otherwise must point at an existing file.
        split_arg = (args.split or "").strip()
        if split_arg in {"", "ALL"}:
            split_path = None
        else:
            split_path = Path(split_arg)
            if not split_path.is_file():
                parser.error(f"--split path not found: {split_path}")
                return 2
        split_ids = _parse_split(split_path)

    advisor = None
    if args.ckpt is not None:
        from app.domain.gnn.inference import GNNAdvisor
        log.info("loading advisor from %s", args.ckpt)
        advisor = GNNAdvisor.from_checkpoint(args.ckpt)

    if real_mode:
        log.info(
            "evaluating real corpus at %s (advisor=%s)",
            args.real_dir,
            advisor.model_version if advisor else "rule-only",
        )
    else:
        log.info(
            "evaluating split=%s (%s samples) advisor=%s",
            split_path,
            len(split_ids) if split_ids else "all",
            advisor.model_version if advisor else "rule-only",
        )

    if real_mode:
        from app.domain.gnn.evaluator import evaluate_real_samples
        report = evaluate_real_samples(
            args.real_dir,
            advisor=advisor,
            seal_threshold=args.seal_threshold,
            limit=args.limit,
        )
    else:
        report = evaluate_split(
            args.label_dir,
            split_ids=split_ids,
            advisor=advisor,
            seal_threshold=args.seal_threshold,
            limit=args.limit,
            netlist_v2_dir=args.netlist_dir,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    (args.output / "report.md").write_text(report.to_markdown(), encoding="utf-8")

    print(
        f"wrote {args.output / 'metrics.json'} and {args.output / 'report.md'}\n"
        f"rule_false_pass_rate={report.rule_false_pass_rate:.4f} | "
        f"rule_accuracy={report.rule_accuracy:.4f} | "
        f"seal_f1={report.seal_edge_f1!r}"
    )

    if report.rule_false_pass_rate > args.false_pass_gate:
        print(
            f"❌ false_pass_rate {report.rule_false_pass_rate:.4f} > "
            f"gate {args.false_pass_gate}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
