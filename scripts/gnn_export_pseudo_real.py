"""Sim → real Phase 1+2: export the synthetic dataset as production-shaped
``netlist_v2`` JSON files, optionally aged through a realism profile.

What it produces:

    datasets/pseudo_real/<profile>/<ref_id>/<sample_id>.json   # netlist_v2
    datasets/pseudo_real/<profile>/<ref_id>/<sample_id>.meta.json
    datasets/pseudo_real/<profile>/manifest.json

The ``.meta.json`` sidecar carries ``ref_id`` / ``expected_outcome`` /
``perturbation_chain`` / ``alignment`` so the evaluator can score
against ground truth without re-reading the original label file.

This is **Phase 1+2** of plan §十 R6: we run the comparator on
production-shaped inputs *generated from* synthetic samples, before
real student exports arrive. Phase 3 (loading real exports) is a
strict subset of this flow — same evaluator entry point, just a
different source directory.

Usage:
    python -m scripts.gnn_export_pseudo_real \
        --label-dir datasets/circuit_compare/labels \
        --output-root datasets/pseudo_real \
        --profile clean,low,high

Exit codes: 0 ok, 2 bad args.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.domain.gnn.evaluator import (
    DEFAULT_REF_PAYLOAD_PATHS,
    DEFAULT_SUBTYPES_BY_REF,
    _build_ref_artifacts,
    _hcg_to_netlist_v2,
)
from app.domain.gnn.pyg_dataset import reconstruct_cur_hcg
from app.domain.gnn.realism_noise import PROFILES, get_profile

log = logging.getLogger("gnn.export.pseudo_real")


def _iter_label_files(label_dir: Path) -> list[Path]:
    return sorted(label_dir.rglob("*.json"))


def _export_one(
    label_path: Path,
    *,
    profile_name: str,
    output_root: Path,
    ref_cache: dict,
    seed_base: int,
) -> dict[str, str] | None:
    label = json.loads(label_path.read_text())
    ref_id = label["ref_id"]
    sample_id = label["sample_id"]
    cur_meta = label.get("cur_metadata") or {}

    if ref_id not in ref_cache:
        try:
            ref_cache[ref_id] = _build_ref_artifacts(
                ref_id, DEFAULT_REF_PAYLOAD_PATHS, DEFAULT_SUBTYPES_BY_REF,
            )
        except KeyError:
            log.warning("skip %s: no ref payload registered", sample_id)
            return None
    _, ref_hcg, _, subtypes = ref_cache[ref_id]

    try:
        cur_hcg = reconstruct_cur_hcg(
            ref_hcg, cur_meta, subtype_by_source_id=subtypes,
        )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "skip %s: cur reconstruction failed (%s)",
            sample_id, type(e).__name__,
        )
        return None

    netlist_v2 = _hcg_to_netlist_v2(cur_hcg, subtype_by_source_id=subtypes)
    profile = get_profile(profile_name)
    # Deterministic per-sample seed so re-runs are byte-identical.
    seed = (hash(sample_id) ^ seed_base) & 0xFFFFFFFF
    noisy = profile.apply(netlist_v2, seed=seed)

    out_dir = output_root / profile_name / ref_id
    out_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = out_dir / f"{sample_id}.json"
    meta_path = out_dir / f"{sample_id}.meta.json"

    netlist_path.write_text(json.dumps(noisy, indent=2, default=str))
    meta_path.write_text(json.dumps({
        "sample_id": sample_id,
        "ref_id": ref_id,
        "expected_outcome": cur_meta.get("expected_outcome"),
        "perturbation_chain": list(cur_meta.get("perturbation_chain", []) or []),
        "alignment": cur_meta.get("alignment", {}),
        "realism_profile": profile_name,
        "source_label_path": str(label_path.relative_to(label_path.parents[2])),
    }, indent=2, default=str))

    return {"sample_id": sample_id, "ref_id": ref_id}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export synthetic dataset as pseudo-real netlist_v2 files."
    )
    parser.add_argument(
        "--label-dir", type=Path,
        default=Path("datasets/circuit_compare/labels"),
        help="root of synthetic label JSON files",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("datasets/pseudo_real"),
        help="output root; profiles land under <root>/<profile>/<ref>/...",
    )
    parser.add_argument(
        "--profile", type=str, default="clean,low,high",
        help=(
            "comma-separated profile names. Built-ins: "
            f"{', '.join(sorted(PROFILES))}"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="seed base XORed with sample_id hash (default 0)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap samples per profile (smoke test)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not args.label_dir.is_dir():
        parser.error(f"--label-dir not found: {args.label_dir}")
        return 2

    profile_names = [n.strip() for n in args.profile.split(",") if n.strip()]
    for name in profile_names:
        try:
            get_profile(name)
        except KeyError as e:
            parser.error(str(e))
            return 2

    label_files = _iter_label_files(args.label_dir)
    if args.limit is not None:
        label_files = label_files[: args.limit]
    if not label_files:
        parser.error(f"no label files under {args.label_dir}")
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    ref_cache: dict = {}

    summary: dict[str, dict] = {}
    for profile_name in profile_names:
        log.info("→ profile=%s (%d samples)", profile_name, len(label_files))
        n_ok = 0
        n_skip = 0
        for i, lp in enumerate(label_files):
            r = _export_one(
                lp,
                profile_name=profile_name,
                output_root=args.output_root,
                ref_cache=ref_cache,
                seed_base=args.seed,
            )
            if r is None:
                n_skip += 1
            else:
                n_ok += 1
            if (i + 1) % 200 == 0:
                log.info("  %s: %d / %d (skips %d)",
                         profile_name, i + 1, len(label_files), n_skip)
        summary[profile_name] = {
            "n_input": len(label_files),
            "n_exported": n_ok,
            "n_skipped": n_skip,
        }
        manifest = args.output_root / profile_name / "manifest.json"
        manifest.write_text(json.dumps({
            "profile": profile_name,
            "n_input": len(label_files),
            "n_exported": n_ok,
            "n_skipped": n_skip,
            "label_dir": str(args.label_dir),
            "seed_base": args.seed,
        }, indent=2))

    print("=== Export summary ===")
    for name, info in summary.items():
        print(
            f"  {name}: exported {info['n_exported']} / "
            f"{info['n_input']} (skipped {info['n_skipped']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
