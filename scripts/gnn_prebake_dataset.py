"""P3.2 · Prebake the P1 SEAL dataset into a single PyG tensor blob.

Reads ``<dataset-dir>/labels/<ref>/<sample>.json`` + a refs config (same
shape as ``scripts.gnn_generate_dataset.DEFAULT_CONFIG``) + the splits
JSON (so we bake exactly the rows the training will iterate), runs
cur_hcg replay + ``seal_subgraph_to_pyg_data`` **once per row**, and
saves the result to a single ``.pt`` blob.

After prebaking, ``scripts/gnn_train_full.py --prebaked <blob>.pt`` skips
the per-row replay entirely and trains ~25× faster on Windows CPU
(measured: 250 s/epoch → 10 s/epoch on the 23 k-row P1 dataset).

Usage::

    python -m scripts.gnn_prebake_dataset \\
        --dataset-dir datasets/circuit_compare \\
        --output-path datasets/circuit_compare/prebaked.pt

By default uses the built-in MVP refs config; override with
``--refs-config <file.json>``.

Exit codes: 0 ok, 2 bad args, 3 zero rows baked (likely refs mismatch).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.domain.gnn.prebaked_dataset import PREBAKED_SCHEMA_VERSION, prebake_to_disk
from app.domain.gnn.pyg_dataset import RefEntry, RefRegistry

log = logging.getLogger("gnn.prebake")


def build_refs_registry(
    refs_cfg: list[dict], subtypes_by_ref_id: dict[str, dict[str, str]]
) -> RefRegistry:
    reg = RefRegistry()
    for r in refs_cfg:
        reg.register(
            RefEntry(
                ref_id=r["ref_id"],
                payload_path=Path(r["payload_path"]),
                subtype_by_source_id=dict(subtypes_by_ref_id.get(r["ref_id"], {})),
            )
        )
    return reg


def collect_entries(splits_dir: Path) -> list[str]:
    """Concatenate train+val+test splits so the bake covers everything
    train_full might iterate. Order is preserved (train, val, test) so
    callers slicing by index can reproduce splits, but downstream code
    should filter by `split_entries=` instead."""

    all_entries: list[str] = []
    for name in ("train.json", "val.json", "test.json"):
        path = splits_dir / name
        if path.is_file():
            all_entries.extend(json.loads(path.read_text(encoding="utf-8")))
        else:
            log.warning("missing splits file: %s", path)
    return all_entries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument(
        "--output-path", type=Path, default=None,
        help="defaults to <dataset-dir>/prebaked.pt",
    )
    p.add_argument(
        "--refs-config", type=Path, default=None,
        help="JSON with `refs` list + optional `subtypes_by_ref_id`. "
             "Defaults to the built-in DEFAULT_CONFIG.",
    )
    p.add_argument(
        "--entries-file", type=Path, default=None,
        help="optional pre-filtered entries list (one entry per line "
             "or JSON array). Defaults to train+val+test from splits/.",
    )
    p.add_argument("--num-hops", type=int, default=2)
    p.add_argument(
        "--drop-drnl-at-bake", action="store_true",
        help="bake without DRNL (rare; prefer to bake DRNL ON and use "
             "PrebakedSealDataset(..., drop_drnl=True) at train time)",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.dataset_dir.is_dir():
        log.error("--dataset-dir not found: %s", args.dataset_dir)
        return 2

    if args.refs_config is None:
        from scripts.gnn_generate_dataset import DEFAULT_CONFIG
        refs_cfg = DEFAULT_CONFIG["refs"]
        subtypes = DEFAULT_CONFIG.get("subtypes_by_ref_id", {})
    else:
        cfg = json.loads(args.refs_config.read_text(encoding="utf-8"))
        refs_cfg = cfg["refs"]
        subtypes = cfg.get("subtypes_by_ref_id", {})

    refs = build_refs_registry(refs_cfg, subtypes)
    labels_dir = args.dataset_dir / "labels"

    if args.entries_file is not None:
        text = args.entries_file.read_text(encoding="utf-8").strip()
        try:
            entries = json.loads(text)
        except json.JSONDecodeError:
            entries = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        entries = collect_entries(args.dataset_dir / "splits")

    log.info(
        "prebaking %d entries with refs=%s", len(entries),
        sorted(refs.entries),
    )
    output_path = args.output_path or (args.dataset_dir / "prebaked.pt")
    stats = prebake_to_disk(
        labels_dir,
        refs,
        entries,
        output_path,
        drop_drnl=args.drop_drnl_at_bake,
        num_hops=args.num_hops,
    )
    print(
        f"[prebake] rows_baked={stats.n_rows_baked} "
        f"samples_processed={stats.n_samples_processed} "
        f"rows_dropped={stats.n_rows_dropped} "
        f"load_fail={stats.n_samples_failed_to_load} "
        f"replay_fail={stats.n_samples_failed_to_replay} "
        f"→ {output_path} ({output_path.stat().st_size / 1e6:.1f} MB) "
        f"[version={PREBAKED_SCHEMA_VERSION}]"
    )
    if stats.n_rows_baked == 0:
        log.error("zero rows baked — refs config or splits likely mismatch")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
