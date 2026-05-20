"""P1 Phase C · 数据集生成 CLI 驱动。

把 ``app.domain.gnn.dataset_builder`` 的端到端流程封装成可重复运行的命令行
脚本：读取 refs + perturbation plan → 生成 labels JSON → 写 train/val/test
splits → assert_manifest_healthy。

支持 ``--resume`` 在长任务中断后继续；同 ``--base-seed`` 多次 run 必出
同样的 labels。

Usage
=====

最简单（用内置 MVP 配置）::

    python -m scripts.gnn_generate_dataset --output-dir datasets/circuit_compare

自定义配置（JSON file）::

    python -m scripts.gnn_generate_dataset \\
        --output-dir datasets/circuit_compare \\
        --config configs/gnn_dataset.json \\
        --base-seed 42 \\
        --resume

配置文件 schema（示例见末尾 :data:`DEFAULT_CONFIG`）::

    {
      "refs": [
        {"ref_id": "rc_lowpass", "payload_path": "tests/fixtures/references/test_rc_v1.json"}
      ],
      "plan": {"counts": {"identity": 50, "wrong_connection": 50, ...}},
      "test_ref_ids": ["opamp_buffer"],
      "val_fraction": 0.1,
      "enforce_healthy": true,
      "negatives_per_positive": 1.0,
      "forbidden_negative_samples": 4,
      "missing_edge_group_size": 5,
      "include_optional": false,
      "num_hops": 2,
      "checkpoint_every": 100
    }

不引入 torch / torch_geometric。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from app.domain.gnn.dataset_builder import (
    DatasetSpec,
    DatasetSpecError,
    PerturbationPlan,
    RefSpec,
    generate_dataset,
)
from app.domain.gnn.splits import (
    SplitsError,
    SplitSpec,
    build_splits,
    discover_samples,
    write_splits,
)

log = logging.getLogger("gnn.generate_dataset")


# ---------------------------------------------------------------------------
# Default MVP config — 4 fixtures × 4 perturbations × 25 samples = 400
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "references"


DEFAULT_CONFIG: dict[str, Any] = {
    # **Phase D · v5 (2026-05-21 prune)**: dropped 4 non-demo refs
    # (`divider`, `all_signal`, `npn_switch`, `lm358_dual_buffer`) because
    # they don't appear in the Intel-cup demo circuit set. Total refs:
    # 10 → 6. Dataset size: 6000 → 3600 samples, ~115k rows, ~410 MB
    # prebaked. Cut motivation: v5 first training attempt on 6000 samples
    # got stuck on Windows DataLoader I/O bottleneck (9 min prebaked load
    # alone, GPU at 34% utilization). Smaller dataset + demo-aligned refs
    # makes the training tractable and focuses signal on edges the demo
    # actually exercises.
    "refs": [
        # === Demo-aligned refs (kept) ===
        # 1) 一阶 RC
        {
            "ref_id": "rc_lowpass",
            "payload_path": str(FIXTURES_DIR / "test_rc_v1.json"),
        },
        # 2) 电压跟随器 (op-amp buffer; still used as held-out test set
        #    per v4 convention — see `test_ref_ids` below)
        {
            "ref_id": "opamp_buffer",
            "payload_path": str(FIXTURES_DIR / "test_opamp_buffer_v1.json"),
        },
        # 3) UA741 反相放大器
        {
            "ref_id": "opamp_inverting",
            "payload_path": str(FIXTURES_DIR / "test_opamp_inverting_v1.json"),
        },
        # 4) 差分放大器 (BJT diff pair w/ Potentiometer-as-trim + current
        #    source) — exercises Transistor.base→input, Pot.wiper, VEE rail.
        {
            "ref_id": "bjt_diff_amp",
            "payload_path": str(FIXTURES_DIR / "test_bjt_diff_amp_v1.json"),
        },
        # 5) 反相一阶 LPF (Rf ∥ C1 feedback) — exercises Capacitor in
        #    feedback loop + dual-supply pin4=VEE convention.
        {
            "ref_id": "opamp_inverting_lpf",
            "payload_path": str(FIXTURES_DIR / "test_opamp_inverting_lpf_v1.json"),
        },
        # 6) 反相加法器 (two-input summing, R11/R12 = VCC→VI1→GND divider)
        #    — exercises multi-resistor fan-in OOD.
        {
            "ref_id": "opamp_summing",
            "payload_path": str(FIXTURES_DIR / "test_opamp_summing_v1.json"),
        },
        # === Refs dropped from v4 → v5 (not in demo set):
        #     divider, all_signal, npn_switch, lm358_dual_buffer.
        #     If signal quality issues arise on these topologies during
        #     evaluation, restore them in a v5.1 retrain.
    ],
    # Plan: 600 samples / ref × 10 refs = 6000 total. Tuned so pos_neg_ratio
    # stays in the [0.3, 3.0] gate and every required source fires on ≥ 1
    # ref. Sums to 600 exactly.
    #
    # **Stage 3 (Phase C)**: `insert_same_net_wire` is added at 100/ref
    # (~17% of mix) to close the Wire-OOD gap measured on real student
    # netlist samples.
    #
    # **Phase D · v5 (this revision)**: 4 new operators added to fix v4 OOD
    # measured on bjt_diff_amp / opamp_inverting_lpf / opamp_summing
    # _correct_v1 fixtures (v4 fails 13 edges; see commit message). Budgets
    # tuned so the new ops total 140/ref (~23% of mix), other ops scaled
    # back proportionally so total stays at 600. New-op budgets bias toward
    # ops that depend on specific topology (only fire on certain refs);
    # operators that always-fire (identity / wrong_connection) take the cuts.
    "plan": {
        "counts": {
            # Phase D · v5 — budget rebalance based on dry-run trigger-rate
            # analysis (see scripts/gnn_generate_dataset.py dry-run report).
            # `misplace_pot_wiper` and `swap_diff_pair_bases` each only fire
            # on 1/10 refs (diff_amp), so their effective WRONG_OBS sample
            # count is much smaller than `extra_fanin_resistor` which fires
            # on 10/10. Bumped their budgets to 60/ref so each gets ~60
            # actual samples (~120 WRONG_OBS rows) — still less than v4's
            # 2800 wire-positive rows, but enough signal to break the OOD.
            "identity": 40,                  # was 80→60; positives are
                                             #   already over-represented
                                             #   via identity-fallbacks of
                                             #   Phase D ops on the 7-9 refs
                                             #   that lack the prerequisite.
            "pin_swap_symmetric": 30,        # was 40
            "wrong_connection": 50,          # was 70
            "pin_reversed": 25,              # was 30
            "missing_component": 25,         # was 30
            "extra_component": 30,           # was 40
            "floating_net": 30,              # was 40
            "short_circuit": 30,             # was 40
            "power_swapped": 20,             # unchanged (V↔GND ≠ VCC↔VEE)
            "input_output_swapped": 20,
            "extra_wire_bridge": 30,         # was 40
            "chained": 20,                   # was 50; cut hard — chains
                                             #   compose existing ops, low
                                             #   marginal value for v5.
            "insert_same_net_wire": 60,      # was 100; v4 already learned
                                             #   this — keep enough to
                                             #   prevent regression.
            # Phase D — v5 OOD-fix (190 total = 31.7% of 600/ref).
            "swap_dual_supply_rail": 30,     # fires on 3/10 refs (diff_amp,
                                             #   lpf, summing) → ~90 actual
                                             #   samples, ~180 WRONG_OBS rows.
            "misplace_pot_wiper": 60,        # fires only on diff_amp → 60
                                             #   actual samples, ~60-120 rows.
            "swap_diff_pair_bases": 60,      # fires only on diff_amp → 60
                                             #   actual samples, ~120 rows.
            "extra_fanin_resistor": 40,      # fires on 10/10 → 400 actual
                                             #   samples, ~800 WRONG_OBS rows.
        },
    },
    # Plan §五: hold out one ref entirely so test ≡ new topology
    "test_ref_ids": ["opamp_buffer"],
    "val_fraction": 0.1,
    "enforce_healthy": True,
    "negatives_per_positive": 1.0,
    "forbidden_negative_samples": 4,
    "missing_edge_group_size": 5,
    "include_optional": False,
    "num_hops": 2,
    "checkpoint_every": 50,
    "subtypes_by_ref_id": {
        # Make payload-less subtype overrides explicit. lm358_dual_buffer
        # removed in v5 prune (see refs list comment above).
        "opamp_buffer": {"U1": "UA741"},
        "opamp_inverting": {"U1": "UA741"},
        "opamp_inverting_lpf": {"U1": "UA741"},
        "opamp_summing": {"U1": "UA741"},
    },
}


# ---------------------------------------------------------------------------
# Config → spec
# ---------------------------------------------------------------------------


def load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return DEFAULT_CONFIG
    parsed: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    return parsed


def build_dataset_spec(
    cfg: dict[str, Any],
    output_dir: Path,
    base_seed: int,
) -> DatasetSpec:
    refs_cfg = cfg["refs"]
    subtypes = cfg.get("subtypes_by_ref_id", {})
    refs = tuple(
        RefSpec(
            ref_id=r["ref_id"],
            payload_path=Path(r["payload_path"]),
            subtype_by_source_id=subtypes.get(r["ref_id"], {}),
        )
        for r in refs_cfg
    )
    plan = PerturbationPlan(counts=dict(cfg["plan"]["counts"]))
    return DatasetSpec(
        refs=refs,
        plan=plan,
        output_dir=output_dir,
        base_seed=base_seed,
        negatives_per_positive=float(cfg.get("negatives_per_positive", 1.0)),
        forbidden_negative_samples=int(cfg.get("forbidden_negative_samples", 4)),
        missing_edge_group_size=int(cfg.get("missing_edge_group_size", 5)),
        include_optional=bool(cfg.get("include_optional", False)),
        num_hops=int(cfg.get("num_hops", 2)),
        checkpoint_every=int(cfg.get("checkpoint_every", 100)),
        enforce_healthy=bool(cfg.get("enforce_healthy", True)),
    )


def build_split_spec(cfg: dict[str, Any], base_seed: int) -> SplitSpec:
    return SplitSpec(
        test_ref_ids=tuple(cfg.get("test_ref_ids", ())),
        val_fraction=float(cfg.get("val_fraction", 0.1)),
        seed=base_seed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="dataset root (will contain labels/<ref>/<sample>.json + "
             "manifest.json + splits/{train,val,test}.json)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="optional JSON config file; defaults to the built-in MVP config",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="dataset-wide seed (default: 0)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="replay existing labels in output_dir, regenerate only missing",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="parallel worker processes (default 1 = serial). Stats are "
             "guaranteed identical to the serial path across runs",
    )
    p.add_argument(
        "--skip-splits",
        action="store_true",
        help="don't build splits/ subdir (useful for incremental data gen)",
    )
    p.add_argument(
        "--no-healthy",
        action="store_true",
        help="disable assert_manifest_healthy (still writes manifest)",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="print stdout progress lines every checkpoint_every samples",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="set log level to INFO",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    if args.no_healthy:
        cfg["enforce_healthy"] = False

    try:
        spec = build_dataset_spec(cfg, args.output_dir, args.base_seed)
    except KeyError as e:
        log.error("config missing required key: %s", e)
        return 2

    try:
        manifest = generate_dataset(
            spec,
            progress=args.progress,
            resume=args.resume,
            workers=args.workers,
        )
    except DatasetSpecError as e:
        log.error("spec validation failed:\n%s", e)
        return 2
    except ValueError as e:
        # health gate failure — manifest.json was already written
        log.error("manifest health gate failed:\n%s", e)
        return 3

    print(
        f"[generate] {manifest.n_processed} samples processed, "
        f"{manifest.n_skipped_failures} failures, "
        f"pos_neg_ratio={manifest.pos_neg_ratio:.2f}, "
        f"avg_samples_per_build={manifest.avg_samples_per_build:.1f}"
    )

    if args.skip_splits:
        return 0

    # Build + write splits
    try:
        samples_by_ref = discover_samples(args.output_dir / "labels")
        split_spec = build_split_spec(cfg, args.base_seed)
        splits = build_splits(samples_by_ref, split_spec)
    except SplitsError as e:
        log.error("split build failed: %s", e)
        return 4
    splits_dir = write_splits(splits, args.output_dir)
    print(
        f"[splits] train={len(splits.train)} val={len(splits.val)} "
        f"test={len(splits.test)} → {splits_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
