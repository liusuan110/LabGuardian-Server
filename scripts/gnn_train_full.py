"""P3 · CircuitMatchNet 多任务训练（MVP · plan §四 L2 主头）.

把 P1 dataset_builder 写盘的 ``datasets/circuit_compare/`` + P2.5 预训练
backbone 拼成端到端训练流水线：

1. ``FlatSealDataset(labels_dir, refs, split_entries)`` 提供 PyG Data 行
2. 可选加载 P2.5 ``backbone.pt``（``--pretrain-ckpt``）
3. ``CircuitMatchNet`` 主头 BCE 训练（``WRONG_EDGE`` 点式 + ``MISSING_EDGE``
   组式样本共享同一 SEAL DGCNN logit）
4. 评估按 ``task_type`` 分桶：
   - ``WRONG_EDGE`` : F1@0.5 / AUC / accuracy
   - ``MISSING_EDGE``: 按 ``group_id`` 排序 → top-1 / top-3 accuracy

Plan §九 P3 gate: ``val SEAL F1 ≥ 0.88``, ``suggested_target top-3 ≥ 0.85``.

Usage::

    # Smoke (1 epoch, no backbone)
    python -m scripts.gnn_train_full \\
        --dataset-dir datasets/circuit_compare \\
        --output-dir checkpoints/p3_smoke \\
        --epochs 1

    # Full + backbone
    python -m scripts.gnn_train_full \\
        --dataset-dir datasets/circuit_compare \\
        --pretrain-ckpt checkpoints/pretrain_v1/backbone.pt \\
        --output-dir checkpoints/p3_v1 \\
        --epochs 20 --batch-size 64 --lr 1e-3

Exit codes: 0 = ok, 2 = bad args, 3 = both gates failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader  # type: ignore[import-untyped]

from app.domain.gnn.model import CircuitMatchNet
from app.domain.gnn.pyg_dataset import FlatSealDataset, RefEntry, RefRegistry
from scripts.gnn_pretrain_seal import roc_auc

log = logging.getLogger("gnn.train_full")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def f1_at_threshold(
    scores: list[float], labels: list[int], threshold: float = 0.5
) -> tuple[float, float, float]:
    """Returns (precision, recall, f1) for binary classification at the
    given decision threshold."""

    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def top_k_accuracy(
    group_scores: dict[str, list[tuple[float, int]]], k: int
) -> float:
    """For each group: ``[(score, label), ...]``. Top-k accuracy = the
    fraction of groups where the ``label==1`` sample is in the top-k by
    score. Groups with no positive (correct_index=None) are skipped.
    """

    n_groups = 0
    n_hit = 0
    for _gid, pairs in group_scores.items():
        if not any(y == 1 for _, y in pairs):
            continue  # no positive in this group
        n_groups += 1
        # Sort by score descending, take top-k labels
        sorted_pairs = sorted(pairs, key=lambda t: -t[0])
        top_k_labels = [y for _, y in sorted_pairs[:k]]
        if 1 in top_k_labels:
            n_hit += 1
    return n_hit / max(1, n_groups)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: CircuitMatchNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out["seal_logits"], batch.y.float().view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: CircuitMatchNet, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    """Returns a dict of metrics split by task_type."""

    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    losses: list[float] = []
    wrong_scores: list[float] = []
    wrong_labels: list[int] = []
    # group_id → [(score, label)]
    missing_groups: dict[str, list[tuple[float, int]]] = defaultdict(list)

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        logits = out["seal_logits"]
        y = batch.y.float().view(-1)
        losses.append(loss_fn(logits, y).item())
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
        labels = y.cpu().numpy().astype(int).tolist()
        task_types = batch.task_type
        group_ids = batch.group_id
        # task_type / group_id are list[str] (PyG collates strings as list)
        for prob, label, tt, gid in zip(probs, labels, task_types, group_ids):
            if tt == "wrong_edge":
                wrong_scores.append(prob)
                wrong_labels.append(label)
            elif tt == "missing_edge":
                missing_groups[gid].append((prob, label))

    metrics: dict[str, Any] = {
        "mean_loss": sum(losses) / max(1, len(losses)),
    }

    if wrong_scores:
        prec, rec, f1 = f1_at_threshold(wrong_scores, wrong_labels)
        metrics["wrong_edge"] = {
            "n": len(wrong_scores),
            "auc": roc_auc(wrong_scores, wrong_labels),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": sum(
                1 for s, y in zip(wrong_scores, wrong_labels)
                if (s >= 0.5) == bool(y)
            ) / len(wrong_scores),
        }
    if missing_groups:
        metrics["missing_edge"] = {
            "n_groups": len(missing_groups),
            "n_samples": sum(len(g) for g in missing_groups.values()),
            "top1": top_k_accuracy(missing_groups, 1),
            "top3": top_k_accuracy(missing_groups, 3),
            "top5": top_k_accuracy(missing_groups, 5),
        }
    return metrics


# ---------------------------------------------------------------------------
# Dataset wiring
# ---------------------------------------------------------------------------


def build_refs_registry(
    refs_config: list[dict], subtypes_by_ref_id: dict[str, dict[str, str]]
) -> RefRegistry:
    reg = RefRegistry()
    for r in refs_config:
        reg.register(
            RefEntry(
                ref_id=r["ref_id"],
                payload_path=Path(r["payload_path"]),
                subtype_by_source_id=dict(subtypes_by_ref_id.get(r["ref_id"], {})),
            )
        )
    return reg


def build_loaders(
    dataset_dir: Path,
    refs_config: list[dict],
    subtypes_by_ref_id: dict[str, dict[str, str]],
    *,
    batch_size: int,
    num_workers: int = 0,
    drop_drnl: bool = False,
    prebaked_path: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders.

    If ``prebaked_path`` is provided, uses :class:`PrebakedSealDataset`
    (zero per-row replay, ~25× faster training on Windows CPU). Otherwise
    falls back to :class:`FlatSealDataset` which replays cur_hcg per row.
    """

    splits_dir = dataset_dir / "splits"
    train_entries = json.loads((splits_dir / "train.json").read_text(encoding="utf-8"))
    val_entries = json.loads((splits_dir / "val.json").read_text(encoding="utf-8"))
    test_entries = json.loads((splits_dir / "test.json").read_text(encoding="utf-8"))

    if prebaked_path is not None:
        from app.domain.gnn.prebaked_dataset import PrebakedSealDataset

        log.info("loading prebaked dataset from %s", prebaked_path)
        train_ds = PrebakedSealDataset(
            prebaked_path, train_entries, drop_drnl=drop_drnl
        )
        val_ds = PrebakedSealDataset(
            prebaked_path, val_entries, drop_drnl=drop_drnl
        )
        test_ds = PrebakedSealDataset(
            prebaked_path, test_entries, drop_drnl=drop_drnl
        )
    else:
        refs = build_refs_registry(refs_config, subtypes_by_ref_id)
        labels_dir = dataset_dir / "labels"
        train_ds = FlatSealDataset(labels_dir, refs, train_entries, drop_drnl=drop_drnl)
        val_ds = FlatSealDataset(labels_dir, refs, val_entries, drop_drnl=drop_drnl)
        test_ds = FlatSealDataset(labels_dir, refs, test_entries, drop_drnl=drop_drnl)
    log.info(
        "dataset rows: train=%d val=%d test=%d (%s)",
        len(train_ds), len(val_ds), len(test_ds),
        "prebaked" if prebaked_path else "live replay",
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="P1 dataset root (contains labels/ + splits/)",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="checkpoint + history JSON output dir",
    )
    p.add_argument(
        "--pretrain-ckpt", type=Path, default=None,
        help="path to P2.5 backbone.pt (optional)",
    )
    p.add_argument(
        "--prebaked", type=Path, default=None,
        help="path to a prebaked .pt blob from scripts.gnn_prebake_dataset "
             "(skips per-row cur_hcg replay; ~25x faster training on "
             "Windows CPU per P3.2)",
    )
    p.add_argument(
        "--refs-config", type=Path, default=None,
        help="JSON file with same shape as scripts.gnn_generate_dataset "
             "DEFAULT_CONFIG['refs'] + subtypes_by_ref_id. Defaults to "
             "the built-in MVP config.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--sort-k", type=int, default=30)
    p.add_argument(
        "--min-f1", type=float, default=0.0,
        help="exit 3 if val WRONG_EDGE F1 < this at the end",
    )
    p.add_argument(
        "--min-top3", type=float, default=0.0,
        help="exit 3 if val MISSING_EDGE top-3 < this at the end",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    # P3.1 ablation flags ----
    p.add_argument(
        "--no-drnl", action="store_true",
        help="ablation: zero the DRNL one-hot channels in SEAL input "
             "(per plan §九 '去 DRNL' ablation; expected -3%% to -5%% F1)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    log.info("device = %s", device)

    if not args.dataset_dir.is_dir():
        log.error("--dataset-dir not found: %s", args.dataset_dir)
        return 2

    # Refs config — default to DEFAULT_CONFIG so the script Just Works
    # against the P1 acceptance dataset.
    if args.refs_config is None:
        from scripts.gnn_generate_dataset import DEFAULT_CONFIG
        refs_config = DEFAULT_CONFIG["refs"]
        subtypes = DEFAULT_CONFIG.get("subtypes_by_ref_id", {})
    else:
        cfg = json.loads(args.refs_config.read_text(encoding="utf-8"))
        refs_config = cfg["refs"]
        subtypes = cfg.get("subtypes_by_ref_id", {})

    train_loader, val_loader, test_loader = build_loaders(
        args.dataset_dir,
        refs_config,
        subtypes,
        batch_size=args.batch_size,
        drop_drnl=args.no_drnl,
        prebaked_path=args.prebaked,
    )
    if args.no_drnl:
        log.info("ablation: --no-drnl active (DRNL one-hot zeroed)")

    # Probe feature width
    probe = next(iter(train_loader))
    in_channels = probe.x.shape[1]

    if args.pretrain_ckpt is not None:
        log.info("loading backbone from %s", args.pretrain_ckpt)
        model = CircuitMatchNet.from_pretrained_backbone(
            args.pretrain_ckpt,
            strict=False,
            override_in_channels=in_channels,
        ).to(device)
    else:
        model = CircuitMatchNet(
            in_channels=in_channels,
            hidden_channels=args.hidden,
            sort_k=args.sort_k,
        ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_f1 = 0.0
    best_top3 = 0.0
    for ep in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        ep_summary = {
            "epoch": ep,
            "train_loss": train_loss,
            "val": val_metrics,
            "elapsed_s": time.time() - t0,
        }
        history.append(ep_summary)
        we = val_metrics.get("wrong_edge", {})
        me = val_metrics.get("missing_edge", {})
        log.info(
            "ep %02d/%d  train_loss=%.4f  val_loss=%.4f  "
            "F1=%.3f AUC=%.3f  top1=%.3f top3=%.3f  (%.1fs)",
            ep, args.epochs, train_loss, val_metrics.get("mean_loss", 0.0),
            we.get("f1", 0.0), we.get("auc", 0.0),
            me.get("top1", 0.0), me.get("top3", 0.0),
            ep_summary["elapsed_s"],
        )
        if we.get("f1", 0.0) > best_f1:
            best_f1 = we["f1"]
            model.save(args.output_dir / "best_f1.pt", extra={
                "epoch": ep, "val_metrics": val_metrics
            })
        if me.get("top3", 0.0) > best_top3:
            best_top3 = me["top3"]

    # Final test eval (using whatever checkpoint is loaded at end)
    test_metrics = evaluate(model, test_loader, device)
    log.info("TEST metrics: %s", test_metrics)

    final = {
        "history": history,
        "best_val_f1": best_f1,
        "best_val_top3": best_top3,
        "test_metrics": test_metrics,
        "config": vars(args) | {
            "dataset_dir": str(args.dataset_dir),
            "output_dir": str(args.output_dir),
            "pretrain_ckpt": str(args.pretrain_ckpt) if args.pretrain_ckpt else None,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(final, indent=2, default=str)
    , encoding="utf-8")

    we_t = test_metrics.get("wrong_edge", {})
    me_t = test_metrics.get("missing_edge", {})
    print(
        f"[train_full] best val F1={best_f1:.3f}  best val top3={best_top3:.3f}  "
        f"test F1={we_t.get('f1', 0):.3f}  test top3={me_t.get('top3', 0):.3f}"
    )
    # Plan §九 P3 verdict: BOTH gates must pass. Exit 3 if ANY gate fails.
    # (Previous version used `and`, which only failed when *both* gates
    # missed — a real false-positive bug spotted in audit. Fixed to `or`
    # below so a single missed gate trips the failure path.)
    failed_gates = []
    if best_f1 < args.min_f1:
        failed_gates.append(f"best_f1={best_f1:.3f} < min_f1={args.min_f1:.2f}")
    if best_top3 < args.min_top3:
        failed_gates.append(
            f"best_top3={best_top3:.3f} < min_top3={args.min_top3:.2f}"
        )
    if failed_gates:
        log.error("gate failure (%d): %s", len(failed_gates), "; ".join(failed_gates))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
