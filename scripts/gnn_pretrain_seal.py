"""P2.5 · SpiceNetlist self-supervised SEAL pretraining (GNN-ACLP range).

Loads the GNN-ACLP SpiceNetlist JSON dump, builds a masked-edge link
prediction dataset, trains :class:`SealDGCNN` with 5-fold CV (split by
circuit, **never** by edge — plan §五 hard constraint), reports per-fold
+ aggregate AUC.

Usage::

    # Smoke (1 fold, 2 epochs, tiny subset — finishes in ~10s on CPU)
    python -m scripts.gnn_pretrain_seal \\
        --spicenetlist-json /path/to/SpiceNetlist/JSON \\
        --output-dir checkpoints/pretrain_smoke \\
        --epochs 2 --folds 1 --max-circuits 20

    # Production (5-fold CV, target AUC ≥ 0.95 per plan §九)
    python -m scripts.gnn_pretrain_seal \\
        --spicenetlist-json /path/to/SpiceNetlist/JSON \\
        --output-dir checkpoints/pretrain_v1 \\
        --epochs 50 --folds 5

Exit codes: 0 = ok, 2 = bad args, 3 = AUC below ``--min-auc`` target.

Requires ``[gnn]`` extras (torch + torch_geometric).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader  # type: ignore[import-untyped]

from app.domain.gnn.pretrain_dataset import SpiceNetlistPretrainDataset
from app.domain.gnn.seal_dgcnn import SealDGCNN
from app.domain.gnn.spicenetlist_loader import (
    SpiceNetlistCircuit,
    load_spicenetlist_dir,
)

log = logging.getLogger("gnn.pretrain_seal")


# ---------------------------------------------------------------------------
# Manual AUC (avoids sklearn dependency)
# ---------------------------------------------------------------------------


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U-based AUC. Works for binary labels in {0,1}."""

    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    n_pairs = len(pos) * len(neg)
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / n_pairs


# ---------------------------------------------------------------------------
# K-fold by circuit
# ---------------------------------------------------------------------------


def kfold_circuit_split(
    circuits: list[SpiceNetlistCircuit], k: int, seed: int = 0
) -> list[tuple[list[SpiceNetlistCircuit], list[SpiceNetlistCircuit]]]:
    """Yields ``[(train_circuits, val_circuits), ...]`` per fold.

    Splits by circuit id (NOT by edge) so masked-edge contamination is
    impossible across folds.

    Special case k=1: simple 80/20 train/val random split (for smoke runs;
    not real cross-validation)."""

    rng = random.Random(seed)
    if k == 1:
        idxs = list(range(len(circuits)))
        rng.shuffle(idxs)
        cut = max(1, int(round(len(idxs) * 0.8)))
        train = [circuits[i] for i in idxs[:cut]]
        val = [circuits[i] for i in idxs[cut:]] or [circuits[idxs[-1]]]
        return [(train, val)]

    idxs = list(range(len(circuits)))
    rng.shuffle(idxs)
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, ci in enumerate(idxs):
        folds[i % k].append(ci)
    out: list[tuple[list[SpiceNetlistCircuit], list[SpiceNetlistCircuit]]] = []
    for f in range(k):
        val_ids = set(folds[f])
        train = [c for i, c in enumerate(circuits) if i not in val_ids]
        val = [c for i, c in enumerate(circuits) if i in val_ids]
        out.append((train, val))
    return out


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: SealDGCNN,
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
        logits = model(batch.x, batch.edge_index, batch.batch)
        y = batch.y.float().view(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: SealDGCNN, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Returns ``(mean_bce_loss, auc)``."""

    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_scores: list[float] = []
    all_labels: list[int] = []
    losses: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        y = batch.y.float().view(-1)
        losses.append(loss_fn(logits, y).item())
        scores = torch.sigmoid(logits).cpu().numpy().tolist()
        labels = y.cpu().numpy().astype(int).tolist()
        all_scores.extend(scores)
        all_labels.extend(labels)
    return (
        float(sum(losses) / max(1, len(losses))),
        roc_auc(all_scores, all_labels),
    )


# ---------------------------------------------------------------------------
# Single-fold runner
# ---------------------------------------------------------------------------


def run_one_fold(
    train_circuits: list[SpiceNetlistCircuit],
    val_circuits: list[SpiceNetlistCircuit],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    sort_k: int,
    num_hops: int,
    max_pairs: int | None,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    train_ds = SpiceNetlistPretrainDataset(
        train_circuits,
        max_pairs_per_circuit=max_pairs,
        num_hops=num_hops,
        seed=seed,
    )
    val_ds = SpiceNetlistPretrainDataset(
        val_circuits,
        max_pairs_per_circuit=max_pairs,
        num_hops=num_hops,
        seed=seed + 1,
    )
    log.info(
        "fold dataset: train=%d val=%d (circuits %d/%d)",
        len(train_ds), len(val_ds), len(train_circuits), len(val_circuits),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Probe feature width from a real batch (matches converter output)
    probe = next(iter(train_loader))
    in_channels = probe.x.shape[1]

    model = SealDGCNN(
        in_channels=in_channels,
        hidden_channels=hidden,
        sort_k=sort_k,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    history: list[dict[str, float]] = []
    best_auc = 0.0
    best_state: dict | None = None
    for ep in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, device)
        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "elapsed_s": time.time() - t0,
        })
        log.info(
            "ep %02d/%d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  (%.1fs)",
            ep, epochs, train_loss, val_loss, val_auc, history[-1]["elapsed_s"],
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    return {
        "best_val_auc": best_auc,
        "final_val_auc": history[-1]["val_auc"],
        "history": history,
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "in_channels": in_channels,
        "best_state": best_state,
        "hidden": hidden,
        "sort_k": sort_k,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--spicenetlist-json",
        type=Path,
        required=True,
        help="path to GNN-ACLP SpiceNetlist/JSON directory",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="write per-fold history + final aggregate JSON here",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--sort-k", type=int, default=30)
    p.add_argument("--num-hops", type=int, default=2)
    p.add_argument(
        "--max-pairs-per-circuit", type=int, default=40,
        help="cap each circuit's contribution to avoid 237-edge dominators",
    )
    p.add_argument(
        "--max-circuits", type=int, default=None,
        help="optional cap on circuit count (for smoke runs)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--min-auc", type=float, default=0.0,
        help="exit 3 if mean fold AUC < this (plan §九 gate = 0.95)",
    )
    p.add_argument("--cpu", action="store_true", help="force CPU even if cuda available")
    p.add_argument("--verbose", "-v", action="store_true")
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

    if not args.spicenetlist_json.is_dir():
        log.error("--spicenetlist-json not found: %s", args.spicenetlist_json)
        return 2
    circuits = load_spicenetlist_dir(args.spicenetlist_json)
    if args.max_circuits is not None:
        circuits = circuits[: args.max_circuits]
    log.info("loaded %d circuits", len(circuits))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = kfold_circuit_split(circuits, args.folds, seed=args.seed)
    fold_summaries: list[dict[str, Any]] = []
    for fold_idx, (train_c, val_c) in enumerate(folds):
        log.info("===== fold %d/%d =====", fold_idx + 1, args.folds)
        summary = run_one_fold(
            train_c, val_c,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden=args.hidden,
            sort_k=args.sort_k,
            num_hops=args.num_hops,
            max_pairs=args.max_pairs_per_circuit,
            device=device,
            seed=args.seed + fold_idx,
        )
        summary["fold"] = fold_idx
        fold_summaries.append(summary)
        # Save state_dict separately as .pt; strip from JSON history
        if summary.get("best_state") is not None:
            torch.save(
                {
                    "state_dict": summary["best_state"],
                    "hidden": summary["hidden"],
                    "sort_k": summary["sort_k"],
                    "in_channels": summary["in_channels"],
                    "best_val_auc": summary["best_val_auc"],
                    "fold": fold_idx,
                },
                args.output_dir / f"fold_{fold_idx}.pt",
            )
        summary_for_json = {
            k: v for k, v in summary.items() if k != "best_state"
        }
        (args.output_dir / f"fold_{fold_idx}.json").write_text(
            json.dumps(summary_for_json, indent=2)
        , encoding="utf-8")

    mean_auc = sum(f["best_val_auc"] for f in fold_summaries) / len(fold_summaries)
    # Symlink the best fold's checkpoint as backbone.pt for P3 to pick up
    best_fold = max(range(len(fold_summaries)), key=lambda i: fold_summaries[i]["best_val_auc"])
    backbone_link = args.output_dir / "backbone.pt"
    if backbone_link.exists() or backbone_link.is_symlink():
        backbone_link.unlink()
    try:
        backbone_link.symlink_to(f"fold_{best_fold}.pt")
    except OSError:  # Windows / restricted filesystems
        import shutil
        shutil.copyfile(args.output_dir / f"fold_{best_fold}.pt", backbone_link)
    # Strip large best_state tensors from JSON summary (already saved as .pt)
    folds_for_json = [
        {k: v for k, v in f.items() if k != "best_state"}
        for f in fold_summaries
    ]
    final = {
        "folds": folds_for_json,
        "mean_best_val_auc": mean_auc,
        "best_fold": best_fold,
        "backbone_path": str(backbone_link),
        "config": vars(args) | {
            "spicenetlist_json": str(args.spicenetlist_json),
            "output_dir": str(args.output_dir),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")

    print(
        f"[pretrain] folds={len(fold_summaries)} "
        f"mean_best_val_auc={mean_auc:.4f} "
        f"(target ≥ {args.min_auc:.2f})"
    )
    if mean_auc < args.min_auc:
        log.error(
            "mean AUC %.4f below --min-auc %.4f", mean_auc, args.min_auc
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
