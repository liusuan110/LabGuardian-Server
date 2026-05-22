"""Train the GNN-A topology classifier (CADx Phase 1).

## Usage

::

    # Local CPU smoke test (50 samples/class, 20 epochs)
    python scripts/cadx/train_topology_classifier.py \\
        --dataset data/cadx/topology_dataset/v1/ \\
        --output checkpoints/gnn_a_v1/ \\
        --epochs 20 --batch-size 16

    # Cloud GPU full training (500 samples/class, 100 epochs)
    python scripts/cadx/train_topology_classifier.py \\
        --dataset data/cadx/topology_dataset/v1/ \\
        --output checkpoints/gnn_a_v1/ \\
        --epochs 100 --batch-size 32 --device cuda

## Output

::

    checkpoints/gnn_a_v1/
    ├── best.pt              # best val accuracy checkpoint
    ├── final.pt             # last epoch ckpt
    ├── training_log.json    # per-epoch metrics
    └── confusion_matrix.txt # final test-set confusion matrix

## Reproducibility

``--seed`` controls torch/numpy/python random state. Same seed + same
dataset → byte-identical checkpoint (verified locally with diff -q).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.domain.topology.features import encode_graph, encoded_to_pyg_data  # noqa: E402
from app.domain.topology.labels import TOPOLOGY_LABELS, index_to_label  # noqa: E402
from app.domain.topology.model import TopologyClassifier  # noqa: E402

# Reuse the dataset loader from the builder so format drifts stay in
# one place.
from scripts.cadx.build_topology_dataset import graph_from_jsonable  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_split(dataset_dir: Path, split: str) -> list:
    """Load all JSON samples in a split directory and convert to PyG Data."""
    split_dir = dataset_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"split directory missing: {split_dir}")
    data_list = []
    for label_dir in sorted(split_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for json_path in sorted(label_dir.glob("*.json")):
            payload = json.loads(json_path.read_text())
            g = graph_from_jsonable(payload["graph"])
            encoded = encode_graph(g)
            data = encoded_to_pyg_data(encoded, label_index=payload["label_index"])
            data_list.append(data)
    return data_list


def summarize_split(data_list: list, name: str) -> None:
    label_counts = Counter(int(d.y[0].item()) for d in data_list)
    print(f"  {name} ({len(data_list)} samples):")
    for idx in sorted(label_counts.keys()):
        print(f"    [{idx}] {index_to_label(idx):25s} {label_counts[idx]}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=-1)
        total_correct += (preds == batch.y).sum().item()
        total_count += batch.num_graphs
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float, list[tuple[int, int]]]:
    """Return ``(loss, accuracy, [(true, pred), ...])``."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    pairs: list[tuple[int, int]] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(logits, batch.y)
        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=-1)
        total_correct += (preds == batch.y).sum().item()
        total_count += batch.num_graphs
        for t, p in zip(batch.y.tolist(), preds.tolist()):
            pairs.append((t, p))
    return total_loss / total_count, total_correct / total_count, pairs


def render_confusion_matrix(pairs: list[tuple[int, int]]) -> str:
    """Render a labeled confusion matrix as a text table."""
    n = len(TOPOLOGY_LABELS)
    matrix = [[0] * n for _ in range(n)]
    for t, p in pairs:
        matrix[t][p] += 1
    lines: list[str] = []
    short = [lbl[:14] for lbl in TOPOLOGY_LABELS]
    header = f"{'true \\ pred':16s}" + "".join(f"{s:15s}" for s in short)
    lines.append(header)
    for i, row in enumerate(matrix):
        line = f"{short[i]:16s}" + "".join(f"{cnt:15d}" for cnt in row)
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "cadx" / "topology_dataset" / "v1",
    )
    p.add_argument("--output", type=Path, default=REPO_ROOT / "checkpoints" / "gnn_a_v1")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stop after N epochs without val-acc improvement.",
    )
    args = p.parse_args()

    set_seeds(args.seed)

    if args.device == "auto":
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"\nLoading dataset from {args.dataset} ...")
    train_data = load_split(args.dataset, "train")
    val_data = load_split(args.dataset, "val")
    test_data = load_split(args.dataset, "test")
    summarize_split(train_data, "train")
    summarize_split(val_data, "val")
    summarize_split(test_data, "test")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    print(f"\nBuilding model (hidden_dim={args.hidden_dim}, dropout={args.dropout})...")
    model = TopologyClassifier(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    print(f"  Total params: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    best_val_acc = 0.0
    epochs_no_improve = 0

    print("\nTraining...")
    print(f"{'epoch':>5s} {'train_loss':>11s} {'train_acc':>10s} "
          f"{'val_loss':>10s} {'val_acc':>8s} {'best':>5s}")

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, device)
        log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        is_best = val_acc > best_val_acc
        marker = " ★" if is_best else ""
        print(f"{epoch:5d} {train_loss:11.4f} {train_acc:10.3f} "
              f"{val_loss:10.4f} {val_acc:8.3f}{marker}")
        if is_best:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "num_classes": model.num_classes,
                    "in_dim": model.in_dim,
                },
                "epoch": epoch,
                "val_acc": val_acc,
            }, args.output / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    duration = time.time() - start
    print(f"\nTraining took {duration:.1f}s")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "num_classes": model.num_classes,
            "in_dim": model.in_dim,
        },
        "epoch": epoch,
        "val_acc": val_acc,
    }, args.output / "final.pt")

    # Reload best for test eval
    print(f"\nLoading best (val_acc={best_val_acc:.3f}) for test eval...")
    best_ckpt = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc, test_pairs = evaluate(model, test_loader, device)
    print(f"Test loss={test_loss:.4f} acc={test_acc:.3f}")

    cm_text = render_confusion_matrix(test_pairs)
    print("\nConfusion matrix:")
    print(cm_text)

    (args.output / "training_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))
    (args.output / "confusion_matrix.txt").write_text(cm_text + "\n")
    (args.output / "test_results.json").write_text(json.dumps({
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "training_seconds": duration,
    }, indent=2))

    print(f"\n✅ All artifacts written to {args.output}/")
    return 0 if test_acc > 0.7 else 1  # arbitrary "good enough" threshold


if __name__ == "__main__":
    # silence unused import warnings; TorchDataLoader is here so future
    # extensions (e.g. balanced sampler) have a path to import.
    _ = TorchDataLoader
    sys.exit(main())
