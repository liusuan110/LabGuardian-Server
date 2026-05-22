"""Build the GNN-A training dataset from canonical references + perturbations.

## What this script produces

A directory tree::

    data/cadx/topology_dataset/v1/
    ├── train/
    │   ├── rc_first_order/
    │   │   ├── rc_first_order__canonical__0001.json
    │   │   ├── rc_first_order__perturbed__0002.json
    │   │   └── ...
    │   ├── common_emitter/
    │   └── ...
    ├── val/
    ├── test/
    └── manifest.json

Each sample JSON contains the graph as a node/edge list (NOT netlist_v2
or logical_reference format — the encoder operates on the same
``nx.Graph`` shape both formats reduce to, so we serialize one level up).

## Run

::

    python scripts/cadx/build_topology_dataset.py \\
        --output data/cadx/topology_dataset/v1/ \\
        --samples-per-class 500 \\
        --seed 42

## Determinism

The ``--seed`` argument controls both the random perturbation chains and
the train/val/test split. Re-running with the same seed produces
byte-identical output (modulo dict-key ordering — Python 3.7+ dicts are
insertion-ordered so this is stable in practice).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.domain.dsl.loader import load_dsl_reference  # noqa: E402
from app.domain.logical_reference import (  # noqa: E402
    current_netlist_v2_to_graph,
    logical_reference_to_graph,
)
from app.domain.topology.labels import (  # noqa: E402
    DEFAULT_UNKNOWN_LABEL,
    TOPOLOGY_LABELS,
    get_label_spec,
)
from app.domain.topology.perturbations import (  # noqa: E402
    PERTURBATIONS,
    apply_random_chain,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "cadx" / "topology_dataset" / "v1"
DEFAULT_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "real_student"
DEFAULT_REFERENCES_DIR = REPO_ROOT / "knowledge" / "references"


# ---------------------------------------------------------------------------
# Sample dataclass + JSON helpers
# ---------------------------------------------------------------------------


@dataclass
class DatasetSample:
    """One labeled training sample, serialized to JSON.

    See ``docs/topology_label_spec.md`` for the schema rationale.
    """

    sample_id: str
    label: str
    label_index: int
    source: str  # "canonical_reference" | "perturbation" | "real_student"
    graph: dict  # serialized nx.Graph (node/edge lists)
    metadata: dict


def graph_to_jsonable(g: nx.Graph) -> dict:
    """Serialize an ``nx.Graph`` to JSON-safe nested dicts.

    Avoid pickle / GraphML — the JSON round-trip keeps the dataset
    debuggable and reduces accidental incompatibility across PyG /
    networkx versions.
    """
    return {
        "nodes": [
            {"id": n, **{k: _scrub(v) for k, v in data.items()}}
            for n, data in g.nodes(data=True)
        ],
        "edges": [
            {"u": u, "v": v, **{k: _scrub(val) for k, val in data.items()}}
            for u, v, data in g.edges(data=True)
        ],
    }


def graph_from_jsonable(d: dict) -> nx.Graph:
    g = nx.Graph()
    for node in d.get("nodes", []):
        nid = node.pop("id")
        g.add_node(nid, **node)
    for edge in d.get("edges", []):
        u = edge.pop("u")
        v = edge.pop("v")
        g.add_edge(u, v, **edge)
    return g


def _scrub(value):
    """Make a value JSON-safe (drop None-valued allowed_role_labels, etc.)."""
    if isinstance(value, (list, tuple)):
        return [_scrub(x) for x in value]
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------


def load_canonical_seed(label: str) -> nx.Graph:
    """Load the canonical reference DSL for a label and convert to graph."""
    spec = get_label_spec(label)
    if spec.reference_id is None:
        raise ValueError(f"label {label!r} has no reference_id (unknown class)")
    ref_path = DEFAULT_REFERENCES_DIR / f"{spec.reference_id}.py"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"reference DSL not found: {ref_path}. "
            f"label={label!r} reference_id={spec.reference_id!r}"
        )
    payload = load_dsl_reference(ref_path)
    return logical_reference_to_graph(payload)


def load_real_student_fixtures() -> list[tuple[str, nx.Graph, str]]:
    """Load all real_student fixtures, return list of
    ``(label, graph, fixture_name)``.

    The label is inferred from the filename prefix using the same heuristic
    as ``scripts/cadx/phase0_comparison_report.py``. Files that don't map
    to any label are skipped with a warning.
    """
    prefix_to_label = {
        "inverting_amp": "inverting_amp_ua741",
        "opamp_inverting_lpf_correct": "integrator_ua741",  # lossy integrator
        "opamp_inverting_lpf_wrong": "inverting_amp_ua741",  # broken LPF == inv amp
        "opamp_summing": "summing_amp_ua741",
        "bjt_diff_amp": "differential_pair",
    }

    out: list[tuple[str, nx.Graph, str]] = []
    for path in sorted(DEFAULT_FIXTURES_DIR.glob("*.json")):
        if path.name.endswith(".expected.json"):
            continue
        # Longest prefix wins so the more-specific "..._correct" key
        # outranks the generic prefix.
        name = path.stem
        matches = sorted(
            (
                (prefix, label)
                for prefix, label in prefix_to_label.items()
                if name.startswith(prefix)
            ),
            key=lambda p: -len(p[0]),
        )
        if not matches:
            print(f"  ⚠️  skipping {path.name} — no label heuristic matches")
            continue
        label = matches[0][1]
        payload = json.loads(path.read_text())
        try:
            g = current_netlist_v2_to_graph(payload)
        except Exception as exc:
            print(f"  ❌ failed to load {path.name}: {exc}")
            continue
        out.append((label, g, name))
    return out


# ---------------------------------------------------------------------------
# Sample emitter
# ---------------------------------------------------------------------------


def emit_canonical_sample(
    label: str,
    seed_graph: nx.Graph,
    sample_index: int,
) -> DatasetSample:
    """Emit the unperturbed canonical sample for a label."""
    return DatasetSample(
        sample_id=f"{label}__canonical__{sample_index:04d}",
        label=label,
        label_index=get_label_spec(label).index,
        source="canonical_reference",
        graph=graph_to_jsonable(seed_graph),
        metadata={
            "base_reference_id": get_label_spec(label).reference_id,
        },
    )


def emit_perturbed_sample(
    label: str,
    seed_graph: nx.Graph,
    rng: random.Random,
    sample_index: int,
    chain_length: int,
) -> DatasetSample:
    perturbed, chain = apply_random_chain(seed_graph, rng, chain_length=chain_length)
    return DatasetSample(
        sample_id=f"{label}__perturbed__{sample_index:04d}",
        label=label,
        label_index=get_label_spec(label).index,
        source="perturbation",
        graph=graph_to_jsonable(perturbed),
        metadata={
            "base_reference_id": get_label_spec(label).reference_id,
            "perturbation_chain": chain,
            "chain_length": chain_length,
        },
    )


def emit_real_student_sample(
    label: str,
    student_graph: nx.Graph,
    fixture_name: str,
) -> DatasetSample:
    return DatasetSample(
        sample_id=f"{label}__real_student__{fixture_name}",
        label=label,
        label_index=get_label_spec(label).index,
        source="real_student",
        graph=graph_to_jsonable(student_graph),
        metadata={
            "fixture_name": fixture_name,
        },
    )


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def assign_split(
    samples: Iterable[DatasetSample],
    rng: random.Random,
    val_fraction: float = 0.1,
) -> dict[str, list[DatasetSample]]:
    """Split perturbed samples into train/val. Canonical + real_student
    go to specific splits per label-spec policy.

    Policy (mirrors ``docs/topology_label_spec.md``):
      * **canonical samples**: always in train (anchor)
      * **real_student samples**: always in test (real-distribution eval)
      * **perturbed samples**: 90% train, 10% val (random per-sample)
    """
    out = {"train": [], "val": [], "test": []}
    for sample in samples:
        if sample.source == "real_student":
            out["test"].append(sample)
        elif sample.source == "canonical_reference":
            out["train"].append(sample)
        else:  # perturbation
            if rng.random() < val_fraction:
                out["val"].append(sample)
            else:
                out["train"].append(sample)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_dataset(
    output_dir: Path,
    samples_per_class: int,
    chain_length: int,
    seed: int,
    val_fraction: float = 0.1,
) -> dict:
    """Build the full dataset and write to disk. Returns a manifest dict."""
    rng = random.Random(seed)

    print("Loading canonical seeds...")
    label_seeds: dict[str, nx.Graph] = {}
    for label in TOPOLOGY_LABELS:
        if label == DEFAULT_UNKNOWN_LABEL:
            continue
        try:
            label_seeds[label] = load_canonical_seed(label)
            print(
                f"  ✓ {label:25s} seed={get_label_spec(label).reference_id}"
            )
        except Exception as exc:
            print(f"  ❌ {label}: {exc}")
            raise

    print("\nGenerating samples...")
    all_samples: list[DatasetSample] = []

    # Canonical samples (one per label)
    for label, seed_graph in label_seeds.items():
        all_samples.append(emit_canonical_sample(label, seed_graph, 0))

    # Perturbed samples
    perturbation_count = max(samples_per_class - 1, 0)
    for label, seed_graph in label_seeds.items():
        for i in range(perturbation_count):
            sample = emit_perturbed_sample(
                label, seed_graph, rng,
                sample_index=i + 1,
                chain_length=chain_length,
            )
            all_samples.append(sample)
        print(f"  ✓ {label:25s} canonical=1 + perturbed={perturbation_count}")

    # Real student fixtures → test split
    print("\nLoading real_student fixtures...")
    for label, g, name in load_real_student_fixtures():
        all_samples.append(emit_real_student_sample(label, g, name))
        print(f"  ✓ {name:40s} → label={label}")

    # Split
    print("\nSplitting train/val/test...")
    splits = assign_split(all_samples, rng, val_fraction=val_fraction)
    for split, samples in splits.items():
        per_label_count = Counter(s.label for s in samples)
        print(f"  {split}: {len(samples)} samples, by label: {dict(per_label_count)}")

    # Write to disk
    print(f"\nWriting to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in splits.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            label_dir = split_dir / sample.label
            label_dir.mkdir(parents=True, exist_ok=True)
            (label_dir / f"{sample.sample_id}.json").write_text(
                json.dumps(asdict(sample), ensure_ascii=False, indent=2)
            )

    # Manifest
    manifest = {
        "version": "v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "samples_per_class": samples_per_class,
        "chain_length": chain_length,
        "val_fraction": val_fraction,
        "total_samples": len(all_samples),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "per_label_total": dict(Counter(s.label for s in all_samples)),
        "perturbation_pool": list(PERTURBATIONS.keys()),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"✅ manifest: {manifest_path}")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--samples-per-class",
        type=int,
        default=500,
        help="Total samples per (non-unknown) label, including 1 canonical.",
    )
    p.add_argument(
        "--chain-length",
        type=int,
        default=3,
        help="Number of perturbations composed per synthetic sample.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of perturbed samples assigned to val split.",
    )
    args = p.parse_args()

    manifest = build_dataset(
        output_dir=args.output,
        samples_per_class=args.samples_per_class,
        chain_length=args.chain_length,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    print(f"\nManifest summary: {json.dumps(manifest, ensure_ascii=False, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
