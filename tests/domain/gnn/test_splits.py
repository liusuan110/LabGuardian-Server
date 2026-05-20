"""P1 Phase C · train/val/test splits.

Verifies the ref-disjoint test set invariant from plan §五:
- test sample_ids never come from a train/val ref
- train + val + test = all discovered samples
- deterministic given (samples_by_ref, spec)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.gnn.splits import (
    DatasetSplits,
    SplitsError,
    SplitSpec,
    build_splits,
    discover_samples,
    load_splits,
    write_splits,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_labels_dir(
    tmp_path: Path, samples_by_ref: dict[str, list[str]]
) -> Path:
    """Touch empty <labels>/<ref>/<sample>.json files for discover."""

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for ref_id, sample_ids in samples_by_ref.items():
        ref_dir = labels_dir / ref_id
        ref_dir.mkdir()
        for sid in sample_ids:
            (ref_dir / f"{sid}.json").write_text("{}", encoding="utf-8")
    return labels_dir


# ---------------------------------------------------------------------------
# SplitSpec validation
# ---------------------------------------------------------------------------


def test_split_spec_rejects_out_of_range_val_fraction() -> None:
    with pytest.raises(SplitsError, match="val_fraction"):
        SplitSpec(test_ref_ids=("a",), val_fraction=1.5)
    with pytest.raises(SplitsError, match="val_fraction"):
        SplitSpec(test_ref_ids=("a",), val_fraction=-0.1)
    # 0 is allowed (no val)
    SplitSpec(test_ref_ids=("a",), val_fraction=0.0)


# ---------------------------------------------------------------------------
# discover_samples
# ---------------------------------------------------------------------------


def test_discover_samples_returns_sorted_dict(tmp_path: Path) -> None:
    _make_labels_dir(
        tmp_path,
        {
            "ref_b": ["s2", "s1", "s3"],
            "ref_a": ["x1"],
        },
    )
    out = discover_samples(tmp_path / "labels")
    # keys sorted by ref_id, values sorted by sample_id
    assert list(out.keys()) == ["ref_a", "ref_b"]
    assert out["ref_b"] == ["s1", "s2", "s3"]


def test_discover_samples_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert discover_samples(tmp_path / "does_not_exist") == {}


def test_discover_samples_ignores_non_json(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    (labels / "ref_a").mkdir(parents=True)
    (labels / "ref_a" / "s1.json").write_text("{}", encoding="utf-8")
    (labels / "ref_a" / "s1.bak").write_text("ignore", encoding="utf-8")
    (labels / "ref_a" / "subdir").mkdir()  # nested dirs also ignored
    assert discover_samples(labels) == {"ref_a": ["s1"]}


# ---------------------------------------------------------------------------
# build_splits — ref disjointness (plan §五 hard constraint)
# ---------------------------------------------------------------------------


def _basic_corpus() -> dict[str, list[str]]:
    return {
        "rc": [f"rc__id_{i:04d}" for i in range(10)],
        "div": [f"div__id_{i:04d}" for i in range(10)],
        "opamp": [f"opamp__id_{i:04d}" for i in range(8)],
    }


def test_test_ref_samples_are_completely_held_out() -> None:
    corpus = _basic_corpus()
    splits = build_splits(
        corpus, SplitSpec(test_ref_ids=("opamp",), val_fraction=0.2)
    )
    # Every test entry comes from opamp
    assert all(s.startswith("opamp/") for s in splits.test)
    # No train/val entry comes from opamp
    assert all(not s.startswith("opamp/") for s in splits.train)
    assert all(not s.startswith("opamp/") for s in splits.val)


def test_splits_partition_total_samples_exactly() -> None:
    corpus = _basic_corpus()
    splits = build_splits(
        corpus, SplitSpec(test_ref_ids=("opamp",), val_fraction=0.3)
    )
    expected_total = sum(len(v) for v in corpus.values())
    assert splits.total() == expected_total
    # No overlap
    train_set = set(splits.train)
    val_set = set(splits.val)
    test_set = set(splits.test)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_zero_val_fraction_assigns_everything_to_train() -> None:
    corpus = _basic_corpus()
    splits = build_splits(
        corpus, SplitSpec(test_ref_ids=("opamp",), val_fraction=0.0)
    )
    assert len(splits.val) == 0
    # train = all non-opamp samples
    n_non_opamp = sum(len(v) for k, v in corpus.items() if k != "opamp")
    assert len(splits.train) == n_non_opamp


def test_empty_corpus_raises() -> None:
    with pytest.raises(SplitsError, match="empty"):
        build_splits({}, SplitSpec(test_ref_ids=()))


def test_test_ref_id_must_exist_in_corpus() -> None:
    with pytest.raises(SplitsError, match="not in labels dir"):
        build_splits(
            _basic_corpus(), SplitSpec(test_ref_ids=("does_not_exist",))
        )


def test_test_ref_ids_empty_is_ok_just_no_test_set() -> None:
    """SplitSpec with no test refs → splits.test is empty, train+val cover all."""

    corpus = _basic_corpus()
    splits = build_splits(
        corpus, SplitSpec(test_ref_ids=(), val_fraction=0.2)
    )
    assert len(splits.test) == 0
    assert splits.total() == sum(len(v) for v in corpus.values())


def test_build_splits_is_deterministic_given_same_spec() -> None:
    corpus = _basic_corpus()
    s1 = build_splits(
        corpus, SplitSpec(test_ref_ids=("opamp",), val_fraction=0.3, seed=42)
    )
    s2 = build_splits(
        corpus, SplitSpec(test_ref_ids=("opamp",), val_fraction=0.3, seed=42)
    )
    assert s1.train == s2.train
    assert s1.val == s2.val
    assert s1.test == s2.test


def test_different_seeds_produce_different_val_picks() -> None:
    corpus = _basic_corpus()
    s1 = build_splits(
        corpus, SplitSpec(test_ref_ids=("opamp",), val_fraction=0.3, seed=1)
    )
    s2 = build_splits(
        corpus, SplitSpec(test_ref_ids=("opamp",), val_fraction=0.3, seed=2)
    )
    # train/val partition differs (test is fixed by spec, not seed)
    assert s1.val != s2.val
    assert s1.test == s2.test


def test_val_fraction_does_not_starve_train() -> None:
    """Even with val_fraction approaching 1, train keeps ≥1 per ref."""

    corpus = {"r1": ["a", "b"]}
    s = build_splits(
        corpus, SplitSpec(test_ref_ids=(), val_fraction=0.99, seed=0)
    )
    # Each ref must contribute at least 1 to train
    assert len(s.train) >= 1


def test_splits_stats_record_per_ref_breakdown() -> None:
    corpus = _basic_corpus()
    s = build_splits(
        corpus, SplitSpec(test_ref_ids=("opamp",), val_fraction=0.2, seed=0)
    )
    assert s.stats["n_train"] == len(s.train)
    assert s.stats["n_val"] == len(s.val)
    assert s.stats["n_test"] == len(s.test)
    assert "opamp" in s.stats["by_ref"]
    assert s.stats["by_ref"]["opamp"] == {"test": 8}
    assert "test_ref_ids" in s.stats


# ---------------------------------------------------------------------------
# write_splits / load_splits round-trip
# ---------------------------------------------------------------------------


def test_write_then_load_round_trip(tmp_path: Path) -> None:
    corpus = _basic_corpus()
    splits = build_splits(
        corpus, SplitSpec(test_ref_ids=("opamp",), val_fraction=0.3, seed=0)
    )
    splits_dir = write_splits(splits, tmp_path)
    assert splits_dir == tmp_path / "splits"
    assert (splits_dir / "train.json").is_file()
    assert (splits_dir / "val.json").is_file()
    assert (splits_dir / "test.json").is_file()
    assert (splits_dir / "stats.json").is_file()
    loaded = load_splits(tmp_path)
    assert loaded.train == splits.train
    assert loaded.val == splits.val
    assert loaded.test == splits.test
    assert loaded.stats == splits.stats


def test_load_splits_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(SplitsError, match="splits dir not found"):
        load_splits(tmp_path)


def test_write_splits_overwrites_in_place(tmp_path: Path) -> None:
    """Re-writing should produce same file paths, not duplicate dirs."""

    corpus = _basic_corpus()
    s1 = build_splits(corpus, SplitSpec(test_ref_ids=("opamp",), seed=0))
    s2 = build_splits(corpus, SplitSpec(test_ref_ids=("opamp",), seed=1))
    write_splits(s1, tmp_path)
    write_splits(s2, tmp_path)
    loaded = load_splits(tmp_path)
    # second write wins
    assert loaded.train == s2.train


# ---------------------------------------------------------------------------
# End-to-end integration with dataset_builder
# ---------------------------------------------------------------------------


def test_splits_integrate_with_real_dataset_builder_output(tmp_path: Path) -> None:
    """Generate a tiny dataset, then split it. Verifies the labels-dir
    contract (`<labels>/<ref>/<sample>.json`) is exactly what discover_samples
    expects."""

    from app.domain.gnn import (
        DatasetSpec,
        PerturbationPlan,
        RefSpec,
        generate_dataset,
    )

    fixtures_dir = Path(__file__).resolve().parents[2] / "fixtures" / "references"
    spec = DatasetSpec(
        refs=(
            RefSpec(ref_id="rc", payload_path=fixtures_dir / "test_rc_v1.json"),
            RefSpec(
                ref_id="div",
                payload_path=fixtures_dir / "test_voltage_divider_v1.json",
            ),
            RefSpec(
                ref_id="opamp",
                payload_path=fixtures_dir / "test_opamp_buffer_v1.json",
            ),
        ),
        plan=PerturbationPlan(
            counts={"identity": 2, "wrong_connection": 2}
        ),
        output_dir=tmp_path / "ds",
        enforce_healthy=False,
    )
    generate_dataset(spec)

    samples = discover_samples(tmp_path / "ds" / "labels")
    assert set(samples) == {"rc", "div", "opamp"}
    assert all(len(v) == 4 for v in samples.values())

    splits = build_splits(
        samples,
        SplitSpec(test_ref_ids=("opamp",), val_fraction=0.25, seed=0),
    )
    assert len(splits.test) == 4
    assert len(splits.train) + len(splits.val) == 8
    # Every train/val entry must be a real file
    for entry in (*splits.train, *splits.val, *splits.test):
        ref_id, sid = entry.split("/", 1)
        assert (tmp_path / "ds" / "labels" / ref_id / f"{sid}.json").is_file()


def test_build_splits_is_deterministic_across_processes(tmp_path: Path) -> None:
    """**Regression** for the cross-process determinism bug: the original
    implementation derived per-ref RNG seeds from Python's builtin
    ``hash((seed, ref_id))``, which is randomised per process via
    PYTHONHASHSEED. Two python subprocesses with the same SplitSpec
    would produce different val splits.

    The fix uses ``hashlib.sha256`` so the bytes are identical regardless
    of process. This test runs two subprocesses, both seeded fresh with
    PYTHONHASHSEED=random, and asserts identical splits.
    """

    import os
    import subprocess
    import sys

    script = (
        "import json, sys; sys.path.insert(0, '.');"
        "from app.domain.gnn.splits import build_splits, SplitSpec;"
        "corpus = {'rc': [f'rc_{i:03d}' for i in range(30)],"
        "          'div': [f'div_{i:03d}' for i in range(30)],"
        "          'opamp': [f'opamp_{i:03d}' for i in range(20)]};"
        "s = build_splits(corpus, SplitSpec("
        "    test_ref_ids=('opamp',), val_fraction=0.25, seed=12345));"
        "print(json.dumps({'val': list(s.val), 'test': list(s.test),"
        "                  'train': list(s.train)}))"
    )
    env = {**os.environ, "PYTHONHASHSEED": "random"}
    repo_root = Path(__file__).resolve().parents[3]
    out1 = subprocess.check_output(
        [sys.executable, "-c", script], cwd=repo_root, env=env
    )
    out2 = subprocess.check_output(
        [sys.executable, "-c", script], cwd=repo_root, env=env
    )
    s1 = json.loads(out1)
    s2 = json.loads(out2)
    assert s1["val"] == s2["val"], (
        "val list differs across processes → string hash leaked again. "
        f"p1[:5]={s1['val'][:5]} p2[:5]={s2['val'][:5]}"
    )
    assert s1["test"] == s2["test"]
    assert s1["train"] == s2["train"]


def test_isinstance_dataset_splits() -> None:
    """DatasetSplits is a dataclass with the expected public attrs."""

    s = build_splits(_basic_corpus(), SplitSpec(test_ref_ids=("opamp",)))
    assert isinstance(s, DatasetSplits)
    # round-trip dict-ish
    payload = {"train": list(s.train), "val": list(s.val), "test": list(s.test)}
    assert json.loads(json.dumps(payload)) == payload
