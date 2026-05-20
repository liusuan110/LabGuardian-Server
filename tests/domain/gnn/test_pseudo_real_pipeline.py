"""End-to-end sim→real Phase 1+2 pipeline test (plan §十 R6).

Exercises:
1. ``scripts.gnn_export_pseudo_real.main`` writes a per-ref / per-sample
   netlist_v2 corpus with sidecar metadata, partitioned by profile.
2. ``evaluate_split(..., netlist_v2_dir=...)`` consumes that corpus
   instead of synthesising from the perturbation pipeline, and still
   produces a well-formed :class:`EvaluationReport`.
3. Noisy profiles produce a **different** netlist_v2 on disk vs clean,
   confirming the noise actually leaks through the export.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

LABEL_ROOT = (
    Path(__file__).resolve().parents[3]
    / "datasets" / "circuit_compare" / "labels"
)


def _label_dir_available() -> bool:
    return LABEL_ROOT.is_dir() and any(LABEL_ROOT.iterdir())


pytestmark = pytest.mark.skipif(
    not _label_dir_available(),
    reason="circuit_compare label dataset not generated on this box",
)


def test_export_writes_netlist_meta_and_manifest_per_profile(tmp_path: Path):
    """Smoke: run the CLI on a slice and inspect file layout."""

    from scripts.gnn_export_pseudo_real import main

    rc = main([
        "--label-dir", str(LABEL_ROOT),
        "--output-root", str(tmp_path),
        "--profile", "clean,low,high",
        "--limit", "8",
    ])
    assert rc == 0

    for profile in ("clean", "low", "high"):
        profile_dir = tmp_path / profile
        assert profile_dir.is_dir(), f"missing profile dir {profile_dir}"
        manifest = profile_dir / "manifest.json"
        assert manifest.is_file()
        m = json.loads(manifest.read_text(encoding="utf-8"))
        assert m["profile"] == profile
        assert m["n_input"] == 8
        assert m["n_exported"] >= 6, (
            "expected nearly all 8 samples to export; "
            f"got {m['n_exported']}"
        )

        # at least one (netlist, meta) pair exists
        netlist_files = list(profile_dir.rglob("*.json"))
        netlist_files = [
            p for p in netlist_files
            if p.name != "manifest.json" and not p.name.endswith(".meta.json")
        ]
        assert netlist_files, f"no netlist_v2 outputs under {profile_dir}"

        sample = netlist_files[0]
        meta = sample.with_name(sample.stem + ".meta.json")
        assert meta.is_file(), f"missing sidecar {meta}"

        doc = json.loads(sample.read_text(encoding="utf-8"))
        # Adapter-shape netlist_v2 has these top-level keys
        assert "components" in doc and "nets" in doc
        assert doc["metadata"]["realism_profile"] == profile


def test_high_profile_differs_from_clean_on_same_sample(tmp_path: Path):
    """Noise must actually mutate the netlist relative to clean."""

    from scripts.gnn_export_pseudo_real import main

    main([
        "--label-dir", str(LABEL_ROOT),
        "--output-root", str(tmp_path),
        "--profile", "clean,high",
        "--limit", "4",
    ])
    # find any sample exported under both profiles
    clean_dir = tmp_path / "clean"
    high_dir = tmp_path / "high"
    clean_samples = {
        p.relative_to(clean_dir).as_posix()
        for p in clean_dir.rglob("*.json")
        if not p.name.endswith(".meta.json")
        and p.name != "manifest.json"
    }
    high_samples = {
        p.relative_to(high_dir).as_posix()
        for p in high_dir.rglob("*.json")
        if not p.name.endswith(".meta.json")
        and p.name != "manifest.json"
    }
    common = sorted(clean_samples & high_samples)
    assert common, "expected at least one sample exported under both profiles"

    rel = common[0]
    clean_doc = json.loads((clean_dir / rel).read_text(encoding="utf-8"))
    high_doc = json.loads((high_dir / rel).read_text(encoding="utf-8"))
    # Different metadata.realism_profile is guaranteed; the rest of the
    # change set proves noise actually flowed through.
    assert clean_doc["metadata"]["realism_profile"] == "clean"
    assert high_doc["metadata"]["realism_profile"] == "high"
    # IC subtype: clean=UA741/LM358 or empty, high=""
    high_ic = [
        c for c in high_doc["components"]
        if c["component_type"] == "IC"
    ]
    for c in high_ic:
        assert c["part_subtype"] == "", (
            "high profile must clear IC part_subtype"
        )
    # Components renamed
    if any(c["component_type"] == "IC" for c in high_doc["components"]):
        # at least one renamed _001-style id appears
        renamed = any(
            c["component_id"].endswith("_001")
            for c in high_doc["components"]
        )
        assert renamed
    # Wire clutter present
    assert any(
        c["component_type"] == "Wire"
        and c.get("metadata", {}).get("realism") == "wire_clutter"
        for c in high_doc["components"]
    )


def test_evaluator_netlist_v2_dir_mode_runs(tmp_path: Path):
    """Wire the exported corpus into ``evaluate_split`` via
    ``netlist_v2_dir`` and check the report comes back well-formed."""

    from app.domain.gnn.evaluator import EvaluationReport, evaluate_split
    from scripts.gnn_export_pseudo_real import main

    main([
        "--label-dir", str(LABEL_ROOT),
        "--output-root", str(tmp_path),
        "--profile", "clean",
        "--limit", "6",
    ])
    clean_dir = tmp_path / "clean"

    # split_ids inferred from what we just exported
    split_ids: list[str] = []
    for f in sorted(clean_dir.rglob("*.json")):
        if f.name == "manifest.json" or f.name.endswith(".meta.json"):
            continue
        rel = f.relative_to(clean_dir)
        # rel = "<ref_id>/<sample_id>.json"
        split_ids.append(rel.with_suffix("").as_posix())
    assert split_ids, "expected at least one split_id from the export"

    report = evaluate_split(
        LABEL_ROOT,
        split_ids=split_ids,
        advisor=None,
        netlist_v2_dir=clean_dir,
    )
    assert isinstance(report, EvaluationReport)
    assert report.n_samples >= 1
    # Clean profile should match the synthetic baseline exactly on rule path
    # (no noise → identical netlist_v2 except for added metadata tag)
    assert 0.0 <= report.rule_false_pass_rate <= 1.0
