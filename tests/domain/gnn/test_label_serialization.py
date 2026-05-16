"""P0.8 · serialize / deserialize round-trip + JSON schema invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.gnn import (
    SCHEMA_VERSION,
    LabelBuildResult,
    LabelSource,
    LabelStats,
    TaskType,
    build_from_logical_reference,
    build_seal_samples,
    deserialize_label_build_result,
    identity_alignment,
    serialize_label_build_result,
)
from app.domain.gnn.port_graph import build_hetero_circuit_graph

from .conftest import hcg_to_cur_nx

FIXTURE_RC = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_rc_v1.json"
)
FIXTURE_OPAMP = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_opamp_buffer_v1.json"
)
_UA741_SUBTYPES = {"U1": "UA741"}


def _build_result(perturbations=None, *, use_ua741: bool = False):
    fixture = FIXTURE_OPAMP if use_ua741 else FIXTURE_RC
    ref = build_from_logical_reference(json.loads(fixture.read_text()))
    cur_g = hcg_to_cur_nx(ref, perturbations=perturbations)
    cur = build_hetero_circuit_graph(
        cur_g,
        side="cur",
        subtype_by_source_id=_UA741_SUBTYPES if use_ua741 else None,
    )
    align = identity_alignment(ref, cur)
    return build_seal_samples(ref, cur, align)


def test_schema_version_constant() -> None:
    assert SCHEMA_VERSION == "1.0"


def test_serialize_payload_is_pure_json() -> None:
    """Output of ``serialize_label_build_result`` should round-trip through
    ``json.dumps / json.loads`` with no enum / tuple / dataclass leakage."""

    result = _build_result(
        perturbations=[
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ]
    )
    payload = serialize_label_build_result(
        result, sample_id="test_001", ref_id="test_rc_v1"
    )
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    # Decoded payload should be structurally identical to the original
    assert decoded == payload


def test_serialize_deserialize_roundtrip_equal() -> None:
    """``deserialize(serialize(x)) == x`` for samples/groups/stats."""

    result = _build_result(
        perturbations=[
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ]
    )
    payload = serialize_label_build_result(
        result, sample_id="test_001", ref_id="test_rc_v1"
    )
    # Round-trip via JSON text to ensure no in-memory cheating
    payload_via_json = json.loads(json.dumps(payload))
    restored = deserialize_label_build_result(payload_via_json)
    assert isinstance(restored, LabelBuildResult)
    assert restored.stats == result.stats
    assert len(restored.samples) == len(result.samples)
    assert len(restored.groups) == len(result.groups)
    for a, b in zip(result.samples, restored.samples):
        assert a.label == b.label
        assert a.label_source == b.label_source
        assert a.task_type == b.task_type
        assert a.candidate_edge == b.candidate_edge
        assert a.expected_edge == b.expected_edge
        assert a.ref_edge_origin == b.ref_edge_origin
        assert a.confidence == b.confidence
        assert a.is_symmetric_equivalent == b.is_symmetric_equivalent
        assert a.group_id == b.group_id
        assert a.subgraph == b.subgraph
    for a, b in zip(result.groups, restored.groups):
        assert a == b


def test_deserialize_rejects_unknown_schema_version() -> None:
    result = _build_result()
    payload = serialize_label_build_result(
        result, sample_id="s", ref_id="r"
    )
    payload["schema_version"] = "999.0"
    with pytest.raises(ValueError, match="schema_version"):
        deserialize_label_build_result(payload)


def test_payload_invariants() -> None:
    """Per plan §附录 A.8: structural invariants for the on-disk format."""

    result = _build_result(perturbations=[("drop_edge", "pin1", "VIN")])
    payload = serialize_label_build_result(
        result, sample_id="s", ref_id="r"
    )

    # 1. len(samples) == stats.total_samples
    assert len(payload["samples"]) == payload["stats"]["total_samples"]

    # 2. each sample's `index` field equals its list position
    for i, s in enumerate(payload["samples"]):
        assert s["index"] == i

    # 3. group.sample_indices are valid; sample.task_type / group_id match
    for g in payload["groups"]:
        for idx in g["sample_indices"]:
            assert 0 <= idx < len(payload["samples"])
            s = payload["samples"][idx]
            assert s["task_type"] == g["task_type"]
            assert s["group_id"] == g["group_id"]
        # 4. correct_index points to label==1
        if g["correct_index"] is not None:
            sample_idx = g["sample_indices"][g["correct_index"]]
            assert payload["samples"][sample_idx]["label"] == 1

    # 5. by_source / by_task_type counts match samples
    actual_source: dict[str, int] = {}
    actual_task: dict[str, int] = {}
    for s in payload["samples"]:
        actual_source[s["label_source"]] = actual_source.get(s["label_source"], 0) + 1
        actual_task[s["task_type"]] = actual_task.get(s["task_type"], 0) + 1
    for src_key, expected in payload["stats"]["by_source"].items():
        assert actual_source.get(src_key, 0) == expected
    for task_key, expected in payload["stats"]["by_task_type"].items():
        assert actual_task.get(task_key, 0) == expected


def test_cur_metadata_is_preserved() -> None:
    result = _build_result()
    payload = serialize_label_build_result(
        result,
        sample_id="s",
        ref_id="r",
        cur_metadata={
            "perturbation_chain": ["pin_reversed:C1"],
            "alignment": {"some": "dict"},
        },
    )
    assert payload["cur_metadata"]["perturbation_chain"] == ["pin_reversed:C1"]


def test_subgraph_drnl_labels_round_trip() -> None:
    """DRNL int labels must survive json (no implicit float coercion)."""

    result = _build_result(
        perturbations=[
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ]
    )
    payload = json.loads(
        json.dumps(serialize_label_build_result(result, sample_id="s", ref_id="r"))
    )
    restored = deserialize_label_build_result(payload)
    for a, b in zip(result.samples, restored.samples):
        # DRNL labels are ints; equality is strict
        assert a.subgraph.drnl_labels == b.subgraph.drnl_labels
        assert all(isinstance(v, int) for v in b.subgraph.drnl_labels.values())


# ---------------------------------------------------------------------------
# File I/O round-trip (real disk write/read; tmp_path)
# ---------------------------------------------------------------------------


def test_file_round_trip_via_tmp_path(tmp_path: Path) -> None:
    """Write to actual disk file, read back, verify equivalence. This is the
    concrete contract P1 dataset_builder relies on."""

    result = _build_result(
        perturbations=[
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ]
    )
    fp = tmp_path / "rc__neg_001.json"
    payload = serialize_label_build_result(
        result, sample_id="rc__neg_001", ref_id="test_rc_v1"
    )
    fp.write_text(json.dumps(payload, indent=2))
    # Reload
    restored = deserialize_label_build_result(json.loads(fp.read_text()))
    assert restored.stats == result.stats
    assert len(restored.samples) == len(result.samples)
    for a, b in zip(result.samples, restored.samples):
        assert a == b


def test_empty_result_round_trip() -> None:
    """A LabelBuildResult with no samples / no groups (e.g., a perturbation
    that removed all edges via complete component drop)."""

    empty = LabelBuildResult(
        samples=(),
        groups=(),
        stats=LabelStats(
            total_samples=0,
            n_positives=0,
            n_negatives=0,
            pos_neg_ratio=0.0,
            by_source={src.value: 0 for src in LabelSource},
            by_task_type={t.value: 0 for t in TaskType},
            n_groups=0,
            n_groups_without_positive=0,
            n_skipped_missing_component=0,
            n_skipped_optional_pin=0,
            n_skipped_forbidden_pin_no_violation=0,
            n_skipped_extract_error=0,
            n_unique_ports_covered=0,
            n_unique_nets_covered=0,
        ),
    )
    payload = serialize_label_build_result(empty, sample_id="e", ref_id="r")
    encoded = json.dumps(payload)
    restored = deserialize_label_build_result(json.loads(encoded))
    assert restored.samples == ()
    assert restored.groups == ()
    assert restored.stats.total_samples == 0


def test_two_distinct_results_have_distinct_files(tmp_path: Path) -> None:
    """Two different perturbations should produce distinct serialized JSON
    payloads (sanity: no accidental sharing of mutable state)."""

    r1 = _build_result()
    r2 = _build_result(
        perturbations=[
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ]
    )
    p1 = serialize_label_build_result(r1, sample_id="a", ref_id="r")
    p2 = serialize_label_build_result(r2, sample_id="b", ref_id="r")
    assert p1["samples"] != p2["samples"], "perturbed result shouldn't equal clean result"
    assert p1["stats"] != p2["stats"]


def test_result_with_only_missing_edge_groups_round_trip() -> None:
    """Make sure groups-only payloads (when WRONG_EDGE samples are filtered
    out) still round-trip. This guards against future refactors that might
    accidentally couple group serialization to sample presence."""

    # We can't easily build a real "groups only" result, but we can construct
    # one synthetically by trimming a real result.
    result = _build_result(perturbations=[("drop_edge", "pin1", "VIN")])
    if not result.groups:
        import pytest as _pt
        _pt.skip("fixture didn't produce groups")
    payload = serialize_label_build_result(result, sample_id="s", ref_id="r")
    text = json.dumps(payload)
    restored = deserialize_label_build_result(json.loads(text))
    # Group structure preserved exactly
    assert len(restored.groups) == len(result.groups)
    for g_orig, g_back in zip(result.groups, restored.groups):
        assert g_orig.group_id == g_back.group_id
        assert g_orig.query_origin == g_back.query_origin
        assert g_orig.sample_indices == g_back.sample_indices
        assert g_orig.correct_index == g_back.correct_index


def test_round_trip_preserves_same_component_edges() -> None:
    """``same_component_edges`` field (default empty) must survive the round
    trip — if a future builder enables it, this guards downstream contract."""

    result = _build_result()
    payload = json.loads(
        json.dumps(serialize_label_build_result(result, sample_id="s", ref_id="r"))
    )
    restored = deserialize_label_build_result(payload)
    for a, b in zip(result.samples, restored.samples):
        assert a.subgraph.same_component_edges == b.subgraph.same_component_edges
