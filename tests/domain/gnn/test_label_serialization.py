"""P0.8 · serialize / deserialize round-trip + JSON schema invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.gnn import (
    SCHEMA_VERSION,
    LabelBuildResult,
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
