"""P0.8 · ComponentAlignment tests.

Covers:
- identity_alignment auto-aligns matching source_ids
- partial mismatch → notes records unmatched_*
- alignment_from_dicts honors explicit dicts; filters invalid entries
- map_ref_port_to_cur_port_id resolves correctly
- map_ref_net_to_cur_net_id resolves correctly
- to_dict / alignment_from_dict_payload round-trip
"""

from __future__ import annotations

import json
from pathlib import Path

from app.domain.gnn import (
    alignment_from_dict_payload,
    alignment_from_dicts,
    build_from_logical_reference,
    identity_alignment,
)
from app.domain.gnn.port_graph import build_hetero_circuit_graph

from .conftest import hcg_to_cur_nx

FIXTURE_RC = Path(__file__).resolve().parents[2] / "fixtures" / "references" / "test_rc_v1.json"


def _build_ref_and_cur(perturbations=None):
    ref = build_from_logical_reference(json.loads(FIXTURE_RC.read_text(encoding="utf-8")))
    cur_g = hcg_to_cur_nx(ref, perturbations=perturbations)
    cur = build_hetero_circuit_graph(cur_g, side="cur")
    return ref, cur


def test_identity_alignment_self_pair() -> None:
    ref, cur = _build_ref_and_cur()
    a = identity_alignment(ref, cur)
    assert a.ref_to_cur_component == {"R1": "R1", "C1": "C1"}
    assert a.ref_to_cur_net == {"VIN": "VIN", "VC": "VC", "GND": "GND"}
    assert a.cur_to_ref_component == {"R1": "R1", "C1": "C1"}
    assert a.notes["constructor"] == "identity_alignment"
    assert a.notes["unmatched_ref_components"] == []
    assert a.notes["unmatched_cur_components"] == []


def test_identity_alignment_missing_component_in_cur() -> None:
    """If perturbation drops C1 from cur, alignment notes record it."""

    ref, cur = _build_ref_and_cur(perturbations=[("remove_component", "C1")])
    a = identity_alignment(ref, cur)
    assert "R1" in a.ref_to_cur_component
    assert "C1" not in a.ref_to_cur_component
    assert "C1" in a.notes["unmatched_ref_components"]


def test_alignment_from_dicts_explicit_renaming() -> None:
    ref, cur = _build_ref_and_cur(
        perturbations=[
            ("rename_component", "R1", "U_R_3"),
            ("rename_component", "C1", "U_C_1"),
            ("rename_net", "VIN", "n_07"),
        ]
    )
    a = alignment_from_dicts(
        ref,
        cur,
        component_map={"R1": "U_R_3", "C1": "U_C_1"},
        net_map={"VIN": "n_07", "VC": "VC", "GND": "GND"},
    )
    assert a.ref_to_cur_component["R1"] == "U_R_3"
    assert a.ref_to_cur_net["VIN"] == "n_07"
    # reverse cache
    assert a.cur_to_ref_component["U_R_3"] == "R1"


def test_alignment_from_dicts_filters_invalid_entries() -> None:
    """Entries pointing to non-existent ids on either side are dropped."""

    ref, cur = _build_ref_and_cur()
    a = alignment_from_dicts(
        ref,
        cur,
        component_map={"R1": "R1", "BOGUS_REF": "BOGUS_CUR", "C1": "MISSING"},
        net_map={"VIN": "VIN"},
    )
    # R1 valid; BOGUS_REF / C1→MISSING invalid → dropped
    assert "R1" in a.ref_to_cur_component
    assert "BOGUS_REF" not in a.ref_to_cur_component
    assert "C1" not in a.ref_to_cur_component
    assert "BOGUS_REF" in a.notes["unmatched_ref_components"]


def test_map_ref_port_to_cur_port_id_identity() -> None:
    ref, cur = _build_ref_and_cur()
    a = identity_alignment(ref, cur)
    assert (
        a.map_ref_port_to_cur_port_id("ref_port:R1.pin1", ref, cur)
        == "cur_port:R1.pin1"
    )
    # Missing ref port → None
    assert (
        a.map_ref_port_to_cur_port_id("ref_port:BOGUS.pin1", ref, cur) is None
    )


def test_map_ref_net_to_cur_net_id_identity_and_missing() -> None:
    ref, cur = _build_ref_and_cur()
    a = identity_alignment(ref, cur)
    assert a.map_ref_net_to_cur_net_id("ref_net:VIN", cur) == "cur_net:VIN"
    # Unmapped net
    assert a.map_ref_net_to_cur_net_id("ref_net:BOGUS", cur) is None
    # Wrong prefix
    assert a.map_ref_net_to_cur_net_id("not_a_ref_net", cur) is None


def test_map_cur_port_back_to_ref_port_id() -> None:
    ref, cur = _build_ref_and_cur()
    a = identity_alignment(ref, cur)
    assert (
        a.map_cur_port_to_ref_port_id("cur_port:R1.pin1", ref, cur)
        == "ref_port:R1.pin1"
    )
    # Renamed scenario
    ref2, cur2 = _build_ref_and_cur(
        perturbations=[("rename_component", "R1", "U_R_3")]
    )
    a2 = alignment_from_dicts(
        ref2,
        cur2,
        component_map={"R1": "U_R_3", "C1": "C1"},
        net_map={"VIN": "VIN", "VC": "VC", "GND": "GND"},
    )
    assert (
        a2.map_cur_port_to_ref_port_id("cur_port:U_R_3.pin1", ref2, cur2)
        == "ref_port:R1.pin1"
    )


def test_alignment_to_dict_roundtrip() -> None:
    ref, cur = _build_ref_and_cur()
    a = identity_alignment(ref, cur)
    payload = a.to_dict()
    assert payload["ref_to_cur_component"] == a.ref_to_cur_component
    assert payload["ref_to_cur_net"] == a.ref_to_cur_net
    a2 = alignment_from_dict_payload(payload, ref, cur)
    assert a2.ref_to_cur_component == a.ref_to_cur_component
    assert a2.ref_to_cur_net == a.ref_to_cur_net


def test_alignment_is_frozen() -> None:
    ref, cur = _build_ref_and_cur()
    a = identity_alignment(ref, cur)
    import pytest

    with pytest.raises((AttributeError, TypeError)):
        a.ref_to_cur_component = {}  # type: ignore[misc]
