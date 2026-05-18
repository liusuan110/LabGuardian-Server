"""Tests for ``app.domain.gnn.realism_noise`` (plan §十 R6 Phase 2)."""

from __future__ import annotations

import copy

import pytest

from app.domain.gnn.realism_noise import (
    ADD_WIRE_CLUTTER,
    CLEAN_PROFILE,
    DROP_IC_SUBTYPE,
    DROP_ROLE_KEEP_CANONICAL,
    HIGH_NOISE_PROFILE,
    IDENTITY,
    LOW_NOISE_PROFILE,
    LOWER_PIN_CONFIDENCE,
    PROFILES,
    RENAME_COMPONENTS,
    STRIP_ROLE_LABELS,
    RealismProfile,
    get_profile,
)


def _sample_netlist() -> dict:
    """Synthesised UA741 buffer netlist_v2 shaped like the evaluator
    adapter produces."""

    return {
        "scene_id": "test_synth",
        "board_schema_id": "test",
        "components": [
            {
                "component_id": "U1",
                "component_type": "IC",
                "package_type": "DIP8",
                "part_subtype": "UA741",
                "polarity": "none",
                "pins": [
                    {"pin_id": 0, "pin_name": "2", "hole_id": "H1",
                     "electrical_net_id": "VOUT", "confidence": 1.0},
                    {"pin_id": 1, "pin_name": "3", "hole_id": "H2",
                     "electrical_net_id": "VIN", "confidence": 1.0},
                ],
            },
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "package_type": "",
                "part_subtype": "",
                "polarity": "none",
                "pins": [
                    {"pin_id": 0, "pin_name": "pin1", "hole_id": "H3",
                     "electrical_net_id": "VOUT", "confidence": 1.0},
                ],
            },
        ],
        "nets": [
            {"electrical_net_id": "VIN", "canonical_name": "VIN",
             "role": "input", "manual_role": "input", "role_label": "VIN"},
            {"electrical_net_id": "VOUT", "canonical_name": "VOUT",
             "role": "output", "manual_role": "output", "role_label": "VOUT",
             "power_role": ""},
        ],
    }


# ---------------------------------------------------------------------------
# Per-operator behaviour
# ---------------------------------------------------------------------------


def test_identity_is_noop():
    nl = _sample_netlist()
    snapshot = copy.deepcopy(nl)
    out = CLEAN_PROFILE.apply(nl, seed=42)
    # CLEAN attaches a metadata tag but nothing else changes
    out_clean = copy.deepcopy(out)
    out_clean.pop("metadata", None)
    assert out_clean == snapshot
    # Original untouched
    assert nl == snapshot


def test_strip_role_labels_drops_label_role_label_and_power_role():
    nl = _sample_netlist()
    out = RealismProfile("t", (STRIP_ROLE_LABELS,)).apply(nl, seed=0)
    for net in out["nets"]:
        assert "role_label" not in net
        assert "manual_role" not in net
        assert "power_role" not in net
        # canonical_name re-rolled into NET_xxx pattern
        assert net["canonical_name"].startswith("NET_")


def test_drop_role_keep_canonical_keeps_canonical_name():
    nl = _sample_netlist()
    out = RealismProfile("t", (DROP_ROLE_KEEP_CANONICAL,)).apply(nl, seed=0)
    for net in out["nets"]:
        assert "role" not in net
        assert "manual_role" not in net
        assert net["canonical_name"]  # preserved


def test_lower_pin_confidence_drops_into_band():
    nl = _sample_netlist()
    out = RealismProfile("t", (LOWER_PIN_CONFIDENCE,)).apply(nl, seed=7)
    for comp in out["components"]:
        for pin in comp.get("pins", []):
            assert 0.30 <= pin["confidence"] <= 0.95


def test_rename_components_generates_type_prefixed_ids():
    nl = _sample_netlist()
    out = RealismProfile("t", (RENAME_COMPONENTS,)).apply(nl, seed=0)
    ids = sorted(c["component_id"] for c in out["components"])
    assert ids == ["IC_001", "Resistor_001"]
    # mapping is preserved in metadata for downstream alignment
    assert "U1" in out["metadata"]["realism_renames"]
    assert out["metadata"]["realism_renames"]["U1"] == "IC_001"


def test_drop_ic_subtype_clears_only_ic_components():
    nl = _sample_netlist()
    out = RealismProfile("t", (DROP_IC_SUBTYPE,)).apply(nl, seed=0)
    for comp in out["components"]:
        if comp["component_type"] == "IC":
            assert comp["part_subtype"] == ""
        # Non-IC subtypes untouched (R1 has "" already, no change)


def test_add_wire_clutter_appends_one_to_three_wires():
    nl = _sample_netlist()
    n_before = len(nl["components"])
    out = RealismProfile("t", (ADD_WIRE_CLUTTER,)).apply(nl, seed=3)
    n_after = len(out["components"])
    n_added = n_after - n_before
    assert 1 <= n_added <= 3
    wires = [c for c in out["components"] if c["component_type"] == "Wire"]
    assert len(wires) == n_added
    for w in wires:
        assert w["metadata"]["realism"] == "wire_clutter"


# ---------------------------------------------------------------------------
# Profile behaviour
# ---------------------------------------------------------------------------


def test_low_noise_profile_keeps_subtype_and_ids():
    nl = _sample_netlist()
    out = LOW_NOISE_PROFILE.apply(nl, seed=0)
    # subtype preserved
    ic = next(c for c in out["components"] if c["component_type"] == "IC")
    assert ic["part_subtype"] == "UA741"
    # component_id preserved
    ids = {c["component_id"] for c in out["components"]}
    assert ids == {"U1", "R1"}
    # role_label preserved
    assert all(n.get("role_label") for n in out["nets"])
    # but pin confidences degraded
    confidences = [
        p["confidence"]
        for c in out["components"]
        for p in c.get("pins", [])
    ]
    assert all(c < 1.0 for c in confidences)


def test_high_noise_profile_applies_all_relevant_operators():
    nl = _sample_netlist()
    out = HIGH_NOISE_PROFILE.apply(nl, seed=0)
    # IC subtype gone
    ic = next(c for c in out["components"] if c["component_type"] == "IC")
    assert ic["part_subtype"] == ""
    # IDs renamed
    assert all(
        c["component_id"].endswith("_001") or c["component_type"] == "Wire"
        for c in out["components"]
        if not c.get("metadata", {}).get("realism") == "wire_clutter"
    )
    # Wire clutter present
    n_clutter = sum(
        1 for c in out["components"]
        if c.get("metadata", {}).get("realism") == "wire_clutter"
    )
    assert n_clutter >= 1
    # Net labels dropped
    for net in out["nets"]:
        assert "role_label" not in net
        assert "manual_role" not in net


def test_profile_apply_is_deterministic():
    nl = _sample_netlist()
    out1 = HIGH_NOISE_PROFILE.apply(nl, seed=42)
    out2 = HIGH_NOISE_PROFILE.apply(nl, seed=42)
    assert out1 == out2
    out3 = HIGH_NOISE_PROFILE.apply(nl, seed=43)
    assert out1 != out3, "different seed must produce different output"


def test_profile_apply_does_not_mutate_input():
    nl = _sample_netlist()
    snap = copy.deepcopy(nl)
    HIGH_NOISE_PROFILE.apply(nl, seed=0)
    assert nl == snap, "apply() must not mutate caller's dict"


def test_get_profile_lookup():
    assert get_profile("clean") is CLEAN_PROFILE
    assert get_profile("low") is LOW_NOISE_PROFILE
    assert get_profile("high") is HIGH_NOISE_PROFILE
    with pytest.raises(KeyError, match="unknown realism profile"):
        get_profile("nope")


def test_profile_records_name_in_metadata():
    nl = _sample_netlist()
    out = LOW_NOISE_PROFILE.apply(nl, seed=0)
    assert out["metadata"]["realism_profile"] == "low"


def test_profiles_dict_lists_all_builtins():
    assert set(PROFILES) >= {"clean", "low", "high"}
    # No accidental sharing — each profile is a distinct object
    assert len({id(p) for p in PROFILES.values()}) == len(PROFILES)


def test_identity_operator_in_clean_profile():
    assert IDENTITY in CLEAN_PROFILE.operators
