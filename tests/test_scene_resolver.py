"""Unit tests for ``app.services.scene_resolver``.

WP-1 (2026-05-24): pins the fail-open behavior described in
``docs/retrieval-contract.md``. Topology-unknown turns MUST resolve to
``None`` (caller skips fault_case retrieval) rather than defaulting to
``exp_first_order_rc``.
"""

from __future__ import annotations

import pytest

from app.services.scene_resolver import (
    SCENE_ID_TO_DISPLAY_NAME,
    TOPOLOGY_LABEL_TO_SCENE_ID,
    VALID_SCENE_IDS,
    resolve_scene_id,
    scene_display_name,
)


# ---------------------------------------------------------------------------
# Static contract
# ---------------------------------------------------------------------------


def test_six_canonical_scenes_are_mapped() -> None:
    """Each of the 6 demo scenes must have a topology label that maps to it."""
    assert len(TOPOLOGY_LABEL_TO_SCENE_ID) == 6
    assert len(VALID_SCENE_IDS) == 6


def test_scene_ids_align_with_display_names() -> None:
    """Every resolved scene_id must have a Chinese display name."""
    for scene_id in VALID_SCENE_IDS:
        assert scene_id in SCENE_ID_TO_DISPLAY_NAME
        assert SCENE_ID_TO_DISPLAY_NAME[scene_id], f"empty name for {scene_id}"


def test_topology_labels_match_canonical_set() -> None:
    """The 6 mapped topology labels are the canonical classroom scene hints."""
    assert set(TOPOLOGY_LABEL_TO_SCENE_ID.keys()) == {
        "rc_first_order",
        "common_emitter",
        "differential_pair",
        "inverting_amp_ua741",
        "summing_amp_ua741",
        "integrator_ua741",
    }


# ---------------------------------------------------------------------------
# Resolution priority
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topology_label,expected_scene_id",
    list(TOPOLOGY_LABEL_TO_SCENE_ID.items()),
)
def test_resolve_from_station_topology_label(topology_label: str, expected_scene_id: str) -> None:
    """station['topology_label'] resolves to the matched scene_id."""
    assert resolve_scene_id(station={"topology_label": topology_label}) == expected_scene_id


def test_explicit_scene_id_takes_precedence() -> None:
    """station['scene_id'] (explicit override) beats topology_label."""
    result = resolve_scene_id(
        station={
            "scene_id": "exp_ua741_integrator",
            "topology_label": "rc_first_order",  # would otherwise win
        }
    )
    assert result == "exp_ua741_integrator"


def test_falls_back_to_comparison_report_topology_label() -> None:
    """When station has no label, validator report can supply one."""
    result = resolve_scene_id(
        station={},
        comparison_report={"topology_label": "common_emitter"},
    )
    assert result == "exp_common_emitter_amplifier"


# ---------------------------------------------------------------------------
# Fail-open contract — the WP-1 hard rule
# ---------------------------------------------------------------------------


def test_returns_none_when_no_topology_context() -> None:
    """No topology hint anywhere → None. MUST NOT default to RC."""
    assert resolve_scene_id() is None
    assert resolve_scene_id(station={}, comparison_report={}) is None


def test_returns_none_for_unknown_label() -> None:
    """Open-set 'unknown' label → None (don't guess a scene)."""
    assert resolve_scene_id(station={"topology_label": "unknown"}) is None
    assert resolve_scene_id(station={"topology_label": "totally_made_up"}) is None


def test_returns_none_for_invalid_explicit_scene_id() -> None:
    """An invalid explicit scene_id is rejected, not passed through."""
    assert resolve_scene_id(station={"scene_id": "exp_nonsense"}) is None


def test_empty_strings_and_whitespace_dont_count() -> None:
    """Whitespace-only labels must not match anything."""
    assert resolve_scene_id(station={"scene_id": "   ", "topology_label": ""}) is None


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------


def test_scene_display_name_for_none() -> None:
    assert scene_display_name(None) == ""
    assert scene_display_name("") == ""


def test_scene_display_name_for_known_scene() -> None:
    assert scene_display_name("exp_ua741_inverting_amplifier") == "UA741 反相放大器"
