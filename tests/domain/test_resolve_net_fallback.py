"""R11 follow-up regression — ``_resolve_net`` must fall through on miss.

Pre-2026-05-19 the resolver short-circuited on the first non-empty
locator field, so a frontend that sent a stale or synthetic
``electrical_net_id`` (e.g. the ``LOCAL_NET_<i>`` IDs the R11
frontend recompute mints) would have its whole annotation silently
dropped — even though the same payload carried a perfectly valid
``hole_id`` / ``component_id`` + ``pin_name`` / ``electrical_node_id``.

The user reported "I annotated UI1/UO1 but nothing reacted"; this
file pins the fix so we never regress.
"""

from __future__ import annotations

from app.domain.net_normalization import _resolve_net


def _make_indexes() -> dict[str, dict]:
    net_a = {"electrical_net_id": "NET_009", "member_hole_ids": ["B16", "C16"]}
    net_b = {"electrical_net_id": "NET_000", "member_hole_ids": ["F21", "G21"]}
    return {
        "id": {"NET_009": net_a, "NET_000": net_b},
        "hole": {"B16": net_a, "C16": net_a, "F21": net_b, "G21": net_b},
        "node": {"ROW_16_R": net_a, "ROW_21_L": net_b},
        "comp_pin": {
            ("R1", "pin1"): net_a,
            ("R_f", "pin2"): net_b,
        },
    }


# ---------------------------------------------------------------------------
# Happy path — every locator type still resolves on its own
# ---------------------------------------------------------------------------


def test_resolves_by_electrical_net_id():
    idx = _make_indexes()
    net, src = _resolve_net({"electrical_net_id": "NET_009"}, idx)
    assert net is not None
    assert net["electrical_net_id"] == "NET_009"
    assert src == "electrical_net_id"


def test_resolves_by_component_pin_when_net_id_absent():
    idx = _make_indexes()
    net, src = _resolve_net({"component_id": "R1", "pin_name": "pin1"}, idx)
    assert net is not None and net["electrical_net_id"] == "NET_009"
    assert src == "component_pin"


def test_resolves_by_hole_id_when_others_absent():
    idx = _make_indexes()
    net, src = _resolve_net({"hole_id": "F21"}, idx)
    assert net is not None and net["electrical_net_id"] == "NET_000"
    assert src == "hole_id"


def test_resolves_by_electrical_node_id_when_others_absent():
    idx = _make_indexes()
    net, src = _resolve_net({"electrical_node_id": "ROW_21_L"}, idx)
    assert net is not None and net["electrical_net_id"] == "NET_000"
    assert src == "electrical_node_id"


# ---------------------------------------------------------------------------
# Fallback — the bug fix
# ---------------------------------------------------------------------------


def test_falls_through_to_hole_id_when_electrical_net_id_unknown():
    """The bug. Frontend sends a synthetic ``LOCAL_NET_X`` plus a
    correct ``hole_id``; old code returned (None, electrical_net_id)
    and the annotation was dropped. Now it must fall through to
    hole_id and return the real net."""

    idx = _make_indexes()
    net, src = _resolve_net(
        {
            "electrical_net_id": "LOCAL_NET_0",      # synthetic, not in indexes
            "hole_id": "B16",                          # stable, resolvable
            "electrical_node_id": "ROW_16_R",
            "component_id": "R1",
            "pin_name": "pin1",
        },
        idx,
    )
    assert net is not None, "fallback must succeed; old code returned None"
    assert net["electrical_net_id"] == "NET_009"
    # Source comes from whichever field actually resolved; here component_pin
    # is checked before hole_id, so that wins.
    assert src == "component_pin"


def test_falls_through_skipping_unresolvable_component_pin():
    """Mid-priority fallback: net_id and component_pin both miss; hole_id
    is the surviving locator."""

    idx = _make_indexes()
    net, src = _resolve_net(
        {
            "electrical_net_id": "LOCAL_NET_0",
            "component_id": "RENAMED_R1",            # ID frontend made up
            "pin_name": "pin1",
            "hole_id": "B16",                          # truth
        },
        idx,
    )
    assert net is not None
    assert net["electrical_net_id"] == "NET_009"
    assert src == "hole_id"


def test_falls_through_to_node_id_as_last_resort():
    """node_id is the lowest-priority locator; should still work."""

    idx = _make_indexes()
    net, src = _resolve_net(
        {
            "electrical_net_id": "LOCAL_NET_99",       # miss
            "component_id": "GHOST",                   # miss
            "pin_name": "pinX",
            "hole_id": "NOT_A_HOLE",                   # miss
            "electrical_node_id": "ROW_16_R",          # hit
        },
        idx,
    )
    assert net is not None
    assert net["electrical_net_id"] == "NET_009"
    assert src == "electrical_node_id"


def test_returns_none_when_every_locator_misses():
    """Hard miss is still a hard miss; the function must not pretend to
    resolve. ``source`` should reflect the first attempted locator so
    downstream log shape stays compatible."""

    idx = _make_indexes()
    net, src = _resolve_net(
        {
            "electrical_net_id": "LOCAL_NET_99",
            "hole_id": "NOT_A_HOLE",
            "electrical_node_id": "ROW_99_L",
        },
        idx,
    )
    assert net is None
    # First attempted locator was electrical_net_id; preserve its tag for log compatibility.
    assert src == "electrical_net_id"


def test_returns_empty_source_when_no_locators_provided():
    idx = _make_indexes()
    net, src = _resolve_net({}, idx)
    assert net is None
    assert src == ""
