"""Regression guards for the 2026-05-19 hole→net pipeline audit.

Three independent fixes land together; this file pins their contract so
no future refactor silently re-introduces the bugs:

- **B0** — ``netlist_v2.board_topology`` exposes the full
  ``node→holes`` map and the effective rail_assignments. Frontend uses
  it to highlight conducting strips on drag.
- **B1** — ``hole_id`` is the source of truth for
  ``electrical_node_id``. Stale upstream values never override the
  schema's recomputation as long as ``hole_id`` is set.
- **B2** — default rail labels live on ``BoardSchema``, not as a
  module-level constant. Each schema can ship its own defaults.
"""

from __future__ import annotations

from app.domain.board_schema import BoardSchema
from app.domain.circuit import CircuitAnalyzer
from app.domain.netlist_models import ComponentInstance, PinAssignment


def _make_resistor(component_id: str, pin1_hole: str, pin2_hole: str,
                   *, pin2_stale_node: str | None = None) -> ComponentInstance:
    """Build a minimal Resistor with optional **deliberately stale**
    electrical_node_id on pin2 — used to prove B1 ignores the stale
    value."""
    return ComponentInstance(
        component_id=component_id,
        component_type="Resistor",
        package_type="axial",
        pins=[
            PinAssignment(
                pin_id=1, pin_name="pin1",
                hole_id=pin1_hole, electrical_node_id=None,
            ),
            PinAssignment(
                pin_id=2, pin_name="pin2",
                hole_id=pin2_hole, electrical_node_id=pin2_stale_node,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# B0 — board_topology export
# ---------------------------------------------------------------------------


def test_b0_board_topology_present_in_netlist_v2():
    """Every CircuitAnalyzer.export_netlist_v2 response must carry the
    board_topology field; frontend can't render strip highlights
    without it."""

    schema = BoardSchema.default_breadboard()
    analyzer = CircuitAnalyzer(board_schema=schema)
    analyzer.add_component_instance(_make_resistor("R1", "B5", "D5"))

    netlist = analyzer.export_netlist_v2()
    assert "board_topology" in netlist
    topo = netlist["board_topology"]
    assert topo["schema_id"] == schema.schema_id
    assert topo["board_type"] == schema.board_type
    assert "node_to_holes" in topo
    assert "rail_assignments" in topo


def test_b0_node_to_holes_lists_full_strip_not_just_pin_holes():
    """Even though only B5 has a pin, ``ROW_5_L`` must enumerate
    A5..E5 — the whole 5-hole conducting strip — so the frontend can
    highlight legal drag targets."""

    schema = BoardSchema.default_breadboard()
    analyzer = CircuitAnalyzer(board_schema=schema)
    analyzer.add_component_instance(_make_resistor("R1", "B5", "D7"))

    netlist = analyzer.export_netlist_v2()
    n2h = netlist["board_topology"]["node_to_holes"]
    assert n2h["ROW_5_L"] == ["A5", "B5", "C5", "D5", "E5"]
    # Right-side strip on same row stays separate (trough between E/F)
    assert n2h["ROW_5_R"] == ["F5", "G5", "H5", "I5", "J5"]
    # Power rail segments expose all rail holes
    assert n2h["TRACK_LP_SEG1"][:3] == ["LP1", "LP10", "LP11"]
    assert len(n2h["TRACK_LP_SEG2"]) == 32   # rows 32-63


def test_b0_node_index_unchanged_only_pinned_holes():
    """B0 is purely additive — the existing ``node_index`` field still
    contains **only** holes with pins (per its docstring contract).
    Frontend that already reads node_index keeps working."""

    schema = BoardSchema.default_breadboard()
    analyzer = CircuitAnalyzer(board_schema=schema)
    analyzer.add_component_instance(_make_resistor("R1", "B5", "D7"))

    netlist = analyzer.export_netlist_v2()
    ni = netlist["node_index"]
    # B5 is the only pin in ROW_5_L; the rest of the strip is NOT in node_index
    assert ni["ROW_5_L"] == ["B5"]
    assert "A5" not in ni.get("ROW_5_L", [])


# ---------------------------------------------------------------------------
# B1 — hole_id is source of truth
# ---------------------------------------------------------------------------


def test_b1_hole_id_beats_stale_electrical_node_id():
    """**The core fix**: if the caller sent ``hole_id=B5`` (→ ROW_5_L)
    AND ``electrical_node_id="WRONG_STALE_NODE"``, the analyzer must
    follow hole_id, not the stale annotation. Previously the stale
    value won and silently broke manual-correction flows."""

    schema = BoardSchema.default_breadboard()
    analyzer = CircuitAnalyzer(board_schema=schema)
    # R1.pin2 carries a deliberately wrong electrical_node_id
    analyzer.add_component_instance(
        _make_resistor("R1", "B5", "D5", pin2_stale_node="STALE_OLD_NODE"),
    )

    # Both pins should land on ROW_5_L (same 5-hole strip)
    pin_nodes = analyzer._instance_pin_nodes(analyzer.component_instances[0])
    assert [node_id for _pin, node_id in pin_nodes] == ["ROW_5_L", "ROW_5_L"]
    # And export confirms only ONE electrical net (the strip)
    netlist = analyzer.export_netlist_v2()
    pin_net_ids = [
        pin["electrical_net_id"]
        for comp in netlist["components"] for pin in comp["pins"]
    ]
    assert pin_net_ids[0] == pin_net_ids[1]


def test_b1_missing_hole_id_falls_back_to_supplied_node_id():
    """Pins genuinely without a hole_id (vision miss) must still be
    placeable via an explicit electrical_node_id — that's the only
    sane fallback. The change in B1 is about *priority*, not removing
    the fallback entirely."""

    schema = BoardSchema.default_breadboard()
    comp = ComponentInstance(
        component_id="R_floating",
        component_type="Resistor",
        package_type="axial",
        pins=[
            PinAssignment(pin_id=1, pin_name="pin1",
                          hole_id="", electrical_node_id="SUPPLIED_NODE"),
            PinAssignment(pin_id=2, pin_name="pin2", hole_id="A2"),
        ],
    )
    analyzer = CircuitAnalyzer(board_schema=schema)
    analyzer.add_component_instance(comp)
    pin_nodes = analyzer._instance_pin_nodes(analyzer.component_instances[0])
    # pin1 has no hole → uses supplied "SUPPLIED_NODE"
    assert pin_nodes[0][1] == "SUPPLIED_NODE"
    # pin2 has hole_id A2 → schema-resolved ROW_2_L
    assert pin_nodes[1][1] == "ROW_2_L"


# ---------------------------------------------------------------------------
# B2 — schema-level rail defaults
# ---------------------------------------------------------------------------


def test_b2_default_rail_assignments_on_schema():
    """The competition breadboard ships sensible defaults so callers
    that don't pass rail_assignments still get reasonable behaviour."""

    schema = BoardSchema.default_breadboard()
    defaults = schema.default_rail_assignments()
    assert defaults == {
        "top_plus": "VCC",
        "top_minus": "VCC",
        "bot_plus": "GND",
        "bot_minus": "VEE",
    }
    # Defensive: returned dict must be a copy so caller mutation doesn't
    # leak into the schema's permanent state
    defaults["top_plus"] = "MUTATED"
    assert schema.default_rail_assignments()["top_plus"] == "VCC"


def test_b2_run_topology_echoes_effective_rails_into_board_topology():
    """When the caller passes explicit rail_assignments they must show
    up in ``board_topology.rail_assignments`` so the frontend can
    render the right rail labels."""

    from app.pipeline.stages.s3_topology import run_topology

    components = [
        {
            "component_id": "R1",
            "component_type": "Resistor",
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "A5"},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "LP10"},
            ],
        },
    ]
    out = run_topology(components, rail_assignments={"top_plus": "V_BIAS"})
    netlist = out["netlist_v2"]
    rails = netlist["board_topology"]["rail_assignments"]
    # Caller override is respected
    assert rails["top_plus"] == "V_BIAS"
    # Defaults still in effect for the keys the caller didn't pass
    assert rails["bot_minus"] == "VEE"
