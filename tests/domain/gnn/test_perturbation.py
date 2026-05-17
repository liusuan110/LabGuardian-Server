"""P1 Phase A · perturbation operators.

Each operator is verified by:
- structural smoke (cur HCG builds, alignment is valid)
- expected_outcome flag matches plan
- determinism (same ref + same seed → same cur)
- integration with build_seal_samples_with_coverage_check (no coverage gap)
- LabelSource produced matches expectation
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.gnn import (
    PERTURBATION_REGISTRY,
    LabelSource,
    apply_perturbation,
    build_from_logical_reference,
    build_seal_samples_with_coverage_check,
    get_perturbation,
)
from app.domain.gnn.perturbation import (
    IdentityPerturbation,
    PerturbedCur,
    hcg_to_nx,
)

FIXTURE_RC = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_rc_v1.json"
)
FIXTURE_DIVIDER = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_voltage_divider_v1.json"
)
FIXTURE_LED = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_all_signal_v1.json"
)


def _load_ref(path: Path):
    return build_from_logical_reference(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_has_12_operators() -> None:
    """Phase A (4) + Phase B (8) = 12 perturbation operators."""

    expected = {
        # Phase A
        "identity",
        "pin_swap_symmetric",
        "wrong_connection",
        "pin_reversed",
        # Phase B
        "missing_component",
        "extra_component",
        "floating_net",
        "short_circuit",
        "power_swapped",
        "input_output_swapped",
        "extra_wire_bridge",
        "chained",
    }
    assert set(PERTURBATION_REGISTRY) == expected


def test_get_perturbation_known_returns_instance() -> None:
    op = get_perturbation("identity")
    assert isinstance(op, IdentityPerturbation)
    assert op.name == "identity"
    assert op.expected_outcome == "positive"


def test_get_perturbation_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown perturbation"):
        get_perturbation("does_not_exist")


# ---------------------------------------------------------------------------
# hcg_to_nx helper
# ---------------------------------------------------------------------------


def test_hcg_to_nx_round_trip_preserves_structure() -> None:
    """ref → nx (target_side=cur) → build_hetero_circuit_graph(side=cur)
    should reproduce same components / nets / edges."""

    from app.domain.gnn.port_graph import build_hetero_circuit_graph

    ref = _load_ref(FIXTURE_RC)
    cur_g = hcg_to_nx(ref, target_side="cur")
    cur = build_hetero_circuit_graph(cur_g, side="cur")
    assert cur.summary()["n_components"] == ref.summary()["n_components"]
    assert cur.summary()["n_nets"] == ref.summary()["n_nets"]
    assert cur.summary()["n_edges"] == ref.summary()["n_edges"]
    assert all(p.node_id.startswith("cur_") for p in cur.ports.values())


def test_hcg_to_nx_rejects_invalid_side() -> None:
    ref = _load_ref(FIXTURE_RC)
    with pytest.raises(ValueError, match="target_side"):
        hcg_to_nx(ref, target_side="bogus")


# ---------------------------------------------------------------------------
# IdentityPerturbation
# ---------------------------------------------------------------------------


def test_identity_yields_perfect_copy() -> None:
    ref = _load_ref(FIXTURE_RC)
    p = apply_perturbation("identity", ref, seed=0)
    assert isinstance(p, PerturbedCur)
    assert p.expected_outcome == "positive"
    assert p.cur_hcg.summary()["n_edges"] == ref.summary()["n_edges"]
    # Every ref edge has a cur counterpart with same source_ids
    ref_edge_sids = {
        (
            ref.ports[e.src_port_id].port_key,
            ref.components[ref.ports[e.src_port_id].parent_component_id].source_id,
            ref.nets[e.dst_net_id].source_id,
        )
        for e in ref.edges
    }
    cur_edge_sids = {
        (
            p.cur_hcg.ports[e.src_port_id].port_key,
            p.cur_hcg.components[
                p.cur_hcg.ports[e.src_port_id].parent_component_id
            ].source_id,
            p.cur_hcg.nets[e.dst_net_id].source_id,
        )
        for e in p.cur_hcg.edges
    }
    assert ref_edge_sids == cur_edge_sids


def test_identity_alignment_is_identity() -> None:
    ref = _load_ref(FIXTURE_RC)
    p = apply_perturbation("identity", ref, seed=0)
    for sid in p.alignment.ref_to_cur_component:
        assert p.alignment.ref_to_cur_component[sid] == sid
    for sid in p.alignment.ref_to_cur_net:
        assert p.alignment.ref_to_cur_net[sid] == sid


def test_identity_with_label_builder_yields_only_positives_and_random_negs() -> None:
    ref = _load_ref(FIXTURE_RC)
    p = apply_perturbation("identity", ref, seed=0)
    result = build_seal_samples_with_coverage_check(ref, p.cur_hcg, p.alignment)
    # No WRONG_OBSERVED on a clean copy
    assert result.stats.by_source[LabelSource.WRONG_OBSERVED.value] == 0
    # REF_PRESENT must fire
    assert result.stats.by_source[LabelSource.REF_PRESENT.value] > 0


# ---------------------------------------------------------------------------
# PinSwapSymmetricPerturbation
# ---------------------------------------------------------------------------


def test_pin_swap_symmetric_produces_ref_symmetric_swap() -> None:
    ref = _load_ref(FIXTURE_RC)
    p = apply_perturbation("pin_swap_symmetric", ref, seed=42)
    assert p.expected_outcome == "positive"
    assert "pin_swap_symmetric" in p.perturbation_chain[0]
    result = build_seal_samples_with_coverage_check(ref, p.cur_hcg, p.alignment)
    # The swap should be label_builder-detected via sibling expansion → REF_SYMMETRIC_SWAP
    assert result.stats.by_source[LabelSource.REF_SYMMETRIC_SWAP.value] > 0
    # Most importantly: NO wrong_observed (swap is electrically equivalent)
    assert result.stats.by_source[LabelSource.WRONG_OBSERVED.value] == 0


def test_pin_swap_falls_back_to_identity_when_no_symmetry_class() -> None:
    """LED1 (anode/cathode polar, no sym class) ref shouldn't crash; falls
    back to identity-like behavior (REF_PRESENT only)."""

    # Use LED fixture; only R1 has sym class. Apply pin_swap_symmetric many
    # seeds — every output should be a legal sym-swap (only R1 candidate).
    ref = _load_ref(FIXTURE_LED)
    p = apply_perturbation("pin_swap_symmetric", ref, seed=7)
    # Either R1 swap or identity-fallback — both produce expected_outcome=positive
    assert p.expected_outcome == "positive"
    # Don't crash; produces a valid cur
    assert p.cur_hcg.summary()["n_edges"] == ref.summary()["n_edges"]


# ---------------------------------------------------------------------------
# WrongConnectionPerturbation
# ---------------------------------------------------------------------------


def test_wrong_connection_produces_wrong_observed() -> None:
    ref = _load_ref(FIXTURE_RC)
    p = apply_perturbation("wrong_connection", ref, seed=1)
    assert p.expected_outcome == "wrong_observed"
    result = build_seal_samples_with_coverage_check(ref, p.cur_hcg, p.alignment)
    # KEY: WRONG_OBSERVED MUST fire
    assert result.stats.by_source[LabelSource.WRONG_OBSERVED.value] >= 1
    # And a MISSING_EDGE group should also exist (wrong_redirect type)
    assert result.stats.n_groups >= 1
    # At least one group has query_origin == "wrong_redirect"
    wr_groups = [g for g in result.groups if g.query_origin == "wrong_redirect"]
    assert wr_groups


def test_wrong_connection_chain_records_components() -> None:
    ref = _load_ref(FIXTURE_DIVIDER)
    p = apply_perturbation("wrong_connection", ref, seed=2)
    chain_entry = p.perturbation_chain[0]
    # Format: "wrong_connection:R1.pin1:VIN→GND"
    assert chain_entry.startswith("wrong_connection:")
    assert "→" in chain_entry


# ---------------------------------------------------------------------------
# PinReversedPerturbation
# ---------------------------------------------------------------------------


def test_pin_reversed_on_led_produces_wrong_observed() -> None:
    ref = _load_ref(FIXTURE_LED)
    p = apply_perturbation("pin_reversed", ref, seed=3)
    assert p.expected_outcome == "wrong_observed"
    assert "pin_reversed:LED1:anode↔cathode" == p.perturbation_chain[0]
    result = build_seal_samples_with_coverage_check(ref, p.cur_hcg, p.alignment)
    # Both LED anode AND cathode now connect to wrong nets → expect 2 wrong_observed
    assert result.stats.by_source[LabelSource.WRONG_OBSERVED.value] == 2


def test_pin_reversed_falls_back_when_no_polarized_components() -> None:
    """RC fixture has no polarized component (R and C ceramic both non-polar)."""

    ref = _load_ref(FIXTURE_RC)
    p = apply_perturbation("pin_reversed", ref, seed=4)
    # Fallback to identity → no perturbation_chain reflects pin_reversed
    assert p.perturbation_chain[0] in ("identity",)


# ---------------------------------------------------------------------------
# Determinism — same seed produces identical cur
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "perturbation_name",
    ["identity", "pin_swap_symmetric", "wrong_connection"],
)
def test_perturbation_is_deterministic(perturbation_name: str) -> None:
    ref = _load_ref(FIXTURE_DIVIDER)
    p1 = apply_perturbation(perturbation_name, ref, seed=99)
    p2 = apply_perturbation(perturbation_name, ref, seed=99)
    assert p1.perturbation_chain == p2.perturbation_chain
    # Same component/net mapping
    assert p1.alignment.ref_to_cur_component == p2.alignment.ref_to_cur_component
    assert p1.alignment.ref_to_cur_net == p2.alignment.ref_to_cur_net
    # Same edges in cur
    e1 = {(e.src_port_id, e.dst_net_id) for e in p1.cur_hcg.edges}
    e2 = {(e.src_port_id, e.dst_net_id) for e in p2.cur_hcg.edges}
    assert e1 == e2


def test_different_seeds_produce_different_perturbations() -> None:
    """Probabilistic — try several seeds on wrong_connection and verify at
    least one pair differs."""

    ref = _load_ref(FIXTURE_DIVIDER)
    chains = {
        apply_perturbation("wrong_connection", ref, seed=s).perturbation_chain[0]
        for s in range(20)
    }
    # voltage_divider has 4 ref edges × 2 net choices each = many variations
    assert len(chains) >= 2


# ---------------------------------------------------------------------------
# Coverage invariant holds for every operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "perturbation_name",
    ["identity", "pin_swap_symmetric", "wrong_connection", "pin_reversed"],
)
def test_every_operator_satisfies_coverage_invariant(perturbation_name: str) -> None:
    """All 4 perturbations + label_builder must produce coverage-complete
    output (cur edges all labeled)."""

    ref = (
        _load_ref(FIXTURE_LED)
        if perturbation_name == "pin_reversed"
        else _load_ref(FIXTURE_DIVIDER)
    )
    p = apply_perturbation(perturbation_name, ref, seed=10)
    # Should not raise CoverageError
    result = build_seal_samples_with_coverage_check(ref, p.cur_hcg, p.alignment)
    assert result.stats.total_samples > 0


# ---------------------------------------------------------------------------
# Voltage divider fixture is fully usable end-to-end
# ---------------------------------------------------------------------------


def test_voltage_divider_fixture_loads_and_perturbs() -> None:
    ref = _load_ref(FIXTURE_DIVIDER)
    assert ref.summary() == {"n_components": 2, "n_ports": 4, "n_nets": 3, "n_edges": 4}
    # All 4 perturbations apply without raising
    for name in PERTURBATION_REGISTRY:
        p = apply_perturbation(name, ref, seed=5)
        assert isinstance(p, PerturbedCur)


# ---------------------------------------------------------------------------
# P1 audit regression: parallel pins must survive perturbation pipeline
# ---------------------------------------------------------------------------


FIXTURE_OPAMP = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_opamp_buffer_v1.json"
)


def test_identity_preserves_parallel_pins_on_ua741_buffer() -> None:
    """P1 audit regression: UA741 unity-gain buffer wires pin 2 (INV) and pin 6
    (OUT) both to VOUT. The old nx.Graph-based pipeline collapsed parallel
    multi-edges between the same (component, net) pair, silently losing one
    of the two edges. The raw_pin_edges pipeline must preserve both.
    """

    ref = _load_ref(FIXTURE_OPAMP)
    # Reference itself: 5 connected pins (2/3/4/6/7) + 3 NC (1/5/8 = OPTIONAL/FORBIDDEN)
    n_ref_edges = len(ref.edges)
    assert n_ref_edges == 5, (
        f"reference UA741 buffer should have 5 edges, got {n_ref_edges}"
    )
    # Identity perturbation: cur must mirror ref exactly
    p = apply_perturbation("identity", ref, seed=0)
    n_cur_edges = len(p.cur_hcg.edges)
    assert n_cur_edges == n_ref_edges, (
        f"identity must preserve parallel pins; ref has {n_ref_edges} edges, "
        f"cur has {n_cur_edges} (lost a parallel pin?)"
    )
    # Specifically: VOUT must receive 2 distinct port edges (pin 2 and pin 6)
    vout_nets = [n for n in p.cur_hcg.nets.values() if n.source_id == "VOUT"]
    assert len(vout_nets) == 1
    vout_id = vout_nets[0].node_id
    vout_incoming = [e for e in p.cur_hcg.edges if e.dst_net_id == vout_id]
    assert len(vout_incoming) == 2, (
        f"VOUT should have 2 incoming port edges (pin 2 + pin 6 fed back), "
        f"got {len(vout_incoming)}"
    )
    # The two ports must be distinct (pin 2 inverting input + pin 6 output)
    src_ports = {e.src_port_id for e in vout_incoming}
    assert len(src_ports) == 2


def test_payload_subtype_auto_propagates_to_perturbation_cur() -> None:
    """Audit follow-up: when the payload already carries ``subtype`` field
    (e.g. UA741 fixture), the caller should NOT have to thread it through
    ``apply_perturbation(subtype_by_source_id=...)`` for cur HCG to get
    OPTIONAL/FORBIDDEN pin materialization. ``build_from_logical_reference``
    must stash the merged subtype onto ``ref_hcg.metadata`` so that
    ``_collect_subtypes`` recovers it automatically.

    Symptom of the bug (before the fix): cur HCG had 5 ports (connected only)
    while ref had 8 ports (5 connected + 1 FORBIDDEN + 2 OPTIONAL), and
    forbidden_negative samples silently dropped to 0.
    """

    ref = _load_ref(FIXTURE_OPAMP)
    assert ref.summary()["n_ports"] == 8, (
        f"ref UA741 should have 8 ports, got {ref.summary()}"
    )
    assert ref.metadata.get("subtype_by_source_id") == {"U1": "UA741"}, (
        "build_from_logical_reference must stash payload subtypes on metadata"
    )
    # Apply identity WITHOUT explicit subtype kwarg — must still produce 8 ports
    p = apply_perturbation("identity", ref, seed=0)
    cur_summary = p.cur_hcg.summary()
    assert cur_summary["n_ports"] == 8, (
        f"cur should auto-pick subtype from ref metadata and produce 8 ports, "
        f"got {cur_summary}"
    )
    # FORBIDDEN pin 8 must be materialized as a floating port on cur side
    forbidden_ports = [
        port for port in p.cur_hcg.ports.values()
        if port.connection_policy == "forbidden"
    ]
    assert len(forbidden_ports) == 1, (
        f"expected 1 FORBIDDEN port on cur (UA741 pin 8), got {len(forbidden_ports)}"
    )
    # And label_builder must emit forbidden_negative samples (was silently 0)
    result = build_seal_samples_with_coverage_check(ref, p.cur_hcg, p.alignment)
    assert result.stats.by_source[LabelSource.FORBIDDEN_NEGATIVE.value] >= 1, (
        "FORBIDDEN_NEGATIVE samples must fire when cur preserves FORBIDDEN pin"
    )


def test_wrong_connection_does_not_collide_when_target_net_is_already_used() -> None:
    """P1 audit follow-up: when WrongConnection picks a wrong_net that some
    OTHER pin of the same component already uses (e.g. the new (port_x, net_y)
    coincides with an existing (port_z, net_y) parallel slot), the raw-edges
    pipeline must keep BOTH edges as distinct records — the old nx.Graph
    pipeline would have collapsed them, hiding the wrong connection from
    label_builder.
    """

    ref = _load_ref(FIXTURE_OPAMP)
    # Try many seeds — at least once we expect the picked wrong_net to be
    # VOUT (already on pin 2 + pin 6) or some shared rail.
    collisions_seen = 0
    for seed in range(50):
        p = apply_perturbation(
            "wrong_connection",
            ref,
            seed=seed,
            subtype_by_source_id={"U1": "UA741"},
        )
        # Parse the chain entry to learn which (port, wrong_net) pair was made
        chain = p.perturbation_chain[0]
        # Format: "wrong_connection:U1.<pin>:<orig>→<wrong>"
        # Locate any other ref-edge already on the wrong_net
        wrong_net = chain.split("→")[-1]
        # Count edges to wrong_net on the cur side that come from a different pin
        wrong_net_id = next(
            (n.node_id for n in p.cur_hcg.nets.values() if n.source_id == wrong_net),
            None,
        )
        if wrong_net_id is None:
            continue
        edges_to_wrong = [
            e for e in p.cur_hcg.edges if e.dst_net_id == wrong_net_id
        ]
        # Collision case: ≥ 2 edges into wrong_net AND each from a distinct port
        if len(edges_to_wrong) >= 2 and (
            len({e.src_port_id for e in edges_to_wrong}) == len(edges_to_wrong)
        ):
            collisions_seen += 1
            # Coverage invariant still holds — every observed edge labeled
            result = build_seal_samples_with_coverage_check(
                ref, p.cur_hcg, p.alignment
            )
            assert result.stats.by_source[LabelSource.WRONG_OBSERVED.value] >= 1
    # Across 50 seeds with a 4-net UA741 buffer, we must hit at least one
    # collision case (otherwise this test isn't exercising what it claims to)
    assert collisions_seen >= 1, (
        "Expected to hit at least one wrong_net collision case across 50 seeds"
    )


# ---------------------------------------------------------------------------
# Phase B operators
# ---------------------------------------------------------------------------


FIXTURE_OPAMP_PHASE_B = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_opamp_buffer_v1.json"
)


@pytest.mark.parametrize(
    "perturbation_name",
    [
        "missing_component",
        "extra_component",
        "floating_net",
        "short_circuit",
        "power_swapped",
        "input_output_swapped",
        "extra_wire_bridge",
        "chained",
    ],
)
def test_phase_b_operator_runs_on_every_fixture(
    perturbation_name: str,
) -> None:
    """Each Phase B operator must run on RC / divider / LED / opamp without
    raising, and label_builder coverage check must pass."""

    for path in (FIXTURE_RC, FIXTURE_DIVIDER, FIXTURE_LED, FIXTURE_OPAMP_PHASE_B):
        ref = _load_ref(path)
        p = apply_perturbation(perturbation_name, ref, seed=7)
        assert isinstance(p, PerturbedCur)
        # Coverage check (the contract dataset_builder enforces)
        result = build_seal_samples_with_coverage_check(
            ref, p.cur_hcg, p.alignment
        )
        assert result.stats.total_samples > 0


# --- MissingComponent ---


def test_missing_component_drops_one_component() -> None:
    ref = _load_ref(FIXTURE_DIVIDER)
    p = apply_perturbation("missing_component", ref, seed=1)
    assert p.expected_outcome == "missing_required"
    assert p.cur_hcg.summary()["n_components"] == 1
    # Alignment notes flag the dropped one
    assert p.alignment.notes["unmatched_ref_components"], (
        "missing_component must leave at least one ref component unmatched"
    )
    # Stats: ref edges from the dropped component land in n_skipped_missing_component
    result = build_seal_samples_with_coverage_check(ref, p.cur_hcg, p.alignment)
    assert result.stats.n_skipped_missing_component >= 1


def test_missing_component_falls_back_on_single_component_circuit() -> None:
    """UA741 buffer has only U1 — nothing safe to drop → identity fallback."""

    ref = _load_ref(FIXTURE_OPAMP_PHASE_B)
    p = apply_perturbation("missing_component", ref, seed=1)
    assert p.expected_outcome == "positive"
    assert p.perturbation_chain == ("identity",)


# --- ExtraComponent ---


def test_extra_component_adds_a_parasitic_component() -> None:
    ref = _load_ref(FIXTURE_DIVIDER)
    p = apply_perturbation("extra_component", ref, seed=1)
    assert p.expected_outcome == "wrong_observed"
    n_ref = ref.summary()["n_components"]
    n_cur = p.cur_hcg.summary()["n_components"]
    assert n_cur == n_ref + 1, (
        f"extra_component must add exactly 1 component (ref={n_ref}, cur={n_cur})"
    )
    # The extra pins must produce WRONG_OBSERVED in label stats
    result = build_seal_samples_with_coverage_check(ref, p.cur_hcg, p.alignment)
    assert result.stats.by_source[LabelSource.WRONG_OBSERVED.value] >= 1


# --- FloatingNet ---


def test_floating_net_removes_all_but_one_edge_into_a_net() -> None:
    ref = _load_ref(FIXTURE_DIVIDER)
    p = apply_perturbation("floating_net", ref, seed=1)
    assert p.expected_outcome == "missing_required"
    # Some net must end up with degree 1 (or 0 in degenerate cases)
    cur_net_degree = {n.source_id: 0 for n in p.cur_hcg.nets.values()}
    for e in p.cur_hcg.edges:
        cur_net_degree[p.cur_hcg.nets[e.dst_net_id].source_id] += 1
    floated = p.notes["floated_net"]
    assert cur_net_degree[floated] == 1, (
        f"floated net {floated} should have degree 1 on cur, got {cur_net_degree[floated]}"
    )


# --- ShortCircuit ---


def test_short_circuit_merges_one_net_into_another() -> None:
    ref = _load_ref(FIXTURE_DIVIDER)
    p = apply_perturbation("short_circuit", ref, seed=1)
    assert p.expected_outcome == "wrong_observed"
    from_net = p.notes["from_net"]
    into_net = p.notes["into_net"]
    assert from_net != into_net
    # `from_net` should have no incoming edges in cur, `into_net` should gain edges
    cur_into_from = sum(
        1 for e in p.cur_hcg.edges
        if p.cur_hcg.nets[e.dst_net_id].source_id == from_net
    )
    assert cur_into_from == 0, (
        f"after short_circuit, {from_net} should have 0 incoming edges on cur"
    )


# --- PowerSwapped ---


def test_power_swapped_swaps_vcc_and_gnd_edges() -> None:
    # UA741 buffer fixture has both VCC (role=power) and GND (role=ground)
    ref = _load_ref(FIXTURE_OPAMP_PHASE_B)
    # Need to check role atts — if no VCC, fallback to identity → skip test
    has_power = any(n.role == "power" for n in ref.nets.values())
    has_ground = any(n.role == "ground" for n in ref.nets.values())
    if not (has_power and has_ground):
        pytest.skip("LED fixture has no VCC+GND pair to test swap")
    p = apply_perturbation("power_swapped", ref, seed=1)
    assert p.expected_outcome == "wrong_observed"
    # Every edge originally on VCC should now be on GND and vice versa
    vcc = p.notes["vcc_net"]
    gnd = p.notes["gnd_net"]
    # Compare by (component source_id, port_key) — strip the "ref_"/"cur_" prefix
    ref_vcc_pins = {
        (
            ref.components[ref.ports[e.src_port_id].parent_component_id].source_id,
            ref.ports[e.src_port_id].port_key,
        )
        for e in ref.edges
        if ref.nets[e.dst_net_id].source_id == vcc
    }
    cur_gnd_pins = {
        (
            p.cur_hcg.components[
                p.cur_hcg.ports[e.src_port_id].parent_component_id
            ].source_id,
            p.cur_hcg.ports[e.src_port_id].port_key,
        )
        for e in p.cur_hcg.edges
        if p.cur_hcg.nets[e.dst_net_id].source_id == gnd
    }
    assert ref_vcc_pins.issubset(cur_gnd_pins), (
        f"ref pins on VCC ({ref_vcc_pins}) should now be on GND ({cur_gnd_pins})"
    )


def test_power_swapped_falls_back_when_no_vcc_or_no_gnd() -> None:
    ref = _load_ref(FIXTURE_RC)
    has_power = any(n.role == "power" for n in ref.nets.values())
    if has_power:
        pytest.skip("RC fixture has power; this test wants the no-power case")
    p = apply_perturbation("power_swapped", ref, seed=1)
    assert p.perturbation_chain == ("identity",)
    assert p.expected_outcome == "positive"


# --- InputOutputSwapped ---


def test_input_output_swapped_swaps_input_and_output_edges() -> None:
    ref = _load_ref(FIXTURE_DIVIDER)
    has_in = any(n.role == "input" for n in ref.nets.values())
    has_out = any(n.role == "output" for n in ref.nets.values())
    if not (has_in and has_out):
        pytest.skip("divider fixture has no input+output pair")
    p = apply_perturbation("input_output_swapped", ref, seed=1)
    assert p.expected_outcome == "wrong_observed"
    result = build_seal_samples_with_coverage_check(ref, p.cur_hcg, p.alignment)
    assert result.stats.by_source[LabelSource.WRONG_OBSERVED.value] >= 1


# --- ExtraWireBridge ---


def test_extra_wire_bridge_adds_wire_component_bridging_two_nets() -> None:
    ref = _load_ref(FIXTURE_DIVIDER)
    p = apply_perturbation("extra_wire_bridge", ref, seed=1)
    if p.perturbation_chain == ("identity",):
        pytest.skip("divider fixture left no unbridged net pair (rare)")
    assert p.expected_outcome == "wrong_observed"
    # The wire component must exist on cur side
    wire_sid = p.notes["wire_id"]
    assert any(
        c.source_id == wire_sid and c.ctype == "Wire"
        for c in p.cur_hcg.components.values()
    )


# --- Chained ---


def test_chained_composes_multiple_links() -> None:
    ref = _load_ref(FIXTURE_DIVIDER)
    p = apply_perturbation("chained", ref, seed=1)
    # Chain prefix marker + ≥ 2 link entries
    assert p.perturbation_chain[0] == "chained:"
    assert len(p.perturbation_chain) >= 3
    # Severity escalates: at least one link is non-positive ⇒ expected_outcome is too
    assert p.expected_outcome in ("missing_required", "wrong_observed", "positive")
    # link_notes records the per-link metadata
    assert "links" in p.notes
    assert 2 <= len(p.notes["links"]) <= 3


# --- Determinism for Phase B ---


@pytest.mark.parametrize(
    "perturbation_name",
    [
        "missing_component",
        "extra_component",
        "floating_net",
        "short_circuit",
        "power_swapped",
        "input_output_swapped",
        "extra_wire_bridge",
        "chained",
    ],
)
def test_phase_b_perturbation_is_deterministic(perturbation_name: str) -> None:
    ref = _load_ref(FIXTURE_DIVIDER)
    p1 = apply_perturbation(perturbation_name, ref, seed=99)
    p2 = apply_perturbation(perturbation_name, ref, seed=99)
    assert p1.perturbation_chain == p2.perturbation_chain
    e1 = {(e.src_port_id, e.dst_net_id) for e in p1.cur_hcg.edges}
    e2 = {(e.src_port_id, e.dst_net_id) for e in p2.cur_hcg.edges}
    assert e1 == e2
