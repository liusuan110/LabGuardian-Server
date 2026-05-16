"""P0.7 · SEAL Enclosing Subgraph + DRNL labeling tests.

Covers:
- DRNL formula (Zhang & Chen 2018) — hand-computed labels on small graphs
- Anchor convention (target_port_id / target_net_id → label 1)
- Unreachable nodes → label 0
- Candidate edge excluded from BFS and from the returned ``edges`` tuple
- h-hop boundary respected (3-hop nodes absent for num_hops=2)
- Auto vs explicit ``edge_present``
- Batched: ``extract_subgraphs_for_observed_edges`` yields exactly one
  subgraph per observed edge
- Batched: ``extract_subgraphs_for_floating_ports`` skips FORBIDDEN ports
  (UA741 NC pin 8) and enumerates every other floating × candidate net pair
- UA741 fixture end-to-end (regression)
- Performance: 50 candidate edges < 30 ms on CPU
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.domain.gnn import (
    SealSubgraph,
    build_from_logical_reference,
    extract_seal_subgraph,
    extract_subgraphs_for_floating_ports,
    extract_subgraphs_for_observed_edges,
)
from app.domain.gnn.port_graph import build_hetero_circuit_graph
from app.domain.gnn.seal_subgraph import _drnl_label

FIXTURE_OPAMP = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_opamp_buffer_v1.json"
)
FIXTURE_RC = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_rc_v1.json"
)


# ---------------------------------------------------------------------------
# DRNL formula — hand-computed values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("du", "dv", "expected"),
    [
        # (1,1): d=2, d_half=1; 1 + 1 + 1*(1+0-1) = 2
        (1, 1, 2),
        # (1,2): d=3, d_half=1; 1 + 1 + 1*(1+1-1) = 3
        (1, 2, 3),
        # (2,1): symmetric
        (2, 1, 3),
        # (2,2): d=4, d_half=2; 1 + 2 + 2*(2+0-1) = 5
        (2, 2, 5),
        # (1,3): d=4, d_half=2; 1 + 1 + 2*(2+0-1) = 4
        (1, 3, 4),
        # (3,1): symmetric
        (3, 1, 4),
        # (2,3): d=5, d_half=2; 1 + 2 + 2*(2+1-1) = 7
        (2, 3, 7),
        # (3,3): d=6, d_half=3; 1 + 3 + 3*(3+0-1) = 10
        (3, 3, 10),
    ],
)
def test_drnl_formula_matches_paper(du: int, dv: int, expected: int) -> None:
    assert _drnl_label(du, dv) == expected


def test_drnl_unreachable_returns_zero() -> None:
    import math

    assert _drnl_label(math.inf, 1) == 0
    assert _drnl_label(2, math.inf) == 0
    assert _drnl_label(math.inf, math.inf) == 0


# ---------------------------------------------------------------------------
# extract_seal_subgraph — basic anchor + structure
# ---------------------------------------------------------------------------


def test_extract_rejects_missing_anchor() -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_RC.read_text()))
    with pytest.raises(KeyError, match="port"):
        extract_seal_subgraph(hcg, "ref_port:DOES_NOT_EXIST", "ref_net:GND")
    with pytest.raises(KeyError, match="net"):
        extract_seal_subgraph(hcg, "ref_port:R1.pin1", "ref_net:NO_SUCH_NET")


def test_extract_rc_anchors_get_label_1() -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_RC.read_text()))
    sg = extract_seal_subgraph(hcg, "ref_port:R1.pin1", "ref_net:VIN")
    assert sg.target_port_id == "ref_port:R1.pin1"
    assert sg.target_net_id == "ref_net:VIN"
    assert sg.drnl_labels["ref_port:R1.pin1"] == 1
    assert sg.drnl_labels["ref_net:VIN"] == 1
    assert sg.is_target["ref_port:R1.pin1"] is True
    assert sg.is_target["ref_net:VIN"] is True
    # anchor ports / nets sit at index 0 of their respective tuples
    assert sg.port_ids[0] == "ref_port:R1.pin1"
    assert sg.net_ids[0] == "ref_net:VIN"


def test_extract_candidate_edge_excluded_from_edge_list() -> None:
    """SEAL convention: the model must not see the link it's predicting."""

    hcg = build_from_logical_reference(json.loads(FIXTURE_RC.read_text()))
    sg = extract_seal_subgraph(hcg, "ref_port:R1.pin1", "ref_net:VIN")
    assert sg.edge_present is True
    for src_port, dst_net in sg.edges:
        assert not (src_port == "ref_port:R1.pin1" and dst_net == "ref_net:VIN"), (
            sg.edges
        )


def test_extract_edge_present_auto_detection() -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_RC.read_text()))
    # (R1.pin1, VIN) does exist
    sg_existing = extract_seal_subgraph(hcg, "ref_port:R1.pin1", "ref_net:VIN")
    assert sg_existing.edge_present is True
    # (R1.pin1, GND) does NOT exist (R1.pin1 only goes to VIN)
    sg_missing = extract_seal_subgraph(hcg, "ref_port:R1.pin1", "ref_net:GND")
    assert sg_missing.edge_present is False


def test_extract_edge_present_explicit_override() -> None:
    """Passing edge_present=False for an actually-existing edge forces the
    "negative sample" interpretation (used in training data construction)."""

    hcg = build_from_logical_reference(json.loads(FIXTURE_RC.read_text()))
    sg = extract_seal_subgraph(
        hcg, "ref_port:R1.pin1", "ref_net:VIN", edge_present=False
    )
    assert sg.edge_present is False
    # Because we passed edge_present=False, the candidate edge is NOT
    # excluded from BFS / edges — so it should appear in the edge list.
    assert ("ref_port:R1.pin1", "ref_net:VIN") in sg.edges


# ---------------------------------------------------------------------------
# extract_seal_subgraph — DRNL on RC fixture (hand-traceable)
# ---------------------------------------------------------------------------


def test_extract_rc_drnl_on_subgraph() -> None:
    """RC circuit: R1.pin1—VIN, R1.pin2—VC, C1.pin1—VC, C1.pin2—GND.

    Take candidate edge (R1.pin1, VIN):
    - With this edge removed, R1.pin1 becomes isolated → d_v(R1.pin1, VIN) = ∞
      and d_u(any_other_node, R1.pin1) = ∞.
    - The 2-hop enclosing subgraph from VIN reaches no other nodes (VIN now
      has no edges).
    - The 2-hop enclosing subgraph from R1.pin1 reaches no other nodes (it's
      isolated after removal).
    So only the two anchors should be in the subgraph.
    """

    hcg = build_from_logical_reference(json.loads(FIXTURE_RC.read_text()))
    sg = extract_seal_subgraph(hcg, "ref_port:R1.pin1", "ref_net:VIN")
    assert set(sg.node_ids()) == {"ref_port:R1.pin1", "ref_net:VIN"}
    assert sg.num_edges() == 0  # the only edge between them was the candidate
    assert sg.drnl_labels == {"ref_port:R1.pin1": 1, "ref_net:VIN": 1}


def test_extract_rc_two_hop_radius_reaches_through_pin2() -> None:
    """Take candidate (R1.pin2, VC). With it removed:
    - R1.pin2 connects only via the removed edge → BFS from R1.pin2 reaches
      no other node.
    - VC still connects to C1.pin1 → 1 hop. C1.pin1 connects to C1 (none in
      bipartite subgraph) and... only VC. So VC reaches {C1.pin1} at hop 1,
      no further reach at hop 2 (because C1.pin1 only connects to VC).
    Enclosing subgraph (num_hops=2): {R1.pin2, VC, C1.pin1}.
    """

    hcg = build_from_logical_reference(json.loads(FIXTURE_RC.read_text()))
    sg = extract_seal_subgraph(hcg, "ref_port:R1.pin2", "ref_net:VC")
    assert set(sg.node_ids()) == {
        "ref_port:R1.pin2",
        "ref_net:VC",
        "ref_port:C1.pin1",
    }
    # C1.pin1: d_u (from R1.pin2 with candidate edge removed) = ∞, so → 0.
    assert sg.drnl_labels["ref_port:C1.pin1"] == 0
    # Anchors → 1
    assert sg.drnl_labels["ref_port:R1.pin2"] == 1
    assert sg.drnl_labels["ref_net:VC"] == 1


def test_extract_negative_sample_yields_richer_subgraph() -> None:
    """If we request the (R1.pin1, GND) candidate (which doesn't exist), the
    BFS does NOT remove any edge, so both anchors can reach the rest of the
    RC chain via the actual circuit. Expected ≥ 4 nodes (R1.pin1, VIN reach
    pin2/VC via R1; GND reaches C1.pin2/VC via C1; depending on hops).
    """

    hcg = build_from_logical_reference(json.loads(FIXTURE_RC.read_text()))
    sg = extract_seal_subgraph(hcg, "ref_port:R1.pin1", "ref_net:GND")
    assert sg.edge_present is False
    node_ids = set(sg.node_ids())
    # With nothing removed, BFS from R1.pin1 within 2 hops reaches:
    # 0: R1.pin1; 1: VIN; 2: (VIN has no other edges) → stops.
    # BFS from GND within 2 hops reaches:
    # 0: GND; 1: C1.pin2; 2: C1.pin1 (because C1.pin2→GND only goes back)
    # Wait, C1.pin2 only connects to GND. So 2-hop from GND = {GND, C1.pin2}.
    # Union: {R1.pin1, VIN, GND, C1.pin2}.
    assert "ref_port:R1.pin1" in node_ids
    assert "ref_net:VIN" in node_ids
    assert "ref_net:GND" in node_ids
    assert "ref_port:C1.pin2" in node_ids


def test_extract_h_hop_boundary_respected() -> None:
    """num_hops=1 must not pull in 2-hop neighbors."""

    hcg = build_from_logical_reference(json.loads(FIXTURE_RC.read_text()))
    sg = extract_seal_subgraph(
        hcg, "ref_port:R1.pin2", "ref_net:VC", num_hops=1
    )
    # 1-hop: anchors + immediate neighbors only.
    # Removed (R1.pin2, VC). VC's 1-hop neighbors: C1.pin1.
    assert set(sg.node_ids()) == {
        "ref_port:R1.pin2",
        "ref_net:VC",
        "ref_port:C1.pin1",
    }


# ---------------------------------------------------------------------------
# Batched: observed-edge extraction (wrong-edge detection input)
# ---------------------------------------------------------------------------


def test_observed_edge_batch_count_matches_hcg_edges() -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text()))
    sgs = extract_subgraphs_for_observed_edges(hcg)
    assert len(sgs) == len(hcg.edges) == 5
    # Each subgraph anchors on a real edge
    expected_anchors = {(e.src_port_id, e.dst_net_id) for e in hcg.edges}
    actual_anchors = {(sg.target_port_id, sg.target_net_id) for sg in sgs}
    assert expected_anchors == actual_anchors
    # All have edge_present=True
    assert all(sg.edge_present for sg in sgs)


# ---------------------------------------------------------------------------
# Batched: floating-port extraction (suggested-target / missing-edge input)
# ---------------------------------------------------------------------------


def test_floating_batch_skips_forbidden_pins() -> None:
    """UA741 has 3 floating pins (1/5/8). Pin 8 has policy=FORBIDDEN and
    must NOT appear among the floating-port candidates."""

    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text()))
    sgs = extract_subgraphs_for_floating_ports(hcg)
    candidate_ports = {sg.target_port_id for sg in sgs}
    # Pin 8 (FORBIDDEN) excluded; pin 1 / 5 (OPTIONAL) included.
    assert "ref_port:U1.8" not in candidate_ports
    assert "ref_port:U1.1" in candidate_ports
    assert "ref_port:U1.5" in candidate_ports


def test_floating_batch_enumerates_port_x_net_pairs() -> None:
    """2 floating-but-allowed ports (pin 1 + pin 5) × 4 nets = 8."""

    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text()))
    sgs = extract_subgraphs_for_floating_ports(hcg)
    assert len(sgs) == 2 * 4
    # All have edge_present=False (these are suggested-target candidates)
    assert all(sg.edge_present is False for sg in sgs)


def test_floating_batch_with_explicit_candidate_nets() -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text()))
    only_gnd = ["ref_net:GND"]
    sgs = extract_subgraphs_for_floating_ports(hcg, candidate_nets=only_gnd)
    # 2 floating-allowed ports × 1 net = 2
    assert len(sgs) == 2
    for sg in sgs:
        assert sg.target_net_id == "ref_net:GND"


def test_floating_batch_can_include_forbidden_when_disabled() -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text()))
    sgs = extract_subgraphs_for_floating_ports(hcg, exclude_forbidden=False)
    candidate_ports = {sg.target_port_id for sg in sgs}
    assert "ref_port:U1.8" in candidate_ports


def test_floating_batch_when_no_floating_ports_returns_empty() -> None:
    # RC fixture has no floating ports.
    hcg = build_from_logical_reference(json.loads(FIXTURE_RC.read_text()))
    sgs = extract_subgraphs_for_floating_ports(hcg)
    assert sgs == []


# ---------------------------------------------------------------------------
# UA741 end-to-end regression
# ---------------------------------------------------------------------------


def test_ua741_extract_for_inverting_input_edge() -> None:
    """Spot-check the SEAL extraction on the U1.2 (inverting input) → VOUT
    feedback edge from the unity-gain buffer fixture."""

    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text()))
    sg = extract_seal_subgraph(hcg, "ref_port:U1.2", "ref_net:VOUT")
    assert sg.edge_present is True
    # With (U1.2, VOUT) removed, U1.2 becomes isolated (only edge was the
    # feedback to VOUT). VOUT still reaches U1.6 (parallel feedback pin).
    # Subgraph at num_hops=2: {U1.2, VOUT, U1.6}.
    assert set(sg.node_ids()) == {"ref_port:U1.2", "ref_net:VOUT", "ref_port:U1.6"}
    assert sg.drnl_labels["ref_port:U1.2"] == 1  # anchor
    assert sg.drnl_labels["ref_net:VOUT"] == 1  # anchor
    # U1.6: reachable from VOUT in 1 hop, unreachable from U1.2 (isolated).
    assert sg.drnl_labels["ref_port:U1.6"] == 0


def test_ua741_isinstance_and_immutable() -> None:
    hcg = build_from_logical_reference(json.loads(FIXTURE_OPAMP.read_text()))
    sg = extract_seal_subgraph(hcg, "ref_port:U1.3", "ref_net:VIN")
    assert isinstance(sg, SealSubgraph)
    with pytest.raises((AttributeError, TypeError)):
        sg.num_hops = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Performance — plan says 50 edges < 30 ms
# ---------------------------------------------------------------------------


def test_performance_under_30ms_for_50_edges() -> None:
    """Synthetic stress: build a chain of 25 resistors (≈ 50 edges, 26 nets)
    and time a full ``extract_subgraphs_for_observed_edges`` pass.
    Plan §三.6 budget: < 30 ms / call on CPU."""

    g = build_hetero_circuit_graph
    import networkx as nx

    chain = nx.Graph()
    for i in range(25):
        chain.add_node(f"ref_comp:R{i}", kind="comp", ctype="Resistor", source_id=f"R{i}")
    for i in range(26):
        chain.add_node(f"ref_net:N{i}", kind="net", role="signal", source_id=f"N{i}")
    for i in range(25):
        chain.add_edge(
            f"ref_comp:R{i}",
            f"ref_net:N{i}",
            pin="pin1",
            pin_role="pin1",
            comp_type="Resistor",
        )
        chain.add_edge(
            f"ref_comp:R{i}",
            f"ref_net:N{i+1}",
            pin="pin2",
            pin_role="pin2",
            comp_type="Resistor",
        )
    hcg = g(chain, side="ref")
    assert len(hcg.edges) == 50

    # Warm
    for _ in range(3):
        extract_subgraphs_for_observed_edges(hcg)
    t0 = time.perf_counter()
    sgs = extract_subgraphs_for_observed_edges(hcg)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert len(sgs) == 50
    # Generous bound to avoid CI flakiness; nominal is well under 5 ms.
    assert elapsed_ms < 30.0, f"50-edge extract took {elapsed_ms:.2f} ms > 30 ms"
