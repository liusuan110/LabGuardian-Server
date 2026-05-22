"""End-to-end tests for the template matcher against real_student fixtures.

These tests are the canonical 验收 for CADx Phase 0: the matcher must
identify each correct circuit's topology as the top-1 hypothesis with
confidence > 0.5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.logical_reference import current_netlist_v2_to_graph
from app.domain.templates import get_template_registry, match_all_templates


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "real_student"


# (fixture_basename, expected_template_id, expected_min_confidence)
#
# NOTE on LPF: ``opamp_inverting_lpf_correct_v1`` is structurally identical
# to a "lossy integrator" (R_f ∥ C1 in feedback). After Phase 0 added
# explicit edges to the integrator's ``with_leak_resistor`` variant, the
# matcher correctly identifies the LPF as ``integrator_ua741_v1`` with
# higher confidence (0.94 vs 0.88 for inverting_amp). This is the more
# precise classification — semantically LPF IS a lossy integrator.
GOLDEN_CASES = [
    ("inverting_amp_correct_v1",       "inverting_amp_ua741_v1",  0.5),
    ("bjt_diff_amp_correct_v1",        "differential_pair_v1",    0.5),
    ("opamp_inverting_lpf_correct_v1", "integrator_ua741_v1",     0.5),
    ("opamp_summing_correct_v1",       "summing_amp_ua741_v1",    0.5),
]


@pytest.fixture(autouse=True)
def _reset_registry_cache():
    get_template_registry.cache_clear()
    yield
    get_template_registry.cache_clear()


@pytest.mark.parametrize(
    "fixture_basename,expected_template_id,min_confidence",
    GOLDEN_CASES,
)
def test_top1_matches_expected_topology(
    fixture_basename: str,
    expected_template_id: str,
    min_confidence: float,
) -> None:
    payload = json.loads(
        (FIXTURE_DIR / f"{fixture_basename}.json").read_text()
    )
    graph = current_netlist_v2_to_graph(payload)
    results = match_all_templates(graph)
    assert len(results) > 0, "matcher returned zero results"
    top = results[0]
    assert top.template_id == expected_template_id, (
        f"{fixture_basename}: expected {expected_template_id} as top-1, "
        f"got {top.template_id} (confidence={top.confidence:.3f}). "
        f"Top 3: {[(r.template_id, round(r.confidence, 3)) for r in results[:3]]}"
    )
    assert top.confidence >= min_confidence, (
        f"{fixture_basename}: top-1 confidence {top.confidence:.3f} "
        f"below threshold {min_confidence}"
    )


def test_match_returns_role_assignments_for_top1() -> None:
    payload = json.loads(
        (FIXTURE_DIR / "inverting_amp_correct_v1.json").read_text()
    )
    graph = current_netlist_v2_to_graph(payload)
    results = match_all_templates(graph)
    top = results[0]
    # Inverting amp must assign opamp + R_g + R_f roles to student components.
    assigned_roles = set(top.role_assignments.values())
    assert "opamp" in assigned_roles
    assert "R_g" in assigned_roles
    assert "R_f" in assigned_roles


def test_summing_beats_inverting_on_summing_fixture() -> None:
    """Critical tie-break: summing_amp must outrank inverting_amp on a
    summing board, even though inverting_amp's spec is a strict subgraph
    of summing's. Coverage scoring is what makes this work.
    """
    payload = json.loads(
        (FIXTURE_DIR / "opamp_summing_correct_v1.json").read_text()
    )
    graph = current_netlist_v2_to_graph(payload)
    results = match_all_templates(graph)
    by_id = {r.template_id: r for r in results}
    summing_conf = by_id["summing_amp_ua741_v1"].confidence
    inverting_conf = by_id["inverting_amp_ua741_v1"].confidence
    assert summing_conf > inverting_conf, (
        f"summing ({summing_conf:.3f}) must beat inverting ({inverting_conf:.3f})"
    )


def test_lpf_fixture_surfaces_both_inverting_and_integrator() -> None:
    """An LPF (R + C in feedback) legitimately matches BOTH inverting_amp
    and integrator topologies as subgraph isomorphisms. Top-K should
    surface both as high-confidence hypotheses for downstream UI to show.
    """
    payload = json.loads(
        (FIXTURE_DIR / "opamp_inverting_lpf_correct_v1.json").read_text()
    )
    graph = current_netlist_v2_to_graph(payload)
    results = match_all_templates(graph)
    by_id = {r.template_id: r for r in results}
    assert by_id["inverting_amp_ua741_v1"].confidence > 0.5
    assert by_id["integrator_ua741_v1"].confidence > 0.5


def test_matcher_returns_empty_assignments_on_unrelated_circuit() -> None:
    """A circuit with no UA741 / BJT / RC structure should still return
    a TemplateMatchResult per template, all with confidence 0.0."""
    import networkx as nx

    # Build a trivial dummy graph that won't match anything.
    g = nx.Graph()
    g.add_node("cur_comp:LED1", kind="comp", ctype="LED", subtype=None)
    g.add_node("cur_net:NET_X", kind="net", role="signal")
    g.add_edge("cur_comp:LED1", "cur_net:NET_X", pin="anode", comp_type="LED")

    results = match_all_templates(g)
    # All six templates return results — none should match this circuit.
    assert all(r.confidence == 0.0 for r in results)


def test_to_dict_is_json_serializable() -> None:
    """TemplateMatchResult.to_dict must produce a JSON-safe dict."""
    payload = json.loads(
        (FIXTURE_DIR / "inverting_amp_correct_v1.json").read_text()
    )
    graph = current_netlist_v2_to_graph(payload)
    results = match_all_templates(graph)
    serialized = json.dumps([r.to_dict() for r in results[:3]])
    assert "template_id" in serialized
    assert "confidence" in serialized
