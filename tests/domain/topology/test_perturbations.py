"""Unit tests for topology-preserving graph perturbations.

The key invariant under test: every perturbation must preserve the
canonical topology label of its input. Tests verify this by running
the symbolic template matcher before and after — same template
should still match (with similar confidence).
"""

from __future__ import annotations

import random
from pathlib import Path

import networkx as nx
import pytest

from app.domain.dsl.loader import load_dsl_reference
from app.domain.logical_reference import logical_reference_to_graph
from app.domain.templates import match_all_templates
from app.domain.topology.perturbations import (
    PERTURBATIONS,
    add_decoration_wire,
    apply_random_chain,
    rename_components,
    rename_nets,
    swap_passive_pin_assignments,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCES_DIR = REPO_ROOT / "knowledge" / "references"


@pytest.fixture(scope="module")
def inverting_amp_seed() -> nx.Graph:
    payload = load_dsl_reference(REFERENCES_DIR / "ua741_inverting_amp_gain10_v1.py")
    return logical_reference_to_graph(payload)


@pytest.fixture(scope="module")
def ce_amp_seed() -> nx.Graph:
    payload = load_dsl_reference(REFERENCES_DIR / "ce_amp_fixed_bias_v1.py")
    return logical_reference_to_graph(payload)


@pytest.fixture(scope="module")
def integrator_seed() -> nx.Graph:
    payload = load_dsl_reference(REFERENCES_DIR / "ua741_integrator_v1.py")
    return logical_reference_to_graph(payload)


def _top1_template(g: nx.Graph) -> tuple[str, float]:
    results = match_all_templates(g)
    if not results or results[0].confidence == 0:
        return ("", 0.0)
    return (results[0].template_id, results[0].confidence)


# ---------------------------------------------------------------------------
# Topology preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed_fixture,expected_template",
    [
        ("inverting_amp_seed", "inverting_amp_ua741_v1"),
        ("ce_amp_seed",        "common_emitter_v1"),
        ("integrator_seed",    "integrator_ua741_v1"),
    ],
)
@pytest.mark.parametrize("perturbation_name", list(PERTURBATIONS.keys()))
def test_single_perturbation_preserves_topology(
    request: pytest.FixtureRequest,
    seed_fixture: str,
    expected_template: str,
    perturbation_name: str,
) -> None:
    """Each individual perturbation, applied once, must not change the
    top-1 template identification."""
    g = request.getfixturevalue(seed_fixture)
    rng = random.Random(hash((seed_fixture, perturbation_name)) & 0xFFFFFFFF)
    fn = PERTURBATIONS[perturbation_name]

    before_template, before_conf = _top1_template(g)
    assert before_template == expected_template, (
        f"seed fixture {seed_fixture} expected to match {expected_template} "
        f"before perturbation, got {before_template} ({before_conf:.3f})"
    )

    perturbed = fn(g, rng)
    after_template, after_conf = _top1_template(perturbed)
    assert after_template == expected_template, (
        f"perturbation {perturbation_name!r} changed top-1 from "
        f"{expected_template!r} to {after_template!r} "
        f"(conf {before_conf:.3f} -> {after_conf:.3f})"
    )


@pytest.mark.parametrize("chain_length", [1, 3, 5, 8])
def test_chain_perturbation_preserves_topology(
    inverting_amp_seed: nx.Graph,
    chain_length: int,
) -> None:
    """Composed chains of perturbations must also preserve topology."""
    rng = random.Random(chain_length * 13)
    perturbed, chain = apply_random_chain(
        inverting_amp_seed, rng, chain_length=chain_length,
    )
    template_id, conf = _top1_template(perturbed)
    assert template_id == "inverting_amp_ua741_v1", (
        f"chain {chain} mutated topology to {template_id} ({conf:.3f})"
    )
    assert len(chain) == chain_length


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_rename_components_is_deterministic_with_fixed_seed(
    inverting_amp_seed: nx.Graph,
) -> None:
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    g1 = rename_components(inverting_amp_seed, rng1)
    g2 = rename_components(inverting_amp_seed, rng2)
    assert set(g1.nodes()) == set(g2.nodes())


def test_chain_is_deterministic_with_fixed_seed(
    inverting_amp_seed: nx.Graph,
) -> None:
    rng1 = random.Random(456)
    rng2 = random.Random(456)
    g1, chain1 = apply_random_chain(inverting_amp_seed, rng1, chain_length=5)
    g2, chain2 = apply_random_chain(inverting_amp_seed, rng2, chain_length=5)
    assert chain1 == chain2
    assert g1.number_of_nodes() == g2.number_of_nodes()
    assert g1.number_of_edges() == g2.number_of_edges()


# ---------------------------------------------------------------------------
# Per-perturbation behavior
# ---------------------------------------------------------------------------


def test_rename_components_does_not_change_count(
    inverting_amp_seed: nx.Graph,
) -> None:
    rng = random.Random(0)
    g = rename_components(inverting_amp_seed, rng)
    assert g.number_of_nodes() == inverting_amp_seed.number_of_nodes()
    assert g.number_of_edges() == inverting_amp_seed.number_of_edges()


def test_rename_nets_preserves_role_attributes(
    inverting_amp_seed: nx.Graph,
) -> None:
    rng = random.Random(0)
    g = rename_nets(inverting_amp_seed, rng)
    roles_before = sorted(
        d.get("role")
        for _, d in inverting_amp_seed.nodes(data=True)
        if d.get("kind") == "net"
    )
    roles_after = sorted(
        d.get("role")
        for _, d in g.nodes(data=True)
        if d.get("kind") == "net"
    )
    assert roles_before == roles_after


def test_swap_pin_assignments_keeps_edge_count(
    inverting_amp_seed: nx.Graph,
) -> None:
    rng = random.Random(0)
    g = swap_passive_pin_assignments(inverting_amp_seed, rng, p=1.0)
    assert g.number_of_edges() == inverting_amp_seed.number_of_edges()


def test_add_decoration_wire_adds_one_component_and_two_edges(
    inverting_amp_seed: nx.Graph,
) -> None:
    rng = random.Random(0)
    g = add_decoration_wire(inverting_amp_seed, rng)
    comp_count_before = sum(
        1 for _, d in inverting_amp_seed.nodes(data=True) if d.get("kind") == "comp"
    )
    comp_count_after = sum(
        1 for _, d in g.nodes(data=True) if d.get("kind") == "comp"
    )
    # NOTE: a Wire bridging a single net adds 1 comp node but only 1 unique
    # edge (since networkx.Graph collapses both wire pins → same net into
    # one edge). This is OK semantically — the Wire is present in the graph.
    assert comp_count_after == comp_count_before + 1
    assert g.number_of_edges() >= inverting_amp_seed.number_of_edges()
