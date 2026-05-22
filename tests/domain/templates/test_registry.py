"""Unit tests for the topology template registry singleton."""

from __future__ import annotations

import pytest

from app.domain.templates.base import TopologyTemplate
from app.domain.templates.registry import get_template_registry


EXPECTED_TEMPLATE_IDS = {
    "rc_first_order_v1",
    "common_emitter_v1",
    "differential_pair_v1",
    "inverting_amp_ua741_v1",
    "summing_amp_ua741_v1",
    "integrator_ua741_v1",
}


@pytest.fixture(autouse=True)
def _reset_registry_cache():
    """Clear LRU cache between tests so changes to templates re-validate."""
    get_template_registry.cache_clear()
    yield
    get_template_registry.cache_clear()


def test_registry_contains_six_templates() -> None:
    registry = get_template_registry()
    assert len(registry) == 6
    assert set(registry.keys()) == EXPECTED_TEMPLATE_IDS


def test_every_template_validates() -> None:
    registry = get_template_registry()
    for template_id, template in registry.items():
        errors = template.validate()
        assert errors == [], f"{template_id} failed validation: {errors}"


def test_every_template_has_required_components() -> None:
    registry = get_template_registry()
    for template_id, template in registry.items():
        assert len(template.required_components) > 0, (
            f"{template_id} has no required components"
        )


def test_topology_labels_are_unique() -> None:
    registry = get_template_registry()
    labels = [t.topology_label for t in registry.values()]
    assert len(labels) == len(set(labels)), "duplicate topology labels"


def test_all_templates_are_topology_template_instances() -> None:
    registry = get_template_registry()
    for template in registry.values():
        assert isinstance(template, TopologyTemplate)
