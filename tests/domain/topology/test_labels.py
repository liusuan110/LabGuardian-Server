"""Unit tests for the canonical topology label set.

These tests guard the **single source of truth** for GNN-A class ids.
Failures here usually mean someone reordered ``TOPOLOGY_LABELS`` (which
silently breaks loaded checkpoints) or that a template's
``topology_label`` field drifted from the label table.
"""

from __future__ import annotations

import pytest

from app.domain.templates import get_template_registry
from app.domain.topology.labels import (
    DEFAULT_UNKNOWN_LABEL,
    TOPOLOGY_LABELS,
    get_label_spec,
    index_to_label,
    label_to_index,
    list_labels,
    validate_label_spec,
)


# ---------------------------------------------------------------------------
# Schema self-tests
# ---------------------------------------------------------------------------


def test_label_spec_self_validates() -> None:
    errors = validate_label_spec()
    assert errors == [], f"label spec inconsistencies: {errors}"


def test_label_count_is_seven() -> None:
    """7 = 6 demo topologies + 1 ``unknown``."""
    assert len(TOPOLOGY_LABELS) == 7


def test_unknown_is_last_index() -> None:
    """``unknown`` lives at the highest index so adding new real classes
    can be done by inserting BEFORE it without shifting non-unknown ids.
    """
    assert TOPOLOGY_LABELS[-1] == DEFAULT_UNKNOWN_LABEL


def test_labels_are_unique() -> None:
    assert len(set(TOPOLOGY_LABELS)) == len(TOPOLOGY_LABELS)


def test_indices_are_contiguous() -> None:
    for i, label in enumerate(TOPOLOGY_LABELS):
        assert get_label_spec(label).index == i


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", TOPOLOGY_LABELS)
def test_round_trip_label_index(label: str) -> None:
    assert index_to_label(label_to_index(label)) == label


def test_get_label_spec_raises_on_unknown_string() -> None:
    with pytest.raises(KeyError):
        get_label_spec("not_a_real_topology")


def test_list_labels_includes_unknown_by_default() -> None:
    assert DEFAULT_UNKNOWN_LABEL in list_labels()


def test_list_labels_can_exclude_unknown() -> None:
    labels = list_labels(include_unknown=False)
    assert DEFAULT_UNKNOWN_LABEL not in labels
    assert len(labels) == 6


# ---------------------------------------------------------------------------
# Cross-reference: every template's ``topology_label`` must be registered
# ---------------------------------------------------------------------------


def test_every_template_label_is_registered() -> None:
    """Templates' ``topology_label`` field is the wire that connects the
    symbolic matcher's output to GNN-A's class ids. If a template uses a
    label not in :data:`TOPOLOGY_LABELS`, the eventual consensus layer
    can't compare the two paths.
    """
    registry = get_template_registry()
    template_labels = {tpl.topology_label for tpl in registry.values()}
    drift = template_labels - set(TOPOLOGY_LABELS)
    assert not drift, (
        f"templates use labels not in TOPOLOGY_LABELS: {drift}. "
        "Add them to app/domain/topology/labels.py::TOPOLOGY_LABELS "
        "(append at the end to preserve existing class ids)."
    )


def test_every_non_unknown_label_has_matching_template() -> None:
    """The inverse: every real label should have a template that claims it,
    so the symbolic path can always speak to GNN's verdict.
    """
    registry = get_template_registry()
    template_labels = {tpl.topology_label for tpl in registry.values()}
    expected = set(TOPOLOGY_LABELS) - {DEFAULT_UNKNOWN_LABEL}
    missing = expected - template_labels
    assert not missing, (
        f"these labels have no matching template: {missing}. "
        "Either add a template, or remove the label."
    )


def test_label_spec_reference_ids_resolve() -> None:
    """Reference IDs in label specs must correspond to existing DSL files."""
    from app.services.reference_service import ReferenceService

    svc = ReferenceService()
    available = {item["reference_id"] for item in svc.list_references()}
    for label in TOPOLOGY_LABELS:
        spec = get_label_spec(label)
        if spec.reference_id is None:
            continue
        assert spec.reference_id in available, (
            f"label {label!r} references {spec.reference_id!r} "
            f"but no such DSL file exists in knowledge/references/"
        )
