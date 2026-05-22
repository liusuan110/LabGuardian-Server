"""CADx Phase 1 — GNN-A topology classifier.

This package houses the *learned* topology classifier that maps an entire
circuit graph to one of 7 canonical topology classes (the 6 demo topologies
plus an ``unknown`` catch-all). It complements the symbolic template
matcher (``app/domain/templates``) by providing:

  * **Fast inference**: single forward pass instead of 6× VF2 runs
  * **Generalization**: handles unseen wiring variants without explicit
    template authoring
  * **Edge-friendly**: tiny GraphSAGE model (~50K params), OpenVINO + INT8

The classifier is **complementary, not competitive**: at runtime the
backend runs both and surfaces the top-K hypotheses. When they agree,
confidence rises; when they disagree, the user sees both for verification.

See :mod:`app.domain.topology.labels` for the canonical label set and
schema documentation.
"""

from app.domain.topology.labels import (
    DEFAULT_UNKNOWN_LABEL,
    TOPOLOGY_LABELS,
    TopologyLabel,
    TopologyLabelSpec,
    get_label_spec,
    label_to_index,
    list_labels,
)

__all__ = [
    "DEFAULT_UNKNOWN_LABEL",
    "TOPOLOGY_LABELS",
    "TopologyLabel",
    "TopologyLabelSpec",
    "get_label_spec",
    "label_to_index",
    "list_labels",
]
