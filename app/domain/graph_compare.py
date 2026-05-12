from __future__ import annotations

from .compare import compare_logical_graphs
from .compare.diff_report import (
    _approximate_similarity,
    _build_mappings,
    _component_count,
    _component_progress,
    _component_type_counts,
    _difference_items,
    _extra_items,
    _fallback_comp_mapping,
    _ged_similarity,
    _graph_neighbor_signature,
    _greedy_match_by_score,
    _missing_items,
    _neighbor_signature_similarity,
    _net_count,
)
from .compare.matcher import (
    _component_type_key,
    _component_types_equivalent,
    _contains_subgraph,
    _edge_match,
    _find_isomorphism,
    _is_isomorphic,
    _mapping_uses_allowed_symmetry,
    _node_match,
    _role_labels_equivalent,
    auto_detect_symmetries,
)
from .compare.role_inference import _infer_current_net_roles_from_reference

__all__ = ["compare_logical_graphs"]
