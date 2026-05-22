"""Pydantic schemas for the ``/api/v1/topology/*`` endpoints.

Mirrors the dataclasses in
:mod:`app.services.topology_classifier_service`. We keep the two layered
because:
  * **Dataclasses** are used by the service layer (typed, lightweight,
    easy to test without pydantic dependency).
  * **Pydantic models** are used at the HTTP boundary (FastAPI generates
    OpenAPI schemas from them, and the frontend's TypeScript codegen
    consumes those).

A single source of truth would be nice but mixing the two libraries at
the same layer historically caused subtle ``model_dump`` vs ``asdict``
drift, so we keep them parallel and copy fields by hand.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================================
# Request schemas
# ============================================================================


class TopologySuggestRequest(BaseModel):
    """Body of ``POST /api/v1/topology/suggest``.

    The request carries one of two graph representations:
      * ``netlist_v2``: the standard pipeline S3 output shape — this is
        the production path the frontend uses.
      * ``logical_reference``: a fully-formed reference circuit payload
        (``logical_reference_v1`` schema). Useful for debugging or
        scripted evaluation where there's no student board involved.

    At least one of the two must be provided. ``netlist_v2`` takes
    precedence if both are set.
    """

    netlist_v2: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Current circuit netlist in netlist_v2 shape (S3 pipeline output). "
            "Preferred input format."
        ),
    )
    logical_reference: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional alternative: a logical_reference_v1 payload. "
            "Useful for testing without a real student board."
        ),
    )
    top_k: int = Field(
        default=7,
        ge=1,
        le=7,
        description=(
            "How many predictions to return (model has 7 classes total)."
        ),
    )


# ============================================================================
# Response schemas
# ============================================================================


class TopologyPredictionDTO(BaseModel):
    """One label's GNN softmax probability."""

    label: str = Field(..., description="Canonical topology label (snake_case).")
    display_name_zh: str
    display_name_en: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1, description="1-based rank by confidence.")
    template_id: str | None
    reference_id: str | None


class TopologyConsensusDTO(BaseModel):
    """GNN-vs-template cross-check result."""

    agreed: bool = Field(
        ..., description="Whether GNN top-1 agrees with the top template."
    )
    recommended_label: str
    recommended_template_id: str | None
    recommended_reference_id: str | None
    confidence_band: Literal["high", "medium", "low", "disagreement"]


class TopologyGraphStats(BaseModel):
    """Lightweight stats so the frontend can show "AI saw N components" etc."""

    num_nodes: int = 0
    num_edges: int = 0
    num_comp_nodes: int = 0
    num_net_nodes: int = 0


class TopologySuggestResponse(BaseModel):
    """Body of the ``POST /api/v1/topology/suggest`` response.

    The ``enabled`` + ``disabled_reason`` pattern mirrors the existing
    GNN advisor schema, so the frontend can reuse its "sat out" widget
    with the same UX vocabulary.
    """

    enabled: bool = Field(
        ..., description="False when the model couldn't run; see disabled_reason."
    )
    disabled_reason: str | None = Field(
        default=None,
        description=(
            "Set when enabled is False. Vocabulary: "
            "checkpoint_missing | runtime_unavailable | model_failed | "
            "tiny_graph | invalid_netlist: ... "
        ),
    )
    gnn_predictions: list[TopologyPredictionDTO] = Field(default_factory=list)
    template_matches: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Symbolic template matcher output (top-K). Each item is the "
            "result of TemplateMatchResult.to_dict() — kept as dicts to "
            "avoid duplicating the template result schema here."
        ),
    )
    consensus: TopologyConsensusDTO | None = None
    model_version: str = "gnn_a_v2"
    inference_ms: float = Field(default=0.0, ge=0.0)
    graph_stats: TopologyGraphStats = Field(default_factory=TopologyGraphStats)


# ============================================================================
# Health / metadata
# ============================================================================


class TopologyModelInfoResponse(BaseModel):
    """``GET /api/v1/topology/model-info`` response.

    Used by frontend at boot to decide whether the AI-recommendation
    panel is available, and by ops dashboards to verify ckpt version.
    """

    available: bool
    ckpt_path: str
    ckpt_exists: bool
    model_version: str = "gnn_a_v2"
    num_classes: int
    labels: list[str]
    load_error: str | None = None
