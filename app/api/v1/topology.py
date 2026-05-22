"""``/api/v1/topology/*`` — GNN-A topology classification endpoints.

## Endpoints

* ``POST /api/v1/topology/suggest`` — main entry point. Takes a
  netlist_v2 (or logical_reference) and returns the GNN's top-K topology
  predictions plus a symbolic-template cross-check.

* ``GET /api/v1/topology/model-info`` — frontend boots fetch this to
  decide whether to show the AI-recommendation panel; ops dashboards use
  it to verify ckpt version.

## Layered design

This router is **thin**. All inference + decision logic lives in
:mod:`app.services.topology_classifier_service`. The router only:
  1. Validates request shape (Pydantic).
  2. Hands off to the service.
  3. Converts service dataclasses → pydantic DTOs.
  4. Translates failures into HTTP semantics.

This separation lets non-HTTP callers (tests, jupyter notebooks, the
compare orchestrator's future ``_attach_topology_suggestion`` hook)
import the service directly without spinning up FastAPI.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.domain.topology.labels import TOPOLOGY_LABELS
from app.schemas.topology import (
    TopologyConsensusDTO,
    TopologyGraphStats,
    TopologyModelInfoResponse,
    TopologyPredictionDTO,
    TopologySuggestRequest,
    TopologySuggestResponse,
)
from app.services.topology_classifier_service import (
    DEFAULT_CKPT_PATH,
    _LOAD_ERROR,
    TopologySuggestion,
    suggest_from_graph,
    suggest_from_netlist_v2,
)


router = APIRouter(prefix="/topology", tags=["topology"])


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _suggestion_to_response(suggestion: TopologySuggestion) -> TopologySuggestResponse:
    """Convert the service-layer dataclass to the HTTP DTO."""
    return TopologySuggestResponse(
        enabled=suggestion.enabled,
        disabled_reason=suggestion.disabled_reason,
        gnn_predictions=[
            TopologyPredictionDTO(**p.to_dict()) for p in suggestion.gnn_predictions
        ],
        template_matches=suggestion.template_matches,
        consensus=(
            TopologyConsensusDTO(
                agreed=suggestion.consensus.agreed,
                recommended_label=suggestion.consensus.recommended_label,
                recommended_template_id=suggestion.consensus.recommended_template_id,
                recommended_reference_id=suggestion.consensus.recommended_reference_id,
                confidence_band=suggestion.consensus.confidence_band,
            )
            if suggestion.consensus
            else None
        ),
        model_version=suggestion.model_version,
        inference_ms=suggestion.inference_ms,
        graph_stats=TopologyGraphStats(**suggestion.graph_stats),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/suggest", response_model=TopologySuggestResponse)
async def suggest_topology(payload: TopologySuggestRequest) -> TopologySuggestResponse:
    """Run GNN-A topology classification on a circuit graph.

    The request must include either ``netlist_v2`` (preferred, matches
    the pipeline S3 output shape) or ``logical_reference`` (the DSL
    payload — useful for testing).

    The response always includes ``enabled`` + ``disabled_reason`` so
    the frontend can render an "AI sat out" widget without inspecting
    exceptions. See :mod:`app.services.topology_classifier_service` for
    the disabled_reason vocabulary.
    """
    if payload.netlist_v2 is None and payload.logical_reference is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "must provide either 'netlist_v2' or 'logical_reference'"
            ),
        )

    if payload.netlist_v2 is not None:
        suggestion = suggest_from_netlist_v2(
            payload.netlist_v2, top_k=payload.top_k,
        )
    else:
        # logical_reference branch: build graph via logical_reference_to_graph.
        # Imported lazily so the router has no hard dep on the domain layer
        # at module import time.
        from app.domain.logical_reference import logical_reference_to_graph

        try:
            graph = logical_reference_to_graph(payload.logical_reference)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail=f"invalid logical_reference: {type(exc).__name__}: {exc}",
            )
        suggestion = suggest_from_graph(graph, top_k=payload.top_k)

    return _suggestion_to_response(suggestion)


@router.get("/model-info", response_model=TopologyModelInfoResponse)
async def get_model_info() -> TopologyModelInfoResponse:
    """Return basic info about the loaded GNN-A model.

    Cheap call — does NOT trigger model loading, only checks ckpt presence
    on disk and reports the latest cached load error (if any). The
    frontend uses this on boot to decide whether the AI-recommendation
    panel should appear.
    """
    return TopologyModelInfoResponse(
        available=DEFAULT_CKPT_PATH.exists(),
        ckpt_path=str(DEFAULT_CKPT_PATH),
        ckpt_exists=DEFAULT_CKPT_PATH.exists(),
        model_version="gnn_a_v2",
        num_classes=len(TOPOLOGY_LABELS),
        labels=list(TOPOLOGY_LABELS),
        load_error=_LOAD_ERROR,
    )
