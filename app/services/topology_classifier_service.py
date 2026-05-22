"""GNN-A topology classifier service — singleton model loader + inference.

## What this service does

Wraps the trained ``TopologyClassifier`` ckpt and exposes a clean
``predict(graph) -> list[Prediction]`` / ``suggest(netlist_v2) -> Suggestion``
API for the FastAPI router (``app/api/v1/topology.py``).

## Singleton + lazy loading

The model is loaded **once** at first call and held in module-level
state (``_LOADED_MODEL``). This avoids:
  * Re-deserializing the 57 KB ckpt on every request
  * Re-initializing the PyG layers' RNG-affected internals
  * Holding a copy per worker process (uvicorn workers will each load
    once, which is correct — they're separate processes)

The lazy load is wrapped in a thread lock so two concurrent first
requests can't double-initialize.

## Robustness

If the ckpt file is missing or unloadable, ``suggest()`` returns a
``disabled_reason`` instead of raising — same pattern as the existing
SEAL advisor (``app/domain/gnn/inference.py::should_use_gnn``). The
caller can then decide whether to surface this in the UI.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import torch

from app.domain.logical_reference import current_netlist_v2_to_graph
from app.domain.templates import get_template_registry, match_all_templates
from app.domain.topology.features import encode_graph, encoded_to_pyg_data
from app.domain.topology.labels import (
    DEFAULT_UNKNOWN_LABEL,
    get_label_spec,
    index_to_label,
)
from app.domain.topology.model import TopologyClassifier


log = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


# Default ckpt path. v2 (2026-05-22) introduced neighbor-count features
# (FEATURE_DIM 21 → 23) that dramatically improved the UA741 three-tribe
# (inverting / summing / integrator) discrimination — margins jumped from
# near-tie to >86% on real fixtures. v1 ckpts are kept on disk for
# regression baselines but NOT used by default.
DEFAULT_CKPT_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "gnn_a_v2" / "best.pt"

# When the top-1 softmax probability is below this threshold, the
# classifier hedges to the ``unknown`` label rather than committing to a
# weak guess. See ``docs/topology_label_spec.md`` for the rationale.
UNKNOWN_CONFIDENCE_THRESHOLD = 0.4

# Top-K predictions returned. Frontend ReferenceSelector only shows 3
# but the API returns all 7 so debuggers can inspect the full distribution.
DEFAULT_TOP_K = 7

# Telemetry codes mirror the SEAL advisor's gnn_disabled_reason vocabulary
# so the frontend can reuse its "AI sat out" widget styling.
REASON_CKPT_MISSING = "checkpoint_missing"
REASON_RUNTIME_UNAVAILABLE = "runtime_unavailable"
REASON_MODEL_FAILED = "model_failed"
REASON_TINY_GRAPH = "tiny_graph"


# ============================================================================
# Dataclasses (mirror Pydantic schemas in app/schemas/topology.py)
# ============================================================================


@dataclass
class TopologyPrediction:
    """One label's GNN softmax output."""

    label: str
    display_name_zh: str
    display_name_en: str
    confidence: float
    rank: int  # 1-based: 1 = top-1
    template_id: str | None
    reference_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TopologyConsensus:
    """Cross-validation of GNN vs template matcher.

    Attributes:
        agreed: ``True`` if GNN top-1 == template top-1 (mapped via
            label spec). Disagreement is a yellow flag for the frontend.
        recommended_label: The label to surface as the AI's choice.
            Falls back to ``unknown`` when both paths are low-confidence.
        recommended_template_id: Template id of the recommended label
            (or ``None`` for ``unknown``).
        recommended_reference_id: Reference DSL id (or ``None``).
        confidence_band: ``"high" | "medium" | "low" | "disagreement"``
            for UX banner color selection.
    """

    agreed: bool
    recommended_label: str
    recommended_template_id: str | None
    recommended_reference_id: str | None
    confidence_band: str


@dataclass
class TopologySuggestion:
    """Full response payload for ``POST /api/v1/topology/suggest``."""

    enabled: bool
    disabled_reason: str | None
    gnn_predictions: list[TopologyPrediction] = field(default_factory=list)
    template_matches: list[dict[str, Any]] = field(default_factory=list)
    consensus: TopologyConsensus | None = None
    model_version: str = "gnn_a_v2"
    inference_ms: float = 0.0
    graph_stats: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "gnn_predictions": [p.to_dict() for p in self.gnn_predictions],
            "template_matches": list(self.template_matches),
            "consensus": asdict(self.consensus) if self.consensus else None,
            "model_version": self.model_version,
            "inference_ms": round(self.inference_ms, 3),
            "graph_stats": dict(self.graph_stats),
        }
        return d


# ============================================================================
# Singleton model
# ============================================================================


_LOADED_MODEL: TopologyClassifier | None = None
_LOADED_MODEL_PATH: Path | None = None
_LOAD_LOCK = threading.Lock()
_LOAD_ERROR: str | None = None


def _load_model(ckpt_path: Path) -> TopologyClassifier | None:
    """Internal: load ckpt with double-checked locking.

    Returns ``None`` on failure and sets ``_LOAD_ERROR``. Callers should
    surface ``_LOAD_ERROR`` via the ``disabled_reason`` field.
    """
    global _LOADED_MODEL, _LOADED_MODEL_PATH, _LOAD_ERROR

    if _LOADED_MODEL is not None and _LOADED_MODEL_PATH == ckpt_path:
        return _LOADED_MODEL

    with _LOAD_LOCK:
        # Double-checked locking: another thread may have loaded while we
        # were waiting on the lock.
        if _LOADED_MODEL is not None and _LOADED_MODEL_PATH == ckpt_path:
            return _LOADED_MODEL

        if not ckpt_path.exists():
            _LOAD_ERROR = f"checkpoint not found: {ckpt_path}"
            log.warning("gnn_a_ckpt_missing path=%s", ckpt_path)
            return None

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            config = ckpt.get("config", {})
            model = TopologyClassifier(**config)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            _LOADED_MODEL = model
            _LOADED_MODEL_PATH = ckpt_path
            _LOAD_ERROR = None
            log.info(
                "gnn_a_loaded ckpt=%s epoch=%s val_acc=%.3f params=%d",
                ckpt_path,
                ckpt.get("epoch", "?"),
                ckpt.get("val_acc", 0.0),
                sum(p.numel() for p in model.parameters()),
            )
            return model
        except Exception as exc:  # noqa: BLE001 — never crash the request
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            log.exception("gnn_a_load_failed path=%s", ckpt_path)
            return None


def reset_model_cache() -> None:
    """Clear the loaded model. Tests + reload scripts use this."""
    global _LOADED_MODEL, _LOADED_MODEL_PATH, _LOAD_ERROR
    with _LOAD_LOCK:
        _LOADED_MODEL = None
        _LOADED_MODEL_PATH = None
        _LOAD_ERROR = None


# ============================================================================
# Prediction logic
# ============================================================================


def _predict_on_graph(
    model: TopologyClassifier,
    graph: nx.Graph,
) -> tuple[list[TopologyPrediction], float]:
    """Run a single forward pass and return ranked predictions + latency_ms.

    The returned list is **sorted by confidence descending**, with rank
    starting at 1.
    """
    encoded = encode_graph(graph)
    data = encoded_to_pyg_data(encoded)
    batch = torch.zeros(data.x.shape[0], dtype=torch.long)

    t0 = time.perf_counter()
    with torch.no_grad():
        probs = model.predict_proba(data.x, data.edge_index, batch).squeeze(0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Build predictions ordered by confidence desc.
    indexed = sorted(
        ((int(i), float(p)) for i, p in enumerate(probs)),
        key=lambda x: -x[1],
    )
    predictions: list[TopologyPrediction] = []
    for rank, (idx, prob) in enumerate(indexed, start=1):
        label = index_to_label(idx)
        spec = get_label_spec(label)
        predictions.append(
            TopologyPrediction(
                label=label,
                display_name_zh=spec.display_name_zh,
                display_name_en=spec.display_name_en,
                confidence=prob,
                rank=rank,
                template_id=spec.template_id,
                reference_id=spec.reference_id,
            )
        )
    return predictions, latency_ms


def _consensus(
    gnn_predictions: list[TopologyPrediction],
    template_matches: list[dict[str, Any]],
) -> TopologyConsensus:
    """Decide what to recommend given the GNN + template outputs.

    Rules (ordered):
      1. If GNN top-1 confidence < ``UNKNOWN_CONFIDENCE_THRESHOLD``,
         recommend ``unknown`` (low confidence band).
      2. If template top-1 exists with confidence > 0.5 AND its
         ``topology_label`` matches GNN top-1's label → high agreement
         (high band).
      3. If they disagree but both are strong → disagreement band; we
         still surface GNN's choice as the recommendation (it has the
         broader generalization), but the UI should show both.
      4. If template path failed entirely → fall back to GNN-only
         (medium band).
    """
    if not gnn_predictions:
        return TopologyConsensus(
            agreed=False,
            recommended_label=DEFAULT_UNKNOWN_LABEL,
            recommended_template_id=None,
            recommended_reference_id=None,
            confidence_band="low",
        )

    top1_gnn = gnn_predictions[0]

    if top1_gnn.confidence < UNKNOWN_CONFIDENCE_THRESHOLD:
        unknown_spec = get_label_spec(DEFAULT_UNKNOWN_LABEL)
        return TopologyConsensus(
            agreed=False,
            recommended_label=DEFAULT_UNKNOWN_LABEL,
            recommended_template_id=unknown_spec.template_id,
            recommended_reference_id=unknown_spec.reference_id,
            confidence_band="low",
        )

    # Look at the top template match (template_matches is sorted by
    # confidence descending by ``match_all_templates``).
    top_template = template_matches[0] if template_matches else None
    if top_template and top_template.get("confidence", 0.0) > 0.5:
        template_label = top_template.get("topology_label")
        if template_label == top1_gnn.label:
            return TopologyConsensus(
                agreed=True,
                recommended_label=top1_gnn.label,
                recommended_template_id=top1_gnn.template_id,
                recommended_reference_id=top1_gnn.reference_id,
                confidence_band="high",
            )
        else:
            return TopologyConsensus(
                agreed=False,
                recommended_label=top1_gnn.label,
                recommended_template_id=top1_gnn.template_id,
                recommended_reference_id=top1_gnn.reference_id,
                confidence_band="disagreement",
            )

    # GNN strong, template weak/missing — go with GNN.
    return TopologyConsensus(
        agreed=False,
        recommended_label=top1_gnn.label,
        recommended_template_id=top1_gnn.template_id,
        recommended_reference_id=top1_gnn.reference_id,
        confidence_band="medium",
    )


# ============================================================================
# Public API
# ============================================================================


def suggest_from_graph(
    graph: nx.Graph,
    ckpt_path: Path | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> TopologySuggestion:
    """Run topology suggestion for a pre-built ``nx.Graph``.

    Returned object always has ``enabled`` + ``disabled_reason`` so the
    caller can render an "AI sat out" widget without inspecting exceptions.
    """
    ckpt_path = ckpt_path or DEFAULT_CKPT_PATH

    graph_stats = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_comp_nodes": sum(
            1 for _, d in graph.nodes(data=True) if d.get("kind") == "comp"
        ),
        "num_net_nodes": sum(
            1 for _, d in graph.nodes(data=True) if d.get("kind") == "net"
        ),
    }

    if graph.number_of_nodes() < 2:
        return TopologySuggestion(
            enabled=False,
            disabled_reason=REASON_TINY_GRAPH,
            graph_stats=graph_stats,
        )

    model = _load_model(ckpt_path)
    if model is None:
        reason = REASON_CKPT_MISSING if "not found" in (_LOAD_ERROR or "") else REASON_MODEL_FAILED
        return TopologySuggestion(
            enabled=False,
            disabled_reason=reason,
            graph_stats=graph_stats,
        )

    try:
        gnn_preds, latency_ms = _predict_on_graph(model, graph)
    except Exception as exc:  # noqa: BLE001 — never crash request
        log.exception("gnn_a_inference_failed")
        return TopologySuggestion(
            enabled=False,
            disabled_reason=f"{REASON_MODEL_FAILED}: {type(exc).__name__}",
            graph_stats=graph_stats,
        )

    # Run symbolic templates in parallel (cheap — a few ms).
    try:
        template_results = match_all_templates(graph, get_template_registry())
        template_dicts = [r.to_dict() for r in template_results[:top_k]]
    except Exception:  # noqa: BLE001
        log.warning("template_match_failed_in_suggest", exc_info=True)
        template_dicts = []

    consensus = _consensus(gnn_preds, template_dicts)

    return TopologySuggestion(
        enabled=True,
        disabled_reason=None,
        gnn_predictions=gnn_preds[:top_k],
        template_matches=template_dicts,
        consensus=consensus,
        inference_ms=latency_ms,
        graph_stats=graph_stats,
    )


def suggest_from_netlist_v2(
    netlist_v2: dict[str, Any],
    ckpt_path: Path | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> TopologySuggestion:
    """Entry point that the HTTP route calls. Builds the graph from the
    netlist_v2 payload, then delegates to :func:`suggest_from_graph`.
    """
    try:
        graph = current_netlist_v2_to_graph(netlist_v2)
    except Exception as exc:  # noqa: BLE001
        log.warning("netlist_v2_to_graph_failed err=%s", type(exc).__name__)
        return TopologySuggestion(
            enabled=False,
            disabled_reason=f"invalid_netlist: {type(exc).__name__}",
        )
    return suggest_from_graph(graph, ckpt_path=ckpt_path, top_k=top_k)


__all__ = [
    "DEFAULT_CKPT_PATH",
    "TopologyConsensus",
    "TopologyPrediction",
    "TopologySuggestion",
    "reset_model_cache",
    "suggest_from_graph",
    "suggest_from_netlist_v2",
]
