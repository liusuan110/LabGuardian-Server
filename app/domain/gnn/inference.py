"""GNN 模块 · GNNAdvisor 推理入口（P4 · plan §六 整合 orchestrator）.

把一个 ref + cur :class:`HeteroCircuitGraph` 喂进 :class:`CircuitMatchNet`，
按 cur 中**实际观测到的 (port, net) 边**逐条评分，输出
:class:`GNNAdvice` —— 一个**纯只读、可序列化**的结构化 hint 包，由
orchestrator 注入 ``validator_report_v2.summary.gnn``。

**核心契约 (plan §一 / §六)**：

1. **GNN 永远不决定 pass/fail** —— GNNAdvice 只是 advisory。orchestrator
   仍由规则路径决定 ``logic_correct``。
2. **失败 / 超时静默 fallback** —— ``advise()`` 任何异常或超时都返回
   ``None``，orchestrator 走纯规则路径。
3. **零 import 副作用** —— 没装 ``[gnn]`` extra 时，``GNNAdvisor.get()``
   抛 ``RuntimeError`` 而不是 ImportError 在 import 期就炸。

设计参见 plan §一 (architecture)、§六 (orchestrator merge layer)、
§十 (risks: GNN 误判 / 黑盒可解释性)。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from app.domain.gnn.hetero_circuit import HeteroCircuitGraph
    from app.domain.gnn.model import CircuitMatchNet


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Advice data structure (orchestrator-visible)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GNNAdvice:
    """**Read-only, JSON-serialisable** advisory payload from GNNAdvisor.

    Fields are aligned with plan §六 ``validator_report_v2.summary.gnn``
    schema. Per plan §一, every field carries a confidence so the UI
    layer can weaken / hide low-confidence hints.

    MVP populates:
        - ``edge_predictions``: per (port, net) edge, ``P(edge_correct)``
        - ``hotspots``: per cur port, ``1 - min(p_correct over its edges)``
        - ``graph_similarity``: mean of all ``p_correct`` (rough scalar)
        - ``inference_ms``, ``model_version``, ``n_edges_scored``

    Reserved for P4.1 (heads not yet implemented):
        - ``predicted_error_types``, ``component_mapping_topk``,
          ``net_mapping_topk``, ``progress_score``, ``suggested_target``
          (per wrong port — needs MISSING_EDGE head batched on the fly)
    """

    model_version: str
    inference_ms: float
    n_edges_scored: int
    edge_predictions: tuple[dict[str, Any], ...] = ()
    hotspots: tuple[dict[str, Any], ...] = ()
    graph_similarity: float = 0.0
    graph_similarity_confidence: float = 0.0
    # P4.1 reserved
    predicted_error_types: tuple[dict[str, Any], ...] = ()
    component_mapping_topk: dict[str, list] = field(default_factory=dict)
    net_mapping_topk: dict[str, list] = field(default_factory=dict)
    progress_score: float | None = None
    disagreement_with_rule: bool = False
    enabled: bool = True

    def to_report_dict(
        self, *, min_hotspot_confidence: float = 0.6
    ) -> dict[str, Any]:
        """Render the advice into the dict the orchestrator stuffs into
        ``report.summary.gnn``. Hotspots below ``min_hotspot_confidence``
        are dropped (plan §六 ``MIN_HOTSPOT_CONFIDENCE = 0.6``)."""

        filtered_hotspots = [
            h for h in self.hotspots
            if h.get("score", 0.0) >= min_hotspot_confidence
        ]
        return {
            "enabled": self.enabled,
            "model_version": self.model_version,
            "graph_similarity": self.graph_similarity,
            "graph_similarity_confidence": self.graph_similarity_confidence,
            "progress_score": self.progress_score,
            "n_edges_scored": self.n_edges_scored,
            "inference_ms": self.inference_ms,
            "edge_predictions": list(self.edge_predictions),
            "hotspots": filtered_hotspots,
            "predicted_error_types": list(self.predicted_error_types),
            "component_mapping_topk": dict(self.component_mapping_topk),
            "net_mapping_topk": dict(self.net_mapping_topk),
            "disagreement_with_rule": self.disagreement_with_rule,
        }


# ---------------------------------------------------------------------------
# GNNAdvisor — model load + advise()
# ---------------------------------------------------------------------------


# Environment override for the default checkpoint path; useful in tests
# or when the deployed checkpoint lives outside the repo.
_DEFAULT_CKPT_ENV = "LABGUARDIAN_GNN_CKPT"
_DEFAULT_CKPT_CANDIDATES: tuple[str, ...] = (
    "checkpoints/p3_followup_v2/best_f1.pt",  # current best baseline
    "checkpoints/p3_followup_v1/best_f1.pt",
    "checkpoints/p3_v1/best_f1.pt",
)

_SINGLETON: GNNAdvisor | None = None


class GNNAdvisor:
    """Stateful GNN inference wrapper. Singleton via :meth:`get`.

    Construction is **deliberately strict**: callers must either
    ``get()`` (returns the singleton, lazily loading the default
    checkpoint) or ``from_checkpoint(path)``. Direct construction
    requires a fully-loaded :class:`CircuitMatchNet`.

    Threading: ``advise()`` is *not* thread-safe (the underlying torch
    model isn't). Callers that fan out should clone the advisor or
    serialise calls.
    """

    def __init__(
        self,
        model: CircuitMatchNet,
        *,
        model_version: str = "circuit_match",
        threshold_wrong: float = 0.5,
    ):
        self.model = model
        self.model_version = model_version
        self.threshold_wrong = threshold_wrong
        self.model.eval()

    # ----- Construction -------------------------------------------------

    @classmethod
    def get(cls) -> GNNAdvisor:
        """Return the process-global advisor, lazily loading the default
        checkpoint on first call. Subsequent calls reuse the same model
        instance.

        Raises:
            RuntimeError if torch / torch_geometric aren't installed
                (``[gnn]`` extra missing).
            FileNotFoundError if no default checkpoint can be located
                (caller can override via the ``LABGUARDIAN_GNN_CKPT``
                env var or use ``from_checkpoint`` explicitly).
        """

        global _SINGLETON
        if _SINGLETON is not None:
            return _SINGLETON
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "GNNAdvisor requires the [gnn] extra (torch + "
                "torch_geometric). Install with: pip install -e '.[gnn]'"
            ) from e
        path = _locate_default_checkpoint()
        if path is None:
            raise FileNotFoundError(
                "no GNN checkpoint found. Train one with "
                "`scripts/gnn_train_full.py` or set "
                f"${_DEFAULT_CKPT_ENV} to its path."
            )
        _SINGLETON = cls.from_checkpoint(path)
        return _SINGLETON

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> GNNAdvisor:
        """Build an advisor by loading a P3 CircuitMatchNet from disk.

        Accepts both P3 ``best_f1.pt`` (saved by
        ``scripts/gnn_train_full.py``) and P2.5 SealDGCNN-only
        ``backbone.pt`` (wraps the SEAL head in a fresh CircuitMatchNet
        skeleton).
        """

        from app.domain.gnn.model import CircuitMatchNet

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        # P3 layout has a top-level "config" key. P2.5 layout has
        # "state_dict" + "hidden" / "sort_k" / "in_channels".
        import torch
        head = torch.load(path, map_location="cpu", weights_only=False)
        if "config" in head:
            model = CircuitMatchNet.load(path)
        else:
            model = CircuitMatchNet.from_pretrained_backbone(
                path, strict=False
            )
        return cls(
            model=model,
            model_version=f"circuit_match:{path.stem}",
        )

    @classmethod
    def checkpoint_available(cls) -> bool:
        """Quick check used by :func:`should_use_gnn` — does *any*
        plausible checkpoint exist on disk? Doesn't actually load it."""

        return _locate_default_checkpoint() is not None

    @classmethod
    def reset_singleton(cls) -> None:
        """Drop the cached singleton — for tests that need to swap
        models."""

        global _SINGLETON
        _SINGLETON = None

    # ----- Advice -------------------------------------------------------

    def advise(
        self,
        ref_hcg: HeteroCircuitGraph,
        cur_hcg: HeteroCircuitGraph,
        *,
        timeout_ms: int = 300,
        num_hops: int = 2,
    ) -> GNNAdvice | None:
        """Score every observed ``(port, net)`` edge in cur and return
        a :class:`GNNAdvice`. **Returns None on any failure** so the
        orchestrator can transparently fall back to the rule path.

        Args:
            timeout_ms: best-effort wall-clock budget. **Soft enforcement
                only** — exceeding it logs a warning but still returns the
                result (real cancellation would need threading and isn't
                worth the complexity at this scale; per-call is ~30 ms).
            num_hops: SEAL enclosing subgraph radius (matches the
                training-time setting in
                :func:`build_seal_samples_with_coverage_check`).
        """

        try:
            return self._advise_impl(
                ref_hcg, cur_hcg,
                timeout_ms=timeout_ms,
                num_hops=num_hops,
            )
        except Exception as e:  # noqa: BLE001 — advisory layer, never crash caller
            log.warning(
                "gnn_advisor_failed: %s — falling back to rule path",
                type(e).__name__,
                exc_info=e,
            )
            return None

    def _advise_impl(
        self,
        ref_hcg: HeteroCircuitGraph,
        cur_hcg: HeteroCircuitGraph,
        *,
        timeout_ms: int,
        num_hops: int,
    ) -> GNNAdvice | None:
        import torch
        from torch_geometric.loader import DataLoader  # type: ignore[import-untyped]

        from app.domain.gnn.pyg_converter import seal_subgraph_to_pyg_data
        from app.domain.gnn.seal_subgraph import (
            extract_subgraphs_for_observed_edges,
        )

        t0 = time.time()
        observed = extract_subgraphs_for_observed_edges(
            cur_hcg, num_hops=num_hops
        )
        if not observed:
            log.info("gnn_advise: no observed edges in cur — nothing to score")
            return GNNAdvice(
                model_version=self.model_version,
                inference_ms=(time.time() - t0) * 1000,
                n_edges_scored=0,
            )

        data_list = [
            seal_subgraph_to_pyg_data(sg, cur_hcg) for sg in observed
        ]
        loader = DataLoader(data_list, batch_size=64, shuffle=False)

        all_probs: list[float] = []
        with torch.no_grad():
            for batch in loader:
                logits = self.model(batch.x, batch.edge_index, batch.batch)[
                    "seal_logits"
                ]
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                all_probs.extend(probs)

        # Aggregate
        edge_predictions: list[dict[str, Any]] = []
        port_to_min_p: dict[str, float] = {}
        for sg, prob in zip(observed, all_probs):
            verdict = (
                "likely_wrong" if prob < self.threshold_wrong else "ok"
            )
            edge_predictions.append({
                "edge": [sg.target_port_id, sg.target_net_id],
                "p_correct": float(prob),
                "verdict": verdict,
            })
            prev = port_to_min_p.get(sg.target_port_id, 1.0)
            port_to_min_p[sg.target_port_id] = min(prev, float(prob))

        hotspots = [
            {
                "node": port_id,
                "score": 1.0 - min_p,
                "hint": (
                    "Pin may be wired wrong" if (1.0 - min_p) > 0.6
                    else "Pin connection slightly suspicious"
                ),
            }
            for port_id, min_p in port_to_min_p.items()
        ]
        hotspots.sort(key=lambda h: -float(h["score"]))

        graph_similarity = (
            float(sum(all_probs) / len(all_probs)) if all_probs else 0.0
        )
        # Confidence = 1 - dispersion proxy; a model that's uniformly
        # confident gets high confidence; if predictions are spread out
        # (close to 0.5), confidence drops.
        if all_probs:
            mean_distance_from_decisive = sum(
                abs(p - 0.5) for p in all_probs
            ) / len(all_probs)
            similarity_confidence = float(min(1.0, mean_distance_from_decisive * 2))
        else:
            similarity_confidence = 0.0

        elapsed_ms = (time.time() - t0) * 1000
        if elapsed_ms > timeout_ms:
            log.warning(
                "gnn_advise: elapsed %.1f ms exceeded timeout %d ms "
                "(returning result anyway; soft enforcement)",
                elapsed_ms, timeout_ms,
            )

        return GNNAdvice(
            model_version=self.model_version,
            inference_ms=elapsed_ms,
            n_edges_scored=len(all_probs),
            edge_predictions=tuple(edge_predictions),
            hotspots=tuple(hotspots),
            graph_similarity=graph_similarity,
            graph_similarity_confidence=similarity_confidence,
        )


# ---------------------------------------------------------------------------
# Trigger predicate (plan §七)
# ---------------------------------------------------------------------------


def should_use_gnn(ctx: Any) -> bool:
    """Plan §七 trigger logic — MVP version.

    ``ctx`` is intentionally untyped: orchestrator may pass a dict, a
    NetworkX graph, or a plan-§七 ``CompareContext`` (when that
    eventually lands). We extract whatever we can find via getattr +
    dict.get; missing fields default to neutral.

    MVP triggers (any True → use GNN):
        - non-trivial circuit (≥ 8 total nodes)
        - rule path has fallen to GED or subgraph match
        - circuit has repeated component types (GraphMatcher ambiguity)

    MVP early-exits (any True → skip GNN):
        - safety-critical check pending (rule must decide)
        - polarity violation already detected by rule
        - no checkpoint available on disk

    The full 6-trigger logic from plan §七 needs a real CompareContext;
    that's a P4.1 follow-up.
    """

    if not GNNAdvisor.checkpoint_available():
        return False

    # ctx-agnostic accessors
    def _g(name: str, default: Any = None) -> Any:
        if isinstance(ctx, dict):
            return ctx.get(name, default)
        return getattr(ctx, name, default)

    if _g("has_safety_critical_check_pending"):
        return False
    if _g("deterministic_polarity_violation"):
        return False

    node_count = _g("node_count_total", 0)
    if node_count > 0 and node_count < 8:
        return False

    triggers = (
        bool(_g("full_isomorphism_failed")),
        _g("match_type_so_far") in {
            "current_subgraph_in_reference",
            "equivalent_with_extra",
            "graph_edit_distance_or_fallback",
        },
        bool(_g("has_repeated_component_types")),
        bool(_g("requested_gnn")),  # explicit caller override
    )
    if any(triggers):
        return True

    # Default for the MVP: enable on any non-trivial circuit. Real
    # production gating moves to plan §七 in P4.1.
    return bool(node_count and node_count >= 8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _locate_default_checkpoint() -> Path | None:
    """Find the first existing checkpoint among the env override or the
    bundled candidates."""

    env_path = os.environ.get(_DEFAULT_CKPT_ENV)
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        log.warning(
            "${_DEFAULT_CKPT_ENV}=%s set but file missing; trying defaults",
            env_path,
        )
    for candidate in _DEFAULT_CKPT_CANDIDATES:
        p = Path(candidate)
        if p.is_file():
            return p
    return None


__all__ = [
    "GNNAdvice",
    "GNNAdvisor",
    "should_use_gnn",
]
