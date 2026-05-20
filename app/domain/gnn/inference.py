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
from importlib import import_module
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
        - ``suggested_targets``: per "suspicious" or REQUIRED-floating port,
          the top-K candidate cur nets to wire to (MISSING_EDGE head reused
          on-the-fly — same ``SealDGCNN`` head, ``edge_present=False``).
          Drives the "which wire and where should it go" UX layer.
        - ``n_suggestion_candidates_scored``: total candidate edges run
          through the model for ``suggested_targets`` (for observability /
          performance budgeting).

    Reserved for P4.1 (heads not yet implemented):
        - ``predicted_error_types``, ``component_mapping_topk``,
          ``net_mapping_topk``, ``progress_score``
    """

    model_version: str
    inference_ms: float
    n_edges_scored: int
    edge_predictions: tuple[dict[str, Any], ...] = ()
    hotspots: tuple[dict[str, Any], ...] = ()
    graph_similarity: float = 0.0
    graph_similarity_confidence: float = 0.0
    suggested_targets: tuple[dict[str, Any], ...] = ()
    n_suggestion_candidates_scored: int = 0
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
            "n_suggestion_candidates_scored": self.n_suggestion_candidates_scored,
            "inference_ms": self.inference_ms,
            "edge_predictions": list(self.edge_predictions),
            "hotspots": filtered_hotspots,
            "suggested_targets": list(self.suggested_targets),
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
    # **Phase C Stage 5 (2026-05-20)** — retrained with
    # `insert_same_net_wire` perturbation + WIRE_SAME_NET_POSITIVE
    # label class to fix the wire OOD identified on real student data
    # (tests/fixtures/real_student/inverting_amp_correct_v1.json).
    # Metrics: val F1 0.967 / val top-3 1.000 / test F1 0.989 / test
    # top-3 1.000 (held-out opamp_buffer). Golden sample: 16/17 edges
    # p > 0.9 (vs v3's 7/17 fail). See RISK_REGISTER Phase C for the
    # one remaining outlier (W3.pin1 SEAL subgraph asymmetry).
    "checkpoints/p3_followup_v4/best_f1.pt",
    # R10 (2026-05-18) — retrained against updated opamp_inverting
    # fixture that adds R_p (textbook bias compensation resistor).
    # val F1 0.959 / test F1 0.994. See RISK_REGISTER §5 R10.
    "checkpoints/p3_followup_v3/best_f1.pt",
    "checkpoints/p3_followup_v2/best_f1.pt",
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
        suggestion_top_k: int = 3,
        max_suggestion_candidates: int = 256,
    ):
        self.model = model
        self.model_version = model_version
        self.threshold_wrong = threshold_wrong
        # K passed to the suggested_target ranking; default 3 mirrors
        # plan §九 MISSING_EDGE top-3 reporting.
        self.suggestion_top_k = suggestion_top_k
        # Hard cap on per-call (port × candidate_net) evaluations so a
        # pathologically large circuit can't blow the 100 ms budget. We
        # drop the lowest-priority ports first (`likely_wrong` over
        # floating, then by min P(edge_correct)) when over budget.
        self.max_suggestion_candidates = max_suggestion_candidates
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
        _ensure_gnn_runtime()
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
    def runtime_available(cls) -> bool:
        """Return True only when torch + PyG can both be imported."""

        try:
            _ensure_gnn_runtime()
        except RuntimeError:
            return False
        return True

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

        DataLoader = getattr(import_module("torch_geometric.loader"), "DataLoader")

        from app.domain.gnn.pyg_converter import seal_subgraph_to_pyg_data
        from app.domain.gnn.seal_subgraph import (
            extract_seal_subgraph,
            extract_subgraphs_for_observed_edges,
        )

        t0 = time.time()
        observed = extract_subgraphs_for_observed_edges(
            cur_hcg, num_hops=num_hops
        )
        # A port may legitimately have multiple observed edges (e.g.
        # UA741 pin2 + pin6 both wired to VOUT). Track *all* of them so
        # the suggested-target step can exclude the currently-wired nets
        # from the candidate set.
        port_observed_nets: dict[str, set[str]] = {}
        for sg in observed:
            port_observed_nets.setdefault(sg.target_port_id, set()).add(
                sg.target_net_id
            )

        all_probs: list[float] = []
        edge_predictions: list[dict[str, Any]] = []
        port_to_min_p: dict[str, float] = {}

        if observed:
            data_list = [
                seal_subgraph_to_pyg_data(sg, cur_hcg) for sg in observed
            ]
            loader = DataLoader(data_list, batch_size=64, shuffle=False)
            with torch.no_grad():
                for batch in loader:
                    logits = self.model(batch.x, batch.edge_index, batch.batch)[
                        "seal_logits"
                    ]
                    probs = torch.sigmoid(logits).cpu().numpy().tolist()
                    all_probs.extend(probs)
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
        else:
            log.info("gnn_advise: no observed edges in cur")

        # ----- D + A · suggested_targets ----------------------------------
        suggested_targets, n_candidates_scored = self._compute_suggested_targets(
            cur_hcg,
            port_to_min_p=port_to_min_p,
            port_observed_nets=port_observed_nets,
            num_hops=num_hops,
            extract_seal_subgraph_fn=extract_seal_subgraph,
            seal_subgraph_to_pyg_data_fn=seal_subgraph_to_pyg_data,
            DataLoader=DataLoader,
            torch=torch,
        )

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
            suggested_targets=tuple(suggested_targets),
            n_suggestion_candidates_scored=n_candidates_scored,
        )

    # ------------------------------------------------------------------
    # Suggested-target ranking (plan §六 "应该接哪里")
    # ------------------------------------------------------------------

    def _compute_suggested_targets(
        self,
        cur_hcg: HeteroCircuitGraph,
        *,
        port_to_min_p: dict[str, float],
        port_observed_nets: dict[str, set[str]],
        num_hops: int,
        extract_seal_subgraph_fn: Any,
        seal_subgraph_to_pyg_data_fn: Any,
        DataLoader: Any,
        torch: Any,
    ) -> tuple[list[dict[str, Any]], int]:
        """Reuse the SEAL head as a MISSING_EDGE ranker.

        Two port populations enter the candidate pool:

        1. **likely_wrong observed ports** — at least one observed edge
           scored below ``self.threshold_wrong``. We exclude the
           currently-wired nets from the candidates (we already know
           those exist and have a low score; "where else" is the
           actionable question).
        2. **REQUIRED floating ports** — ``is_floating=True`` +
           ``connection_policy == "required"``. Same MISSING_EDGE
           training distribution; ``edge_present=False`` matches.

        For each `(port, candidate_net)` pair we build a SEAL subgraph
        and batch-score with the same head. Returns the top-K candidates
        per port, sorted descending by ``p_connect``.

        Returns (suggested_targets, n_candidates_scored). Empty list +
        zero when no port qualifies or there are no candidate nets.
        """

        cur_net_ids = list(cur_hcg.nets.keys())
        if not cur_net_ids:
            return [], 0

        # Build the port population. Each entry carries a priority used
        # to decide what to drop when over ``max_suggestion_candidates``.
        # Lower priority value = drop sooner. We keep floating-required
        # ahead of likely-wrong because the "you forgot a wire" hint is
        # more actionable than ranking alternates for an already-present
        # connection.
        port_entries: list[tuple[float, str, str]] = []  # (priority, port_id, reason)
        for port_id, min_p in port_to_min_p.items():
            if min_p < self.threshold_wrong:
                # priority lower than floating's 2.0, ordered by p
                # (the lower p, the more we want to keep)
                port_entries.append((1.0 - min_p, port_id, "likely_wrong"))
        for port in cur_hcg.ports.values():
            if not port.is_floating:
                continue
            if port.connection_policy != "required":
                continue
            # Floating REQUIRED pins outrank wrong-but-present pins.
            port_entries.append((2.0, port.node_id, "floating_required"))
        if not port_entries:
            return [], 0

        port_entries.sort(key=lambda t: -t[0])

        # Build (port, candidate_net) plan with the budget cap. Each
        # port consumes up to ``len(cur_net_ids) - |observed nets|``
        # candidates.
        plan: list[tuple[str, str, str]] = []  # (port_id, net_id, reason)
        budget = self.max_suggestion_candidates
        dropped_ports: list[str] = []
        for _, port_id, reason in port_entries:
            if port_id not in cur_hcg.ports:
                continue
            excluded_nets = port_observed_nets.get(port_id, set())
            port_candidates = [n for n in cur_net_ids if n not in excluded_nets]
            if not port_candidates:
                continue
            if budget <= 0:
                dropped_ports.append(port_id)
                continue
            if len(port_candidates) > budget:
                # Keep budget intact across ports — drop overflow on
                # this port rather than starving later ones.
                port_candidates = port_candidates[:budget]
            budget -= len(port_candidates)
            for net_id in port_candidates:
                plan.append((port_id, net_id, reason))

        if dropped_ports:
            log.warning(
                "gnn_advise: suggested-target candidate budget "
                "(%d) exhausted; dropped %d ports (%s)",
                self.max_suggestion_candidates,
                len(dropped_ports),
                ", ".join(dropped_ports[:3]),
            )

        if not plan:
            return [], 0

        subgraphs = [
            extract_seal_subgraph_fn(
                cur_hcg,
                port_id,
                net_id,
                num_hops=num_hops,
                edge_present=False,
            )
            for port_id, net_id, _ in plan
        ]
        data_list = [seal_subgraph_to_pyg_data_fn(sg, cur_hcg) for sg in subgraphs]
        loader = DataLoader(data_list, batch_size=64, shuffle=False)

        cand_probs: list[float] = []
        with torch.no_grad():
            for batch in loader:
                logits = self.model(batch.x, batch.edge_index, batch.batch)[
                    "seal_logits"
                ]
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                cand_probs.extend(probs)

        # Aggregate per port → top-K
        by_port: dict[str, list[tuple[str, float, str]]] = {}
        for (port_id, net_id, reason), prob in zip(plan, cand_probs):
            by_port.setdefault(port_id, []).append(
                (net_id, float(prob), reason)
            )

        # Preserve the priority ordering established above so the
        # consumer sees floating_required pins first.
        priority_order = {pid: idx for idx, (_, pid, _) in enumerate(port_entries)}
        port_ids_sorted = sorted(by_port.keys(), key=lambda p: priority_order.get(p, 1 << 30))

        out: list[dict[str, Any]] = []
        for port_id in port_ids_sorted:
            candidates = by_port[port_id]
            candidates.sort(key=lambda x: -x[1])
            topk = candidates[: self.suggestion_top_k]
            reason = topk[0][2]
            observed_for_port = sorted(port_observed_nets.get(port_id, set()))
            out.append({
                "port": port_id,
                "reason": reason,
                "current_nets": observed_for_port,
                "top_p_connect": float(topk[0][1]),
                "candidates": [
                    {
                        "net": net,
                        "p_connect": float(p),
                        "rank": rank + 1,
                    }
                    for rank, (net, p, _) in enumerate(topk)
                ],
            })

        return out, len(plan)


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

    if not GNNAdvisor.runtime_available():
        return False
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


def _ensure_gnn_runtime() -> None:
    """Raise RuntimeError when the optional GNN runtime is unavailable."""

    try:
        import torch  # noqa: F401
        import torch_geometric  # type: ignore[import-not-found,import-untyped]  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "GNNAdvisor requires the [gnn] extra (torch + "
            "torch_geometric). Install with: pip install -e '.[gnn]'"
        ) from e


__all__ = [
    "GNNAdvice",
    "GNNAdvisor",
    "should_use_gnn",
]
