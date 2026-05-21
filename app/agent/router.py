"""Lightweight semantic router for tool-routing decisions.

A `SemanticRouter` loads one YAML file per route from ``app/agent/routes/``
and decides which (if any) route a query falls into. The design follows
[aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router)
in spirit, but stays in-process, dependency-free, and gracefully degrades:

- **With embeddings**: positive and negative utterances are encoded once at
  startup. At query time we compute
  ``score = max_cosine(query, positives) - max_cosine(query, negatives)``
  and the route fires when ``score > threshold``. This catches the cases
  Phase 2's keyword check missed (paraphrases, English queries about Chinese
  datasheets, indirect phrasings).
- **Without embeddings** (NullEmbeddingBackend, board with no model): falls
  back to deterministic keyword overlap. ``min_keyword_hits`` of the route's
  ``keywords`` list must appear as substrings in the query.
- **Auto-fire**: ``auto_fire_part_numbers`` short-circuits to True whenever
  the query contains a known part-number string, regardless of cosine /
  keywords. This guarantees that "tell me about NE555" always routes to
  datasheet even if utterance encoding is poor.

The router is *advisory*: it returns a ``RouteDecision``. The caller
(``context_pack.build_context_pack``) decides what to do with it. Today we
use it to gate ``datasheet_lookup_tool``; future routes (concept tutor,
lab guidance, micro-defect inspection) can plug into the same machinery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from app.services.embedding_backend import EmbeddingBackend, NullEmbeddingBackend

logger = logging.getLogger(__name__)

_DEFAULT_ROUTES_DIR = Path(__file__).resolve().parent / "routes"


@dataclass
class RouteDefinition:
    name: str
    description: str = ""
    threshold: float = 0.2
    utterances: list[str] = field(default_factory=list)
    negative_utterances: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    min_keyword_hits: int = 1
    auto_fire_part_numbers: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, payload: dict[str, Any]) -> "RouteDefinition":
        return cls(
            name=str(payload.get("route") or "").strip(),
            description=str(payload.get("description") or ""),
            threshold=float(payload.get("threshold", 0.2)),
            utterances=[str(u) for u in payload.get("utterances", []) or []],
            negative_utterances=[str(u) for u in payload.get("negative_utterances", []) or []],
            keywords=[str(k).lower() for k in payload.get("keywords", []) or []],
            min_keyword_hits=int(payload.get("min_keyword_hits", 1)),
            auto_fire_part_numbers=[
                str(p).lower() for p in payload.get("auto_fire_part_numbers", []) or []
            ],
        )


@dataclass
class RouteDecision:
    name: str
    fired: bool
    score: float
    threshold: float
    matched_via: str  # "auto_fire" | "embedding" | "keyword" | "none"
    keyword_hits: list[str] = field(default_factory=list)
    confidence: float = 0.0
    positive_score: float = 0.0
    negative_score: float = 0.0
    reasons: list[str] = field(default_factory=list)


class SemanticRouter:
    def __init__(
        self,
        routes_dir: str | Path | None = None,
        embedding: EmbeddingBackend | None = None,
    ) -> None:
        self._routes_dir = Path(routes_dir) if routes_dir else _DEFAULT_ROUTES_DIR
        self._embedding = embedding or NullEmbeddingBackend()
        self._routes: dict[str, RouteDefinition] = {}
        # Per-route cached centroids: positive_matrix and negative_matrix are
        # (N, dim) unit-norm float32 arrays. Empty when embeddings are off or
        # the route has no utterances of that polarity.
        self._positive_vecs: dict[str, np.ndarray] = {}
        self._negative_vecs: dict[str, np.ndarray] = {}
        self._load_routes()
        self._maybe_encode()

    def _load_routes(self) -> None:
        if not self._routes_dir.exists():
            return
        for path in sorted(self._routes_dir.glob("*.yaml")):
            try:
                payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                route = RouteDefinition.from_yaml(payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to load route %s: %s", path, exc)
                continue
            if not route.name:
                logger.warning("route file %s missing top-level `route` field", path)
                continue
            self._routes[route.name] = route

    def _maybe_encode(self) -> None:
        if not self._routes:
            return
        if not self._embedding.is_active:
            return
        for name, route in self._routes.items():
            if route.utterances:
                vecs = self._embedding.encode(route.utterances)
                if vecs.size:
                    self._positive_vecs[name] = vecs.astype(np.float32, copy=False)
            if route.negative_utterances:
                vecs = self._embedding.encode(route.negative_utterances)
                if vecs.size:
                    self._negative_vecs[name] = vecs.astype(np.float32, copy=False)
        if self._positive_vecs:
            logger.info(
                "semantic router encoded %d routes (%s)",
                len(self._positive_vecs),
                ",".join(sorted(self._positive_vecs)),
            )

    @property
    def has_routes(self) -> bool:
        return bool(self._routes)

    def has_route(self, name: str) -> bool:
        return name in self._routes

    def decide(self, name: str, query: str) -> RouteDecision:
        route = self._routes.get(name)
        if route is None:
            return RouteDecision(
                name=name,
                fired=False,
                score=0.0,
                threshold=0.0,
                matched_via="none",
                reasons=["route_not_found"],
            )

        q = (query or "").strip()
        ql = q.lower()
        if not q:
            return RouteDecision(
                name=name,
                fired=False,
                score=0.0,
                threshold=route.threshold,
                matched_via="none",
                reasons=["empty_query"],
            )

        # 1. Auto-fire: explicit part-number mention bypasses both routes.
        for part in route.auto_fire_part_numbers:
            if part and part in ql:
                return RouteDecision(
                    name=name,
                    fired=True,
                    score=1.0,
                    threshold=route.threshold,
                    matched_via="auto_fire",
                    keyword_hits=[part],
                    confidence=0.99,
                    reasons=[f"part_number:{part}"],
                )

        # 2. Embedding side, when both encoded utterances and a query encoder
        # are available. We compute pos_max - neg_max; this is the standard
        # semantic-router scoring trick and gives negatives real veto power.
        pos = self._positive_vecs.get(name)
        pos_score = 0.0
        neg_score = 0.0
        if pos is not None and self._embedding.is_active:
            q_vec = self._embedding.encode([q])
            if q_vec.size:
                qv = q_vec[0]
                pos_score = float(np.max(pos @ qv)) if pos.size else 0.0
                neg = self._negative_vecs.get(name)
                neg_score = float(np.max(neg @ qv)) if neg is not None and neg.size else 0.0
                score = pos_score - neg_score
                if score > route.threshold:
                    return RouteDecision(
                        name=name,
                        fired=True,
                        score=score,
                        threshold=route.threshold,
                        matched_via="embedding",
                        confidence=self._confidence_from_embedding(score, route.threshold),
                        positive_score=pos_score,
                        negative_score=neg_score,
                        reasons=[
                            f"embedding_margin={score:.3f}",
                            f"positive={pos_score:.3f}",
                            f"negative={neg_score:.3f}",
                        ],
                    )
                # Embedding said no; fall through to keyword check as a safety
                # net for OOV phrasings. Keyword matches are conservative.

        # 3. Keyword fallback. Used both when embeddings are off and as a
        # safety net when the cosine margin is below threshold.
        hits = [kw for kw in route.keywords if kw in ql]
        if len(hits) >= max(1, route.min_keyword_hits):
            return RouteDecision(
                name=name,
                fired=True,
                score=float(len(hits)),
                threshold=float(route.min_keyword_hits),
                matched_via="keyword",
                keyword_hits=hits,
                confidence=self._confidence_from_keywords(len(hits), route.min_keyword_hits),
                positive_score=pos_score,
                negative_score=neg_score,
                reasons=[f"keyword:{hit}" for hit in hits[:6]],
            )

        return RouteDecision(
            name=name,
            fired=False,
            score=0.0,
            threshold=route.threshold,
            matched_via="none",
            confidence=0.0,
            positive_score=pos_score,
            negative_score=neg_score,
            reasons=(
                [f"embedding_margin_below_threshold={pos_score - neg_score:.3f}"]
                if pos is not None and self._embedding.is_active
                else ["no_matching_signal"]
            ),
        )

    def decide_all(self, query: str) -> list[RouteDecision]:
        decisions = [self.decide(name, query) for name in sorted(self._routes)]
        return sorted(
            decisions,
            key=lambda item: (item.fired, item.confidence, item.score),
            reverse=True,
        )

    @staticmethod
    def _confidence_from_embedding(score: float, threshold: float) -> float:
        margin = max(0.0, score - threshold)
        return round(min(0.99, 0.55 + margin * 0.8), 4)

    @staticmethod
    def _confidence_from_keywords(hit_count: int, min_hits: int) -> float:
        extra = max(0, hit_count - max(1, min_hits))
        return round(min(0.92, 0.55 + extra * 0.08), 4)


_ROUTER_SINGLETON: SemanticRouter | None = None


def get_router() -> SemanticRouter:
    """Lazy singleton. Reuses the same embedding backend as DatasheetKbService
    so the model is loaded once per process. Tests that need a custom router
    should construct ``SemanticRouter`` directly with a test-only backend."""

    global _ROUTER_SINGLETON
    if _ROUTER_SINGLETON is None:
        from app.core.config import settings
        from app.services.embedding_backend import create_embedding_backend

        backend = create_embedding_backend(
            kind=getattr(settings, "DATASHEET_EMBEDDING_BACKEND", "null"),
            model_dir=getattr(settings, "DATASHEET_EMBEDDING_MODEL_DIR", None),
            device=getattr(settings, "DATASHEET_EMBEDDING_DEVICE", "CPU"),
            max_length=getattr(settings, "DATASHEET_EMBEDDING_MAX_LEN", 256),
        )
        _ROUTER_SINGLETON = SemanticRouter(embedding=backend)
    return _ROUTER_SINGLETON
