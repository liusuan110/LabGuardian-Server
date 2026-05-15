"""Phase 4 — semantic router behavior.

Mix of three test paths to cover the router's degradation chain:

1. ``NullEmbeddingBackend`` (default board posture): only the keyword
   fallback should fire. Negative utterances and embeddings have no effect.
2. A deterministic in-test ``_FakeEmbedding`` so we can validate the
   positive/negative cosine scoring path without needing the real bge model.
3. Auto-fire short-circuit on explicit part numbers regardless of either.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from app.agent.router import RouteDefinition, SemanticRouter
from app.services.embedding_backend import EmbeddingBackend, NullEmbeddingBackend


class _FakeEmbedding(EmbeddingBackend):
    """Hash-bucket fake. Different lexicons produce mostly-orthogonal vectors,
    same-token texts produce highly-cosine vectors. Good enough to validate
    the cosine routing logic deterministically."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def is_active(self) -> bool:
        return True

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [t for t in text.lower().split() if t]
            if not tokens:
                out[row] = 1.0 / np.sqrt(self._dim)
                continue
            for tok in tokens:
                bucket = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self._dim
                out[row, bucket] += 1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


def test_router_loads_datasheet_route_from_yaml() -> None:
    router = SemanticRouter(embedding=NullEmbeddingBackend())
    assert router.has_route("datasheet")


def test_keyword_fallback_fires_on_pinout_query() -> None:
    router = SemanticRouter(embedding=NullEmbeddingBackend())
    decision = router.decide("datasheet", "请把这颗芯片的 pinout 给我看一下")
    assert decision.fired
    assert decision.matched_via == "keyword"
    assert "pinout" in decision.keyword_hits


def test_keyword_fallback_silent_on_wiring_query() -> None:
    """The old keyword check returned True on almost anything; the new
    YAML-driven keyword list is tighter — a wiring question with no
    datasheet vocabulary must NOT fire."""

    router = SemanticRouter(embedding=NullEmbeddingBackend())
    decision = router.decide("datasheet", "我电路里这根线接哪个孔位才对")
    assert not decision.fired
    assert decision.matched_via == "none"


def test_auto_fire_part_number_bypasses_other_checks() -> None:
    """Even with no datasheet keywords, mentioning a known part_number must
    route to datasheet — preserves the "tell me about NE555" intent."""

    router = SemanticRouter(embedding=NullEmbeddingBackend())
    decision = router.decide("datasheet", "随便讲讲 ne555 这颗芯片")
    assert decision.fired
    assert decision.matched_via == "auto_fire"
    assert decision.keyword_hits == ["ne555"]


def test_embedding_path_fires_on_paraphrased_datasheet_query(tmp_path) -> None:
    """Phrasing that shares NO surface keywords with the YAML keyword list
    should still route via the embedding side when a backend is active."""

    routes_dir = tmp_path / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    (routes_dir / "datasheet.yaml").write_text(
        """
route: datasheet
description: test
threshold: 0.05
utterances:
  - chip pinout reference voltage power supply
  - datasheet electrical characteristics absolute maximum
negative_utterances:
  - wire connection circuit node
keywords: []
min_keyword_hits: 1
""",
        encoding="utf-8",
    )
    router = SemanticRouter(routes_dir=routes_dir, embedding=_FakeEmbedding())
    decision = router.decide(
        "datasheet", "chip pinout reference voltage power supply"
    )
    assert decision.fired
    assert decision.matched_via == "embedding"


def test_embedding_negative_utterance_vetoes_wiring_query(tmp_path) -> None:
    """A query that lexically matches negative utterances must score below
    threshold even if it tangentially overlaps positive utterances."""

    routes_dir = tmp_path / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    (routes_dir / "datasheet.yaml").write_text(
        """
route: datasheet
description: test
threshold: 0.2
utterances:
  - chip pinout voltage
negative_utterances:
  - wire connection node hole
keywords: []
min_keyword_hits: 1
""",
        encoding="utf-8",
    )
    router = SemanticRouter(routes_dir=routes_dir, embedding=_FakeEmbedding())
    decision = router.decide("datasheet", "wire connection node hole")
    assert not decision.fired


def test_decide_returns_unfired_for_unknown_route() -> None:
    router = SemanticRouter(embedding=NullEmbeddingBackend())
    decision = router.decide("nonexistent_route", "anything")
    assert not decision.fired
    assert decision.matched_via == "none"
