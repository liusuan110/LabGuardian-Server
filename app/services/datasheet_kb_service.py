from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from app.schemas.kb import (
    DatasheetChunk,
    DatasheetChunkModality,
    DatasheetDocument,
    RetrievedChunk,
)
from app.services.embedding_backend import EmbeddingBackend, NullEmbeddingBackend

logger = logging.getLogger(__name__)


_DEFAULT_BASE_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "datasheets"
_DEFAULT_EMBED_DIR = _DEFAULT_BASE_DIR / "embeddings"

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[一-鿿]")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_PATTERN.findall(text)]


def _is_cjk(ch: str) -> bool:
    return bool(ch) and "一" <= ch[0] <= "鿿"


def _strong_tokens(tokens: set[str]) -> set[str]:
    """Drop tokens too weak to act as a part-number signal.

    A single CJK character (e.g. "器") routinely appears across unrelated
    queries and would otherwise let an unrelated chip's alias capture the
    query. Multi-char CJK and any alphanumeric token survive.
    """
    return {tok for tok in tokens if len(tok) > 1 or not _is_cjk(tok)}


class DatasheetKbService:
    """Local, offline multimodal datasheet retrieval.

    Reads pre-parsed DatasheetDocument JSON files from ``knowledge/datasheets/``.
    Scoring is deterministic (keyword + part-number + modality filter). An
    optional EmbeddingBackend can be wired in later (Phase 3) for hybrid scoring;
    Phase 1 ships with NullEmbeddingBackend, so retrieval is pure keyword.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        embedding: EmbeddingBackend | None = None,
        *,
        embeddings_dir: str | Path | None = None,
        fusion_weight: float = 0.55,
    ) -> None:
        self._base_dir = Path(base_dir) if base_dir else _DEFAULT_BASE_DIR
        self._embedding = embedding or NullEmbeddingBackend()
        self._embeddings_dir = (
            Path(embeddings_dir) if embeddings_dir else _DEFAULT_EMBED_DIR
        )
        self._fusion_weight = max(0.0, min(1.0, fusion_weight))
        self._documents: dict[str, DatasheetDocument] = {}
        self._chunk_index: dict[str, tuple[DatasheetDocument, DatasheetChunk]] = {}
        # chunk_id -> 1-D unit-norm vector loaded from the offline .npz cache.
        # Empty when no cache exists or backend is Null; hybrid scoring skips
        # cosine in that case and behaves identically to Phase 1.
        self._chunk_vectors: dict[str, np.ndarray] = {}
        self._load()
        self._load_chunk_embeddings()

    def _load(self) -> None:
        if not self._base_dir.exists():
            logger.info("datasheet kb base dir missing: %s", self._base_dir)
            return
        for path in sorted(self._base_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                document = DatasheetDocument.model_validate(payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to load datasheet %s: %s", path, exc)
                continue
            self._documents[document.document_id] = document
            for chunk in document.chunks:
                self._chunk_index[chunk.chunk_id] = (document, chunk)

    def _load_chunk_embeddings(self) -> None:
        if not self._embeddings_dir.exists():
            return
        for npz_path in sorted(self._embeddings_dir.glob("*.npz")):
            try:
                payload = np.load(npz_path, allow_pickle=False)
                ids = payload["chunk_ids"]
                vectors = payload["vectors"].astype(np.float32, copy=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to load embeddings %s: %s", npz_path, exc)
                continue
            if vectors.shape[0] != len(ids):
                logger.warning(
                    "embeddings %s mismatched: %s ids vs %s vectors",
                    npz_path,
                    len(ids),
                    vectors.shape[0],
                )
                continue
            for chunk_id, vector in zip(ids, vectors, strict=True):
                cid = str(chunk_id)
                if cid in self._chunk_index:
                    self._chunk_vectors[cid] = vector
        if self._chunk_vectors:
            logger.info(
                "loaded %d datasheet chunk embeddings from %s",
                len(self._chunk_vectors),
                self._embeddings_dir,
            )

    @property
    def is_empty(self) -> bool:
        return not self._documents

    @property
    def has_embeddings(self) -> bool:
        """True when both an active backend and at least one cached vector exist.

        ``DatasheetKbService.search`` uses this to decide whether to encode
        the query and fuse cosine into the score. Either piece missing → pure
        keyword path (Phase 1 behavior).
        """
        return bool(self._chunk_vectors) and self._embedding.is_active

    def list_documents(self) -> list[DatasheetDocument]:
        return list(self._documents.values())

    def get_document(self, document_id: str) -> DatasheetDocument | None:
        return self._documents.get(document_id)

    def search(
        self,
        query: str,
        *,
        part_numbers: Iterable[str] | None = None,
        modalities: Iterable[DatasheetChunkModality] | None = None,
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        if not self._documents:
            return []

        query_tokens = set(_tokenize(query))
        explicit_part_terms: set[str] = set()
        for raw in part_numbers or []:
            explicit_part_terms.update(_tokenize(raw))

        modality_filter = {m for m in (modalities or [])}

        # Part-signal detection is per-document: a doc gets the "you meant me"
        # boost when its own part_numbers / document_id overlap query or
        # explicit part_numbers. We strip single CJK characters before the
        # comparison so a token like "器" (from 触发器) cannot accidentally
        # bind to "定时器" in NE555's alias list.
        query_strong = _strong_tokens(query_tokens)
        explicit_strong = _strong_tokens(explicit_part_terms)
        signal_pool = query_strong | explicit_strong
        # `has_part_signal` reflects whether ANY document in the corpus was
        # part-matched; only then do we down-rank non-matching docs.
        per_doc_part_match: dict[str, bool] = {}
        has_part_signal = False
        for document in self._documents.values():
            doc_part_tokens = {
                tok for pn in document.part_numbers for tok in _tokenize(pn)
            }
            doc_part_tokens.update(_tokenize(document.document_id))
            matched = bool(_strong_tokens(doc_part_tokens) & signal_pool)
            per_doc_part_match[document.document_id] = matched
            if matched:
                has_part_signal = True

        # Optional semantic side: encode the query once when both an active
        # backend and a non-empty .npz cache exist. Cosine is computed only
        # against chunks we'll actually score (lazy), so the cost is O(1) per
        # candidate chunk lookup. The board never re-encodes the corpus.
        query_vec: np.ndarray | None = None
        if self.has_embeddings:
            encoded = self._embedding.encode([query])
            if encoded.size:
                query_vec = encoded[0]

        results: list[tuple[float, DatasheetDocument, DatasheetChunk]] = []
        for document in self._documents.values():
            doc_matches_part = per_doc_part_match[document.document_id]
            part_boost = 1.5 if doc_matches_part else 0.0
            doc_score_scale = 1.0
            if has_part_signal and not doc_matches_part:
                doc_score_scale = 0.35

            for chunk in document.chunks:
                if modality_filter and chunk.modality not in modality_filter:
                    continue
                keyword_score = (
                    self._score_chunk(chunk, query_tokens, part_boost) * doc_score_scale
                )
                cosine = self._cosine_against(chunk.chunk_id, query_vec)
                score = self._fuse(keyword_score, cosine, doc_score_scale)
                if score <= 0:
                    continue
                results.append((score, document, chunk))

        results.sort(key=lambda triple: triple[0], reverse=True)
        return [
            self._to_retrieved(doc, chunk, score)
            for score, doc, chunk in results[: max(1, top_k)]
        ]

    def _score_chunk(
        self,
        chunk: DatasheetChunk,
        query_tokens: set[str],
        part_boost: float,
    ) -> float:
        if not query_tokens and part_boost <= 0:
            return 0.0

        keyword_tokens = {tok for kw in chunk.keywords for tok in _tokenize(kw)}
        title_tokens = set(_tokenize(chunk.title))
        text_tokens = set(_tokenize(chunk.text or ""))
        section_tokens = set(_tokenize(chunk.section or ""))

        score = 0.0
        if query_tokens:
            score += 2.0 * len(query_tokens & keyword_tokens)
            score += 1.2 * len(query_tokens & title_tokens)
            score += 0.4 * len(query_tokens & text_tokens)
            score += 0.6 * len(query_tokens & section_tokens)

        if part_boost:
            # Part-number hits still need *some* topical signal; if the query
            # only carries the chip name we accept a small floor so the top
            # chunk of that chip surfaces.
            score += part_boost
            if score < part_boost:
                score = part_boost

        return score

    def _cosine_against(
        self,
        chunk_id: str,
        query_vec: np.ndarray | None,
    ) -> float | None:
        if query_vec is None:
            return None
        chunk_vec = self._chunk_vectors.get(chunk_id)
        if chunk_vec is None:
            return None
        # Vectors are pre-normalized at build time and at encode time, so cosine
        # is just the dot product. Negative values are clipped to 0 — they
        # carry no useful signal for re-ranking.
        return float(max(0.0, np.dot(chunk_vec, query_vec)))

    def _fuse(
        self,
        keyword_score: float,
        cosine: float | None,
        doc_score_scale: float,
    ) -> float:
        """Combine keyword and cosine signals.

        Keyword is unbounded (0..~10), cosine is in [0, 1] after the dot
        product. We squash keyword to (0, 1) via a saturating curve so the
        two terms live on the same scale, then weighted-sum.
        """
        if cosine is None or self._fusion_weight <= 0.0:
            return keyword_score
        keyword_normalized = keyword_score / (1.0 + keyword_score)
        w = self._fusion_weight
        fused = (1.0 - w) * keyword_normalized + w * cosine
        # Apply the same off-doc dampening to the semantic side so a wrong-doc
        # high cosine can't overpower a right-doc keyword hit.
        return fused * doc_score_scale

    def _to_retrieved(
        self,
        document: DatasheetDocument,
        chunk: DatasheetChunk,
        score: float,
    ) -> RetrievedChunk:
        snippet = (chunk.text or chunk.title or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            modality=chunk.modality,
            title=chunk.title,
            snippet=snippet,
            score=round(score, 4),
            document_id=document.document_id,
            page=chunk.page,
            asset_path=chunk.asset_path,
            table_html=chunk.table_html,
            source_ref={
                **(chunk.source_ref or {}),
                "document_id": document.document_id,
                "source_path": document.source_path,
            },
        )
