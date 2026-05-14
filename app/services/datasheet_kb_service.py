from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

from app.schemas.kb import (
    DatasheetChunk,
    DatasheetChunkModality,
    DatasheetDocument,
    RetrievedChunk,
)
from app.services.embedding_backend import EmbeddingBackend, NullEmbeddingBackend

logger = logging.getLogger(__name__)


_DEFAULT_BASE_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "datasheets"

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
    ) -> None:
        self._base_dir = Path(base_dir) if base_dir else _DEFAULT_BASE_DIR
        self._embedding = embedding or NullEmbeddingBackend()
        self._documents: dict[str, DatasheetDocument] = {}
        self._chunk_index: dict[str, tuple[DatasheetDocument, DatasheetChunk]] = {}
        self._load()

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

    @property
    def is_empty(self) -> bool:
        return not self._documents

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
                score = self._score_chunk(chunk, query_tokens, part_boost) * doc_score_scale
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
