from __future__ import annotations

import time

from pydantic import BaseModel, Field

from app.schemas.angnt import AngntCitation, AngntEvidence


class KbDocumentInfo(BaseModel):
    doc_id: str
    filename: str
    sha256: str
    page_count: int = 0
    chunk_count: int = 0
    created_at: float = Field(default_factory=time.time)


class KbStatusResponse(BaseModel):
    storage_dir: str
    collection: str
    doc_count: int
    chunk_count: int


class KbQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=6, ge=1, le=20)
    chip_hint: str | None = None


class KbQueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[AngntCitation] = Field(default_factory=list)
    evidence: list[AngntEvidence] = Field(default_factory=list)
    used_retrieval: bool = False

