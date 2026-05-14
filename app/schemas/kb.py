from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.angnt import AngntCitation, AngntEvidence

DatasheetChunkModality = Literal["text", "table", "figure", "schematic", "waveform"]


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


class DatasheetChunk(BaseModel):
    """Single multimodal evidence unit produced by build-time PDF parsing."""

    chunk_id: str
    modality: DatasheetChunkModality = "text"
    title: str = ""
    text: str | None = None
    page: int | None = None
    section: str | None = None
    keywords: list[str] = Field(default_factory=list)
    asset_path: str | None = None
    table_html: str | None = None
    bbox: list[float] | None = None
    source_ref: dict[str, Any] = Field(default_factory=dict)


class DatasheetDocument(BaseModel):
    """A parsed datasheet: stable identifier + ordered chunks."""

    document_id: str
    title: str = ""
    part_numbers: list[str] = Field(default_factory=list)
    source_path: str | None = None
    chunks: list[DatasheetChunk] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """Uniform retrieval result for both DatasheetKbService and legacy KbService."""

    chunk_id: str
    modality: DatasheetChunkModality = "text"
    title: str = ""
    snippet: str = ""
    score: float = 0.0
    document_id: str = ""
    page: int | None = None
    asset_path: str | None = None
    table_html: str | None = None
    source_ref: dict[str, Any] = Field(default_factory=dict)

