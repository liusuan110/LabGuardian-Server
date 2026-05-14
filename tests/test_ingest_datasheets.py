from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from app.schemas.kb import DatasheetChunk, DatasheetDocument

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_ingest_module():
    spec = importlib.util.spec_from_file_location(
        "ingest_datasheets",
        REPO_ROOT / "scripts" / "ingest_datasheets.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ingest():
    return _load_ingest_module()


def test_pypdf_backend_is_available_in_runtime(ingest) -> None:
    backend = ingest.PypdfBackend()
    assert backend.is_available(), "pypdf is a runtime dep; backend must be ready"


def test_resolve_backend_auto_picks_available(ingest) -> None:
    backend = ingest.resolve_backend("auto")
    assert backend.name in {"pypdf", "mineru"}


def test_pypdf_backend_produces_real_chunks_from_ne555_pdf(ingest, tmp_path) -> None:
    pdf = REPO_ROOT / "knowledge_base" / "C695838_555定时器-计时器_NE555DR_规格书_WJ1799212.PDF"
    if not pdf.exists():
        pytest.skip("NE555 PDF fixture missing")
    backend = ingest.PypdfBackend()

    chunks = backend.parse(pdf, "ne555_smoke", tmp_path / "assets")

    assert chunks, "pypdf should extract at least one chunk"
    assert all(isinstance(c, DatasheetChunk) for c in chunks)
    assert all(c.modality == "text" for c in chunks)
    assert all(c.chunk_id.startswith("ne555_smoke.") for c in chunks)
    assert any(c.page == 1 for c in chunks)
    # Real-text spot check: the first page mentions timing / oscillation.
    page1 = next(c for c in chunks if c.page == 1)
    assert "555" in (page1.text or "")


def test_merge_preserves_non_text_chunks(ingest) -> None:
    existing = DatasheetDocument(
        document_id="demo",
        title="Demo",
        part_numbers=["DEMO"],
        source_path="knowledge_base/demo.pdf",
        chunks=[
            DatasheetChunk(chunk_id="demo.fig.1", modality="figure", title="fig"),
            DatasheetChunk(chunk_id="demo.tbl.1", modality="table", title="tbl"),
            DatasheetChunk(chunk_id="demo.old.text", modality="text", title="old text"),
        ],
    )
    new_chunks = [
        DatasheetChunk(chunk_id="demo.p001.text", modality="text", title="page 1"),
    ]

    merged = ingest.merge_document(
        existing,
        document_id="demo",
        title="",
        part_numbers=[],
        source_path="",
        new_chunks=new_chunks,
        overwrite=False,
    )

    chunk_ids = {c.chunk_id for c in merged.chunks}
    assert "demo.fig.1" in chunk_ids  # preserved figure
    assert "demo.tbl.1" in chunk_ids  # preserved table
    assert "demo.old.text" not in chunk_ids  # old text replaced
    assert "demo.p001.text" in chunk_ids  # new text in
    # Document-level fields fall back to existing values when CLI args empty.
    assert merged.title == "Demo"
    assert merged.part_numbers == ["DEMO"]
    assert merged.source_path == "knowledge_base/demo.pdf"


def test_merge_overwrite_drops_existing_chunks(ingest) -> None:
    existing = DatasheetDocument(
        document_id="demo",
        chunks=[DatasheetChunk(chunk_id="demo.fig.1", modality="figure")],
    )
    merged = ingest.merge_document(
        existing,
        document_id="demo",
        title="Demo",
        part_numbers=["DEMO"],
        source_path="x",
        new_chunks=[DatasheetChunk(chunk_id="demo.p001.text", modality="text")],
        overwrite=True,
    )
    assert {c.chunk_id for c in merged.chunks} == {"demo.p001.text"}


def test_ingested_jsons_load_into_kb_service() -> None:
    """End-to-end: the produced JSON files round-trip through the runtime KB."""

    from app.services.datasheet_kb_service import DatasheetKbService

    kb = DatasheetKbService()
    doc_ids = {d.document_id for d in kb.list_documents()}
    # Phase 2 ingested all three /knowledge_base/ PDFs.
    assert {"ne555", "lm324", "sn74ls74a"}.issubset(doc_ids)

    hits = kb.search("SN74LS74A 触发器 D flip-flop", top_k=3)
    assert any(h.document_id == "sn74ls74a" for h in hits)


def test_existing_json_load_helper_returns_validated_model(ingest) -> None:
    path = REPO_ROOT / "knowledge/datasheets/ne555.json"
    if not path.exists():
        pytest.skip("ne555.json not yet ingested")
    document = ingest.load_existing(path)
    assert document is not None
    assert document.document_id == "ne555"
    # Phase 1 hand-authored figure+table chunks were preserved by the merge.
    modalities = {c.modality for c in document.chunks}
    assert {"text", "figure", "table"}.issubset(modalities)
    # Sanity check that the round-trip JSON is valid against the schema.
    json.loads(path.read_text(encoding="utf-8"))
