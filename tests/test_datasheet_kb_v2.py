from __future__ import annotations

from app.services.datasheet_kb_service import DatasheetKbService


def _kb() -> DatasheetKbService:
    # Default base_dir points to repo's knowledge/datasheets which ships with
    # the Phase 1 hand-written samples. Tests rely on that fixture data.
    return DatasheetKbService()


def test_ne555_pinout_query_returns_text_and_figure_chunks() -> None:
    hits = _kb().search("NE555 引脚 pinout", top_k=4)

    assert hits, "expected at least one chunk for NE555 pinout"
    # The top two hits should be from the NE555 doc (one text, one figure);
    # lower-ranked LM324 'pinout' chunks may follow, that's fine.
    top_two_ids = {hit.document_id for hit in hits[:2]}
    assert top_two_ids == {"ne555"}
    ne555_modalities = {hit.modality for hit in hits if hit.document_id == "ne555"}
    assert "text" in ne555_modalities
    assert "figure" in ne555_modalities


def test_pinout_query_exposes_intent_confidence_and_features() -> None:
    hits = _kb().search("NE555 引脚 pinout", top_k=2)

    assert hits
    top = hits[0]
    assert top.query_intent == "pinout"
    assert top.confidence > 0.0
    assert top.matched_features
    assert any(
        feature in top.matched_features
        for feature in ("intent_section_match", "intent_modality_match", "part_number_match")
    )


def test_lm324_supply_query_surfaces_source_pdf_path() -> None:
    hits = _kb().search("LM324 供电范围 共模", top_k=3)

    assert hits
    top = hits[0]
    assert top.document_id == "lm324"
    assert top.chunk_id.startswith("lm324.")
    assert (top.source_ref or {}).get("source_path", "").endswith(".PDF")
    assert top.query_intent in {"supply", "electrical"}


def test_capacitor_polarity_query_surfaces_safety_text() -> None:
    hits = _kb().search("电解电容 极性 怎么接", top_k=3)

    assert hits
    snippets = " ".join(hit.snippet for hit in hits)
    assert "极性" in snippets
    assert any(hit.chunk_id.startswith("passive.capacitor_polarity.") for hit in hits)


def test_unrelated_query_returns_no_hits() -> None:
    hits = _kb().search("foobarbaz quxquux", top_k=4)
    assert hits == []


def test_part_number_alias_expands_to_doc() -> None:
    hits = _kb().search("555 timer trigger", top_k=3)
    assert any(hit.document_id == "ne555" for hit in hits)


def test_modality_filter_restricts_results() -> None:
    figure_only = _kb().search("NE555 引脚", modalities=["figure"], top_k=4)
    assert figure_only
    assert all(hit.modality == "figure" for hit in figure_only)


def test_package_query_prefers_package_related_chunks() -> None:
    hits = _kb().search("SN74LS74A package outline 封装尺寸", top_k=3)

    assert hits
    top = hits[0]
    assert top.document_id == "sn74ls74a"
    assert top.query_intent == "package"
    assert any(
        token in (top.title + " " + top.snippet).lower()
        for token in ("package", "outline", "materials", "封装")
    )
