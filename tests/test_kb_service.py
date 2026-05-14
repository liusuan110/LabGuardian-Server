from app.services.kb_service import KbService


def _hit(title: str, text: str, *, filename: str = "TI-LM324.pdf", page: int = 0):
    return (
        {
            "title": title,
            "score": 0.0,
            "metadata": {"filename": filename, "page": page},
            "snippet": text[:120],
            "text": text,
        },
        filename,
    )


def test_chip_hints_understand_common_chinese_aliases() -> None:
    service = KbService()

    assert service._chip_hints_from_query("555定时器 reset 引脚") == ["ne555"]
    assert service._chip_hints_from_query("运算放大器供电电压") == ["lm324"]
    assert service._chip_hints_from_query("D触发器真值表") == ["74ls74"]


def test_datasheet_rerank_prefers_pin_function_page_over_addendum() -> None:
    service = KbService()
    addendum = _hit(
        "TI-LM324.pdf p29",
        "Package Option Addendum 可订购器件 封装图 引脚 包装数量",
        page=28,
    )
    pin_page = _hit(
        "TI-LM324.pdf p3",
        "Pin Configuration and Functions 引脚配置和功能 1OUT 1IN VCC terminal functions",
        page=2,
    )

    ranked = service._rank_hits(
        query="LM324 pinout pin configuration pin diagram",
        hits=[addendum, pin_page],
        chip_hints=["lm324"],
    )

    assert ranked[0] == pin_page


def test_query_terms_expand_chinese_datasheet_language() -> None:
    service = KbService()

    terms = service._query_terms("LM324 供电电压")

    assert "supply voltage" in terms
    assert "vcc" in terms


def test_fallback_pdf_retrieval_skips_unreadable_encrypted_pdf(monkeypatch) -> None:
    service = KbService()
    monkeypatch.setattr(service, "_iter_local_pdf_paths", lambda: [object()])

    class _Reader:
        @property
        def pages(self):
            raise RuntimeError("cryptography>=3.1 is required for AES algorithm")

    monkeypatch.setattr("app.services.kb_service.PdfReader", lambda _: _Reader())

    assert service._fallback_retrieve_from_pdfs(query="NE555 引脚", top_k=3) == []
