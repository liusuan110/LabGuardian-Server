from __future__ import annotations

import json
from pathlib import Path

from scripts.distill.analyze_dual_teacher import analyze_dual_teacher


def _make_row(
    *,
    qid: str,
    ok: bool,
    citations: list[str],
    supported: int,
    downgraded: bool,
    latency_ms: float,
) -> dict:
    row = {
        "qid": qid,
        "teacher_output": {
            "answer": "ok" if ok else "",
            "citations": citations,
            "safety_notes": [],
            "reasoning_brief": "",
        },
        "generation": {
            "ok": ok,
            "latency_ms": latency_ms,
        },
    }
    if ok:
        row["generation"]["contract_audit"] = {
            "supported_citation_count": supported,
            "downgraded_to_evidence_insufficient": downgraded,
        }
    return row


def test_analyze_dual_teacher_summarizes_overlap_and_pool_gain(tmp_path: Path) -> None:
    deepseek_path = tmp_path / "teacher_deepseek.jsonl"
    qwen_path = tmp_path / "teacher_qwen.jsonl"

    deepseek_rows = [
        _make_row(qid="q1", ok=True, citations=["c1"], supported=1, downgraded=False, latency_ms=1000),
        _make_row(qid="q2", ok=True, citations=["c2"], supported=1, downgraded=False, latency_ms=1200),
        _make_row(qid="q3", ok=True, citations=[], supported=0, downgraded=True, latency_ms=1300),
        _make_row(qid="q4", ok=False, citations=[], supported=0, downgraded=False, latency_ms=0),
    ]
    qwen_rows = [
        _make_row(qid="q1", ok=True, citations=["c1"], supported=1, downgraded=False, latency_ms=2000),
        _make_row(qid="q2", ok=True, citations=[], supported=0, downgraded=True, latency_ms=2200),
        _make_row(qid="q3", ok=True, citations=["c3"], supported=1, downgraded=False, latency_ms=2300),
        _make_row(qid="q4", ok=False, citations=[], supported=0, downgraded=False, latency_ms=0),
    ]

    deepseek_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in deepseek_rows) + "\n",
        encoding="utf-8",
    )
    qwen_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in qwen_rows) + "\n",
        encoding="utf-8",
    )

    stats = analyze_dual_teacher(deepseek_path=deepseek_path, qwen_path=qwen_path)

    assert stats["teachers"]["deepseek"]["usable"] == 2
    assert stats["teachers"]["qwen"]["usable"] == 2
    assert stats["teachers"]["deepseek"]["usable_rate_pct"] == 50.0
    assert stats["teachers"]["qwen"]["usable_rate_pct"] == 50.0

    overlap = stats["overlap"]
    assert overlap["total"] == 4
    assert overlap["both_usable"] == 1
    assert overlap["deepseek_only_usable"] == 1
    assert overlap["qwen_only_usable"] == 1
    assert overlap["neither_usable"] == 1
    assert overlap["deepseek_usable_rate_pct"] == 50.0
    assert overlap["qwen_usable_rate_pct"] == 50.0
    assert overlap["pooled_usable_rate_pct"] == 75.0
    assert overlap["pooled_gain_vs_deepseek_pp"] == 25.0
