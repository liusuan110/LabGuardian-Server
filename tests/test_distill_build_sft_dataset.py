from __future__ import annotations

import json
from pathlib import Path


def _sample_evidence() -> dict:
    return {
        "qid": "pilot_001",
        "query": "UA741 反相放大输出为什么钉在 +13V？",
        "intent": "diagnostic",
        "scene_id": "exp_ua741_inverting_amplifier",
        "agent_output": {
            "context_pack": {
                "risk_level": "danger",
                "allowed_tools": [{"name": "fault_case_lookup_tool", "required": True}],
            },
            "tool_results": [
                {
                    "tool_name": "fault_case_lookup_tool",
                    "status": "ok",
                    "payload": {
                        "fault_cases": [{"knowledge_id": "inv_vee_pin_not_connected"}]
                    },
                }
            ],
            "evidence_resolved_scene_id": "exp_ua741_inverting_amplifier",
            "evidence_error_codes": ["FLOATING_PIN"],
            "evidence_error_tags": ["floating_connection"],
        },
        "audit": {
            "skipped": False,
            "evidence_only": True,
            "filter": {"kept": True},
            "tool_error_count": 0,
        },
    }


def _sample_teacher(fingerprint: str) -> dict:
    return {
        "qid": "pilot_001",
        "teacher_name": "default",
        "teacher_model": "deepseek-chat",
        "scene_id": "exp_ua741_inverting_amplifier",
        "source_query": "UA741 反相放大输出为什么钉在 +13V？",
        "source_evidence_fingerprint": fingerprint,
        "source_evidence_path": "datasets/distill/mock_evidence.jsonl",
        "teacher_output": {
            "answer": "先检查 U1 的 pin4 负电源是否悬空。",
            "citations": ["FLOATING_PIN", "inv_vee_pin_not_connected"],
            "safety_notes": ["先断电再改线。"],
            "reasoning_brief": "故障码和 fault case 都指向负电源未接。",
        },
        "generation": {
            "ok": True,
            "prompt_version": "teacher_v1_frozen_evidence",
        },
    }


def test_build_assistant_output_can_include_citations_and_safety() -> None:
    from scripts.distill.build_sft_dataset import _build_assistant_output

    text = _build_assistant_output(
        _sample_teacher("abc"),
        include_citations=True,
        include_safety_notes=True,
        include_reasoning_brief=False,
    )
    assert "先检查 U1 的 pin4 负电源是否悬空" in text
    assert "引用依据：" in text
    assert "安全提示：" in text


def test_main_builds_alpaca_records(tmp_path: Path) -> None:
    import scripts.distill.build_sft_dataset as entry

    evidence_path = tmp_path / "mock_evidence.jsonl"
    teacher_path = tmp_path / "mock_teachers.jsonl"
    output_path = tmp_path / "mock_sft.jsonl"

    evidence = _sample_evidence()
    fingerprint = entry._sample_fingerprint(evidence)
    teacher = _sample_teacher(fingerprint)

    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    teacher["source_evidence_path"] = str(evidence_path)
    teacher_path.write_text(
        json.dumps(teacher, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    exit_code = entry.main(
        [
            "--teachers",
            str(teacher_path),
            "--evidence",
            str(evidence_path),
            "--output",
            str(output_path),
            "--min-answer-chars",
            "5",
            "--include-citations",
            "--include-safety-notes",
        ]
    )
    assert exit_code == 0
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["instruction"] == "UA741 反相放大输出为什么钉在 +13V？"
    assert "frozen evidence" in row["input"]
    assert "引用依据：" in row["output"]
    assert row["metadata"]["teacher_model"] == "deepseek-chat"


def test_main_skips_failed_teacher_rows(tmp_path: Path) -> None:
    import scripts.distill.build_sft_dataset as entry

    evidence_path = tmp_path / "mock_evidence.jsonl"
    teacher_path = tmp_path / "mock_teachers.jsonl"
    output_path = tmp_path / "mock_sft.jsonl"

    evidence = _sample_evidence()
    fingerprint = entry._sample_fingerprint(evidence)
    teacher = _sample_teacher(fingerprint)
    teacher["generation"]["ok"] = False
    teacher["generation"]["error_type"] = "TimeoutError"

    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    teacher["source_evidence_path"] = str(evidence_path)
    teacher_path.write_text(
        json.dumps(teacher, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    exit_code = entry.main(
        [
            "--teachers",
            str(teacher_path),
            "--evidence",
            str(evidence_path),
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8").strip() == ""
