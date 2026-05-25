from __future__ import annotations

import json
from pathlib import Path


def _sample_evidence_record() -> dict:
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
                        "fault_cases": [
                            {"knowledge_id": "inv_vee_pin_not_connected"},
                        ]
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
            "filter": {"kept": True, "reason": "matched_target_ids"},
            "tool_error_count": 0,
        },
    }


def test_parse_teacher_spec_reads_api_key_from_env(monkeypatch) -> None:
    from scripts.distill.gen_teacher_answers import _parse_teacher_spec

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-token")
    spec = _parse_teacher_spec(
        "name=deepseek,model=deepseek-chat,base_url=https://api.deepseek.com/v1,api_key_env=DEEPSEEK_API_KEY"
    )
    assert spec.name == "deepseek"
    assert spec.model == "deepseek-chat"
    assert spec.base_url == "https://api.deepseek.com/v1"
    assert spec.api_key == "secret-token"


def test_build_user_prompt_contains_frozen_evidence() -> None:
    from scripts.distill.gen_teacher_answers import _build_user_prompt

    prompt = _build_user_prompt(_sample_evidence_record())
    assert "pilot_001" in prompt
    assert "fault_case_lookup_tool" in prompt
    assert "inv_vee_pin_not_connected" in prompt
    assert "FLOATING_PIN" in prompt


def test_main_writes_teacher_output_jsonl(tmp_path: Path, monkeypatch) -> None:
    import scripts.distill.gen_teacher_answers as entry

    evidence = tmp_path / "evidence.jsonl"
    output = tmp_path / "teachers.jsonl"
    evidence.write_text(
        json.dumps(_sample_evidence_record(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QWEN_API_KEY", "qwen-secret")

    def _fake_call_openai_compatible(*, teacher, messages, timeout_s):
        assert teacher.name == "qwen"
        assert messages[0]["role"] == "system"
        assert "frozen evidence" in messages[1]["content"]
        assert timeout_s == 15.0
        return (
            {
                "answer": "先检查 U1 的 pin4 负电源是否悬空。",
                "citations": ["FLOATING_PIN", "inv_vee_pin_not_connected"],
                "safety_notes": ["先断电再改线。"],
                "reasoning_brief": "故障码和 fault case 都指向负电源未接。",
            },
            {
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "raw_content": "{}",
            },
            123.4,
        )

    monkeypatch.setattr(entry, "_call_openai_compatible", _fake_call_openai_compatible)

    exit_code = entry.main(
        [
            "--evidence",
            str(evidence),
            "--output",
            str(output),
            "--teacher",
            "name=qwen,model=Qwen3-32B,base_url=https://example.invalid/v1,api_key_env=QWEN_API_KEY",
            "--timeout-s",
            "15",
        ]
    )
    assert exit_code == 0
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["teacher_name"] == "qwen"
    assert rows[0]["generation"]["ok"] is True
    assert rows[0]["teacher_output"]["citations"] == [
        "FLOATING_PIN",
        "inv_vee_pin_not_connected",
    ]
