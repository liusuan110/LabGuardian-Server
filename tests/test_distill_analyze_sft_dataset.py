from __future__ import annotations

import json
from pathlib import Path

from scripts.distill.analyze_sft_dataset import analyze_dataset


def _make_row(*, qid: str, intent: str, scene_id: str, risk_level: str, error_code: str, error_tag: str) -> dict:
    evidence = {
        "qid": qid,
        "query": "test",
        "intent": intent,
        "scene_id": scene_id,
        "agent_output": {
            "context_pack": {
                "pack_id": "pcm_unknown_v1",
                "risk_level": risk_level,
                "allowed_tools": [{"name": "fault_case_lookup_tool", "required": True}],
            },
            "evidence_error_codes": [error_code],
            "evidence_error_tags": [error_tag],
        },
    }
    return {
        "id": f"{qid}:default",
        "instruction": "test",
        "input": "下面是同一条课堂问题的 frozen evidence。\n\n" + json.dumps(evidence, ensure_ascii=False),
        "output": "这是一个足够长的回答。",
    }


def test_analyze_dataset_summarizes_core_counts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train_sft_alpaca.jsonl"
    rows = [
        _make_row(
            qid="q1",
            intent="diagnostic",
            scene_id="exp_first_order_rc",
            risk_level="warning",
            error_code="NODE_MISMATCH",
            error_tag="wrong_node_connection",
        ),
        _make_row(
            qid="q2",
            intent="concept_tutor",
            scene_id="exp_ua741_integrator",
            risk_level="safe",
            error_code="FLOATING_PIN",
            error_tag="floating_connection",
        ),
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    stats = analyze_dataset(
        dataset_path=dataset_path,
        candidate_total=10,
        teacher_trainable=4,
        sft_kept_expected=2,
    )

    assert stats["totals"]["candidate_total"] == 10
    assert stats["totals"]["teacher_trainable"] == 4
    assert stats["totals"]["sft_kept"] == 2
    assert stats["totals"]["filtered_before_trainable"] == 6
    assert stats["totals"]["sft_skipped_after_trainable"] == 2
    assert stats["totals"]["matches_expected"] is True
    assert stats["intents"][0]["key"] == "diagnostic"
    assert stats["intents"][0]["count"] == 1
    assert stats["scenes"][0]["key"] == "exp_first_order_rc"
    assert stats["scenes"][1]["key"] == "exp_common_emitter_amplifier"
    assert stats["error_codes"][0]["key"] in {"NODE_MISMATCH", "FLOATING_PIN"}
    assert stats["output_length"]["min_chars"] > 0
