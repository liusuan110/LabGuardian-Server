from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.llm_eval.compare_train_deploy import (
    matched_prefix_length,
    normalize_text_for_compare,
    prefix_match_ratio,
)
from scripts.openvino_export.export_student_openvino import build_export_command


def test_matched_prefix_length_stops_at_first_mismatch() -> None:
    assert matched_prefix_length([1, 2, 3, 4], [1, 2, 9, 4], 4) == 2


def test_prefix_match_ratio_uses_shorter_sequence() -> None:
    ratio = prefix_match_ratio([11, 22, 33], [11, 22], 100)
    assert ratio == 1.0


def test_normalize_text_for_compare_collapses_whitespace() -> None:
    normalized = normalize_text_for_compare("  第一行\r\n\r\n第二行\t  第三行  ")
    assert normalized == "第一行 第二行 第三行"


def test_build_export_command_uses_text_generation_with_past() -> None:
    args = Namespace(
        optimum_cli="optimum-cli",
        weight_format="int4",
        task="text-generation-with-past",
        trust_remote_code=True,
    )
    command = build_export_command(
        args,
        Path("models/labguardian-student-1p5-merged"),
        Path("models/labguardian-student-1p5-int4-ov"),
    )
    assert "--task" in command
    assert "text-generation-with-past" in command
    assert "--trust-remote-code" in command
