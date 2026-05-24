"""WP-4 pilot：gold set 的 schema 与 ID 真实性校验测试。

跑这一组测试就能证明 20 题 (将来 60 题) 全部满足：
  - schema 合法（intent / scene_id 在白名单）
  - 所有 fault_case_id_expected 在 knowledge/fault_cases/** 真实存在
  - 所有 datasheet_chunk_id_expected 在 knowledge/datasheets/*.json 真实存在
  - forbidden_ids 中非 glob 的部分也必须真实存在（防 typo）
  - qid 全局唯一
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.rag_eval.schema import (
    GoldSetEntry,
    load_gold_set,
    validate_gold_set_against_repo,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = REPO_ROOT / "tests" / "rag_eval" / "gold_pilot20.yaml"


@pytest.fixture(scope="module")
def pilot_entries() -> list[GoldSetEntry]:
    assert PILOT_PATH.is_file(), f"gold pilot 文件不存在: {PILOT_PATH}"
    return load_gold_set(PILOT_PATH)


def test_pilot_has_exactly_20_entries(pilot_entries: list[GoldSetEntry]) -> None:
    assert len(pilot_entries) == 20, (
        f"WP-4 pilot 应该恰好 20 题（用户审完才扩到 60）；当前 {len(pilot_entries)} 题"
    )


def test_pilot_covers_all_six_demo_scenes(pilot_entries: list[GoldSetEntry]) -> None:
    """每个 demo 场景至少有一道题。"""
    expected_scenes = {
        "exp_first_order_rc",
        "exp_common_emitter_amplifier",
        "exp_differential_amplifier",
        "exp_ua741_inverting_amplifier",
        "exp_ua741_summing_amplifier",
        "exp_ua741_integrator",
    }
    covered = {e.scene_id_expected for e in pilot_entries if e.scene_id_expected}
    missing = expected_scenes - covered
    assert not missing, f"WP-4 pilot 缺以下场景: {sorted(missing)}"


def test_pilot_covers_all_four_intents(pilot_entries: list[GoldSetEntry]) -> None:
    """每个 intent 至少一题（验证 schema 落得下所有分支）。"""
    intents = {e.intent for e in pilot_entries}
    missing = {"diagnostic", "concept_tutor", "lab_guidance", "mixed"} - intents
    assert not missing, f"WP-4 pilot 缺以下 intent: {sorted(missing)}"


def test_pilot_intent_distribution(pilot_entries: list[GoldSetEntry]) -> None:
    """与 WP-4 设计文档中的分布一致（8/6/3/3）。"""
    counts = {}
    for e in pilot_entries:
        counts[e.intent] = counts.get(e.intent, 0) + 1
    expected = {
        "diagnostic": 8,
        "concept_tutor": 6,
        "lab_guidance": 3,
        "mixed": 3,
    }
    assert counts == expected, (
        f"WP-4 pilot intent 分布偏移：期望 {expected}, 实际 {counts}"
    )


def test_pilot_all_ids_exist_in_repo(pilot_entries: list[GoldSetEntry]) -> None:
    """每个 fault_case_id_expected / datasheet_chunk_id_expected 必须真实存在。"""
    errors = validate_gold_set_against_repo(pilot_entries)
    assert not errors, "WP-4 pilot ID 校验失败:\n  " + "\n  ".join(errors)


def test_pilot_diagnostic_and_mixed_have_scene_id(pilot_entries: list[GoldSetEntry]) -> None:
    """与 WP-3 v4 P1 入口校验一致：diagnostic / mixed 必须有 scene_id_expected。"""
    for e in pilot_entries:
        if e.intent in ("diagnostic", "mixed"):
            assert e.scene_id_expected, (
                f"qid={e.qid}: intent={e.intent} 缺少 scene_id_expected"
            )


def test_pilot_qids_are_unique(pilot_entries: list[GoldSetEntry]) -> None:
    qids = [e.qid for e in pilot_entries]
    duplicates = {q for q in qids if qids.count(q) > 1}
    assert not duplicates, f"qid 重复: {sorted(duplicates)}"


def test_pilot_diagnostic_entries_have_fault_case_or_forbidden(
    pilot_entries: list[GoldSetEntry],
) -> None:
    """diagnostic 题至少要么有 expected fault_case_id（命中目标）
    要么有 forbidden_ids（错检测目标）—— 否则评测时无法区分召回质量。"""
    for e in pilot_entries:
        if e.intent != "diagnostic":
            continue
        assert e.fault_case_ids_expected or e.forbidden_ids, (
            f"qid={e.qid}: diagnostic 题缺少评测锚点（fault_case_ids_expected "
            f"或 forbidden_ids 至少有一个）"
        )
