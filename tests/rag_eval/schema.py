"""Gold-set schema + ID 存在性校验（WP-4）。

加载 ``gold_pilot20.yaml`` / ``gold_full60.yaml`` 时跑：
  1. Pydantic 字段类型 + 枚举校验（intent / scene_id）；
  2. 每个 ``fault_case_ids_expected`` / ``datasheet_chunk_ids_expected``
     都要对应到 ``knowledge/`` 下真实文件 —— 防止 gold 集里有 typo 或
     幽灵 ID；
  3. ``forbidden_ids`` 允许 ``<scene_id>:*`` glob 模式，匹配语义由
     评测脚本（WP-5）实现。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from app.services.scene_resolver import VALID_SCENE_IDS

REPO_ROOT = Path(__file__).resolve().parents[2]

VALID_INTENTS = ("diagnostic", "concept_tutor", "lab_guidance", "mixed")
VALID_REQUIRED_SOURCES = (
    "teaching_scene",
    "fault_case",
    "datasheet_v2",
    "structured_fact",  # error_tags / station_state / pipeline_snapshot
)


class GoldSetEntry(BaseModel):
    """一条 gold 评测样本。

    `*_expected` 字段是评测时的命中目标；`forbidden_ids` 是任何情况下
    不应出现在 evidence 中的 ID（用于检测跨场景/跨芯片泄漏）。
    """

    qid: str
    query: str
    intent: Literal["diagnostic", "concept_tutor", "lab_guidance", "mixed"]
    scene_id_expected: str | None = None
    fault_case_ids_expected: list[str] = Field(default_factory=list)
    datasheet_chunk_ids_expected: list[str] = Field(default_factory=list)
    forbidden_ids: list[str] = Field(default_factory=list)
    required_sources: list[
        Literal["teaching_scene", "fault_case", "datasheet_v2", "structured_fact"]
    ] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _check_scene_id(self) -> "GoldSetEntry":
        # diagnostic / mixed 必须有 scene_id（与 WP-3 v4 P1 一致）
        if self.intent in ("diagnostic", "mixed") and not self.scene_id_expected:
            raise ValueError(
                f"qid={self.qid}: intent={self.intent} 必须设置 scene_id_expected"
            )
        if self.scene_id_expected and self.scene_id_expected not in VALID_SCENE_IDS:
            raise ValueError(
                f"qid={self.qid}: scene_id_expected={self.scene_id_expected!r} "
                f"不在 6 demo 之中 ({sorted(VALID_SCENE_IDS)})"
            )
        return self


# ---------------------------------------------------------------------------
# Repository fact loader — fault_cases + datasheet chunks
# ---------------------------------------------------------------------------


def load_known_fault_case_ids() -> set[str]:
    fault_dir = REPO_ROOT / "knowledge" / "fault_cases"
    ids: set[str] = set()
    for path in fault_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        kid = payload.get("knowledge_id")
        if isinstance(kid, str) and kid:
            ids.add(kid)
    return ids


def load_known_datasheet_chunk_ids() -> set[str]:
    sheet_dir = REPO_ROOT / "knowledge" / "datasheets"
    ids: set[str] = set()
    for path in sheet_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for chunk in payload.get("chunks", []):
            cid = chunk.get("chunk_id") if isinstance(chunk, dict) else None
            if isinstance(cid, str) and cid:
                ids.add(cid)
    return ids


# ---------------------------------------------------------------------------
# Loader + cross-check
# ---------------------------------------------------------------------------


def load_gold_set(path: Path) -> list[GoldSetEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} 顶层必须是 list，得到 {type(raw).__name__}")
    return [GoldSetEntry.model_validate(item) for item in raw]


def validate_gold_set_against_repo(entries: list[GoldSetEntry]) -> list[str]:
    """返回所有未通过的错误信息（空列表 = 全部 OK）。

    校验：
      - 每个 fault_case_id_expected 必须存在于 knowledge/fault_cases/**
      - 每个 datasheet_chunk_id_expected 必须存在于 knowledge/datasheets/*.json
      - forbidden_ids 中的非 glob 项也必须存在（防止 typo）
      - qid 全集唯一
    """
    errors: list[str] = []
    known_faults = load_known_fault_case_ids()
    known_chunks = load_known_datasheet_chunk_ids()

    seen_qids: set[str] = set()
    for entry in entries:
        if entry.qid in seen_qids:
            errors.append(f"qid={entry.qid}: 重复的 qid")
        seen_qids.add(entry.qid)
        for fid in entry.fault_case_ids_expected:
            if fid not in known_faults:
                errors.append(
                    f"qid={entry.qid}: fault_case_id_expected={fid!r} 在 "
                    "knowledge/fault_cases/** 中不存在"
                )
        for cid in entry.datasheet_chunk_ids_expected:
            if cid not in known_chunks:
                errors.append(
                    f"qid={entry.qid}: datasheet_chunk_id_expected={cid!r} 在 "
                    "knowledge/datasheets/*.json 中不存在"
                )
        for fid in entry.forbidden_ids:
            if ":*" in fid or "*" in fid:
                # glob — 由评测脚本解释
                continue
            if fid not in known_faults and fid not in known_chunks:
                errors.append(
                    f"qid={entry.qid}: forbidden_id={fid!r} 既不是已知 fault "
                    "也不是 datasheet chunk（typo？）"
                )
    return errors


__all__ = [
    "GoldSetEntry",
    "VALID_INTENTS",
    "VALID_REQUIRED_SOURCES",
    "load_gold_set",
    "load_known_datasheet_chunk_ids",
    "load_known_fault_case_ids",
    "validate_gold_set_against_repo",
]
