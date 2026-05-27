"""Generate large-scale distillation question candidates with DeepSeek.

This script expands a small seed set of student-like questions into a larger
JSONL candidate pool for distillation. It is designed to satisfy the Day-1
contract in ``breezy-toasting-forest.md`` while staying runnable with the
current repo:

* reuse real ``scene_id`` / ``fault_case_id`` / ``datasheet_chunk_id`` anchors
* generate only question-side fields (no answers)
* optionally add a minimal ``station`` stub so the current ``run_inference.py``
  can consume the output without another conversion pass
* keep wording short and student-like, using ``datasets/distill/train_questions``
  as style seeds

Example::

    python -m scripts.distill.gen_questions ^
      --seed datasets\\distill\\train_questions.jsonl ^
      --output datasets\\distill\\questions_v1_5000.jsonl ^
      --target-total 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings  # noqa: E402
from app.services.scene_resolver import SCENE_TO_ALLOWED_DATASHEETS, VALID_SCENE_IDS  # noqa: E402
from tests.rag_eval.schema import (  # noqa: E402
    VALID_INTENTS,
    load_known_datasheet_chunk_ids,
    load_known_fault_case_ids,
)

logger = logging.getLogger("scripts.distill.gen_questions")

_INTENT_QUOTAS = {
    "diagnostic": 0.40,
    "concept_tutor": 0.25,
    "lab_guidance": 0.25,
    "mixed": 0.10,
}
_GENERAL_RATIO = 0.15
_DEFAULT_BATCH_SIZE = 12
_DEFAULT_MAX_ATTEMPTS = 18
_DEFAULT_MAX_ROUNDS = 4
_PREFERRED_SCENE_CHUNKS = {
    "exp_first_order_rc": [],
    "exp_common_emitter_amplifier": ["bjt_8050.common_emitter.text"],
    "exp_differential_amplifier": ["bjt_8050.diff_pair.text"],
    "exp_ua741_inverting_amplifier": ["ua741.inverting.text"],
    "exp_ua741_summing_amplifier": ["ua741.summing.text"],
    "exp_ua741_integrator": ["ua741.integrator.text"],
}
_FAULT_COMPONENT_HINTS = {
    "rc_probe_x10_not_accounted": ("SCOPE1", "probe"),
    "rc_scope_ground_not_reference_ground": ("SCOPE_GND", "gnd_clip"),
    "rc_wrong_output_node_for_integrator": ("OUT_PROBE", "output_node"),
    "rc_wrong_signal_offset": ("SIG_GEN", "dc_offset"),
    "rc_capacitor_value_mismatch": ("C1", "body_marking"),
    "ce_quiescent_point_saturation": ("Q1", "collector"),
    "ce_missing_emitter_bypass_low_gain": ("C_E", "lead"),
    "ce_bjt_pin_reversed": ("Q1", "pinout"),
    "diff_pair_tail_path_broken": ("TAIL_PATH", "emitter_tail"),
    "diff_pair_collector_resistor_mismatch": ("RC_PAIR", "collector_load_pair"),
    "inv_vee_pin_not_connected": ("U1", "pin4"),
    "inv_input_pins_swapped": ("U1", "pin2_pin3"),
    "inv_gain_calculation_error": ("Rf_Rg", "gain_network"),
    "sum_signal_sources_not_common_ground": ("SOURCE_GND", "common_ground"),
    "sum_input_resistors_shorted_at_node": ("SUM_NODE", "input_mix_node"),
    "int_time_constant_mismatch": ("Rin_Cf", "time_constant"),
    "int_missing_rleak_dc_drift": ("Rleak", "feedback_branch"),
}


@dataclass(frozen=True)
class FaultCase:
    knowledge_id: str
    scene_id: str
    title: str
    related_error_codes: tuple[str, ...]
    error_tags: tuple[str, ...]
    fix_steps: tuple[str, ...]


@dataclass(frozen=True)
class TeacherConfig:
    model: str
    base_url: str
    api_key: str
    api_key_env: str
    temperature: float = 0.7
    max_tokens: int = 3200


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON at {path}:{line_no}: {exc}") from exc


def _load_seed_questions(path: Path) -> list[dict[str, Any]]:
    return [payload for _, payload in iter_jsonl(path)]


def _load_fault_cases() -> list[FaultCase]:
    fault_dir = REPO_ROOT / "knowledge" / "fault_cases"
    cases: list[FaultCase] = []
    for path in sorted(fault_dir.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scene_id = str(payload.get("scene_id") or "").strip()
        knowledge_id = str(payload.get("knowledge_id") or "").strip()
        if not scene_id or not knowledge_id:
            continue
        if scene_id not in VALID_SCENE_IDS:
            continue
        cases.append(
            FaultCase(
                knowledge_id=knowledge_id,
                scene_id=scene_id,
                title=str(payload.get("title") or "").strip(),
                related_error_codes=tuple(
                    str(code).strip()
                    for code in payload.get("related_error_codes", [])
                    if str(code).strip()
                ),
                error_tags=tuple(
                    str(tag).strip()
                    for tag in payload.get("error_tags", [])
                    if str(tag).strip()
                ),
                fix_steps=tuple(
                    str(step).strip()
                    for step in payload.get("fix_steps", [])
                    if str(step).strip()
                ),
            )
        )
    return cases


def _load_datasheet_chunk_map() -> dict[str, list[str]]:
    datasheet_dir = REPO_ROOT / "knowledge" / "datasheets"
    scene_to_chunks: dict[str, list[str]] = defaultdict(list)
    for scene_id, docs in SCENE_TO_ALLOWED_DATASHEETS.items():
        for doc in docs:
            filename = doc.replace(".", "_") + ".json"
            path = datasheet_dir / filename
            if not path.exists():
                alt = datasheet_dir / f"{doc}.json"
                path = alt if alt.exists() else path
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for chunk in payload.get("chunks", []):
                chunk_id = str((chunk or {}).get("chunk_id") or "").strip()
                if chunk_id:
                    scene_to_chunks[scene_id].append(chunk_id)
    return {k: sorted(v) for k, v in scene_to_chunks.items()}


def _default_teacher_config() -> TeacherConfig:
    model = str(getattr(settings, "LLM_MODEL", "") or "").strip()
    base_url = str(getattr(settings, "LLM_BASE_URL", "") or "").rstrip("/")
    api_key = str(getattr(settings, "LLM_API_KEY", "") or "").strip()
    if not model or not base_url or not api_key:
        raise ValueError(
            "LLM_API_KEY / LLM_BASE_URL / LLM_MODEL must be set in .env before running gen_questions.py"
        )
    return TeacherConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env="LLM_API_KEY",
    )


def _scene_to_topology_label(scene_id: str) -> str:
    mapping = {
        "exp_first_order_rc": "rc_first_order",
        "exp_common_emitter_amplifier": "common_emitter",
        "exp_differential_amplifier": "differential_pair",
        "exp_ua741_inverting_amplifier": "inverting_amp_ua741",
        "exp_ua741_summing_amplifier": "summing_amp_ua741",
        "exp_ua741_integrator": "integrator_ua741",
    }
    return mapping.get(scene_id, "")


def _intent_bucket(samples: list[dict[str, Any]]) -> dict[str, list[str]]:
    bucket: dict[str, list[str]] = {intent: [] for intent in VALID_INTENTS}
    for sample in samples:
        intent = str(sample.get("intent") or "").strip()
        query = str(sample.get("query") or "").strip()
        if intent in bucket and query:
            bucket[intent].append(query)
    return bucket


def _normalize_query(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace("？", "?").replace("。", "").replace("，", "，")
    text = text.replace("?", "？")
    return text


def _general_topics() -> list[str]:
    return [
        "反馈为什么能稳住电路",
        "为什么共地这么重要",
        "静态工作点为什么先量",
        "为什么电阻比会影响增益",
        "电容在交流和直流里作用有什么不同",
        "为什么输出会削顶",
        "为什么会漂移",
        "怎么判断是接线错还是参数错",
        "怎么改得更稳",
        "怎么提高抗噪",
        "为什么示波器一接上就出问题",
        "运放为什么会饱和",
    ]


def _fallback_error_code(fault: FaultCase | None) -> str:
    if not fault:
        return ""
    tags = set(fault.error_tags)
    if "wrong_component_value" in tags:
        return "PARAMETRIC_MISMATCH"
    if "missing_or_broken_connection" in tags:
        return "OPEN_CIRCUIT"
    if "wrong_node_connection" in tags:
        return "NODE_MISMATCH"
    if "scope_ground_or_short_risk" in tags:
        return "COMPONENT_SHORTED_SAME_NET"
    return ""


def _build_station(
    scene_id: str,
    qid: str,
    intent: str,
    fault_case_id: str | None,
    station_hint: str,
    fault_lookup: dict[str, FaultCase],
) -> dict[str, Any]:
    station: dict[str, Any] = {
        "station_id": f"S_{qid}",
        "risk_level": "safe",
        "diagnostics": [],
        "comparison_report": {"items": []},
    }
    if scene_id:
        station["scene_id"] = scene_id
        topology = _scene_to_topology_label(scene_id)
        if topology:
            station["topology_label"] = topology
    if intent in {"diagnostic", "mixed", "lab_guidance"} and fault_case_id:
        fault = fault_lookup.get(fault_case_id)
        risk_level = "warning"
        if fault and "scope_ground_or_short_risk" in set(fault.error_tags):
            risk_level = "danger"
        station["risk_level"] = risk_level
        station["diagnostics"] = [fault_case_id]
        component_id, pin_name = _FAULT_COMPONENT_HINTS.get(
            fault_case_id, ("AUTO_COMPONENT", "auto_pin")
        )
        error_code = ""
        if fault and fault.related_error_codes:
            error_code = fault.related_error_codes[0]
        if not error_code:
            error_code = _fallback_error_code(fault)
        if error_code:
            station["comparison_report"] = {
                "items": [
                    {
                        "error_code": error_code,
                        "component_id": component_id,
                        "pin_name": pin_name,
                        "hint": station_hint[:80],
                    }
                ]
            }
    return station


def _make_group_id(scene_id: str, fault_case_id: str | None, intent: str, index: int) -> str:
    scene_token = scene_id or "general"
    fault_token = fault_case_id or "none"
    return f"{scene_token}:{fault_token}:{intent}:g{index:04d}"


def _call_deepseek(
    cfg: TeacherConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    timeout_s: float,
) -> str:
    endpoint = f"{cfg.base_url}/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
    choice = (((body.get("choices") or [{}])[0]).get("message") or {}).get("content")
    if not isinstance(choice, str) or not choice.strip():
        raise ValueError("DeepSeek returned an empty message")
    return choice


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("response is not a JSON object")


def _build_prompts(
    *,
    seed_examples: list[dict[str, Any]],
    batch_spec: dict[str, Any],
    known_fault_ids: set[str],
    known_chunk_ids: set[str],
) -> tuple[str, str]:
    seeds_by_intent = _intent_bucket(seed_examples)
    fault_case_id = batch_spec.get("target_fault_case_id") or ""
    scene_id = batch_spec.get("scene_id") or ""
    intent = batch_spec["intent"]
    examples = seeds_by_intent.get(intent) or []
    examples = examples[:6]
    style_lines = "\n".join(f"- {q}" for q in examples) or "- RC 的 tau 怎么算？"
    chunk_ids = batch_spec.get("target_datasheet_chunk_ids") or []
    fault_title = batch_spec.get("fault_title") or ""
    related_error_codes = batch_spec.get("related_error_codes") or []
    general_topics = batch_spec.get("general_topics") or []

    system_prompt = (
        "你是蒸馏数据问题生成器。目标是生成'像学生真实会问的模电问题'。\n"
        "只输出 JSON 对象，字段为 {\"items\": [...]}。\n"
        "items 中每一项只能包含：query, intent, scene_id, target_fault_case_id, "
        "target_datasheet_chunk_ids, paraphrase_group, station_hint。\n"
        "不要输出答案，不要解释，不要输出额外字段。\n"
        "query 必须短、口语、自然，尽量 8 到 24 个汉字，允许带一个问号。\n"
        "query 不能像作业题，也不要过长的三连问。\n"
        "target_fault_case_id 和 target_datasheet_chunk_ids 必须严格从给定锚点里选择，不能编造。\n"
        "station_hint 用一句中文描述可能的实验现象或排查线索，不要写成完整 station JSON。"
    )

    user_prompt = (
        f"本批次生成 {batch_spec['count']} 条问题。\n"
        f"intent={intent}\n"
        f"scene_id={scene_id or ''}\n"
        f"target_fault_case_id={fault_case_id or ''}\n"
        f"fault_title={fault_title}\n"
        f"related_error_codes={json.dumps(related_error_codes, ensure_ascii=False)}\n"
        f"target_datasheet_chunk_ids={json.dumps(chunk_ids, ensure_ascii=False)}\n"
        f"general_topics={json.dumps(general_topics, ensure_ascii=False)}\n"
        f"paraphrase_group={batch_spec['paraphrase_group']}\n\n"
        "风格参考（越像学生越好，尽量短）：\n"
        f"{style_lines}\n\n"
        "严格要求：\n"
        "1. 如果有 scene_id，就保持不变；如果是通用题，scene_id 留空字符串。\n"
        "2. 如果有 target_fault_case_id，就保持不变，并让 query 能够自然对应这个故障。\n"
        "3. 如果有 target_datasheet_chunk_ids，就从给定列表中原样引用；没有就给空数组。\n"
        "4. paraphrase_group 对本批所有 items 保持完全一致。\n"
        "5. 不要把 fault_case_id、chunk_id 原样塞进学生问句里。\n"
        "6. 多写学生口语：'先查哪''是不是''为什么会这样''怎么改'。\n"
        "7. concept_tutor 更偏原理，通常不要写成具体故障排查题；diagnostic 更偏现象+定位；lab_guidance 更偏操作步骤；mixed 两者结合。\n"
        "8. 如果 target_fault_case_id 为空，query 不要硬扯到某个具体故障。\n"
        "9. 只有当给定了明确 datasheet chunk 时，才围绕对应器件/文档内容问；没有就给空数组。\n"
        "10. 如果 intent=lab_guidance 且给了 target_fault_case_id，问题必须围绕这个故障的检查、确认、修复或预防，不要换成别的现象。\n"
        "11. concept_tutor 优先产出简短原理问句；没有强锚点时 target_datasheet_chunk_ids 应为空数组。\n"
        f"8. 已知合法 fault_case_id 总数={len(known_fault_ids)}，合法 chunk_id 总数={len(known_chunk_ids)}，但本批只能用上面给定的目标锚点。\n\n"
        "只返回 JSON：\n"
        "{\"items\": [ ... ]}"
    )
    return system_prompt, user_prompt


def _validate_generated_item(
    item: dict[str, Any],
    *,
    batch_spec: dict[str, Any],
    known_fault_ids: set[str],
    known_chunk_ids: set[str],
) -> dict[str, Any] | None:
    query = _normalize_query(str(item.get("query") or ""))
    if len(query) < 4 or len(query) > 48:
        return None
    intent = str(item.get("intent") or "").strip()
    if intent != batch_spec["intent"]:
        return None
    scene_id = str(item.get("scene_id") or "").strip()
    expected_scene = str(batch_spec.get("scene_id") or "").strip()
    if scene_id != expected_scene:
        return None
    fault_case_id = str(item.get("target_fault_case_id") or "").strip()
    expected_fault = str(batch_spec.get("target_fault_case_id") or "").strip()
    if fault_case_id != expected_fault:
        return None
    if fault_case_id and fault_case_id not in known_fault_ids:
        return None

    chunk_ids = item.get("target_datasheet_chunk_ids")
    if not isinstance(chunk_ids, list):
        return None
    chunk_ids = [str(x).strip() for x in chunk_ids if str(x).strip()]
    if any(chunk_id not in known_chunk_ids for chunk_id in chunk_ids):
        return None
    expected_chunks = [str(x).strip() for x in batch_spec.get("target_datasheet_chunk_ids") or []]
    if sorted(chunk_ids) != sorted(expected_chunks):
        return None

    paraphrase_group = str(item.get("paraphrase_group") or "").strip()
    if paraphrase_group != batch_spec["paraphrase_group"]:
        return None
    station_hint = str(item.get("station_hint") or "").strip()
    if not station_hint:
        return None
    return {
        "query": query,
        "intent": intent,
        "scene_id": scene_id,
        "target_fault_case_id": fault_case_id,
        "target_datasheet_chunk_ids": chunk_ids,
        "paraphrase_group": paraphrase_group,
        "station_hint": station_hint,
    }


def _dedupe_key(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        payload.get("scene_id", ""),
        payload.get("intent", ""),
        payload.get("target_fault_case_id", ""),
        re.sub(r"[？?！!，,。.\s]+", "", payload.get("query", "")),
    )


def _append_split_specs(batch_specs: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    remaining = int(spec["count"])
    while remaining > 0:
        chunk_size = min(_DEFAULT_BATCH_SIZE, remaining)
        item = dict(spec)
        item["count"] = chunk_size
        batch_specs.append(item)
        remaining -= chunk_size


def _build_batch_specs(
    *,
    target_total: int,
    faults: list[FaultCase],
    scene_to_chunks: dict[str, list[str]],
    group_index_start: int = 1,
) -> list[dict[str, Any]]:
    scene_faults: dict[str, list[FaultCase]] = defaultdict(list)
    for fault in faults:
        scene_faults[fault.scene_id].append(fault)

    target_general = math.ceil(target_total * _GENERAL_RATIO)
    target_scene = target_total - target_general

    batch_specs: list[dict[str, Any]] = []
    group_index = group_index_start

    per_intent_totals = {
        intent: math.ceil(target_scene * ratio) for intent, ratio in _INTENT_QUOTAS.items()
    }
    base_per_fault_intent = {
        intent: max(4, math.ceil(total / max(len(faults), 1)))
        for intent, total in per_intent_totals.items()
    }

    for scene_id in sorted(scene_faults):
        for fault in scene_faults[scene_id]:
            for intent in ("diagnostic", "lab_guidance", "mixed"):
                count = base_per_fault_intent[intent]
                if intent == "diagnostic":
                    count += 1
                preferred = list(_PREFERRED_SCENE_CHUNKS.get(scene_id) or [])
                chunk_pool = scene_to_chunks.get(scene_id, [])
                if fault.knowledge_id == "inv_vee_pin_not_connected" and "ua741.power.text" in chunk_pool:
                    preferred = ["ua741.power.text"]
                target_chunks = preferred[:1] if intent == "mixed" else preferred[:2]
                _append_split_specs(
                    batch_specs,
                    {
                        "kind": "scene_fault",
                        "count": count,
                        "scene_id": scene_id,
                        "intent": intent,
                        "target_fault_case_id": fault.knowledge_id,
                        "target_datasheet_chunk_ids": target_chunks,
                        "fault_title": fault.title,
                        "related_error_codes": list(fault.related_error_codes[:3]),
                        "paraphrase_group": _make_group_id(
                            scene_id, fault.knowledge_id, intent, group_index
                        ),
                    },
                )
                group_index += 1

    concept_total = per_intent_totals["concept_tutor"]
    concept_per_scene = max(8, math.ceil(concept_total / max(len(VALID_SCENE_IDS), 1)))
    for scene_id in sorted(VALID_SCENE_IDS):
        _append_split_specs(
            batch_specs,
            {
                "kind": "scene_concept",
                "count": concept_per_scene,
                "scene_id": scene_id,
                "intent": "concept_tutor",
                "target_fault_case_id": "",
                "target_datasheet_chunk_ids": [],
                "fault_title": "",
                "related_error_codes": [],
                "paraphrase_group": _make_group_id(scene_id, None, "concept_tutor", group_index),
            },
        )
        group_index += 1

    general_topics = _general_topics()
    general_per_intent = {
        "concept_tutor": math.ceil(target_general * 0.6),
        "lab_guidance": math.floor(target_general * 0.4),
    }
    general_batch_count = _DEFAULT_BATCH_SIZE
    for intent in ("concept_tutor", "lab_guidance"):
        remaining = general_per_intent[intent]
        while remaining > 0:
            batch_count = min(general_batch_count, remaining)
            _append_split_specs(
                batch_specs,
                {
                    "kind": "general",
                    "count": batch_count,
                    "scene_id": "",
                    "intent": intent,
                    "target_fault_case_id": "",
                    "target_datasheet_chunk_ids": [],
                    "fault_title": "",
                    "related_error_codes": [],
                    "general_topics": general_topics,
                    "paraphrase_group": _make_group_id("", None, intent, group_index),
                },
            )
            group_index += 1
            remaining -= batch_count

    return batch_specs


def _load_existing_records(output_path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, str, str]], int]:
    records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    max_qid_index = 0
    qid_pattern = re.compile(r"^qv1_(\d+)$")
    for _, payload in iter_jsonl(output_path):
        if not isinstance(payload, dict):
            continue
        records.append(payload)
        seen_keys.add(_dedupe_key(payload))
        qid = str(payload.get("qid") or "").strip()
        match = qid_pattern.match(qid)
        if match:
            max_qid_index = max(max_qid_index, int(match.group(1)))
    return records, seen_keys, max_qid_index


def _to_output_record(
    *,
    item: dict[str, Any],
    qid: str,
    fault_lookup: dict[str, FaultCase],
) -> dict[str, Any]:
    scene_id = item["scene_id"]
    fault_case_id = item["target_fault_case_id"]
    record = {
        "qid": qid,
        "query": item["query"],
        "intent": item["intent"],
        "scene_id": scene_id,
        "target_fault_case_id": fault_case_id,
        "target_datasheet_chunk_ids": item["target_datasheet_chunk_ids"],
        "paraphrase_group": item["paraphrase_group"],
        "station_hint": item["station_hint"],
        "station": _build_station(
            scene_id,
            qid,
            item["intent"],
            fault_case_id or None,
            item["station_hint"],
            fault_lookup,
        ),
    }
    return record


def _generate_questions(
    *,
    teacher_cfg: TeacherConfig,
    seed_examples: list[dict[str, Any]],
    output_path: Path,
    target_total: int,
    timeout_s: float,
    max_attempts: int,
    resume: bool,
    max_rounds: int,
) -> tuple[int, int]:
    known_fault_ids = load_known_fault_case_ids()
    known_chunk_ids = load_known_datasheet_chunk_ids()
    faults = _load_fault_cases()
    fault_lookup = {fault.knowledge_id: fault for fault in faults}
    scene_to_chunks = _load_datasheet_chunk_map()
    existing_records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    next_qid_index = 0
    if resume and output_path.exists():
        existing_records, seen_keys, next_qid_index = _load_existing_records(output_path)
        if existing_records:
            logger.info(
                "resume mode loaded existing=%s remaining=%s from %s",
                len(existing_records),
                max(target_total - len(existing_records), 0),
                output_path.relative_to(REPO_ROOT) if output_path.is_relative_to(REPO_ROOT) else output_path,
            )

    written = len(existing_records)
    discarded = 0
    next_qid_index = max(next_qid_index, written)
    if written >= target_total:
        return written, discarded

    output_path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if resume and output_path.exists() and written > 0 else "w"
    with output_path.open(open_mode, encoding="utf-8", newline="\n") as out_fh:
        for round_index in range(max_rounds):
            if written >= target_total:
                break
            remaining_needed = target_total - written
            batch_specs = _build_batch_specs(
                target_total=remaining_needed,
                faults=faults,
                scene_to_chunks=scene_to_chunks,
                group_index_start=(round_index * 10000) + 1,
            )
            random.Random(42 + round_index).shuffle(batch_specs)
            round_written_before = written
            for batch_index, batch_spec in enumerate(batch_specs, start=1):
                if written >= target_total:
                    break
                remaining_needed = target_total - written
                batch_spec = dict(batch_spec)
                batch_spec["count"] = min(batch_spec["count"], remaining_needed)
                attempts = 0
                batch_kept = 0
                while batch_kept < batch_spec["count"] and attempts < max_attempts:
                    attempts += 1
                    system_prompt, user_prompt = _build_prompts(
                        seed_examples=seed_examples,
                        batch_spec=batch_spec,
                        known_fault_ids=known_fault_ids,
                        known_chunk_ids=known_chunk_ids,
                    )
                    try:
                        raw = _call_deepseek(
                            teacher_cfg,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            timeout_s=timeout_s,
                        )
                        parsed = _extract_json_object(raw)
                        items = parsed.get("items")
                        if not isinstance(items, list):
                            raise ValueError("response JSON does not contain list field items")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "round %s batch %s/%s failed attempt=%s kind=%s scene=%s intent=%s fault=%s: %s",
                            round_index + 1,
                            batch_index,
                            len(batch_specs),
                            attempts,
                            batch_spec["kind"],
                            batch_spec["scene_id"],
                            batch_spec["intent"],
                            batch_spec["target_fault_case_id"],
                            exc,
                        )
                        time.sleep(1.0)
                        continue

                    for item in items:
                        normalized = _validate_generated_item(
                            item,
                            batch_spec=batch_spec,
                            known_fault_ids=known_fault_ids,
                            known_chunk_ids=known_chunk_ids,
                        )
                        if not normalized:
                            discarded += 1
                            continue
                        dedupe_key = _dedupe_key(normalized)
                        if dedupe_key in seen_keys:
                            discarded += 1
                            continue
                        seen_keys.add(dedupe_key)
                        next_qid_index += 1
                        qid = f"qv1_{next_qid_index:05d}"
                        record = _to_output_record(
                            item=normalized,
                            qid=qid,
                            fault_lookup=fault_lookup,
                        )
                        out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1
                        batch_kept += 1
                        if written >= target_total or batch_kept >= batch_spec["count"]:
                            break
                    if batch_kept < batch_spec["count"]:
                        time.sleep(0.4)

                logger.info(
                    "round %s batch %s/%s done kept=%s target=%s cumulative=%s/%s kind=%s scene=%s intent=%s fault=%s",
                    round_index + 1,
                    batch_index,
                    len(batch_specs),
                    batch_kept,
                    batch_spec["count"],
                    written,
                    target_total,
                    batch_spec["kind"],
                    batch_spec["scene_id"] or "<general>",
                    batch_spec["intent"],
                    batch_spec["target_fault_case_id"] or "<none>",
                )
            round_added = written - round_written_before
            logger.info(
                "round %s done added=%s cumulative=%s/%s discarded=%s",
                round_index + 1,
                round_added,
                written,
                target_total,
                discarded,
            )
            if round_added <= 0:
                logger.warning(
                    "stop after round %s because no new unique records were added",
                    round_index + 1,
                )
                break

    return written, discarded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate large candidate question set with DeepSeek."
    )
    parser.add_argument(
        "--seed",
        default="datasets/distill/train_questions.jsonl",
        help="Seed JSONL with short student-style questions.",
    )
    parser.add_argument(
        "--output",
        default="datasets/distill/questions_v1_5000.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=5000,
        help="Target number of generated question rows.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
        help="Per-request timeout for DeepSeek API.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=_DEFAULT_MAX_ATTEMPTS,
        help="Max retries per generation batch.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature passed to the API.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3200,
        help="Max response tokens for one generation batch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append new unique questions to an existing output file until target-total is reached.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=_DEFAULT_MAX_ROUNDS,
        help="Max supplement rounds when generation quality or dedupe prevents filling target-total in one pass.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    seed_path = REPO_ROOT / args.seed
    output_path = REPO_ROOT / args.output

    if args.target_total <= 0:
        print("--target-total must be > 0", file=sys.stderr)
        return 2
    if args.max_rounds <= 0:
        print("--max-rounds must be > 0", file=sys.stderr)
        return 2
    if not seed_path.exists():
        print(f"seed file not found: {seed_path}", file=sys.stderr)
        return 2

    try:
        teacher_cfg = _default_teacher_config()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    teacher_cfg = TeacherConfig(
        model=teacher_cfg.model,
        base_url=teacher_cfg.base_url,
        api_key=teacher_cfg.api_key,
        api_key_env=teacher_cfg.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    seed_examples = _load_seed_questions(seed_path)
    if not seed_examples:
        print(f"seed file is empty: {seed_path}", file=sys.stderr)
        return 2

    written, discarded = _generate_questions(
        teacher_cfg=teacher_cfg,
        seed_examples=seed_examples,
        output_path=output_path,
        target_total=args.target_total,
        timeout_s=args.timeout_s,
        max_attempts=args.max_attempts,
        resume=args.resume,
        max_rounds=args.max_rounds,
    )
    logger.info(
        "done — written=%s discarded=%s output=%s",
        written,
        discarded,
        output_path.relative_to(REPO_ROOT),
    )
    return 0 if written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
