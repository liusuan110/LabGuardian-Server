"""Build SFT training data from teacher-answer JSONL.

The script converts:

1. frozen evidence records (from ``run_inference --evidence-only``), and
2. teacher-answer records (from ``gen_teacher_answers.py``)

into a student fine-tuning dataset. The default output format is
LLaMA-Factory-friendly ``alpaca`` JSONL, with an optional ``chatml`` mode.

Example::

    python -m scripts.distill.build_sft_dataset ^
      --teachers datasets\\distill\\pilot20_teacher_answers.jsonl ^
      --evidence datasets\\distill\\pilot20_evidence_strict.jsonl ^
      --output datasets\\distill\\pilot20_sft_alpaca.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("scripts.distill.build_sft_dataset")

_VALID_OUTPUT_FORMATS = {"alpaca", "chatml"}
_SYSTEM_PROMPT = (
    "你是 LabGuardian 课堂实验助教。"
    "回答时必须严格依据给定 frozen evidence，不得编造器件、引脚、节点、网表或实验现象。"
    "先给结论，再给依据，再给修改或操作步骤。"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _sample_fingerprint(record: dict[str, Any]) -> str:
    payload = {
        "qid": record.get("qid"),
        "query": record.get("query"),
        "scene_id": record.get("scene_id"),
        "context_pack": ((record.get("agent_output") or {}).get("context_pack")),
        "tool_results": ((record.get("agent_output") or {}).get("tool_results")),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_evidence_index(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for _, record in iter_jsonl(path):
            qid = str(record.get("qid") or "").strip()
            if not qid:
                continue
            fingerprint = _sample_fingerprint(record)
            index[(qid, fingerprint)] = record
    return index


def _build_user_input(evidence: dict[str, Any]) -> str:
    compact = {
        "qid": evidence.get("qid"),
        "query": evidence.get("query"),
        "intent": evidence.get("intent"),
        "scene_id": evidence.get("scene_id"),
        "agent_output": {
            "context_pack": ((evidence.get("agent_output") or {}).get("context_pack")),
            "tool_results": ((evidence.get("agent_output") or {}).get("tool_results")),
            "evidence_resolved_scene_id": (
                (evidence.get("agent_output") or {}).get("evidence_resolved_scene_id")
            ),
            "evidence_error_codes": (
                (evidence.get("agent_output") or {}).get("evidence_error_codes")
            ),
            "evidence_error_tags": (
                (evidence.get("agent_output") or {}).get("evidence_error_tags")
            ),
        },
        "audit": {
            "filter": ((evidence.get("audit") or {}).get("filter")),
            "tool_error_count": ((evidence.get("audit") or {}).get("tool_error_count")),
            "evidence_only": ((evidence.get("audit") or {}).get("evidence_only")),
        },
    }
    frozen = json.dumps(compact, ensure_ascii=False, indent=2)
    return (
        "下面是同一条课堂问题的 frozen evidence。\n"
        "请你只依据这些证据回答学生原问题。\n\n"
        f"{frozen}"
    )


def _build_assistant_output(
    teacher_row: dict[str, Any],
    *,
    include_citations: bool,
    include_safety_notes: bool,
    include_reasoning_brief: bool,
) -> str:
    teacher_output = teacher_row.get("teacher_output") or {}
    answer = str(teacher_output.get("answer") or "").strip()
    citations = teacher_output.get("citations") or []
    safety_notes = teacher_output.get("safety_notes") or []
    reasoning_brief = str(teacher_output.get("reasoning_brief") or "").strip()

    sections = [answer]
    if include_citations and citations:
        citation_lines = "\n".join(f"- {str(item).strip()}" for item in citations if str(item).strip())
        if citation_lines:
            sections.append(f"引用依据：\n{citation_lines}")
    if include_safety_notes and safety_notes:
        safety_lines = "\n".join(f"- {str(item).strip()}" for item in safety_notes if str(item).strip())
        if safety_lines:
            sections.append(f"安全提示：\n{safety_lines}")
    if include_reasoning_brief and reasoning_brief:
        sections.append(f"补充说明：{reasoning_brief}")
    return "\n\n".join(section for section in sections if section.strip())


def _teacher_row_error(row: dict[str, Any], *, min_answer_chars: int) -> str | None:
    generation = row.get("generation") or {}
    if not generation.get("ok"):
        return f"teacher generation failed: {generation.get('error_type') or 'unknown'}"
    answer = str(((row.get("teacher_output") or {}).get("answer")) or "").strip()
    if len(answer) < min_answer_chars:
        return f"answer too short ({len(answer)} chars)"
    qid = str(row.get("qid") or "").strip()
    fingerprint = str(row.get("source_evidence_fingerprint") or "").strip()
    if not qid or not fingerprint:
        return "missing qid or source_evidence_fingerprint"
    return None


def _build_alpaca_record(
    *,
    teacher_row: dict[str, Any],
    evidence: dict[str, Any],
    assistant_output: str,
) -> dict[str, Any]:
    qid = str(teacher_row.get("qid") or "")
    teacher_name = str(teacher_row.get("teacher_name") or "teacher")
    return {
        "id": f"{qid}:{teacher_name}",
        "system": _SYSTEM_PROMPT,
        "instruction": str(evidence.get("query") or ""),
        "input": _build_user_input(evidence),
        "output": assistant_output,
        "history": [],
        "metadata": {
            "qid": qid,
            "scene_id": evidence.get("scene_id"),
            "intent": evidence.get("intent"),
            "teacher_name": teacher_name,
            "teacher_model": teacher_row.get("teacher_model"),
            "source_evidence_fingerprint": teacher_row.get("source_evidence_fingerprint"),
            "source_evidence_path": teacher_row.get("source_evidence_path"),
            "prompt_version": ((teacher_row.get("generation") or {}).get("prompt_version")),
            "built_at_iso": _now_iso(),
        },
    }


def _build_chatml_record(
    *,
    teacher_row: dict[str, Any],
    evidence: dict[str, Any],
    assistant_output: str,
) -> dict[str, Any]:
    qid = str(teacher_row.get("qid") or "")
    teacher_name = str(teacher_row.get("teacher_name") or "teacher")
    return {
        "id": f"{qid}:{teacher_name}",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_input(evidence)},
            {"role": "assistant", "content": assistant_output},
        ],
        "metadata": {
            "qid": qid,
            "scene_id": evidence.get("scene_id"),
            "intent": evidence.get("intent"),
            "teacher_name": teacher_name,
            "teacher_model": teacher_row.get("teacher_model"),
            "source_evidence_fingerprint": teacher_row.get("source_evidence_fingerprint"),
            "source_evidence_path": teacher_row.get("source_evidence_path"),
            "prompt_version": ((teacher_row.get("generation") or {}).get("prompt_version")),
            "built_at_iso": _now_iso(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teachers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence", action="append", default=[], type=Path)
    parser.add_argument(
        "--format",
        choices=sorted(_VALID_OUTPUT_FORMATS),
        default="alpaca",
    )
    parser.add_argument("--teacher-name", action="append", default=[])
    parser.add_argument("--min-answer-chars", type=int, default=40)
    parser.add_argument("--include-citations", action="store_true", default=False)
    parser.add_argument("--include-safety-notes", action="store_true", default=False)
    parser.add_argument("--include-reasoning-brief", action="store_true", default=False)
    parser.add_argument("--fail-on-missing-evidence", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.teachers.is_file():
        print(f"--teachers does not exist: {args.teachers}", file=sys.stderr)
        return 2
    if args.min_answer_chars < 1:
        print("--min-answer-chars must be >= 1", file=sys.stderr)
        return 2

    teacher_rows = [row for _, row in iter_jsonl(args.teachers)]
    teacher_filter = {name.strip() for name in args.teacher_name if name.strip()}

    evidence_paths: list[Path] = []
    if args.evidence:
        evidence_paths.extend(args.evidence)
    else:
        seen: set[Path] = set()
        for row in teacher_rows:
            raw = str(row.get("source_evidence_path") or "").strip()
            if not raw:
                continue
            path = _resolve_path(raw)
            if path not in seen:
                seen.add(path)
                evidence_paths.append(path)

    if not evidence_paths:
        print("no evidence paths resolved; pass --evidence explicitly", file=sys.stderr)
        return 2
    for path in evidence_paths:
        if not path.is_file():
            print(f"evidence file does not exist: {path}", file=sys.stderr)
            return 2

    evidence_index = _load_evidence_index(evidence_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as out:
        for row in teacher_rows:
            teacher_name = str(row.get("teacher_name") or "").strip()
            if teacher_filter and teacher_name not in teacher_filter:
                skipped += 1
                continue

            error = _teacher_row_error(row, min_answer_chars=args.min_answer_chars)
            if error:
                logger.info("skip qid=%s: %s", row.get("qid"), error)
                skipped += 1
                continue

            key = (
                str(row.get("qid") or "").strip(),
                str(row.get("source_evidence_fingerprint") or "").strip(),
            )
            evidence = evidence_index.get(key)
            if evidence is None:
                message = (
                    f"missing evidence for qid={key[0]} fingerprint={key[1]}"
                )
                if args.fail_on_missing_evidence:
                    print(message, file=sys.stderr)
                    return 1
                logger.warning(message)
                skipped += 1
                continue

            assistant_output = _build_assistant_output(
                row,
                include_citations=args.include_citations,
                include_safety_notes=args.include_safety_notes,
                include_reasoning_brief=args.include_reasoning_brief,
            )
            if args.format == "alpaca":
                record = _build_alpaca_record(
                    teacher_row=row,
                    evidence=evidence,
                    assistant_output=assistant_output,
                )
            else:
                record = _build_chatml_record(
                    teacher_row=row,
                    evidence=evidence,
                    assistant_output=assistant_output,
                )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    logger.info(
        "done — kept=%d skipped=%d output=%s",
        kept,
        skipped,
        args.output.relative_to(REPO_ROOT)
        if args.output.is_relative_to(REPO_ROOT)
        else args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
