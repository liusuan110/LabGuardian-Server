"""Run local OpenVINO LLM evaluation for a single student model.

This script reuses the shared question bank in ``eval_questions.py`` and
produces:

- a machine-readable JSON result file
- a markdown report for quick manual review

Example:
    .\.venv-export\Scripts\python.exe scripts\llm_eval\run_local_student_eval.py ^
      --model-dir models\labguardian-student-1p5-int4-ov ^
      --device CPU ^
      --limit 5

    .\.venv-export\Scripts\python.exe scripts\llm_eval\run_local_student_eval.py ^
      --device CPU ^
      --question "共射放大电路输出失真，最先该查什么？"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import openvino as ov
import openvino_genai as ov_genai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_questions import QUESTIONS, build_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "labguardian-student-1p5-int4-ov"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "llm_eval"
END_PUNCTUATION = "。！？；.!?;"
MULTISPACE_RE = re.compile(r"[ \t]+")
NOISY_SECTION_PREFIXES = (
    "引用依据",
    "证据引用",
    "参考依据",
    "参考证据",
    "安全提示",
)
SECTION_MARKERS = ("[回答约束]", "[回答格式]")
LEAKY_PREFIXES = (
    "请严格按",
    "严格按",
    "只能二选一回答",
    "只回答",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local OpenVINO student LLM on the shared question bank."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"OpenVINO model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--device",
        default="CPU",
        help="OpenVINO device to use, for example CPU or GPU.",
    )
    parser.add_argument(
        "--question-bank-json",
        type=Path,
        help="Optional JSON question bank path. When set, questions are loaded from this file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of questions to run from the start of the question bank.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Run only the specified question id. Can be provided multiple times.",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Custom question to run from the command line. Can be provided multiple times.",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Optional shared context used with --question mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
        help="Maximum number of generated tokens per question.",
    )
    parser.add_argument(
        "--answer-instruction",
        default="",
        help="Optional extra instruction appended to each prompt to control answer style.",
    )
    parser.add_argument(
        "--concise",
        action="store_true",
        help="Append a concise-answer instruction suited for fixed-length P0 evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON and markdown reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--report-prefix",
        default="local_student_eval",
        help="Prefix for output file names.",
    )
    return parser.parse_args()


def available_devices() -> list[str]:
    return list(ov.Core().available_devices)


def load_question_bank(question_bank_json: Path | None) -> list[dict[str, Any]]:
    if question_bank_json is None:
        return list(QUESTIONS)

    bank_path = question_bank_json.resolve()
    if not bank_path.exists():
        raise SystemExit(f"Question bank not found: {bank_path}")

    data = json.loads(bank_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Question bank JSON must contain a top-level list.")
    return data


def select_questions(questions: list[dict[str, Any]], question_ids: list[str], limit: int) -> list[dict]:
    if question_ids:
        wanted = set(question_ids)
        selected = [q for q in questions if q["id"] in wanted]
        found = {q["id"] for q in selected}
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit(f"Unknown question id(s): {', '.join(missing)}")
        return selected
    if limit <= 0:
        raise SystemExit("--limit must be >= 1 when --question-id is not provided")
    return questions[:limit]


def build_custom_questions(custom_questions: list[str], context: str) -> list[dict]:
    selected: list[dict] = []
    shared_context = context.strip()
    for index, question_text in enumerate(custom_questions, start=1):
        text = question_text.strip()
        if not text:
            continue
        selected.append(
            {
                "id": f"user_{index:02d}",
                "intent": "custom_cli",
                "topology": "custom",
                "source": "custom_cli",
                "scene_id": "",
                "risk_level": "unknown",
                "question": text,
                "context": shared_context,
                "expected_points": [],
            }
        )
    if not selected:
        raise SystemExit("At least one non-empty --question value is required.")
    return selected


def build_eval_prompt(question: dict[str, Any], *, answer_instruction: str) -> str:
    prompt = build_prompt(question)
    answer_format = _answer_format_instruction(question)
    instruction = answer_instruction.strip()
    parts = [prompt]
    if answer_format:
        parts.append(f"[回答格式]\n{answer_format}")
    if instruction:
        parts.append(f"[回答约束]\n{instruction}")
    return "\n".join(parts) + "\n"


def _answer_format_instruction(question: dict[str, Any]) -> str:
    intent = str(question.get("intent", "")).strip()
    if intent == "concept_tutor":
        return "只回答概念、公式或结论；如果题目要求关键等式，必须直接写出等式；不要写操作建议；不要写安全提示；优先用 1 到 2 句话答完。"
    if intent == "diagnostic":
        return "严格按 3 点回答：1. 先直接给故障判断或二选一结论 2. 再给依据 3. 最后给排查方向；每点不超过 1 句话；不要举额外算例。"
    if intent == "lab_guidance":
        return "严格按步骤 1、2、3 回答；每步只写 1 句话；不要补充额外说明。"
    if intent == "mixed":
        return "严格按 3 点回答：1. 结论 2. 原因 3. 操作建议；不要补充额外说明。"
    return ""


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    texts = getattr(result, "texts", None)
    if texts:
        try:
            return str(texts[0])
        except (IndexError, TypeError):
            pass
    return str(result)


def _dedup_repeated_lines(text: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for raw in text.splitlines():
        key = raw.strip()
        if not key:
            out.append(raw)
            continue
        if len(key) >= 2 and key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return "\n".join(out)


def _strip_instruction_leakage(text: str, answer_instruction: str, answer_format: str) -> str:
    leaked_lines = {line.strip() for line in answer_instruction.splitlines() if line.strip()}
    leaked_lines.update(line.strip() for line in answer_format.splitlines() if line.strip())
    cleaned_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if line in SECTION_MARKERS or line in leaked_lines:
            continue
        if any(line.startswith(prefix) for prefix in LEAKY_PREFIXES):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _strip_noisy_sections(text: str) -> str:
    cleaned_lines: list[str] = []
    skipping_tail = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if not skipping_tail and cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        normalized = line.lstrip("*#- ").strip()
        if any(normalized.startswith(prefix) for prefix in NOISY_SECTION_PREFIXES):
            skipping_tail = True
            continue
        if skipping_tail:
            continue
        cleaned_lines.append(line.replace("**", ""))
    return "\n".join(cleaned_lines).strip()


def _limit_to_three_blocks(text: str) -> str:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if not blocks:
        return ""
    return "\n\n".join(blocks[:3]).strip()


def _filter_blocks_by_intent(text: str, intent: str) -> str:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if not blocks:
        return ""

    if intent == "concept_tutor":
        kept: list[str] = []
        for block in blocks:
            normalized = block.lstrip("*#- ").strip()
            if normalized.startswith(("解释", "操作建议", "操作", "步骤")):
                continue
            kept.append(block)
        return (kept[0] if kept else blocks[0]).strip()

    if intent == "lab_guidance":
        primary = blocks[0]
        lines = [line.strip() for line in primary.splitlines() if line.strip()]
        step_lines = [line for line in lines if line[:2] in ("1.", "2.", "3.")]
        if step_lines:
            return "\n".join(step_lines[:3]).strip()
        return primary.strip()

    if intent in {"diagnostic", "mixed"}:
        kept: list[str] = []
        for block in blocks:
            normalized = block.lstrip("*#- ").strip()
            if normalized.startswith(("解释", "补充", "延伸")):
                continue
            kept.append(block)
        return "\n\n".join(kept[:2]).strip() if kept else blocks[0].strip()

    return "\n\n".join(blocks[:2]).strip()


def _trim_to_complete_sentence(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    if not stripped:
        return stripped, False
    if stripped[-1] in END_PUNCTUATION:
        return stripped, False

    last_end = max(stripped.rfind(ch) for ch in END_PUNCTUATION)
    if last_end >= max(20, len(stripped) // 3):
        trimmed = stripped[: last_end + 1].strip()
        if trimmed:
            return trimmed, True
    return stripped, False


def _finalize_response_text(
    text: str,
    *,
    answer_instruction: str,
    answer_format: str,
    intent: str,
) -> tuple[str, bool]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = "\n".join(MULTISPACE_RE.sub(" ", line).strip() for line in normalized.splitlines())
    normalized = _strip_instruction_leakage(normalized, answer_instruction, answer_format)
    normalized = _strip_noisy_sections(normalized)
    normalized = _dedup_repeated_lines(normalized).strip()
    normalized = _limit_to_three_blocks(normalized)
    normalized = _filter_blocks_by_intent(normalized, intent)
    return _trim_to_complete_sentence(normalized)


def generate(
    pipe: ov_genai.LLMPipeline,
    prompt: str,
    max_new_tokens: int,
    *,
    answer_instruction: str,
    answer_format: str,
    intent: str,
) -> tuple[str, float, bool]:
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    if hasattr(cfg, "do_sample"):
        cfg.do_sample = False
    if hasattr(cfg, "temperature"):
        cfg.temperature = 0.0
    if hasattr(cfg, "top_p"):
        cfg.top_p = 1.0
    if hasattr(cfg, "top_k"):
        cfg.top_k = 1
    if hasattr(cfg, "num_beams"):
        cfg.num_beams = 1
    if hasattr(cfg, "repetition_penalty"):
        cfg.repetition_penalty = 1.05
    started = time.time()
    raw = pipe.generate(prompt, cfg)
    finalized, trimmed = _finalize_response_text(
        _extract_text(raw),
        answer_instruction=answer_instruction,
        answer_format=answer_format,
        intent=intent,
    )
    return finalized, time.time() - started, trimmed


def build_report(
    *,
    model_dir: Path,
    device: str,
    selected_questions: list[dict],
    results: list[dict],
    max_new_tokens: int,
    load_seconds: float,
    devices: list[str],
    question_bank_json: Path | None,
) -> str:
    ok_latencies = [r["latency_s"] for r in results if "latency_s" in r]
    lines: list[str] = []
    lines.append("# Local Student Eval Report\n\n")
    lines.append(f"- model_dir: `{model_dir}`\n")
    lines.append(f"- device: `{device}`\n")
    lines.append(f"- available_devices: `{', '.join(devices)}`\n")
    if question_bank_json is not None:
        lines.append(f"- question_bank_json: `{question_bank_json}`\n")
    lines.append(f"- questions: `{len(selected_questions)}`\n")
    lines.append(f"- max_new_tokens: `{max_new_tokens}`\n")
    lines.append(f"- load_seconds: `{load_seconds:.2f}`\n")
    if ok_latencies:
        lines.append(f"- avg_latency_s: `{statistics.mean(ok_latencies):.2f}`\n")
    lines.append("\n---\n\n")

    for result in results:
        lines.append(
            f"## {result['qid']} · {result['intent']} · {result['topology']}\n\n"
        )
        lines.append(f"**Source**: `{result.get('source', '')}`\n\n")
        if result.get("scene_id"):
            lines.append(f"**Scene**: `{result['scene_id']}`\n\n")
        if result.get("risk_level"):
            lines.append(f"**Risk**: `{result['risk_level']}`\n\n")
        lines.append(f"**Question**: {result['question']}\n\n")
        expected_points = result.get("expected_points") or []
        if expected_points:
            lines.append("**Expected Points**:\n\n")
            for point in expected_points:
                lines.append(f"- {point}\n")
            lines.append("\n")
        if result.get("error"):
            lines.append(f"**Error**: `{result['error']}`\n\n")
        else:
            lines.append(f"**Latency**: `{result['latency_s']:.2f}s`\n\n")
            if result.get("trimmed_to_complete_sentence"):
                lines.append("**Postprocess**: `trimmed_to_complete_sentence`\n\n")
            lines.append(result["response"].strip() + "\n\n")
            if expected_points:
                lines.append("**Manual Checklist**:\n\n")
                for point in expected_points:
                    lines.append(f"- [ ] {point}\n")
                lines.append("\n")
        lines.append("---\n\n")
    return "".join(lines)


def write_score_sheet(results: list[dict[str, Any]], output_path: Path) -> bool:
    if not any(result.get("expected_points") for result in results):
        return False

    fieldnames = [
        "qid",
        "source",
        "scene_id",
        "intent",
        "topology",
        "risk_level",
        "question",
        "expected_point_1",
        "expected_point_2",
        "expected_point_3",
        "hit_1",
        "hit_2",
        "hit_3",
        "manual_score",
        "manual_notes",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            expected_points = list(result.get("expected_points") or [])
            writer.writerow(
                {
                    "qid": result["qid"],
                    "source": result.get("source", ""),
                    "scene_id": result.get("scene_id", ""),
                    "intent": result.get("intent", ""),
                    "topology": result.get("topology", ""),
                    "risk_level": result.get("risk_level", ""),
                    "question": result.get("question", ""),
                    "expected_point_1": expected_points[0] if len(expected_points) > 0 else "",
                    "expected_point_2": expected_points[1] if len(expected_points) > 1 else "",
                    "expected_point_3": expected_points[2] if len(expected_points) > 2 else "",
                    "hit_1": "",
                    "hit_2": "",
                    "hit_3": "",
                    "manual_score": "",
                    "manual_notes": "",
                }
            )
    return True


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    devices = available_devices()
    answer_instruction_parts: list[str] = []
    if args.concise:
        answer_instruction_parts.append(
            "请只输出答案正文；控制在3点以内作答，每点1到2句话；先给结论，再给原因或操作；不要重复题干和上下文；不要输出引用依据、证据引用、安全提示或额外说明。"
        )
    if args.answer_instruction.strip():
        answer_instruction_parts.append(args.answer_instruction.strip())
    answer_instruction = "\n".join(answer_instruction_parts).strip()

    if args.question:
        selected_questions = build_custom_questions(args.question, args.context)
    else:
        question_bank = load_question_bank(args.question_bank_json)
        selected_questions = select_questions(question_bank, args.question_id, args.limit)

    print(f"Model dir : {model_dir}")
    print(f"Device    : {args.device}")
    print(f"Questions : {len(selected_questions)}")
    if answer_instruction:
        print(f"Instruction: {answer_instruction}")
    print(f"Devices   : {', '.join(devices)}")
    print("-" * 72)

    load_started = time.time()
    pipe = ov_genai.LLMPipeline(str(model_dir), device=args.device)
    load_seconds = time.time() - load_started
    print(f"Load done : {load_seconds:.2f}s")

    results: list[dict] = []
    for index, question in enumerate(selected_questions, start=1):
        prompt = build_eval_prompt(question, answer_instruction=answer_instruction)
        answer_format = _answer_format_instruction(question)
        try:
            response, latency_s, trimmed_to_sentence = generate(
                pipe,
                prompt,
                args.max_new_tokens,
                answer_instruction=answer_instruction,
                answer_format=answer_format,
                intent=str(question.get("intent", "")),
            )
            result = {
                "qid": question["id"],
                "intent": question["intent"],
                "topology": question["topology"],
                "source": question.get("source", "eval_questions"),
                "scene_id": question.get("scene_id", ""),
                "risk_level": question.get("risk_level", ""),
                "question": question["question"],
                "expected_points": question.get("expected_points", []),
                "latency_s": latency_s,
                "prompt_instruction": answer_instruction,
                "trimmed_to_complete_sentence": trimmed_to_sentence,
                "response": response,
            }
            results.append(result)
            preview = response.strip().replace("\n", " ")
            print(
                f"[{index:02d}/{len(selected_questions):02d}] "
                f"{question['id']:<10} {latency_s:6.2f}s  {preview[:80]}"
            )
        except Exception as exc:  # pragma: no cover - runtime smoke path
            result = {
                "qid": question["id"],
                "intent": question["intent"],
                "topology": question["topology"],
                "source": question.get("source", "eval_questions"),
                "scene_id": question.get("scene_id", ""),
                "risk_level": question.get("risk_level", ""),
                "question": question["question"],
                "expected_points": question.get("expected_points", []),
                "prompt_instruction": answer_instruction,
                "error": str(exc).replace("\n", " ")[:300],
            }
            results.append(result)
            print(
                f"[{index:02d}/{len(selected_questions):02d}] "
                f"{question['id']:<10} FAIL  {result['error']}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.report_prefix}.json"
    md_path = args.output_dir / f"{args.report_prefix}.md"
    score_sheet_path = args.output_dir / f"{args.report_prefix}_score_sheet.csv"

    payload = {
        "model_dir": str(model_dir),
        "device": args.device,
        "available_devices": devices,
        "question_bank_json": str(args.question_bank_json.resolve()) if args.question_bank_json else None,
        "max_new_tokens": args.max_new_tokens,
        "answer_instruction": answer_instruction,
        "load_seconds": round(load_seconds, 4),
        "question_count": len(selected_questions),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        build_report(
            model_dir=model_dir,
            device=args.device,
            selected_questions=selected_questions,
            results=results,
            max_new_tokens=args.max_new_tokens,
            load_seconds=load_seconds,
            devices=devices,
            question_bank_json=args.question_bank_json.resolve() if args.question_bank_json else None,
        ),
        encoding="utf-8",
    )
    wrote_score_sheet = write_score_sheet(results, score_sheet_path)

    print("-" * 72)
    print(f"Saved JSON : {json_path}")
    print(f"Saved MD   : {md_path}")
    if wrote_score_sheet:
        print(f"Saved CSV  : {score_sheet_path}")


if __name__ == "__main__":
    main()
