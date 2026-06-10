r"""Run local OpenVINO LLM evaluation for a single student model.

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
import json
import os
import statistics
import sys
import time
from pathlib import Path

import openvino as ov
import openvino_genai as ov_genai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_questions import QUESTIONS, build_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "labguardian-student-1p5-int4-ov"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "llm_eval"


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


def select_questions(question_ids: list[str], limit: int) -> list[dict]:
    if question_ids:
        wanted = set(question_ids)
        selected = [q for q in QUESTIONS if q["id"] in wanted]
        found = {q["id"] for q in selected}
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit(f"Unknown question id(s): {', '.join(missing)}")
        return selected
    if limit <= 0:
        raise SystemExit("--limit must be >= 1 when --question-id is not provided")
    return QUESTIONS[:limit]


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
                "question": text,
                "context": shared_context,
            }
        )
    if not selected:
        raise SystemExit("At least one non-empty --question value is required.")
    return selected


def generate(pipe: ov_genai.LLMPipeline, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    started = time.time()
    text = pipe.generate(prompt, cfg)
    return str(text), time.time() - started


def build_report(
    *,
    model_dir: Path,
    device: str,
    selected_questions: list[dict],
    results: list[dict],
    max_new_tokens: int,
    load_seconds: float,
    devices: list[str],
) -> str:
    ok_latencies = [r["latency_s"] for r in results if "latency_s" in r]
    lines: list[str] = []
    lines.append("# Local Student Eval Report\n\n")
    lines.append(f"- model_dir: `{model_dir}`\n")
    lines.append(f"- device: `{device}`\n")
    lines.append(f"- available_devices: `{', '.join(devices)}`\n")
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
        lines.append(f"**Question**: {result['question']}\n\n")
        if result.get("error"):
            lines.append(f"**Error**: `{result['error']}`\n\n")
        else:
            lines.append(f"**Latency**: `{result['latency_s']:.2f}s`\n\n")
            lines.append(result["response"].strip() + "\n\n")
        lines.append("---\n\n")
    return "".join(lines)


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    devices = available_devices()
    if args.question:
        selected_questions = build_custom_questions(args.question, args.context)
    else:
        selected_questions = select_questions(args.question_id, args.limit)

    print(f"Model dir : {model_dir}")
    print(f"Device    : {args.device}")
    print(f"Questions : {len(selected_questions)}")
    print(f"Devices   : {', '.join(devices)}")
    print("-" * 72)

    load_started = time.time()
    pipe = ov_genai.LLMPipeline(str(model_dir), device=args.device)
    load_seconds = time.time() - load_started
    print(f"Load done : {load_seconds:.2f}s")

    results: list[dict] = []
    for index, question in enumerate(selected_questions, start=1):
        prompt = build_prompt(question)
        try:
            response, latency_s = generate(pipe, prompt, args.max_new_tokens)
            result = {
                "qid": question["id"],
                "intent": question["intent"],
                "topology": question["topology"],
                "question": question["question"],
                "latency_s": latency_s,
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
                "question": question["question"],
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

    payload = {
        "model_dir": str(model_dir),
        "device": args.device,
        "available_devices": devices,
        "max_new_tokens": args.max_new_tokens,
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
        ),
        encoding="utf-8",
    )

    print("-" * 72)
    print(f"Saved JSON : {json_path}")
    print(f"Saved MD   : {md_path}")


if __name__ == "__main__":
    main()
