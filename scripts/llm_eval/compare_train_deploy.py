"""Compare merged-model output against the deployed OpenVINO student model.

This is the local "train == deploy" check before copying the OpenVINO model to
the board. The script runs one deterministic prompt on:

1. the merged Hugging Face model directory
2. the exported OpenVINO model directory

Then it compares the first N generated tokens using the merged model tokenizer.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_questions import QUESTIONS, build_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERGED_DIR = REPO_ROOT / "models" / "labguardian-student-1p5-merged"
DEFAULT_OPENVINO_DIR = REPO_ROOT / "models" / "labguardian-student-1p5-int4-ov"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "llm_eval" / "train_deploy_compare.json"
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare merged student output with OpenVINO deployed output."
    )
    parser.add_argument(
        "--merged-model-dir",
        type=Path,
        default=DEFAULT_MERGED_DIR,
        help=f"Merged Hugging Face model directory (default: {DEFAULT_MERGED_DIR})",
    )
    parser.add_argument(
        "--openvino-model-dir",
        type=Path,
        default=DEFAULT_OPENVINO_DIR,
        help=f"OpenVINO model directory (default: {DEFAULT_OPENVINO_DIR})",
    )
    parser.add_argument(
        "--hf-device",
        default="auto",
        help="Transformers device choice: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--openvino-device",
        default="CPU",
        help="OpenVINO device to use, for example CPU or GPU.",
    )
    parser.add_argument(
        "--question-id",
        default="ce_04",
        help="Question id from scripts/llm_eval/eval_questions.py.",
    )
    parser.add_argument(
        "--question",
        default="",
        help="Custom question text. When provided, it overrides --question-id.",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Optional context used with --question.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens for both models.",
    )
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=100,
        help="How many generated tokens to compare from the front.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Deterministic comparison uses the same repetition penalty on both backends.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward trust_remote_code to AutoTokenizer/AutoModelForCausalLM.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"JSON report path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def find_question(question_id: str) -> dict[str, str]:
    for question in QUESTIONS:
        if question["id"] == question_id:
            return question
    raise SystemExit(f"Unknown question id: {question_id}")


def build_question_payload(args: argparse.Namespace) -> dict[str, str]:
    if args.question.strip():
        return {
            "id": "custom_cli",
            "intent": "custom_cli",
            "topology": "custom",
            "question": args.question.strip(),
            "context": args.context.strip(),
        }
    return find_question(args.question_id)


def matched_prefix_length(left: list[int], right: list[int], limit: int) -> int:
    capped = min(limit, len(left), len(right))
    matched = 0
    for index in range(capped):
        if left[index] != right[index]:
            break
        matched += 1
    return matched


def prefix_match_ratio(left: list[int], right: list[int], limit: int) -> float:
    capped = min(limit, len(left), len(right))
    if capped == 0:
        return 0.0
    return matched_prefix_length(left, right, limit) / capped


def normalize_text_for_compare(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized


def load_transformers(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Missing dependencies for merged-model comparison. "
            "Install transformers + torch before running this script."
        ) from exc

    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.merged_model_dir),
            fix_mistral_regex=True,
            **tokenizer_kwargs,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.merged_model_dir),
            **tokenizer_kwargs,
        )
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "dtype": "auto",
    }
    if args.hf_device == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(args.merged_model_dir), **load_kwargs)
    if args.hf_device != "auto":
        model = model.to(args.hf_device)
    model.eval()
    return torch, tokenizer, model


def generate_with_transformers(
    *,
    torch_module: Any,
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    repetition_penalty: float,
) -> tuple[list[int], str, float]:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
    input_length = int(encoded["input_ids"].shape[-1])
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.repetition_penalty = repetition_penalty
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    started = time.time()
    with torch_module.inference_mode():
        output = model.generate(
            **encoded,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )
    elapsed = time.time() - started
    continuation_ids = output[0][input_length:].tolist()
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return continuation_ids, continuation_text, elapsed


def load_openvino_pipeline(model_dir: Path, device: str) -> Any:
    try:
        import openvino_genai as ov_genai
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Missing dependency openvino-genai. Install it before running this script."
        ) from exc
    return ov_genai.LLMPipeline(str(model_dir), device=device)


def generate_with_openvino(pipe: Any, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    import openvino_genai as ov_genai

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
    started = time.time()
    output = pipe.generate(prompt, cfg)
    elapsed = time.time() - started
    return str(output), elapsed


def summarize_difference(left: list[int], right: list[int], limit: int) -> dict[str, Any]:
    capped = min(limit, len(left), len(right))
    mismatch_index = None
    for index in range(capped):
        if left[index] != right[index]:
            mismatch_index = index
            break
    return {
        "compared_tokens": capped,
        "matched_prefix_tokens": matched_prefix_length(left, right, limit),
        "prefix_match_ratio": round(prefix_match_ratio(left, right, limit), 4),
        "first_mismatch_index": mismatch_index,
    }


def tokenize_generated_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def main() -> None:
    args = parse_args()
    merged_model_dir = args.merged_model_dir.resolve()
    openvino_model_dir = args.openvino_model_dir.resolve()
    if not merged_model_dir.exists():
        raise SystemExit(f"Merged model directory not found: {merged_model_dir}")
    if not openvino_model_dir.exists():
        raise SystemExit(f"OpenVINO model directory not found: {openvino_model_dir}")

    question_payload = build_question_payload(args)
    prompt = build_prompt(question_payload)

    print(f"Merged model : {merged_model_dir}")
    print(f"OpenVINO dir : {openvino_model_dir}")
    print(f"Question     : {question_payload['id']} - {question_payload['question']}")
    print(f"HF device    : {args.hf_device}")
    print(f"OV device    : {args.openvino_device}")
    print("-" * 72)

    torch_module, tokenizer, model = load_transformers(args)
    hf_ids, hf_text, hf_seconds = generate_with_transformers(
        torch_module=torch_module,
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"HF generate  : {hf_seconds:.2f}s")

    pipe = load_openvino_pipeline(openvino_model_dir, args.openvino_device)
    ov_text, ov_seconds = generate_with_openvino(pipe, prompt, args.max_new_tokens)
    hf_surface_ids = tokenize_generated_text(tokenizer, hf_text)
    ov_surface_ids = tokenize_generated_text(tokenizer, ov_text)
    hf_normalized_text = normalize_text_for_compare(hf_text)
    ov_normalized_text = normalize_text_for_compare(ov_text)
    hf_normalized_ids = tokenize_generated_text(tokenizer, hf_normalized_text)
    ov_normalized_ids = tokenize_generated_text(tokenizer, ov_normalized_text)
    print(f"OV generate  : {ov_seconds:.2f}s")

    strict_summary = summarize_difference(hf_ids, ov_surface_ids, args.prefix_tokens)
    retokenized_summary = summarize_difference(
        hf_surface_ids, ov_surface_ids, args.prefix_tokens
    )
    normalized_summary = summarize_difference(
        hf_normalized_ids, ov_normalized_ids, args.prefix_tokens
    )
    report = {
        "question": question_payload,
        "prompt": prompt,
        "merged_model_dir": str(merged_model_dir),
        "openvino_model_dir": str(openvino_model_dir),
        "hf_device": args.hf_device,
        "openvino_device": args.openvino_device,
        "max_new_tokens": args.max_new_tokens,
        "prefix_tokens": args.prefix_tokens,
        "repetition_penalty": args.repetition_penalty,
        "hf_generate_seconds": round(hf_seconds, 4),
        "openvino_generate_seconds": round(ov_seconds, 4),
        "hf_generated_token_count": len(hf_ids),
        "openvino_generated_token_count": len(ov_surface_ids),
        "comparison": {
            "strict_generated_tokens": strict_summary,
            "retokenized_surface_text": retokenized_summary,
            "normalized_surface_text": normalized_summary,
        },
        "hf_text": hf_text,
        "openvino_text": ov_text,
        "hf_normalized_text": hf_normalized_text,
        "openvino_normalized_text": ov_normalized_text,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 72)
    for name, summary in report["comparison"].items():
        print(
            f"{name}: "
            f"{summary['matched_prefix_tokens']}/{summary['compared_tokens']} "
            f"({summary['prefix_match_ratio']:.2%})"
        )
        if summary["first_mismatch_index"] is not None:
            print(f"  first mismatch token index: {summary['first_mismatch_index']}")
    print(f"Saved report : {args.output.resolve()}")


if __name__ == "__main__":
    main()
