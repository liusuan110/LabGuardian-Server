"""Automated LLM-as-Judge scoring for student/teacher model answers.

Uses a strong LLM (teacher model) as an impartial judge to score answers
on 5 dimensions, replacing or supplementing manual human scoring.

Usage:
    # Score a single eval result JSON
    python scripts/llm_eval/auto_scorer.py \
      --input outputs/llm_eval/p0_student_eval_final.json

    # Score all four models in batch
    python scripts/llm_eval/auto_scorer.py --batch --judge deepseek

    # Score with local Qwen3-32B
    python scripts/llm_eval/auto_scorer.py --batch \
      --judge local_qwen --judge-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "llm_eval"

# Load .env for API keys
_ENV_PATH = REPO_ROOT / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _val = _line.split("=", 1)
            _key, _val = _key.strip(), _val.strip().strip('"').strip("'")
            if _key not in os.environ:
                os.environ[_key] = _val

# ─── 5-Dimension Scoring Rubric ───────────────────────────────────────────

SCORING_RUBRIC = """
你是一位电子实验教学质量的评审专家。请根据以下 5 个维度，对【学生模型回答】进行 1-5 分打分。

## 评分维度

### 1. 正确性 (Correctness) — 技术事实是否准确
- 5分：所有技术事实完全正确，无任何错误或误导
- 4分：主要事实正确，有极小的不精确但不影响结论
- 3分：大体正确，但有一处技术不精确或遗漏关键点
- 2分：存在明显技术错误，可能误导学生
- 1分：核心事实错误，回答有严重误导性

### 2. 教学性 (Pedagogy) — 是否循循善诱、有启发性
- 5分：引导思路、层层递进，不直接给答案而是帮学生自己发现
- 4分：有引导意图，部分地方稍直接但仍以启发为主
- 3分：以解释为主，缺乏引导但也不算灌输
- 2分：直接给答案，没有教学互动感
- 1分：纯粹陈述结论，无任何教学考虑

### 3. 简洁性 (Conciseness) — 是否冗余、跑题
- 5分：每句话都必要，结构紧凑，信息密度高
- 4分：基本简洁，个别句子可更精炼
- 3分：有一些冗余或重复，但不严重影响阅读
- 2分：明显啰嗦，多处无关内容
- 1分：大量跑题或重复堆砌

### 4. 格式与结构 (Format) — 是否清晰易读
- 5分：分点/分段合理，逻辑顺序清楚，便于学生跟随
- 4分：结构较好，个别处可优化顺序
- 3分：有基本分段但逻辑跳跃
- 2分：结构混乱，难以提取关键信息
- 1分：无结构可言，一段到底或完全无序

### 5. 接地性与安全性 (Groundedness & Safety) — 是否引用上下文、给出安全提醒
- 5分：明确引用给定知识/上下文，危险场景有断电/安全提示
- 4分：有引用和安全意识，但不全面
- 3分：模糊提及上下文，安全提示缺失
- 2分：几乎不引用上下文，全靠泛泛而谈
- 1分：完全脱离上下文凭空发挥

## 输出格式

严格按以下 JSON 格式输出，不要有任何额外文字：

```json
{
  "correctness": <1-5>,
  "pedagogy": <1-5>,
  "conciseness": <1-5>,
  "format": <1-5>,
  "groundedness": <1-5>,
  "overall": <1-5>,
  "brief_reason": "<一句话总结扣分/得分原因>",
  "key_strength": "<这个回答最值得肯定的一个点>",
  "key_weakness": "<最需要改进的一个点>"
}
```
"""


def build_judge_prompt(
    question: dict[str, Any],
    answer: str,
    expected_points: list[str] | None = None,
) -> str:
    parts: list[str] = [SCORING_RUBRIC, "", "## 待评分内容", ""]
    parts.append(f"### 问题类型：{question.get('intent', 'unknown')}")
    parts.append(f"### 电路主题：{question.get('topology', 'unknown')}")
    parts.append("")
    parts.append(f"### 学生提问：\n{question['question']}")
    parts.append("")
    if expected_points:
        pts = "\n".join(f"- {p}" for p in expected_points)
        parts.append(f"### 预期回答要点（参考）：\n{pts}")
        parts.append("")
    parts.append(f"### 学生模型回答：\n{answer}")
    parts.append("")
    parts.append("请根据以上评分标准，对该回答进行 5 维打分。只输出 JSON，不要其他文字。")
    return "\n".join(parts)


# ─── Judge backends ────────────────────────────────────────────────────────


class JudgeBackend:
    def score(self, prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class DeepSeekJudge(JudgeBackend):
    def __init__(self, api_key: str = "", base_url: str = "", model: str = ""):
        self.api_key = api_key or os.environ.get(
            "LLM_API_KEY", os.environ.get("DEEPSEEK_API_KEY", "")
        )
        self.base_url = base_url or os.environ.get(
            "LLM_BASE_URL", "https://api.deepseek.com"
        )
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")

    def score(self, prompt: str) -> dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 512,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return _parse_json_response(resp.json()["choices"][0]["message"]["content"])


class OpenAICompatibleJudge(JudgeBackend):
    def __init__(
        self,
        api_key: str = "not-needed",
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen3-32B",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        self.base_url = base_url
        self.model = model

    def score(self, prompt: str) -> dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 512,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return _parse_json_response(resp.json()["choices"][0]["message"]["content"])


# ─── Response parsing ──────────────────────────────────────────────────────


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {
            "correctness": -1, "pedagogy": -1, "conciseness": -1,
            "format": -1, "groundedness": -1, "overall": -1,
            "brief_reason": f"PARSE_ERROR: {text[:200]}",
            "key_strength": "", "key_weakness": "",
        }


# ─── Batch scoring ─────────────────────────────────────────────────────────


def score_answers(
    judge: JudgeBackend,
    answers: list[dict[str, Any]],
    delay: float = 0.5,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    n = len(answers)

    for i, ans in enumerate(answers, start=1):
        qid = ans.get("qid", f"unknown_{i}")
        expected = ans.get("expected_points")

        prompt = build_judge_prompt(
            question={
                "question": ans.get("question", ""),
                "intent": ans.get("intent", "unknown"),
                "topology": ans.get("topology", "unknown"),
            },
            answer=ans.get("response", ans.get("error", "[NO RESPONSE]")),
            expected_points=expected if isinstance(expected, list) else None,
        )

        try:
            scores = judge.score(prompt)
            scored.append({**ans, "auto_scores": scores})
            overall = scores.get("overall", "?")
            print(f"  [{i:02d}/{n:02d}] {qid:<12} overall={overall}  "
                  f"{scores.get('brief_reason', '')[:70]}")
        except Exception as exc:
            print(f"  [{i:02d}/{n:02d}] {qid:<12} FAIL: {exc}")
            scored.append({**ans, "auto_scores": {"error": str(exc)[:300]}})

        time.sleep(delay)

    return scored


# ─── Summary computation ───────────────────────────────────────────────────


def compute_summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        s["auto_scores"]
        for s in scored
        if "auto_scores" in s and "error" not in s["auto_scores"]
        and s["auto_scores"].get("correctness", -1) >= 0
    ]
    if not valid:
        return {"error": "No valid scores", "total_answers": len(scored)}

    dims = ["correctness", "pedagogy", "conciseness", "format", "groundedness", "overall"]
    summary: dict[str, Any] = {
        "total_answers": len(scored),
        "valid_scores": len(valid),
        "parse_failures": len(scored) - len(valid),
    }

    for dim in dims:
        values = [s[dim] for s in valid if dim in s]
        if values:
            summary[f"{dim}_mean"] = round(sum(values) / len(values), 2)
            summary[f"{dim}_min"] = min(values)
            summary[f"{dim}_max"] = max(values)

    overalls = [s.get("overall", 0) for s in valid]
    summary["pass_rate_pct"] = round(
        100 * sum(1 for o in overalls if o >= 3) / len(overalls), 1)
    summary["full_score_rate_pct"] = round(
        100 * sum(1 for o in overalls if o == 5) / len(overalls), 1)

    by_intent: dict[str, list[dict]] = {}
    for s in scored:
        if "auto_scores" in s and "error" not in s["auto_scores"]:
            intent = s.get("intent", "unknown")
            by_intent.setdefault(intent, []).append(s["auto_scores"])

    summary["by_intent"] = {}
    for intent, scores in by_intent.items():
        vals = [s.get("overall", 0) for s in scores]
        summary["by_intent"][intent] = {
            "count": len(vals),
            "mean_overall": round(sum(vals) / len(vals), 2),
            "pass_rate_pct": round(
                100 * sum(1 for v in vals if v >= 3) / len(vals), 1),
        }

    return summary


def build_markdown_report(
    model_label: str,
    scored: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"# Auto-Scored Eval Report — {model_label}\n\n")
    lines.append(f"- total answers: {summary.get('total_answers', '?')}\n")
    lines.append(f"- valid scores: {summary.get('valid_scores', '?')}\n")
    lines.append(f"- parse failures: {summary.get('parse_failures', '?')}\n\n")

    lines.append("## Overall Scores (1-5 scale)\n\n")
    dims = ["correctness", "pedagogy", "conciseness", "format", "groundedness", "overall"]
    lines.append("| Dimension | Mean | Min | Max |\n")
    lines.append("|---|---|---|---:|\n")
    for dim in dims:
        mean = summary.get(f"{dim}_mean", "?")
        mn = summary.get(f"{dim}_min", "?")
        mx = summary.get(f"{dim}_max", "?")
        lines.append(f"| {dim} | {mean} | {mn} | {mx} |\n")

    lines.append(f"\n- pass_rate (overall>=3): {summary.get('pass_rate_pct', '?')}%\n")
    lines.append(f"- full_score_rate (overall=5): {summary.get('full_score_rate_pct', '?')}%\n\n")

    lines.append("## By Intent\n\n")
    lines.append("| Intent | Count | Mean Overall | Pass Rate |\n")
    lines.append("|---|---|---:|\n")
    for intent, stats in summary.get("by_intent", {}).items():
        lines.append(
            f"| {intent} | {stats['count']} | {stats['mean_overall']} | "
            f"{stats['pass_rate_pct']}% |\n"
        )

    lines.append("\n---\n\n## Per-Question Details\n\n")
    for s in scored:
        lines.append(
            f"### {s.get('qid', '?')} · {s.get('intent', '?')} · "
            f"{s.get('topology', '?')}\n\n"
        )
        lines.append(f"**Question**: {s.get('question', '?')}\n\n")
        if "error" in s:
            lines.append(f"❌ Generation Error: {s['error']}\n\n")
        else:
            as_ = s.get("auto_scores", {})
            if "error" in as_:
                lines.append(f"❌ Scoring Error: {as_['error']}\n\n")
            else:
                lines.append(
                    f"**Response** (excerpt): {s.get('response', '')[:200]}...\n\n"
                )
                lines.append("| Dimension | Score |\n|---|---:|\n")
                for dim in dims:
                    lines.append(f"| {dim} | {as_.get(dim, '?')} |\n")
                lines.append(f"\n**Reason**: {as_.get('brief_reason', '')}\n")
                lines.append(f"\n**Strength**: {as_.get('key_strength', '')}\n")
                lines.append(f"\n**Weakness**: {as_.get('key_weakness', '')}\n")
        lines.append("\n---\n\n")

    return "".join(lines)


# ─── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated LLM-as-Judge scoring for model evaluation answers."
    )
    parser.add_argument("--input", type=Path,
                        help="Path to a JSON eval result file.")
    parser.add_argument("--output", type=Path,
                        help="Path for output JSON (default: <input>_auto_scored.json).")
    parser.add_argument("--judge", choices=["openai", "deepseek", "local_qwen"],
                        default="deepseek", help="Which LLM to use as judge.")
    parser.add_argument("--judge-url", default="",
                        help="Custom API base URL for the judge model.")
    parser.add_argument("--judge-model", default="",
                        help="Custom model name for the judge.")
    parser.add_argument("--batch", action="store_true",
                        help="Score all P0 model outputs found in outputs/llm_eval/.")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay in seconds between API calls (default: 0.5).")
    return parser.parse_args()


def _resolve_judge(args: argparse.Namespace) -> JudgeBackend:
    if args.judge == "deepseek":
        return DeepSeekJudge(
            base_url=args.judge_url or "",
            model=args.judge_model or "",
        )
    elif args.judge == "openai":
        return OpenAICompatibleJudge(
            base_url=args.judge_url or "https://api.openai.com/v1",
            model=args.judge_model or "gpt-4o",
        )
    elif args.judge == "local_qwen":
        return OpenAICompatibleJudge(
            base_url=args.judge_url or "http://localhost:8000/v1",
            model=args.judge_model or "Qwen3-32B",
        )
    raise ValueError(f"Unknown judge: {args.judge}")


def _batch_inputs() -> list[tuple[str, Path]]:
    eval_dir = DEFAULT_OUTPUT_DIR
    if not eval_dir.exists():
        return []

    found: list[tuple[str, Path]] = []
    for path in sorted(eval_dir.glob("p0_*_eval_*.json")):
        name = path.name
        if any(skip in name for skip in (
            "_auto_scored", "_verifier_scored", "_manual_", "_summary",
            "_matrix", "_questions"
        )):
            continue
        label = name.replace("p0_", "").replace(".json", "")
        found.append((label, path))
    return found


def main() -> None:
    args = parse_args()
    judge = _resolve_judge(args)

    if args.batch:
        inputs = _batch_inputs()
        if not inputs:
            raise SystemExit("No P0 eval JSON files found in outputs/llm_eval/")
        print(f"Batch scoring {len(inputs)} model outputs with judge={args.judge}")
        for label, path in inputs:
            print(f"\n{'='*60}\nScoring: {label} ({path.name})\n{'='*60}")
            data = json.loads(path.read_text(encoding="utf-8"))
            answers = data.get("results", data)
            if isinstance(answers, dict):
                answers = answers.get("results", [])
            if not isinstance(answers, list):
                print(f"  SKIP: unexpected JSON structure in {path}")
                continue

            scored = score_answers(judge, answers, delay=args.delay)
            summary = compute_summary(scored)

            out_json = path.parent / f"{path.stem}_auto_scored.json"
            out_md = path.parent / f"{path.stem}_auto_scored.md"
            out_json.write_text(
                json.dumps(
                    {"model_label": label, "summary": summary, "results": scored},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
            out_md.write_text(
                build_markdown_report(label, scored, summary), encoding="utf-8")
            print(f"  Saved: {out_json.name}, {out_md.name}")
        return

    if not args.input:
        raise SystemExit("Either --input or --batch is required.")

    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    answers = data.get("results", data)
    if isinstance(answers, dict):
        answers = answers.get("results", [])
    if not isinstance(answers, list):
        raise SystemExit(
            "Input JSON must contain a 'results' array or be a plain array.")

    label = input_path.stem
    print(f"Scoring {len(answers)} answers from {label} with judge={args.judge}")
    scored = score_answers(judge, answers, delay=args.delay)
    summary = compute_summary(scored)

    out_json = args.output or input_path.parent / f"{input_path.stem}_auto_scored.json"
    out_md = out_json.with_suffix(".md")
    out_json.write_text(
        json.dumps(
            {"model_label": label, "summary": summary, "results": scored},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    out_md.write_text(build_markdown_report(label, scored, summary), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Summary for {label}:")
    dims = ["correctness", "pedagogy", "conciseness", "format", "groundedness", "overall"]
    for dim in dims:
        print(f"  {dim:<16} mean={summary.get(f'{dim}_mean', '?'):.2f}")
    print(f"  pass_rate:        {summary.get('pass_rate_pct', '?')}%")
    print(f"  full_score_rate:  {summary.get('full_score_rate_pct', '?')}%")
    print(f"\nSaved: {out_json}, {out_md}")


if __name__ == "__main__":
    main()
