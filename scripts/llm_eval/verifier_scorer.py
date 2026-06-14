"""Rule-based automated scoring using the existing LabGuardian verifier.

Leverages deterministic rules (safety, fault keywords, component references,
expected point hits, structure) to score answers without any LLM judge.

Usage:
    python scripts/llm_eval/verifier_scorer.py \
      --input outputs/llm_eval/p0_student_eval_final.json

    python scripts/llm_eval/verifier_scorer.py --batch
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "llm_eval"


def _keyword_overlap(text: str, keywords: tuple[str, ...]) -> int:
    """Count how many keyword groups appear in text."""
    return sum(1 for kw in keywords if kw in text)


def _expected_point_match(expected_text: str, answer: str) -> bool:
    """Check if an expected point is roughly covered in the answer."""
    # Extract core tokens (Chinese characters + alphanumeric) from expected
    core = re.sub(r"[^一-鿿\w]", "", expected_text)
    ans_core = re.sub(r"[^一-鿿\w]", "", answer)
    if len(core) < 4:
        return core in ans_core
    # Sliding 3-char overlap check
    hits = 0
    for i in range(0, len(core) - 2, 3):
        chunk = core[i:i + 3]
        if chunk in ans_core:
            hits += 1
    return hits >= max(1, len(core) // 5)


def _score_diagnostic(answer: str, question: dict[str, Any]) -> dict[str, Any]:
    text = answer or ""
    risk = question.get("risk_level", "warning")
    checks: list[dict[str, Any]] = []

    # 1. Safety check
    safety_words = ("断电", "电源", "短路", "关闭电源", "先关", "安全")
    has_safety = any(w in text for w in safety_words)
    checks.append({
        "name": "safety_precaution",
        "pass": has_safety,
        "required": risk == "danger",
    })

    # 2. Fault diagnosis keywords
    fault_kw = ("截止", "饱和", "短路", "开路", "悬空", "极性", "偏置",
                "接错", "缺失", "不匹配", "漂移", "失真", "故障", "错误")
    has_fault = any(kw in text for kw in fault_kw)
    checks.append({"name": "fault_diagnosis_present", "pass": has_fault})

    # 3. Expected points
    expected = question.get("expected_points", [])
    if expected and isinstance(expected, list):
        hits = sum(1 for ep in expected if _expected_point_match(ep, text))
        checks.append({
            "name": "expected_points_hit",
            "pass": hits >= 1,
            "value": f"{hits}/{len(expected)}",
        })

    # 4. Component/pin reference
    comp_patterns = ("R_", "C_", "pin", "V_", "VCC", "GND", "管", "脚",
                     "基极", "集电极", "发射极", "反相端", "同相端",
                     "pin7", "pin4", "pin3", "pin2", "pin6")
    has_comp = any(p in text for p in comp_patterns)
    checks.append({"name": "component_reference", "pass": has_comp})

    # 5. Actionable suggestion
    action_kw = ("检查", "测量", "调整", "更换", "确认", "排查", "减小",
                 "增大", "替换", "万用表", "示波器", "重新", "改")
    has_action = any(kw in text for kw in action_kw)
    checks.append({"name": "actionable_suggestion", "pass": has_action})

    pass_count = sum(1 for c in checks if c["pass"])
    penalty = sum(1 for c in checks if c.get("penalty"))
    total = max(0, pass_count - penalty)
    return {
        "total_points": total,
        "max_points": len(checks),
        "score_pct": round(100 * total / len(checks), 1),
        "checks": checks,
    }


def _score_concept_tutor(answer: str, question: dict[str, Any]) -> dict[str, Any]:
    text = answer or ""
    checks: list[dict[str, Any]] = []

    # 1. Explains why
    why_kw = ("因为", "由于", "原因", "原理", "公式", "推导", "因此",
              "所以", "关系", "定律", "作用", "表示")
    has_why = any(kw in text for kw in why_kw)
    checks.append({"name": "explains_why_not_just_what", "pass": has_why})

    # 2. Formula or quantitative
    formula_kw = ("=", "τ", "A_v", "R_f", "R_in", "1/", "f_L", "dB",
                  "V_out", "V_in", "Ω", "VCC", "GND", "RC", "C_E")
    has_formula = any(kw in text for kw in formula_kw)
    checks.append({"name": "quantitative_or_formula", "pass": has_formula})

    # 3. Expected points
    expected = question.get("expected_points", [])
    if expected and isinstance(expected, list):
        hits = sum(1 for ep in expected if _expected_point_match(ep, text))
        checks.append({
            "name": "expected_points_hit",
            "pass": hits >= 1,
            "value": f"{hits}/{len(expected)}",
        })

    # 4. Structure
    struct_kw = ("首先", "其次", "最后", "综上", "总结", "1.", "2.", "3.",
                 "一是", "二是", "三是", "第一", "第二", "第三")
    has_struct = any(kw in text for kw in struct_kw)
    checks.append({"name": "structured_explanation", "pass": has_struct})

    # 5. Safety awareness
    safety_kw = ("安全", "注意", "小心", "断电", "勿")
    has_safety = any(kw in text for kw in safety_kw)
    checks.append({"name": "safety_awareness", "pass": has_safety, "required": False})

    pass_count = sum(1 for c in checks if c["pass"])
    return {
        "total_points": pass_count,
        "max_points": len(checks),
        "score_pct": round(100 * pass_count / len(checks), 1) if checks else 0,
        "checks": checks,
    }


def _score_lab_guidance(answer: str, question: dict[str, Any]) -> dict[str, Any]:
    text = answer or ""
    checks: list[dict[str, Any]] = []

    # 1. Numbered steps
    step_kw = ("1.", "2.", "3.", "第一步", "第二步", "第三步",
               "①", "②", "③", "首先", "然后", "最后", "步骤")
    has_steps = sum(1 for m in step_kw if m in text) >= 2
    checks.append({"name": "numbered_sequential_steps", "pass": has_steps})

    # 2. Safety
    safety_kw = ("断电", "电源", "安全", "小心", "注意", "先关")
    has_safety = any(w in text for w in safety_kw)
    checks.append({"name": "safety_precaution", "pass": has_safety})

    # 3. Instrument mentions
    instrument_kw = ("万用表", "示波器", "信号源", "电源", "探头", "通道")
    has_instruments = any(w in text for w in instrument_kw)
    checks.append({"name": "instrument_guidance", "pass": has_instruments})

    # 4. Expected points
    expected = question.get("expected_points", [])
    if expected and isinstance(expected, list):
        hits = sum(1 for ep in expected if _expected_point_match(ep, text))
        checks.append({
            "name": "expected_points_hit",
            "pass": hits >= 1,
            "value": f"{hits}/{len(expected)}",
        })

    # 5. Expected observation
    observe_kw = ("观察", "应该看到", "正常", "异常", "判断", "确认",
                  "波形", "电压", "显示")
    has_observe = any(w in text for w in observe_kw)
    checks.append({"name": "expected_observation", "pass": has_observe})

    pass_count = sum(1 for c in checks if c["pass"])
    return {
        "total_points": pass_count,
        "max_points": len(checks),
        "score_pct": round(100 * pass_count / len(checks), 1) if checks else 0,
        "checks": checks,
    }


SCORERS = {
    "diagnostic": _score_diagnostic,
    "concept_tutor": _score_concept_tutor,
    "lab_guidance": _score_lab_guidance,
    "mixed": _score_diagnostic,
}


def score_answer(answer: str, question: dict[str, Any]) -> dict[str, Any]:
    intent = question.get("intent", "diagnostic")
    scorer = SCORERS.get(intent, _score_diagnostic)
    result = scorer(answer, question)
    result["intent"] = intent
    return result


def compute_summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [s for s in scored if "verifier_score" in s
             and "error" not in s["verifier_score"]]
    if not valid:
        return {"error": "No valid scores", "total_answers": len(scored)}

    pcts = [s["verifier_score"]["score_pct"] for s in valid]
    summary = {
        "total_answers": len(scored),
        "valid_scores": len(valid),
        "mean_score_pct": round(sum(pcts) / len(pcts), 1),
        "min_score_pct": min(pcts),
        "max_score_pct": max(pcts),
        "pass_rate_pct": round(
            100 * sum(1 for p in pcts if p >= 60) / len(pcts), 1),
    }

    by_intent: dict[str, list[float]] = {}
    for s in valid:
        intent = s.get("intent", "unknown")
        by_intent.setdefault(intent, []).append(
            s["verifier_score"]["score_pct"])

    summary["by_intent"] = {}
    for intent, pcts_i in by_intent.items():
        summary["by_intent"][intent] = {
            "count": len(pcts_i),
            "mean_score_pct": round(sum(pcts_i) / len(pcts_i), 1),
        }
    return summary


def build_markdown(scored: list[dict[str, Any]], summary: dict[str, Any],
                   model_label: str = "") -> str:
    lines = [f"# Verifier-Based Auto Score Report — {model_label}\n\n"]
    lines.append(f"- total: {summary.get('total_answers', '?')}\n")
    lines.append(f"- mean_score_pct: {summary.get('mean_score_pct', '?')}%\n")
    lines.append(f"- pass_rate_pct: {summary.get('pass_rate_pct', '?')}%\n\n")

    lines.append("## By Intent\n\n")
    for intent, stats in summary.get("by_intent", {}).items():
        lines.append(
            f"- {intent}: {stats['count']}q, mean {stats['mean_score_pct']}%\n")

    lines.append("\n---\n\n## Per-Question\n\n")
    for s in scored:
        lines.append(f"### {s.get('qid', '?')} · {s.get('intent', '?')}\n\n")
        vs = s.get("verifier_score", {})
        lines.append(
            f"**Score**: {vs.get('total_points', '?')}/{vs.get('max_points', '?')} "
            f"({vs.get('score_pct', '?')}%)\n\n")
        for check in vs.get("checks", []):
            icon = "✅" if check.get("pass") else "❌"
            req = " [REQUIRED]" if check.get("required") else ""
            val = f" -> {check.get('value', '')}" if check.get("value") else ""
            lines.append(f"- {icon} {check['name']}{req}{val}\n")
        lines.append("\n---\n\n")

    return "".join(lines)


def _batch_inputs() -> list[tuple[str, Path]]:
    eval_dir = DEFAULT_OUTPUT_DIR
    if not eval_dir.exists():
        return []
    found: list[tuple[str, Path]] = []
    for path in sorted(eval_dir.glob("p0_*_eval_*.json")):
        name = path.name
        if any(skip in name for skip in (
            "_auto_scored", "_verifier_scored", "_manual_",
            "_summary", "_matrix", "_questions"
        )):
            continue
        label = name.replace("p0_", "").replace(".json", "")
        found.append((label, path))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule-based verifier scoring for model evaluation answers.")
    parser.add_argument("--input", type=Path,
                        help="Path to eval JSON result file.")
    parser.add_argument("--output", type=Path,
                        help="Output path (default: <input>_verifier_scored.json).")
    parser.add_argument("--batch", action="store_true",
                        help="Score all P0 eval JSON files.")
    args = parser.parse_args()

    if args.batch:
        inputs = _batch_inputs()
        if not inputs:
            raise SystemExit("No P0 eval JSON files found.")
        for label, path in inputs:
            print(f"Scoring: {label}")
            data = json.loads(path.read_text(encoding="utf-8"))
            answers = data.get("results", data)
            if isinstance(answers, dict):
                answers = answers.get("results", [])
            if not isinstance(answers, list):
                print(f"  SKIP: {path.name}")
                continue

            scored = []
            for ans in answers:
                response = ans.get("response", "")
                if not response:
                    scored.append(
                        {**ans, "verifier_score": {"error": "no response"}})
                    continue
                ctx = {
                    "intent": ans.get("intent", "diagnostic"),
                    "topology": ans.get("topology", ""),
                    "risk_level": ans.get("risk_level", "warning"),
                    "expected_points": ans.get("expected_points", []),
                }
                vs = score_answer(response, ctx)
                scored.append({**ans, "verifier_score": vs})

            summary = compute_summary(scored)
            out_json = path.parent / f"{path.stem}_verifier_scored.json"
            out_md = out_json.with_suffix(".md")
            out_json.write_text(
                json.dumps({
                    "model_label": label, "summary": summary, "results": scored,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            out_md.write_text(
                build_markdown(scored, summary, label), encoding="utf-8")
            print(f"  mean={summary.get('mean_score_pct', '?')}%  "
                  f"pass={summary.get('pass_rate_pct', '?')}%  "
                  f"-> {out_json.name}")
        return

    if not args.input:
        raise SystemExit("Either --input or --batch is required.")

    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    answers = data.get("results", data)
    if isinstance(answers, dict):
        answers = answers.get("results", [])

    scored = []
    for ans in answers:
        response = ans.get("response", "")
        if not response:
            scored.append({**ans, "verifier_score": {"error": "no response"}})
            continue
        ctx = {
            "intent": ans.get("intent", "diagnostic"),
            "topology": ans.get("topology", ""),
            "risk_level": ans.get("risk_level", "warning"),
            "expected_points": ans.get("expected_points", []),
        }
        vs = score_answer(response, ctx)
        scored.append({**ans, "verifier_score": vs})

    summary = compute_summary(scored)
    out_json = args.output or input_path.parent / f"{input_path.stem}_verifier_scored.json"
    out_md = out_json.with_suffix(".md")
    out_json.write_text(
        json.dumps({
            "model_label": input_path.stem, "summary": summary, "results": scored,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_md.write_text(
        build_markdown(scored, summary, input_path.stem), encoding="utf-8")

    print(f"Verifier scoring: {summary['valid_scores']} answers")
    print(f"  mean: {summary.get('mean_score_pct', '?')}%")
    print(f"  pass: {summary.get('pass_rate_pct', '?')}%")
    print(f"Saved: {out_json}, {out_md}")


if __name__ == "__main__":
    main()
