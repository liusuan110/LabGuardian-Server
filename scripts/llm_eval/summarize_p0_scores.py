from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize manual P0 score-sheet results into JSON and Markdown."
    )
    parser.add_argument("--score-sheet", type=Path, required=True, help="CSV score sheet to read.")
    parser.add_argument("--output-json", type=Path, help="Optional JSON summary output path.")
    parser.add_argument("--output-md", type=Path, help="Optional Markdown summary output path.")
    return parser.parse_args()


def _parse_hit(value: str) -> bool | None:
    text = value.strip().lower()
    if not text:
        return None
    if text in {"1", "y", "yes", "true", "t", "hit"}:
        return True
    if text in {"0", "n", "no", "false", "f", "miss"}:
        return False
    return None


def _parse_score(row: dict[str, str]) -> tuple[float | None, int]:
    max_points = sum(1 for key in ("expected_point_1", "expected_point_2", "expected_point_3") if row.get(key, "").strip())
    if max_points == 0:
        max_points = 3

    manual_score = row.get("manual_score", "").strip()
    if manual_score:
        try:
            return float(manual_score), max_points
        except ValueError:
            pass

    hits = [_parse_hit(row.get(key, "")) for key in ("hit_1", "hit_2", "hit_3")]
    if any(hit is not None for hit in hits):
        return float(sum(1 for hit in hits if hit)), max_points
    return None, max_points


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["score"] is not None]
    payload: dict[str, Any] = {
        "question_count": len(rows),
        "completed_count": len(completed),
        "completion_rate_pct": round(100.0 * len(completed) / len(rows), 2) if rows else 0.0,
    }
    if not completed:
        payload.update(
            {
                "avg_score": None,
                "avg_score_pct": None,
                "pass_rate_pct": None,
                "full_score_rate_pct": None,
            }
        )
        return payload

    avg_score = mean(row["score"] for row in completed)
    avg_score_pct = mean(100.0 * row["score"] / row["max_points"] for row in completed)
    pass_rate_pct = 100.0 * sum(1 for row in completed if row["score"] >= 2.0) / len(completed)
    full_score_rate_pct = (
        100.0 * sum(1 for row in completed if row["score"] >= row["max_points"]) / len(completed)
    )
    payload.update(
        {
            "avg_score": round(avg_score, 3),
            "avg_score_pct": round(avg_score_pct, 2),
            "pass_rate_pct": round(pass_rate_pct, 2),
            "full_score_rate_pct": round(full_score_rate_pct, 2),
        }
    )
    return payload


def load_rows(score_sheet: Path) -> list[dict[str, Any]]:
    with score_sheet.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            score, max_points = _parse_score(row)
            rows.append(
                {
                    **row,
                    "score": score,
                    "max_points": max_points,
                }
            )
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "overall": _summarize_group(rows),
        "by_source": {},
        "by_intent": {},
        "by_topology": {},
        "by_risk_level": {},
        "unscored_qids": [row["qid"] for row in rows if row["score"] is None],
    }

    for field in ("source", "intent", "topology", "risk_level"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row.get(field, "") or "unknown"].append(row)
        summary[f"by_{field}"] = {
            key: _summarize_group(group_rows)
            for key, group_rows in sorted(grouped.items())
        }

    return summary


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# P0 Manual Score Summary",
        "",
        "## Overall",
        "",
    ]
    overall = summary["overall"]
    lines.extend(
        [
            f"- question_count: {overall['question_count']}",
            f"- completed_count: {overall['completed_count']}",
            f"- completion_rate_pct: {overall['completion_rate_pct']}",
            f"- avg_score: {overall['avg_score']}",
            f"- avg_score_pct: {overall['avg_score_pct']}",
            f"- pass_rate_pct: {overall['pass_rate_pct']}",
            f"- full_score_rate_pct: {overall['full_score_rate_pct']}",
            "",
        ]
    )

    for field in ("source", "intent", "topology", "risk_level"):
        lines.append(f"## By {field}")
        lines.append("")
        lines.append("| Group | Questions | Completed | Avg Score % | Pass % | Full Score % |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for key, payload in summary[f"by_{field}"].items():
            lines.append(
                "| {key} | {question_count} | {completed_count} | {avg_score_pct} | {pass_rate_pct} | {full_score_rate_pct} |".format(
                    key=key,
                    question_count=payload["question_count"],
                    completed_count=payload["completed_count"],
                    avg_score_pct=payload["avg_score_pct"],
                    pass_rate_pct=payload["pass_rate_pct"],
                    full_score_rate_pct=payload["full_score_rate_pct"],
                )
            )
        lines.append("")

    lines.append("## Pending")
    lines.append("")
    pending = summary["unscored_qids"]
    if pending:
        lines.append(f"- unscored_count: {len(pending)}")
        lines.append(f"- unscored_qids: {', '.join(pending)}")
    else:
        lines.append("- unscored_count: 0")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    score_sheet = args.score_sheet.resolve()
    if not score_sheet.exists():
        raise SystemExit(f"Score sheet not found: {score_sheet}")

    rows = load_rows(score_sheet)
    summary = build_summary(rows)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.output_json}")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(build_markdown(summary), encoding="utf-8")
        print(f"wrote {args.output_md}")

    overall = summary["overall"]
    print(
        "overall:",
        {
            "question_count": overall["question_count"],
            "completed_count": overall["completed_count"],
            "avg_score_pct": overall["avg_score_pct"],
            "pass_rate_pct": overall["pass_rate_pct"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
