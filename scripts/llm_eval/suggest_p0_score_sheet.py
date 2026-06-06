from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a draft/manual-review P0 score sheet from eval outputs."
    )
    parser.add_argument("--eval-json", type=Path, required=True, help="P0 eval JSON output.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Draft score-sheet CSV path.")
    parser.add_argument("--output-md", type=Path, help="Optional markdown review summary path.")
    return parser.parse_args()


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    return text


def _contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _match_groups(text: str, groups: list[list[str]]) -> list[int]:
    hits: list[int] = []
    for group in groups:
        hits.append(1 if _contains_any(text, group) else 0)
    return hits


def _rule_rc_01(text: str) -> tuple[list[int], str]:
    hits = [
        1 if ("τ=rc" in text or "tau=rc" in text or _contains_all(text, ["时间常数", "rc"])) else 0,
        1 if ("ω·f=[s]" in text or "ω·f=s" in text or "ω·f=[s]" in text or "ω·f=[秒]" in text or "[v/a]·[c/v]=[s]" in text or "欧姆" in text and "法拉" in text and "秒" in text) else 0,
        1 if _contains_any(text, ["充电", "放电", "指数", "e^{-t/rc}", "63.2%"]) else 0,
    ]
    return hits, ""


def _rule_rc_02(text: str) -> tuple[list[int], str]:
    wrong = _contains_any(text, ["τ太小", "tau太小"])
    hits = [
        1 if _contains_any(text, ["τ太大", "tau太大", "远大于输入周期", "大于输入周期"]) and not wrong else 0,
        1 if _contains_any(text, ["τ<<t", "tau<<t", "τ<<周期", "应满足τ<<t", "应满足tau<<t"]) else 0,
        1 if _contains_any(text, ["减小r", "减小c", "减小电阻", "减小电容", "使τ降低", "使tau降低"]) and not _contains_any(text, ["τ近似等于输入信号周期"]) else 0,
    ]
    note = "疑似方向性错误" if wrong else ""
    return hits, note


def _rule_ce_02(text: str) -> tuple[list[int], str]:
    wrong = _contains_any(text, ["饱和区", "不是截止", "进入饱和"])
    hits = [
        1 if _contains_any(text, ["截止", "vce≈vcc说明截止", "vce=11v说明截止"]) and not wrong else 0,
        1 if _contains_any(text, ["基极偏置", "检查vb", "检查vbe", "rb1", "rb2"]) and not wrong else 0,
        1 if _contains_any(text, ["vb", "vbe", "0.7v"]) and not wrong else 0,
    ]
    note = "把截止答成饱和" if wrong else ""
    return hits, note


def _rule_diff_03(text: str) -> tuple[list[int], str]:
    hits = [
        1 if _contains_any(text, ["反相", "+vin", "-vin", "差模"]) else 0,
        1 if _contains_any(text, ["同相同幅", "共模", "同时接两个输入端"]) else 0,
        1 if _contains_any(text, ["ad", "ac", "vout/vid", "vout/vic"]) else 0,
    ]
    return hits, ""


def _rule_ua_inv_01(text: str) -> tuple[list[int], str]:
    hits = [
        1 if _contains_any(text, ["反相", "180°", "180度"]) else 0,
        1 if _contains_any(text, ["虚地", "虚短", "v-=0", "v-=0"]) else 0,
        1 if _contains_any(text, ["(vin-0)/rin=(0-vout)/rf", "av=-rf/rin", "-rf/rin"]) else 0,
    ]
    return hits, ""


def _rule_ua_inv_02(text: str) -> tuple[list[int], str]:
    hits = [
        1 if _contains_any(text, ["pin7", "pin4", "电源脚", "+vcc", "-vee"]) else 0,
        1 if _contains_any(text, ["v+", "pin3", "接地", "gnd"]) else 0,
        1 if _contains_any(text, ["rf", "pin6", "pin2", "输出回到反相端"]) else 0,
    ]
    note = "过度归因增益设置" if _contains_any(text, ["增益设置错误"]) else ""
    return hits, note


def _rule_ua_int_01(text: str) -> tuple[list[int], str]:
    hits = [
        1 if _contains_any(text, ["1/rc", "积分", "∫", "vin/r=-c", "vout(t)"]) else 0,
        1 if _contains_any(text, ["反馈电容", "c是反馈", "电容c"]) else 0,
        1 if _contains_any(text, ["r_leak", "泄放", "偏置电流", "防漂移"]) else 0,
    ]
    return hits, ""


def _rule_generic(result: dict[str, Any]) -> tuple[list[int], str]:
    text = _normalize(str(result.get("response", "")))
    groups = []
    for expected in result.get("expected_points", []):
        normalized = _normalize(str(expected))
        tokens = [token for token in re.split(r"[，、；或/()\s]+", normalized) if len(token) >= 2]
        if not tokens:
            groups.append([normalized])
        else:
            groups.append(tokens[:3])
    hits = _match_groups(text, groups[:3])
    return hits, ""


CUSTOM_RULES: dict[str, Any] = {
    "rc_01": _rule_rc_01,
    "rc_02": _rule_rc_02,
    "ce_02": _rule_ce_02,
    "diff_03": _rule_diff_03,
    "ua_inv_01": _rule_ua_inv_01,
    "ua_inv_02": _rule_ua_inv_02,
    "ua_int_01": _rule_ua_int_01,
}


def suggest_hits(result: dict[str, Any]) -> tuple[list[int], str]:
    text = _normalize(str(result.get("response", "")))
    rule = CUSTOM_RULES.get(str(result.get("qid", "")))
    if rule is not None:
        return rule(text)
    return _rule_generic(result)


def score_band(score: int) -> str:
    if score >= 3:
        return "明显通过"
    if score == 2:
        return "部分通过"
    return "明显失败"


def build_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        hits, note = suggest_hits(result)
        while len(hits) < 3:
            hits.append(0)
        score = sum(hits[:3])
        rows.append(
            {
                "qid": result["qid"],
                "source": result.get("source", ""),
                "scene_id": result.get("scene_id", ""),
                "intent": result.get("intent", ""),
                "topology": result.get("topology", ""),
                "risk_level": result.get("risk_level", ""),
                "question": result.get("question", ""),
                "expected_point_1": (result.get("expected_points") or ["", "", ""])[0] if len(result.get("expected_points") or []) > 0 else "",
                "expected_point_2": (result.get("expected_points") or ["", "", ""])[1] if len(result.get("expected_points") or []) > 1 else "",
                "expected_point_3": (result.get("expected_points") or ["", "", ""])[2] if len(result.get("expected_points") or []) > 2 else "",
                "suggested_hit_1": hits[0],
                "suggested_hit_2": hits[1],
                "suggested_hit_3": hits[2],
                "suggested_score": score,
                "suggested_band": score_band(score),
                "suggested_notes": note,
                "manual_hit_1": "",
                "manual_hit_2": "",
                "manual_hit_3": "",
                "manual_score": "",
                "manual_notes": "",
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
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
        "suggested_hit_1",
        "suggested_hit_2",
        "suggested_hit_3",
        "suggested_score",
        "suggested_band",
        "suggested_notes",
        "manual_hit_1",
        "manual_hit_2",
        "manual_hit_3",
        "manual_score",
        "manual_notes",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P0 Score Suggestion Draft",
        "",
        "| QID | Intent | Score | Band | Notes |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['qid']} | {row['intent']} | {row['suggested_score']} | {row['suggested_band']} | {row['suggested_notes']} |"
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    eval_json = args.eval_json.resolve()
    if not eval_json.exists():
        raise SystemExit(f"Eval JSON not found: {eval_json}")

    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise SystemExit("Eval JSON missing top-level results list.")

    rows = build_rows(results)
    write_csv(rows, args.output_csv.resolve())
    if args.output_md:
        write_markdown(rows, args.output_md.resolve())

    bands: dict[str, int] = {"明显通过": 0, "部分通过": 0, "明显失败": 0}
    for row in rows:
        bands[row["suggested_band"]] += 1
    print("draft bands:", bands)
    print(f"wrote {args.output_csv}")
    if args.output_md:
        print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
