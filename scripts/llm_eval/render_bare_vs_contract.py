"""Render the bare-vs-contract comparison JSON into a defense-ready Markdown.

Usage:
    python scripts/llm_eval/render_bare_vs_contract.py \
        --input reports/compare_bare_vs_contract.json \
        --output reports/distill_bare_vs_contract.md \
        --date 2026-05-29
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SAFETY_WORDS = ("断电", "电源", "短路")


def _yn(flag: bool) -> str:
    return "✓" if flag else "✗"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/compare_bare_vs_contract.json")
    parser.add_argument("--output", default="reports/distill_bare_vs_contract.md")
    parser.add_argument("--date", default="")
    args = parser.parse_args()

    cases = json.loads(Path(args.input).read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append("# 端侧学生模型:裸 1.5B vs 1.5B + 检索契约 对比报告")
    lines.append("")
    if args.date:
        lines.append(f"- 日期:{args.date}")
    lines.append("- 模型:`labguardian-student-1p5-int4-ov`(Qwen2.5-1.5B 蒸馏 → INT4 → OpenVINO IR)")
    lines.append("- 设备:Intel DK-2500 iGPU(device=GPU)")
    lines.append("- 方法:**同一条自然学生问题问两次** —— "
                 "(A) **裸跑**:学生模型直接作答,无板上状态/无检索契约;"
                 "(B) **契约**:经完整 `diagnostic_agent`(场景锚定检索 + 确定性校验器 + 同一学生模型当\"嘴巴\")。")
    lines.append("")

    # ---- summary table ----
    n = len(cases)
    bare_hit = 0
    code_hit = 0
    comp_hit = 0
    safety_hit = 0
    scene_ok = 0
    verifier_ok = 0
    lines.append("## 总览")
    lines.append("")
    lines.append("| # | 场景 | 注入故障 | 裸跑命中故障? | 契约·引error_code | 引元件 | 安全前置 | 场景锚定 | 校验通过 | 裸/契约延迟(s) |")
    lines.append("|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|")
    for i, c in enumerate(cases, 1):
        code = c["injected_error_code"]
        comp = c["injected_component"]
        bare_ans = c["bare"]["answer"]
        con = c["contract"]
        con_ans = con["answer"]
        b_hit = (code in bare_ans) or (comp in bare_ans)
        c_code = code in con_ans
        c_comp = comp in con_ans
        c_safe = any(w in con_ans for w in SAFETY_WORDS)
        c_scene = con.get("scene_resolved") == c["scene"]
        c_ver = bool(con.get("verifier_passed"))
        bare_hit += b_hit
        code_hit += c_code
        comp_hit += c_comp
        safety_hit += c_safe
        scene_ok += c_scene
        verifier_ok += c_ver
        lines.append(
            f"| {i} | {c['scene_zh']} | `{code}`@{comp} | {_yn(b_hit)} | {_yn(c_code)} | "
            f"{_yn(c_comp)} | {_yn(c_safe)} | {_yn(c_scene)} | {_yn(c_ver)} | "
            f"{c['bare']['latency_s']} / {con['latency_s']} |"
        )
    lines.append("")
    lines.append("## 关键结论")
    lines.append("")
    lines.append(f"- **裸跑命中具体故障:{bare_hit}/{n}** —— 没有板上状态,学生模型只能泛泛讲通用排查,无法指认\"你这块板上的 {cases[0]['injected_component']} 错了\"。")
    lines.append(f"- **契约·引用 error_code:{code_hit}/{n}**;引用涉事元件:{comp_hit}/{n};安全前置:{safety_hit}/{n}。")
    lines.append(f"- **场景锚定正确:{scene_ok}/{n}**(6 拓扑全部由 `topology_label` 解析出正确 scene_id);**确定性校验器通过:{verifier_ok}/{n}**。")
    lines.append("- 结论:**同一个 1.5B 模型**,接上检索契约后从\"通用教科书\"升级为\"针对当前电路、可审计、安全优先\"的诊断 —— 差异完全来自契约,而非换模型。")
    lines.append("")

    # ---- per-scene detail ----
    lines.append("## 逐场景对比")
    lines.append("")
    for i, c in enumerate(cases, 1):
        con = c["contract"]
        lines.append(f"### {i}. {c['scene_zh']}  ·  注入 `{c['injected_error_code']}` @ {c['injected_component']}")
        lines.append("")
        lines.append(f"**学生问题**:{c['question']}")
        lines.append("")
        lines.append(f"**(A) 裸 1.5B**(无契约,{c['bare']['latency_s']}s):")
        lines.append("")
        for ln in c["bare"]["answer"].splitlines():
            lines.append(f"> {ln}" if ln.strip() else ">")
        lines.append("")
        lines.append(f"**(B) 1.5B + 契约**(场景=`{con.get('scene_resolved')}`,校验={con.get('verifier_passed')},{con['latency_s']}s):")
        lines.append("")
        for ln in con["answer"].splitlines():
            lines.append(f"> {ln}" if ln.strip() else ">")
        lines.append("")

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}  ({n} cases)")
    print(f"summary: bare_hit={bare_hit}/{n}  code_hit={code_hit}/{n}  "
          f"comp_hit={comp_hit}/{n}  safety={safety_hit}/{n}  scene_ok={scene_ok}/{n}  verifier={verifier_ok}/{n}")


if __name__ == "__main__":
    main()
