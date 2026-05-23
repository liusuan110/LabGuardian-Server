"""端侧基线评测 — 在板上跑 20 题 × 2 模型 = 40 generations.

输出:
  ~/llm_eval_results.json  (机读)
  ~/llm_eval_report.md     (人读，并排对比 + 评分模板)
"""
import os, sys, time, json, gc
from pathlib import Path

# 把 eval_questions.py 嵌入脚本本身（远程跑不方便 import）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_questions import QUESTIONS, build_prompt


import openvino as ov
import openvino_genai as ov_genai

os.environ.setdefault("OV_CACHE_DIR", os.path.expanduser("~/ov_cache"))

MODELS = [
    {
        "name": "Gemma3-4B-INT4 (iGPU)",
        "tag": "gemma3-4b-gpu",
        "path": os.path.expanduser("~/models/gemma-3-4b-it-int4-ov"),
        "device": "GPU",
        "pipeline": "VLM",  # 多模态模型用 VLMPipeline
    },
    {
        "name": "Qwen2.5-1.5B-INT4 (NPU)",
        "tag": "qwen1.5b-npu",
        "path": os.path.expanduser("~/models/qwen2.5-1.5b-int4-ov"),
        "device": "NPU",
        "pipeline": "LLM",
    },
]

MAX_NEW_TOKENS = 256


def load_pipeline(model_cfg):
    if model_cfg["pipeline"] == "VLM":
        return ov_genai.VLMPipeline(model_cfg["path"], device=model_cfg["device"])
    return ov_genai.LLMPipeline(model_cfg["path"], device=model_cfg["device"])


def generate(pipe, prompt, is_vlm):
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = MAX_NEW_TOKENS
    t0 = time.time()
    if is_vlm:
        out = pipe.generate(prompt, generation_config=cfg)
    else:
        out = pipe.generate(prompt, cfg)
    elapsed = time.time() - t0
    return str(out), elapsed


def main():
    out_json = os.path.expanduser("~/llm_eval_results.json")
    out_md = os.path.expanduser("~/llm_eval_report.md")
    results = []

    for model_cfg in MODELS:
        if not Path(model_cfg["path"]).exists():
            print(f"SKIP {model_cfg['name']}: {model_cfg['path']} not found")
            continue
        print(f"\n{'=' * 70}\nLoading: {model_cfg['name']}\n{'=' * 70}")
        t0 = time.time()
        pipe = load_pipeline(model_cfg)
        print(f"  load: {time.time() - t0:.2f}s")
        is_vlm = (model_cfg["pipeline"] == "VLM")

        for i, q in enumerate(QUESTIONS, start=1):
            prompt = build_prompt(q)
            try:
                response, latency = generate(pipe, prompt, is_vlm)
                results.append({
                    "qid": q["id"],
                    "intent": q["intent"],
                    "topology": q["topology"],
                    "question": q["question"],
                    "context": q.get("context", ""),
                    "model": model_cfg["tag"],
                    "model_name": model_cfg["name"],
                    "device": model_cfg["device"],
                    "latency_s": latency,
                    "response": response,
                })
                print(f"  [{i:2d}/{len(QUESTIONS)}] {q['id']:<12} {latency:5.1f}s  {response[:60]!r}")
            except Exception as e:
                msg = str(e).replace("\n", " ")[:200]
                results.append({"qid": q["id"], "model": model_cfg["tag"], "error": msg})
                print(f"  [{i:2d}/{len(QUESTIONS)}] {q['id']:<12} FAIL: {msg}")

        del pipe
        gc.collect()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out_json}")

    # 生成并排 markdown 报告
    md = []
    md.append("# LabGuardian 端侧 LLM 基线评测报告\n\n")
    md.append(f"- 题数: {len(QUESTIONS)}\n")
    md.append(f"- 模型: " + " · ".join(m["name"] for m in MODELS) + "\n")
    md.append(f"- 生成 max_new_tokens: {MAX_NEW_TOKENS}\n\n")

    md.append("## 评分规则\n\n")
    md.append("每个回答按 5 维 1-5 打分:\n")
    md.append("1. **正确性 (Correctness)**: 技术事实是否对，有无错误信息\n")
    md.append("2. **教学性 (Pedagogy)**: 是否循循善诱、有启发，而非直给答案\n")
    md.append("3. **简洁 (Conciseness)**: 是否冗余、跑题\n")
    md.append("4. **格式 (Format)**: 是否结构清晰、易读\n")
    md.append("5. **总体 (Overall)**: 综合教学场景可用性\n\n")
    md.append("打分后请填到每题下方 `[ ]` 里。统计 mean 看哪个模型在哪个维度强。\n\n")
    md.append("---\n\n")

    # 按 qid 重组：每题 一段
    by_qid = {}
    for r in results:
        by_qid.setdefault(r["qid"], []).append(r)

    for q in QUESTIONS:
        rs = by_qid.get(q["id"], [])
        md.append(f"## Q{q['id']} · {q['intent']} · {q['topology']}\n\n")
        md.append(f"**学生提问**：{q['question']}\n\n")
        if q.get("context"):
            md.append("<details><summary>📚 KB 上下文 (点开)</summary>\n\n")
            md.append("```\n" + q["context"] + "\n```\n\n</details>\n\n")
        for r in rs:
            md.append(f"### 🤖 {r.get('model_name', r.get('model'))}  ({r.get('latency_s', 0):.1f}s)\n\n")
            if "error" in r:
                md.append(f"❌ ERROR: {r['error']}\n\n")
            else:
                md.append(r["response"].strip() + "\n\n")
            md.append("**评分**: 正确[ ]  教学[ ]  简洁[ ]  格式[ ]  总分[ ]\n\n")
        md.append("---\n\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"[saved] {out_md}")
    print(f"\nrun: scp bupt@10.133.22.42:~/llm_eval_report.md ./")


if __name__ == "__main__":
    main()
