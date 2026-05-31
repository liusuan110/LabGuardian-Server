"""Board probe: does the distilled student model classify intent (source='llm')?

Validates decision ② (B): with AGENT_LLM_PROVIDER=openvino_genai_text, the
`classify_intent_smart` path should now route through the student model's
generate() and return source='llm' — falling back to keyword only on failure.
Also times each classification so we can quantify B's per-question latency cost.

Run on the board:
    cd /home/bupt/LabGuardian-Server
    /home/bupt/miniconda3/envs/labguardian/bin/python -m scripts.board.probe_intent_student
"""

from __future__ import annotations

import time

from app.agent.contracts import RuntimeEvidence
from app.agent.intent_llm import classify_intent_smart
from app.core.config import settings

# (question, evidence_has_diag_context, expected-ish label for eyeballing)
CASES = [
    ("什么是RC电路的时间常数？", False, "concept_tutor"),
    ("我这个电路到底哪里接错了？", True, "diagnostic"),
    ("怎么用示波器测一下输出波形的下一步该做什么？", False, "lab_guidance"),
    ("这个共射放大电路输出不对，顺便能讲讲静态工作点是干嘛的吗？", True, "mixed"),
    # paraphrased / colloquial — the cases keyword tables tend to miss
    ("帮我瞅瞅这运放咋回事，还有虚短虚断啥意思", True, "mixed"),
    ("这玩意儿为啥不工作", True, "diagnostic"),
]


def main() -> None:
    print(f"AGENT_LLM_PROVIDER = {getattr(settings, 'AGENT_LLM_PROVIDER', '?')}")
    print(f"model_dir          = {getattr(settings, 'AGENT_LLM_OPENVINO_MODEL_DIR', '?')}")
    print("=" * 78)

    for q, has_ctx, expect in CASES:
        ev = None
        if has_ctx:
            ev = RuntimeEvidence(
                station_id="PROBE",
                risk_level="warning",
                error_codes=["NODE_MISMATCH"],
            )
        t = time.time()
        d = classify_intent_smart(q, evidence=ev)
        dt = (time.time() - t) * 1000.0
        flag = "✓" if d.intent == expect else "≠"
        print(f"[{d.source:7s}] {d.intent:13s} {flag}(期望 {expect:13s}) "
              f"conf={d.confidence:.2f} {dt:7.0f}ms  reason={d.reason!r}")
        print(f"          Q: {q}")
    print("=" * 78)
    print("source=llm 表示学生模型在分类；source=keyword 表示退回关键词兜底。")


if __name__ == "__main__":
    main()
