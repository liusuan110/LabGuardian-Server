"""Bare 1.5B vs 1.5B + retrieval-contract comparison (board).

For each of the 6 demo scenes, ask the SAME natural student question two ways:

  A) BARE      — the distilled student model answers the raw question with no
                 board state / no retrieval contract (``provider.generate``).
  B) CONTRACT  — the full ``diagnostic_agent`` graph: scene-anchored retrieval
                 + deterministic verifier + the same student model as the
                 "mouth" (``actual_llm_provider == openvino_genai_text``).

Emits a JSON array to ``--output`` (default /tmp). Pull it back to the report
machine and render Markdown there. This is the headline "why the contract
matters" comparison for the paper Table / defense slide.

Run on the board:
    cd /home/bupt/LabGuardian-Server
    /home/bupt/miniconda3/envs/labguardian/bin/python -m scripts.board.compare_bare_vs_contract \
        --output /tmp/compare_bare_vs_contract.json
"""

from __future__ import annotations

import argparse
import json
import time

from app.agent.llm.factory import clear_llm_provider_cache, get_llm_provider
from app.core import deps
from app.schemas.angnt import AngntAskRequest


_SYS = "你是 LabGuardian 电路实验助教，回答电路诊断/原理/操作问题，涉及危险先提醒断电。"

CASES = [
    {
        "scene": "exp_first_order_rc", "scene_zh": "一阶 RC 滤波器",
        "topology_label": "rc_first_order",
        "question": "我搭的一阶RC低通滤波器输出端几乎没有信号，怎么排查？",
        "error_code": "FLOATING_PIN", "severity": "warning", "component_id": "C1",
        "suggested_action": "断电后检查 C1 接地脚是否插实、是否有一脚悬空",
    },
    {
        "scene": "exp_common_emitter_amplifier", "scene_zh": "共射放大电路",
        "topology_label": "common_emitter",
        "question": "这个共射放大电路现在危险吗？我该怎么处理？",
        "error_code": "COMPONENT_SHORTED_SAME_NET", "severity": "danger", "component_id": "R2",
        "suggested_action": "断电后将 R2 两端跨接到不同行",
    },
    {
        "scene": "exp_differential_amplifier", "scene_zh": "BJT 差分放大器",
        "topology_label": "differential_pair",
        "question": "差分放大器两路输出不对称，可能是哪里接错了？",
        "error_code": "NODE_MISMATCH", "severity": "warning", "component_id": "Q2",
        "suggested_action": "断电后核对 Q2 基极是否接到正确的输入节点",
    },
    {
        "scene": "exp_ua741_inverting_amplifier", "scene_zh": "UA741 反相放大器",
        "topology_label": "inverting_amp_ua741",
        "question": "我搭的UA741反相放大器输出不对，怎么排查？",
        "error_code": "POLARITY_REVERSED", "severity": "warning", "component_id": "C1",
        "suggested_action": "断电后按极性标记重新插接 C1",
    },
    {
        "scene": "exp_ua741_summing_amplifier", "scene_zh": "UA741 反相加法器",
        "topology_label": "summing_amp_ua741",
        "question": "反相加法器好像少加了一路输入，怎么定位？",
        "error_code": "COMPONENT_MISSING", "severity": "warning", "component_id": "R2",
        "suggested_action": "断电后补接第二路输入电阻 R2 到反相输入节点",
    },
    {
        "scene": "exp_ua741_integrator", "scene_zh": "UA741 反相积分器",
        "topology_label": "integrator_ua741",
        "question": "积分器输出一直往电源轨饱和，不积分，怎么查？",
        "error_code": "NODE_MISMATCH", "severity": "warning", "component_id": "Cf",
        "suggested_action": "断电后核对反馈电容 Cf 是否接在输出与反相输入之间",
    },
]


def _station(case: dict) -> dict:
    return {
        "station_id": case["station_id"],
        "risk_level": case["severity"] if case["severity"] == "danger" else "warning",
        "topology_label": case["topology_label"],
        "diagnostics": [f"{case['component_id']} {case['error_code']}"],
        "risk_reasons": [case["error_code"]],
        "comparison_report": {
            "topology_label": case["topology_label"],
            "items": [
                {
                    "error_code": case["error_code"],
                    "severity": case["severity"],
                    "component_id": case["component_id"],
                    "suggested_action": case["suggested_action"],
                }
            ],
        },
        "netlist_v2": {
            "components": [{"component_id": case["component_id"], "pins": []}],
            "nets": [{"net_id": "N1", "members": [f"{case['component_id']}.pin1"]}],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/compare_bare_vs_contract.json")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()

    clear_llm_provider_cache()
    provider = get_llm_provider()
    classroom = deps.get_classroom()
    agent = deps.get_agent_service()

    results = []
    for i, case in enumerate(CASES):
        case["station_id"] = f"CMP{i:02d}"

        # A) BARE — raw model, no contract
        t = time.time()
        try:
            bare_answer = provider.generate(
                case["question"], system=_SYS, max_new_tokens=args.max_new_tokens
            )
        except Exception as exc:  # provider fell back to template / no model
            bare_answer = f"(bare generate unavailable: {type(exc).__name__})"
        bare_s = time.time() - t

        # B) CONTRACT — full diagnostic_agent
        classroom.update_station(_station(case))
        t = time.time()
        accepted = agent.submit(
            AngntAskRequest(
                station_id=case["station_id"],
                query=case["question"],
                mode="diagnostic_agent",
            ),
            classroom,
        )
        res = agent.get_status(accepted.job_id).result
        contract_s = time.time() - t

        scene_resolved = ""
        verifier_passed = None
        for ev in res.evidence:
            payload = ev.payload or {}
            for key in ("current_scene_id", "scene_id"):
                if payload.get(key):
                    scene_resolved = str(payload[key])
            if ev.evidence_type == "verification_report":
                verifier_passed = bool(payload.get("passed"))

        record = {
            "scene": case["scene"],
            "scene_zh": case["scene_zh"],
            "question": case["question"],
            "injected_error_code": case["error_code"],
            "injected_component": case["component_id"],
            "risk": case["severity"],
            "bare": {
                "answer": bare_answer,
                "latency_s": round(bare_s, 2),
            },
            "contract": {
                "answer": res.answer,
                "provider": res.actual_llm_provider,
                "model": res.actual_llm_model,
                "scene_resolved": scene_resolved,
                "verifier_passed": verifier_passed,
                "latency_s": round(contract_s, 2),
            },
        }
        results.append(record)
        print(f"[{i+1}/6] {case['scene_zh']}: bare {bare_s:.1f}s / contract {contract_s:.1f}s "
              f"(provider={res.actual_llm_provider}, scene={scene_resolved}, verifier={verifier_passed})")

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"saved {len(results)} cases -> {args.output}")


if __name__ == "__main__":
    main()
