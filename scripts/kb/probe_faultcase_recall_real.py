"""Verify the diagnostic→teaching-case recall under REAL diff_report codes.

The demo includes a "build a faulty board → diagnose → surface the teaching
case" segment. The validator (diff_report) emits diff-vocabulary codes
(WRONG_CONNECTION / OPEN_CIRCUIT / SHORT_CIRCUIT / *_NODE_MISMATCH /
COMPONENT_MISSING ...), but the fault_case KB ``related_error_codes`` are
tagged with the s5/ERC vocabulary — disjoint except COMPONENT_MISSING. This
probe feeds the *real* diff codes (per the fault family, confirmed by
scripts/kb/probe_diff_codes.py) into the actual recall path and reports
whether the intended teaching case is recalled and at what rank.

Run locally (no model needed — recall is deterministic Python):
    .venv/bin/python -m scripts.kb.probe_faultcase_recall_real
"""

from __future__ import annotations

from app.services.error_tag_service import ErrorTagService
from app.services.teaching_kb_service import TeachingKbService

# (scene_id, demo_fault_label, intended_case_id, real_diff_codes, student_question)
DEMO_FAULTS = [
    ("exp_common_emitter_amplifier", "三极管引脚装反", "ce_bjt_pin_reversed",
     ["WRONG_CONNECTION", "OPEN_CIRCUIT", "SHORT_CIRCUIT"],
     "我这个共射放大电路三极管好像装反了，输出不正常"),
    ("exp_differential_amplifier", "尾电阻未接VEE(开路)", "diff_pair_tail_path_broken",
     ["OPEN_CIRCUIT", "WRONG_CONNECTION"],
     "差分放大器两路输出不对称，是不是哪里断了"),
    ("exp_first_order_rc", "积分输出节点接错", "rc_wrong_output_node_for_integrator",
     ["OUTPUT_NODE_MISMATCH", "WRONG_CONNECTION", "OPEN_CIRCUIT"],
     "RC 积分电路输出节点接错了怎么办"),
    ("exp_ua741_integrator", "缺 R_leak(漏电阻)", "int_missing_rleak_dc_drift",
     ["COMPONENT_MISSING"],
     "积分器输出一直饱和漂移，是不是少接了电阻"),
    ("exp_ua741_inverting_amplifier", "输入脚 pin2/pin3 接反", "inv_input_pins_swapped",
     ["WRONG_CONNECTION", "OPEN_CIRCUIT"],
     "UA741 反相放大器输出不对，怀疑输入脚接反了"),
    ("exp_ua741_summing_amplifier", "各路输入短接", "sum_input_resistors_shorted_at_node",
     ["SHORT_CIRCUIT", "EXTRA_CONNECTION"],
     "加法器几路输入好像短接在一起了"),
]


def main() -> None:
    kb = TeachingKbService()
    tagger = ErrorTagService()

    hit_top1 = 0
    hit_any = 0
    for scene_id, fault, intended, codes, q in DEMO_FAULTS:
        # mirror production: codes -> tags via ErrorTagService, then recall
        cmp_report = {"items": [{"error_code": c, "severity": "error"} for c in codes]}
        tags = [t["error_tag"] for t in tagger.extract_tags(cmp_report)]
        recalled = kb.search_fault_cases(
            query=q, scene_id=scene_id, error_tags=tags, error_codes=codes, top_k=5
        )
        ids = [c.get("knowledge_id") for c in recalled]
        rank = (ids.index(intended) + 1) if intended in ids else 0
        # diagnose WHY: does the intended case share any code/tag with the real diff codes?
        intended_case = next((c for c in kb.list_fault_cases(scene_id=scene_id)
                              if c.get("knowledge_id") == intended), {})
        case_codes = set(intended_case.get("related_error_codes", []))
        code_overlap = case_codes & set(codes)

        if rank == 1:
            hit_top1 += 1
        if rank >= 1:
            hit_any += 1
        mark = "✅TOP1" if rank == 1 else (f"⚠️#{rank}" if rank else "❌MISS")
        print("=" * 92)
        print(f"[{mark}] {scene_id}  ·  演示故障：{fault}")
        print(f"  真实 diff 码: {codes}")
        print(f"  → tags(经ErrorTagService): {tags or '（全落空）'}")
        print(f"  KB 该案例标注码: {sorted(case_codes)}  ∩真实码 = {sorted(code_overlap) or '∅（桥断）'}")
        print(f"  召回排序: {ids}")
        print(f"  想要的案例 [{intended}] 命中: {'是，第'+str(rank) if rank else '否'}")
    print("=" * 92)
    print(f"汇总：TOP1 命中 {hit_top1}/{len(DEMO_FAULTS)}，任意命中 {hit_any}/{len(DEMO_FAULTS)}")
    print("（code_overlap=∅ 表示 error_code 桥在生产里对这条故障完全不触发，只靠问题词在撑）")


if __name__ == "__main__":
    main()
