from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_questions import QUESTIONS as BASE_QUESTIONS


TOPOLOGY_TO_SCENE = {
    "rc_first_order": "exp_first_order_rc",
    "common_emitter": "exp_common_emitter_amplifier",
    "differential_amplifier": "exp_differential_amplifier",
    "ua741_inverting": "exp_ua741_inverting_amplifier",
    "ua741_summing": "exp_ua741_summing_amplifier",
    "ua741_integrator": "exp_ua741_integrator",
}


BASE_METADATA: dict[str, dict[str, Any]] = {
    "rc_01": {
        "risk_level": "safe",
        "question_override": "一阶 RC 电路里，τ 为什么等于 RC？请只答 2 点：1. 物理含义 2. 为什么 Ω·F = s。",
        "expected_points": ["点出 τ=RC", "说明单位 Ω·F = s", "提到充放电或指数响应"],
    },
    "rc_02": {
        "risk_level": "warning",
        "question_override": "RC 微分电路输出几乎跟输入一样、没有尖峰。只能二选一回答：`τ 太大` 或 `τ 太小`。然后再给出一句调整方向。",
        "context_override": (
            "[fault_case differentiator_no_spike]\n"
            "现象：微分电路输出无尖峰，输出几乎跟输入一样。\n"
            "明确结论：这是 τ = RC 太大，不是 τ 太小；因为 τ 远大于输入周期 T 时，电路失去微分特性，输出会更接近输入。\n"
            "正确条件：应满足 τ << T。\n"
            "调整方向：减小 R 或减小 C，让 τ 下降。"
        ),
        "expected_points": ["指出 τ 相对周期过大", "提到应满足 τ << T", "给出调小 R 或 C 的方向"],
    },
    "rc_03": {
        "risk_level": "warning",
        "expected_points": ["先测元件标称值", "先接输入再接双通道示波器", "提到先低频后高频或探头衰减"],
    },
    "ce_01": {
        "risk_level": "safe",
        "question_override": "共射放大器里 C_E 的作用是什么？去掉后最明显的两个变化是什么？请只答 3 点。",
        "expected_points": ["说明 C_E 旁路 R_E 的交流作用", "指出去掉后增益下降", "提到线性度变好或更稳定"],
    },
    "ce_02": {
        "risk_level": "warning",
        "expected_points": ["识别 V_CE≈VCC 为截止", "优先检查基极偏置", "提到 V_B 或 V_BE 应接近导通值"],
    },
    "ce_03": {
        "risk_level": "warning",
        "expected_points": ["先量 V_B、V_E、V_C", "提到 V_CE 应接近 VCC/2", "指出截止/饱和两种异常"],
    },
    "ce_04": {
        "risk_level": "warning",
        "expected_points": ["识别顶部削平与偏置或输入过大有关", "优先检查静态工作点", "给出减小输入或调整偏置的办法"],
    },
    "diff_01": {
        "risk_level": "safe",
        "question_override": "差模增益 A_d、共模增益 A_c、CMRR 分别表示什么？请各用一句话回答。",
        "expected_points": ["区分差模与共模", "解释 A_d 和 A_c", "说明高 CMRR 用于抑制共模干扰"],
    },
    "diff_02": {
        "risk_level": "warning",
        "expected_points": ["指出输出偏移与失配有关", "提到电阻或管子不匹配", "提到尾电流源不理想会影响 CMRR"],
    },
    "diff_03": {
        "risk_level": "warning",
        "expected_points": ["说明差模要反相信号", "说明共模要同相同幅", "提到 A_d 或 A_c 的测法"],
    },
    "ua_inv_01": {
        "risk_level": "safe",
        "question_override": "UA741 反相放大器里，负号表示什么？请直接写出虚地条件下的关键等式 `(Vin-0)/Rin=(0-Vout)/Rf` 和结论 `Av=-Rf/Rin`。",
        "expected_points": ["解释负号表示反相", "提到虚短或虚地", "给出 (Vin-0)/Rin=(0-Vout)/Rf 或 Av=-Rf/Rin"],
    },
    "ua_inv_02": {
        "risk_level": "warning",
        "expected_points": ["先查电源脚 pin7/pin4", "检查 V+ 是否接地", "检查 R_f 是否从输出回到反相端"],
    },
    "ua_inv_03": {
        "risk_level": "warning",
        "expected_points": ["指出电阻不能一味做大", "给出 R_in 常用范围", "提到偏置电流补偿或 V+ 端补偿电阻"],
    },
    "ua_sum_01": {
        "risk_level": "safe",
        "question_override": "为什么加法器更常用反相结构？请只答 2 到 3 点，并顺带说明同相结构哪里更难调。",
        "expected_points": ["说明反相结构靠虚地叠加电流", "指出各输入通道互不影响", "说明同相结构耦合更强更难调"],
    },
    "ua_sum_02": {
        "risk_level": "warning",
        "expected_points": ["先查 R_f 是否等于各输入电阻", "指出未用输入不能悬空", "提到未用通道要接地"],
    },
    "ua_int_01": {
        "risk_level": "safe",
        "question_override": "积分器里反馈电容 C 和并联 R_leak 各起什么作用？请只答 3 点。",
        "expected_points": ["写出积分关系或 1/RC", "指出 C 是反馈元件", "说明 R_leak 用于泄放偏置电流防漂移"],
    },
    "ua_int_02": {
        "risk_level": "warning",
        "expected_points": ["识别为积分器漂移到饱和", "优先检查 R_leak", "提到电容漏电或器件问题"],
    },
    "mixed_01": {
        "risk_level": "warning",
        "question_override": "共射放大器里 C_in 太小会出现什么现象？选 C_in 时最少要看哪两个量？",
        "expected_points": ["指出 C_in 太小会抬高低频截止", "写出高通截止频率公式或提到截止频率", "提到阻抗和目标频率"],
    },
    "mixed_02": {
        "risk_level": "warning",
        "expected_points": ["识别输出钉在 +VCC 为饱和", "优先查反馈是否开路或 V+ 是否错误", "提到输入偏置或极性问题"],
    },
}


COMPARE_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "p0_rc_fault_01",
        "source": "bare_vs_contract",
        "scene_id": "exp_first_order_rc",
        "topology": "rc_first_order",
        "intent": "diagnostic",
        "risk_level": "warning",
        "question": "我搭的一阶 RC 低通滤波器输出端几乎没有信号，怎么排查？",
        "context": (
            "[fault_case rc_output_missing]\n"
            "输出端几乎没有信号；错误码提示 FLOATING_PIN，涉事元件为 C1。\n"
            "优先排查：C1 是否悬空、输出端是否接错节点、输入与地是否共参考。\n"
            "操作顺序：先断电，再检查元件接触、孔位和节点连通性。"
        ),
        "expected_points": ["先断电", "优先检查 C1 悬空或接触不良", "提到节点连通或接地参考"],
    },
    {
        "id": "p0_ce_fault_01",
        "source": "bare_vs_contract",
        "scene_id": "exp_common_emitter_amplifier",
        "topology": "common_emitter",
        "intent": "mixed",
        "risk_level": "danger",
        "question": "这个共射放大电路现在危险吗？我该怎么处理？",
        "context": (
            "[fault_case ce_short_risk]\n"
            "错误码为 COMPONENT_SHORTED_SAME_NET，涉事元件为 R2，属于 danger 风险。\n"
            "处理优先级：先断电，再检查是否存在短路、误接电源轨或反馈电阻错连。\n"
            "任何带电排查都应避免。"
        ),
        "expected_points": ["必须先断电", "指出存在短路或同网短接风险", "给出检查电阻和供电节点的步骤"],
    },
    {
        "id": "p0_diff_fault_01",
        "source": "bare_vs_contract",
        "scene_id": "exp_differential_amplifier",
        "topology": "differential_amplifier",
        "intent": "diagnostic",
        "risk_level": "warning",
        "question": "差分放大器两路输出不对称，可能是哪里接错了？",
        "context": (
            "[fault_case diff_pair_node_mismatch]\n"
            "错误码为 NODE_MISMATCH，涉事元件为 Q2。\n"
            "对称性破坏常见来源：两侧节点接法不一致、集电极负载不匹配、尾电流支路异常。\n"
            "应优先比较两支路静态工作点。"
        ),
        "expected_points": ["指出节点接法不一致", "提到两支路静态工作点或集电极电压", "提到负载失配或尾支路问题"],
    },
    {
        "id": "p0_uainv_fault_01",
        "source": "bare_vs_contract",
        "scene_id": "exp_ua741_inverting_amplifier",
        "topology": "ua741_inverting",
        "intent": "diagnostic",
        "risk_level": "warning",
        "question": "我搭的 UA741 反相放大器输出不对，怎么排查？",
        "context": (
            "[fault_case ua741_polarity_reversed]\n"
            "错误码为 POLARITY_REVERSED，涉事元件为 C1。\n"
            "优先检查：输入耦合电容极性、pin2 反相输入、pin6 输出反馈回路、供电极性。\n"
            "带电拔插前必须断电。"
        ),
        "expected_points": ["先断电", "优先检查 C1 极性", "提到 pin2、pin6 或反馈回路"],
    },
    {
        "id": "p0_uasum_fault_01",
        "source": "bare_vs_contract",
        "scene_id": "exp_ua741_summing_amplifier",
        "topology": "ua741_summing",
        "intent": "diagnostic",
        "risk_level": "warning",
        "question": "反相加法器好像少加了一路输入，怎么定位？",
        "context": (
            "[fault_case ua741_summing_component_missing]\n"
            "错误码为 COMPONENT_MISSING，涉事元件为 R2。\n"
            "应先核对缺失通道对应输入电阻是否存在、阻值是否正确、焊点是否可靠。\n"
            "未装或开路会导致该路输入不参与求和。"
        ),
        "expected_points": ["优先检查对应输入电阻是否缺失或开路", "提到阻值与焊点", "指出该路不参与求和的原因"],
    },
    {
        "id": "p0_uaint_fault_01",
        "source": "bare_vs_contract",
        "scene_id": "exp_ua741_integrator",
        "topology": "ua741_integrator",
        "intent": "diagnostic",
        "risk_level": "warning",
        "question": "积分器输出一直往电源轨饱和，不积分，怎么查？",
        "context": (
            "[fault_case ua741_integrator_node_mismatch]\n"
            "错误码为 NODE_MISMATCH，涉事元件为 Cf。\n"
            "应优先检查积分反馈支路是否接错、输入回路是否开路、运放供电与极性是否正确。\n"
            "积分器异常饱和时，R_f/C_f/R_leak 和同相端平衡支路都需要复核。"
        ),
        "expected_points": ["指出积分反馈支路接错或开路", "提到供电或极性检查", "提到 R_f/C_f/R_leak 或平衡支路"],
    },
]


CUSTOM_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "p0_rc_04",
        "source": "custom_extension",
        "scene_id": "exp_first_order_rc",
        "topology": "rc_first_order",
        "intent": "diagnostic",
        "risk_level": "warning",
        "question": "RC 低通实验里我测到截止频率明显比理论值高，最先怀疑什么？",
        "context": (
            "[teaching_scene exp_first_order_rc · cutoff_shift]\n"
            "理论截止频率 f_c = 1/(2πRC)。\n"
            "实测高于理论常见原因：R 或 C 实际值偏小、示波器探头/负载改变等效阻抗、接线点取错导致测到输入而非输出。\n"
            "排查应先从元件实测和测量方式入手。"
        ),
        "expected_points": ["先核对 R、C 实测值", "提到负载或探头影响", "检查输出测点是否接对"],
    },
    {
        "id": "p0_diff_04",
        "source": "custom_extension",
        "scene_id": "exp_differential_amplifier",
        "topology": "differential_amplifier",
        "intent": "mixed",
        "risk_level": "warning",
        "question": "差分放大器一边集电极电压高、一边低，这说明什么？应该按什么顺序查？",
        "context": (
            "[teaching_scene exp_differential_amplifier · collector_imbalance]\n"
            "两侧集电极静态电压明显不对称，通常说明尾电流分配失衡。\n"
            "优先排查顺序：两侧电阻值是否匹配 → 两个三极管是否装反/失配 → 尾支路或恒流源是否正常。\n"
            "测量时先看静态电压，再看交流波形。"
        ),
        "expected_points": ["判断为尾电流分配或器件失配问题", "先查电阻匹配", "再查三极管与尾支路"],
    },
    {
        "id": "p0_uasum_03",
        "source": "custom_extension",
        "scene_id": "exp_ua741_summing_amplifier",
        "topology": "ua741_summing",
        "intent": "concept_tutor",
        "risk_level": "safe",
        "question": "反相加法器里不用的输入端为什么不能悬空？标准处理方式是什么？请只答 2 到 3 点。",
        "context": (
            "[teaching_scene exp_ua741_summing_amplifier · unused_input]\n"
            "反相加法器未用输入若悬空，会通过输入电阻引入噪声与不确定偏置，破坏求和结果。\n"
            "标准做法：未用输入通道通过与其他输入同量级的电阻接地。\n"
            "这样可保持虚地节点条件和通道阻抗一致。"
        ),
        "expected_points": ["指出悬空会引入噪声或偏置", "说明未用输入应经电阻接地", "提到保持虚地条件或阻抗一致"],
    },
    {
        "id": "p0_uaint_03",
        "source": "custom_extension",
        "scene_id": "exp_ua741_integrator",
        "topology": "ua741_integrator",
        "intent": "concept_tutor",
        "risk_level": "safe",
        "question": "积分器输入正弦波时，输出为什么会有 90 度相移？在什么条件下这个结论会失效？请只答 3 点。",
        "context": (
            "[teaching_scene exp_ua741_integrator · sinusoidal_response]\n"
            "理想积分器对正弦输入的传递函数近似为 -1/(jωRC)，输出相位相对输入滞后 90 度。\n"
            "若频率太低、R_leak 影响明显，或运放带宽不足、输出已接近饱和，则积分近似失效，相位关系会偏离 90 度。\n"
            "观察时应保证在线性工作区。"
        ),
        "expected_points": ["指出积分传递函数带来 90 度相移", "提到频率太低或 R_leak 影响会破坏近似", "提到带宽或饱和导致偏离"],
    },
    {
        "id": "p0_uaint_04",
        "source": "custom_extension",
        "scene_id": "exp_ua741_integrator",
        "topology": "ua741_integrator",
        "intent": "lab_guidance",
        "risk_level": "warning",
        "question": "做 UA741 积分器实验时，先量哪几个直流点，才能最快判断它为什么老是漂移？",
        "context": (
            "[teaching_scene exp_ua741_integrator · dc_checks]\n"
            "优先直流检查点：pin7/pin4 供电、电容两端平均电压、pin3 同相端电位、pin2 反相端是否接近虚地、输出 pin6 是否已偏到电源轨。\n"
            "若 pin2 不在虚地附近、pin6 长期贴近电源轨，通常提示偏置/泄放支路异常。\n"
            "测量前应先确认接线稳固并注意断电改线。"
        ),
        "expected_points": ["先量供电脚和 pin2/pin3/pin6", "判断虚地是否成立", "把漂移归因到偏置或泄放支路异常"],
    },
]


def _base_question_to_p0(question: dict[str, Any]) -> dict[str, Any]:
    metadata = BASE_METADATA[question["id"]]
    return {
        "id": question["id"],
        "source": "eval_questions",
        "scene_id": TOPOLOGY_TO_SCENE[question["topology"]],
        "topology": question["topology"],
        "intent": question["intent"],
        "risk_level": metadata["risk_level"],
        "question": metadata.get("question_override", question["question"]),
        "context": metadata.get("context_override", question.get("context", "")),
        "expected_points": metadata["expected_points"],
    }


def build_questions() -> list[dict[str, Any]]:
    questions = [_base_question_to_p0(question) for question in BASE_QUESTIONS]
    questions.extend(COMPARE_QUESTIONS)
    questions.extend(CUSTOM_QUESTIONS)
    if len(questions) != 30:
        raise ValueError(f"expected 30 P0 questions, got {len(questions)}")
    return questions


QUESTIONS = build_questions()


def _write_markdown(questions: list[dict[str, Any]], output_path: Path) -> None:
    by_source = Counter(question["source"] for question in questions)
    by_intent = Counter(question["intent"] for question in questions)
    by_topology = Counter(question["topology"] for question in questions)
    by_risk = Counter(question["risk_level"] for question in questions)

    lines = [
        "# P0 Eval Question Set",
        "",
        f"- total questions: {len(questions)}",
        f"- by source: {dict(by_source)}",
        f"- by intent: {dict(by_intent)}",
        f"- by topology: {dict(by_topology)}",
        f"- by risk: {dict(by_risk)}",
        "",
        "## Questions",
        "",
        "| ID | Source | Intent | Topology | Risk | Question |",
        "|---|---|---|---|---|---|",
    ]
    for question in questions:
        lines.append(
            "| {id} | {source} | {intent} | {topology} | {risk_level} | {question} |".format(**question)
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the fixed P0 evaluation question set.")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)

    questions = build_questions()
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.output_json}")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(questions, args.output_md)
        print(f"wrote {args.output_md}")

    by_source = Counter(question["source"] for question in questions)
    by_intent = Counter(question["intent"] for question in questions)
    by_topology = Counter(question["topology"] for question in questions)
    print(f"total questions: {len(questions)}")
    print("by source:", dict(by_source))
    print("by intent:", dict(by_intent))
    print("by topology:", dict(by_topology))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
