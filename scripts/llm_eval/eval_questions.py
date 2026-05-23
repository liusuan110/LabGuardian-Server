"""20 道端侧 LLM 评测题库 (6 demo × 4 intent)."""

# 极简的 system prompt — 跟实际 agent 一致的风格
SYSTEM_PROMPT = """你是 LabGuardian，一个面向高校电子类基础实验的智能教学助教。
你的回答要做到：(1) 技术事实正确；(2) 教学性强，循循善诱不直给答案；
(3) 简洁结构清晰，分点回答；(4) 引用提供的 [上下文] 而不是凭空发挥。"""


def build_prompt(q):
    parts = [SYSTEM_PROMPT, ""]
    if q.get("context"):
        parts.append(f"[上下文]\n{q['context']}\n")
    parts.append(f"[学生提问]\n{q['question']}\n")
    parts.append("[你的回答]")
    return "\n".join(parts)


QUESTIONS = [
    # ========== 1. RC 一阶电路 ==========
    {
        "id": "rc_01",
        "topology": "rc_first_order",
        "intent": "concept_tutor",
        "question": "为什么一阶 RC 电路的时间常数是 τ = RC？这两个量为什么相乘能给出一个时间？",
        "context": (
            "[teaching_scene exp_first_order_rc]\n"
            "一阶 RC 电路时间常数 τ=RC，单位为秒。\n"
            "物理含义：当电容从 0 充到 63.2% 的稳态电压所需的时间。\n"
            "数学：dv/dt = (V-v)/RC，解得 v(t) = V(1 - e^{-t/RC})。\n"
            "单位推导：[Ω]·[F] = [V/A]·[C/V] = [s]。"
        ),
    },
    {
        "id": "rc_02",
        "topology": "rc_first_order",
        "intent": "diagnostic",
        "question": "我搭了一个 RC 微分电路，输入是 5V 1kHz 方波，但示波器看到输出几乎跟输入一样，没有微分尖峰。原因是什么？",
        "context": (
            "[fault_case differentiator_no_spike]\n"
            "现象：微分电路输出无尖峰。\n"
            "可能原因：(1) τ = RC 太大（远大于输入周期 T），RC 失去微分特性，变成耦合电容；\n"
            "(2) 应满足 τ << T，对 1kHz (T=1ms) 而言 τ < 0.1ms，如 R=1kΩ, C=10nF 时 τ=10μs OK。"
        ),
    },
    {
        "id": "rc_03",
        "topology": "rc_first_order",
        "intent": "lab_guidance",
        "question": "我做一阶 RC 实验，仪器有信号发生器、示波器、数字万用表。我应该按什么顺序连线和测量？",
        "context": (
            "[teaching_scene exp_first_order_rc · 测量步骤]\n"
            "推荐顺序：(1) 万用表先测电阻、电容标称是否准；(2) 信号源接 RC 输入端、示波器 CH1 接输入、CH2 接输出；\n"
            "(3) 先低频（10 倍 1/τ）观察稳态、再高频（0.1 倍 1/τ）观察微分/积分；(4) 注意示波器探头 ×10 衰减。"
        ),
    },
    # ========== 2. 共射放大器 ==========
    {
        "id": "ce_01",
        "topology": "common_emitter",
        "intent": "concept_tutor",
        "question": "共射放大器的发射极旁路电容 C_E 起什么作用？我把它去掉会怎样？",
        "context": (
            "[teaching_scene exp_common_emitter_amplifier · C_E 作用]\n"
            "C_E 旁路 R_E 的交流通路，避免 R_E 引入的负反馈使增益严重下降。\n"
            "有 C_E：A_v ≈ -R_C/r_e (高增益，~100×); \n"
            "无 C_E：A_v ≈ -R_C/(R_E+r_e) (低增益，~10×，但线性度提升)。"
        ),
    },
    {
        "id": "ce_02",
        "topology": "common_emitter",
        "intent": "diagnostic",
        "question": "我搭的共射放大器，输入 10mV 正弦波，输出却几乎是 0。三极管 V_CE 测得是 11V（VCC=12V），是什么问题？",
        "context": (
            "[fault_case bjt_cutoff]\n"
            "现象：V_CE ≈ VCC 说明三极管处于截止区，没有静态电流。\n"
            "常见原因：(1) R_B1/R_B2 比例错，V_B 太低 (<0.7V)；(2) 上拉电阻 R_B1 缺失或开路；\n"
            "(3) NPN/PNP 装反了；(4) BE 结开路（管子坏）。\n"
            "判断：万用表测 V_B 应该 ≈ 0.7V，V_BE > 0.7V 才正常导通。"
        ),
    },
    {
        "id": "ce_03",
        "topology": "common_emitter",
        "intent": "lab_guidance",
        "question": "做共射放大器实验，万用表先测哪几个直流电压可以快速判断静态工作点是否正常？",
        "context": (
            "[teaching_scene exp_common_emitter_amplifier · 静态测量]\n"
            "三表必测：V_B (≈0.7V+V_E)、V_E (≈I_E·R_E)、V_C (≈VCC-I_C·R_C)。\n"
            "正确特征：V_CE = V_C - V_E ≈ VCC/2，三极管在放大区中点。\n"
            "异常：V_CE → 0 (饱和)、V_CE → VCC (截止) 都说明偏置错。"
        ),
    },
    {
        "id": "ce_04",
        "topology": "common_emitter",
        "intent": "mixed",
        "question": "我的共射放大器输出失真严重（顶部削平），是什么原因？应该怎么调？",
        "context": (
            "[fault_case bjt_top_clipping]\n"
            "顶部削平 = 输出正半周饱和。原因：静态工作点偏 VCC 太近、或输入信号过大、或 V_CE 不在 VCC/2。\n"
            "调整方法：(1) 减小输入幅度；(2) 调整 R_B1 让 V_C 落在 VCC/2；(3) 适当减小 R_C。"
        ),
    },
    # ========== 3. 差分放大器 ==========
    {
        "id": "diff_01",
        "topology": "differential_amplifier",
        "intent": "concept_tutor",
        "question": "差分放大器的「差模增益」和「共模增益」分别是什么？为什么共模抑制比 CMRR 是个重要指标？",
        "context": (
            "[teaching_scene exp_differential_amplifier · CMRR]\n"
            "差模信号：两个输入端反向变化（如 +V 和 -V）。\n"
            "共模信号：两个输入端同向变化（如 +V 和 +V，常见噪声）。\n"
            "A_d = 差模增益, A_c = 共模增益；CMRR = |A_d/A_c| (dB)。\n"
            "CMRR 高 = 放大有用信号、抑制共模干扰（如电源纹波、温漂），这是差分对的核心价值。"
        ),
    },
    {
        "id": "diff_02",
        "topology": "differential_amplifier",
        "intent": "diagnostic",
        "question": "我做差分对，两路相同输入信号，本应输出 0，但实际测出来有几十 mV 偏移。问题在哪？",
        "context": (
            "[fault_case diff_pair_imbalance]\n"
            "现象：共模输入下输出非零 = 差分对失配。\n"
            "常见原因：(1) 两个 R_C 阻值不匹配（容差 5% 也会差几十 mV）；\n"
            "(2) 两个三极管 β 不一致；(3) 尾电流源不理想（如用单 R_E 替代恒流源，CMRR 退化）。\n"
            "解决：选配对管 / 用恒流源做尾。"
        ),
    },
    {
        "id": "diff_03",
        "topology": "differential_amplifier",
        "intent": "lab_guidance",
        "question": "差分放大器实验里我有两台信号源，怎么生成差模信号？怎么生成共模信号？",
        "context": (
            "[teaching_scene exp_differential_amplifier · 信号注入]\n"
            "差模：两路信号源反相（一路接 +V_in，另一路接 -V_in 或反相输出端口）。\n"
            "共模：两路信号源同相同幅（最简单是把同一路信号源同时接两个输入端，串相同电阻隔离）。\n"
            "测增益：A_d = V_out / V_id (V_id 是两输入差)；A_c = V_out / V_ic (V_ic 是两输入平均)。"
        ),
    },
    # ========== 4. UA741 反相放大器 ==========
    {
        "id": "ua_inv_01",
        "topology": "ua741_inverting",
        "intent": "concept_tutor",
        "question": "UA741 反相放大器电压增益是 -R_f/R_in，这个负号代表什么？为什么是这个表达式，能简单推导一下吗？",
        "context": (
            "[teaching_scene exp_ua741_inverting_amplifier · 推导]\n"
            "假设：理想运放，V+ = V-（虚短）、I_in = 0（虚断）。\n"
            "V+ = GND, 所以 V- = 0 (虚地)。\n"
            "节点电流：(V_in - 0)/R_in = (0 - V_out)/R_f\n"
            "解得：V_out = -(R_f/R_in)·V_in，所以 A_v = -R_f/R_in。\n"
            "负号 = 输出与输入反相（180° 相移）。"
        ),
    },
    {
        "id": "ua_inv_02",
        "topology": "ua741_inverting",
        "intent": "diagnostic",
        "question": "我搭 UA741 反相放大器，期望增益 -10，但实测输出是 0V 不动。怎么排查？",
        "context": (
            "[fault_case ua741_no_output]\n"
            "排查顺序：(1) 电源：pin7 +VCC、pin4 -VEE 是否到位；用万用表测 IC 实际供电；\n"
            "(2) pin3 (V+) 是否接 GND；\n"
            "(3) 信号路径：V_in 实际接在 R_in 还是接在 pin3 (反相接成同相)；\n"
            "(4) R_f 是否接对：必须从 pin6 (输出) → R_f → pin2 (V-)。\n"
            "(5) 芯片本身：用万用表测 pin6 对地电压，应该跟 V+ 一致 (虚短)。"
        ),
    },
    {
        "id": "ua_inv_03",
        "topology": "ua741_inverting",
        "intent": "lab_guidance",
        "question": "做 UA741 反相放大器，怎样选 R_in 和 R_f 才合理？是不是越大越好？",
        "context": (
            "[teaching_scene exp_ua741_inverting_amplifier · 选阻原则]\n"
            "R_in 太小 → 输入信号源被过载（要看信号源驱动能力）；R_in 太大 → 输入偏置电流 I_B 引入误差。\n"
            "经验：R_in ∈ [1kΩ, 100kΩ]，UA741 典型 10kΩ。\n"
            "R_f = A_v·R_in，比如 A_v=-10, R_in=10kΩ → R_f=100kΩ。\n"
            "另：建议 V+ 端接 R_p ≈ R_in∥R_f 补偿偏置电流，减小输出失调。"
        ),
    },
    # ========== 5. UA741 加法器 ==========
    {
        "id": "ua_sum_01",
        "topology": "ua741_summing",
        "intent": "concept_tutor",
        "question": "UA741 加法器为什么用反相结构？同相加法器不行吗？",
        "context": (
            "[teaching_scene exp_ua741_summing_amplifier · 反相 vs 同相]\n"
            "反相加法器：所有输入电阻都接到 V- (虚地)，节点电流叠加得 V_out = -R_f(V1/R1 + V2/R2 + ...)。\n"
            "各输入通道互不影响（虚地隔离），系数可独立设。\n"
            "同相加法器：输入接到 V+，节点不是虚地，各通道阻抗互相耦合，公式复杂、调试难。\n"
            "教学上反相加法器是标准做法。"
        ),
    },
    {
        "id": "ua_sum_02",
        "topology": "ua741_summing",
        "intent": "diagnostic",
        "question": "我搭了 3 路输入的加法器 (V_out = -(V1+V2+V3))，但是只在 V1 输入信号，V_out 振幅几乎是 V1 的一半，不是 1 倍，为什么？",
        "context": (
            "[fault_case ua741_summing_attenuation]\n"
            "可能原因：(1) R_f 不等于各 R_in；要让 V_out=-V1 需 R_f = R1 = R2 = R3；\n"
            "(2) V2, V3 输入端没接 GND 而是悬空 → 通过 R2, R3 引入噪声 + 改变虚地阻抗；\n"
            "未用的输入端应短接到 GND。"
        ),
    },
    # ========== 6. UA741 积分器 ==========
    {
        "id": "ua_int_01",
        "topology": "ua741_integrator",
        "intent": "concept_tutor",
        "question": "UA741 积分器输出是输入的时间积分，能简单证明吗？为什么实际电路要并联 R_leak？",
        "context": (
            "[teaching_scene exp_ua741_integrator · 推导 + R_leak]\n"
            "理想推导：虚地 V- = 0；I_in = V_in/R; I_C = C·dV_out/dt; KCL: V_in/R = -C·dV_out/dt;\n"
            "积分得 V_out(t) = -(1/RC)·∫V_in dt。\n"
            "R_leak (与 C 并联) 的作用：直流通路给输入偏置电流 I_B 提供泄放，否则 I_B 持续给 C 充电 → V_out 漂移到饱和。\n"
            "经验：R_leak >> 1/(2πf_min·C)，对最低工作频率仍是高阻。"
        ),
    },
    {
        "id": "ua_int_02",
        "topology": "ua741_integrator",
        "intent": "diagnostic",
        "question": "我做积分器，输入 1kHz 方波，期望输出三角波，但实际看到的是一个慢慢往上漂、最后撞到 +VCC 的斜坡。问题在哪？",
        "context": (
            "[fault_case ua741_integrator_drift]\n"
            "现象描述完全符合「积分器漂移到饱和」。\n"
            "原因：(1) 没接 R_leak（输入失调 + 偏置电流持续积分）；\n"
            "(2) R_leak 太大（不起作用）；\n"
            "(3) C 漏电（劣质电解电容方向也会反常）。\n"
            "建议：并联 R_leak = 10·R 试，比如 R=10kΩ → R_leak=100kΩ；用陶瓷/聚酯电容。"
        ),
    },
    # ========== 通用 / 混合 ==========
    {
        "id": "mixed_01",
        "topology": "common_emitter",
        "intent": "mixed",
        "question": "共射放大器的输入耦合电容 C_in 太小会怎样？怎么算多大才合适？",
        "context": (
            "[teaching_scene exp_common_emitter_amplifier · 耦合电容]\n"
            "C_in 与下级输入阻抗 (R_B1∥R_B2∥r_be) 构成高通滤波器。\n"
            "下限截止频率 f_L = 1/(2π·R·C_in)。\n"
            "如 R≈1kΩ, 要让 f_L=20Hz → C_in ≥ 1/(2π·1k·20) ≈ 8μF (取 10μF 标称)。\n"
            "C_in 太小：低频信号被衰减，输出「瘦」或失真。"
        ),
    },
    {
        "id": "mixed_02",
        "topology": "ua741_inverting",
        "intent": "diagnostic",
        "question": "UA741 反相放大器输出始终在 +12V，跟着输入完全不变。万用表测 pin6 = +VCC，怎么回事？",
        "context": (
            "[fault_case ua741_output_saturated]\n"
            "输出钉在 +VCC = 正向饱和。\n"
            "原因：(1) V- 比 V+ 低得多，运放输出最大化；(2) V+ 端电压不对（如悬空、接错电源）；\n"
            "(3) 反馈环路开路（R_f 没接好、或接到错的引脚）；\n"
            "(4) 输入信号偏置错误（如直流偏置 + 信号已经超出输入范围）。"
        ),
    },
]


if __name__ == "__main__":
    print(f"total questions: {len(QUESTIONS)}")
    by_intent = {}
    by_topo = {}
    for q in QUESTIONS:
        by_intent[q["intent"]] = by_intent.get(q["intent"], 0) + 1
        by_topo[q["topology"]] = by_topo.get(q["topology"], 0) + 1
    print("\nby intent:")
    for k, v in by_intent.items():
        print(f"  {k:<16} {v}")
    print("\nby topology:")
    for k, v in by_topo.items():
        print(f"  {k:<24} {v}")
    print("\n--- sample prompt (rc_01) ---")
    print(build_prompt(QUESTIONS[0]))
