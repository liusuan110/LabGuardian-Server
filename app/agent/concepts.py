"""Local static concept library for the concept_tutor / lab_guidance flow.

All knowledge is hand-curated and considered generic. The library never claims
specific facts about the current circuit — it only carries reusable teaching
content. `lookup_concept` is a deterministic keyword-match; misses return None.
"""

from __future__ import annotations

from app.agent.contracts import ConceptPack

CONCEPT_LIBRARY: dict[str, ConceptPack] = {
    "breadboard_basics": ConceptPack(
        concept_id="breadboard_basics",
        title="面包板导通规则",
        level="basic",
        summary="面包板中间区按列纵向导通，两侧电源轨横向导通，中间分隔条断开两侧。",
        key_points=[
            "中间区每一列（同一字母行内 a-e 或 f-j）5 个孔位内部相连。",
            "两侧电源轨（红/蓝长条）整列横向相连，少数面包板中间会有断点，需按板规格确认。",
            "中央槽两侧不连通，跨槽必须用跳线。",
        ],
        formulas=[],
        examples=[
            "把元件两端插在同一列 → 元件两端短路。",
            "把元件两端跨在中央槽两侧 → 通过中间空槽自然断开。",
        ],
        common_mistakes=[
            "把电阻两端插在同一列内，导致短路。",
            "误以为整条电源轨没有断点，结果断点处电源不通。",
        ],
        lab_guidance=[
            "插接前先确认面包板分区与电源轨断点位置。",
            "用万用表通断挡测两个孔位是否在同一节点。",
        ],
        safety_notes=[
            "插接或调整接线时先断电。",
        ],
        citations=["local:breadboard_basics"],
    ),
    "ohms_law": ConceptPack(
        concept_id="ohms_law",
        title="欧姆定律",
        level="basic",
        summary="对线性电阻，电压、电流、电阻三者满足 V = I × R。",
        key_points=[
            "V 单位为伏特(V)，I 单位为安培(A)，R 单位为欧姆(Ω)。",
            "在串联回路中各元件电流相同；在并联节点电压相同。",
        ],
        formulas=[
            "V = I × R",
            "I = V / R",
            "R = V / I",
        ],
        examples=[
            "5V 电源串联 1kΩ 电阻接 LED，限流电流约为 (5-Vf)/1000 A。",
        ],
        common_mistakes=[
            "把欧姆定律直接套用到非线性元件（如 LED 本身）。",
            "忘记把电阻换算成欧姆单位。",
        ],
        lab_guidance=[
            "用万用表电压挡测两端电压，电流挡串入回路测电流。",
            "由 V 和 I 反推 R 时确认电流是流过该电阻而非旁路。",
        ],
        safety_notes=[
            "测电流必须串联接入，禁止把万用表电流挡并联到电源上，否则会短路。",
        ],
        citations=["local:ohms_law"],
    ),
    "led_current_limit": ConceptPack(
        concept_id="led_current_limit",
        title="LED 为什么需要串联限流电阻",
        level="basic",
        summary="LED 的伏安特性非线性，电压超过导通阈值后电流会急剧上升，必须靠串联电阻把电流限制在额定值内。",
        key_points=[
            "LED 有正向导通电压 Vf（红/绿约 1.8-2.2V，蓝/白约 3.0-3.4V）。",
            "限流电阻阻值 R ≈ (Vsource - Vf) / Iled。",
            "典型直插式 LED 工作电流 5-20mA，超过会烧坏。",
        ],
        formulas=[
            "R = (Vsource - Vf) / I_LED",
        ],
        examples=[
            "5V 供电、Vf=2V、目标电流 10mA → R = (5-2)/0.01 = 300Ω，实际选 330Ω 标准值。",
        ],
        common_mistakes=[
            "直接把 LED 接到电源上（无限流电阻），瞬间烧毁。",
            "极性接反，LED 不亮。",
            "限流电阻太小，电流超过 LED 额定值。",
        ],
        lab_guidance=[
            "先用万用表二极管挡确认 LED 正负极。",
            "上电前用电阻挡确认限流电阻已经串入回路。",
            "通电后用电流挡或测电阻两端电压反推电流。",
        ],
        safety_notes=[
            "更换或调整 LED 接线时先断电。",
            "怀疑短路或电流异常时立即断电再检查。",
        ],
        citations=["local:led_current_limit"],
    ),
    "rc_time_constant": ConceptPack(
        concept_id="rc_time_constant",
        title="RC 时间常数",
        level="basic",
        summary="RC 一阶电路充放电的特征时间 τ = R × C，决定响应快慢。",
        key_points=[
            "τ 单位为秒，R 单位为欧姆，C 单位为法拉。",
            "充电 1τ 达到约 63%，5τ 后基本完成（>99%）。",
            "放电 1τ 衰减到约 37%。",
        ],
        formulas=[
            "τ = R × C",
            "充电 v(t) = V_final × (1 - e^(-t/τ))",
            "放电 v(t) = V0 × e^(-t/τ)",
        ],
        examples=[
            "R=10kΩ, C=10μF → τ = 0.1s，约 0.5s 可视为充满。",
        ],
        common_mistakes=[
            "把 μF 当 F 计算，τ 量级差百万倍。",
            "忽视电容初始电压，结果与预测偏离。",
        ],
        lab_guidance=[
            "示波器接电容两端观察充放电曲线。",
            "用方波激励测 τ：从 0 升到约 63% 的时间近似 τ。",
        ],
        safety_notes=[
            "大容量电解电容断电后仍可能保有电荷，操作前先短路放电。",
        ],
        citations=["local:rc_time_constant"],
    ),
    "voltage_divider": ConceptPack(
        concept_id="voltage_divider",
        title="电阻分压",
        level="basic",
        summary="两个电阻串联接电源，中点电压按电阻比分压：Vout = Vin × R2 / (R1+R2)。",
        key_points=[
            "成立前提：分压点几乎不带负载，否则负载会拉低中点电压。",
            "总电流 I = Vin / (R1+R2)，注意电阻功耗。",
        ],
        formulas=[
            "Vout = Vin × R2 / (R1 + R2)",
        ],
        examples=[
            "5V 输入、R1=R2=10kΩ → Vout=2.5V。",
        ],
        common_mistakes=[
            "在分压点直接驱动较大电流负载，输出电压塌陷。",
            "R1+R2 太小，电源到地静态电流过大。",
        ],
        lab_guidance=[
            "用万用表电压挡直接测分压点对地电压。",
            "对比理论值与实测值，差异大时检查负载或接线。",
        ],
        safety_notes=[
            "改接分压电阻时先断电。",
        ],
        citations=["local:voltage_divider"],
    ),
    "capacitor_filtering": ConceptPack(
        concept_id="capacitor_filtering",
        title="电容滤波",
        level="basic",
        summary="并联在电源轨上的电容可吸收高频纹波；与串联电阻配合形成 RC 低通。",
        key_points=[
            "去耦电容就近放在 IC 电源-地引脚之间，越短越好。",
            "大容量电解负责低频，0.1μF 陶瓷负责高频。",
        ],
        formulas=[
            "fc = 1 / (2π × R × C)  (一阶低通截止频率)",
        ],
        examples=[
            "IC 电源脚旁加 0.1μF 陶瓷+10μF 电解，平衡高低频纹波。",
        ],
        common_mistakes=[
            "电解电容极性接反，可能漏液或炸开。",
            "去耦电容离 IC 太远，等效串联电感削弱效果。",
        ],
        lab_guidance=[
            "用示波器在电源轨观察纹波峰峰值。",
            "断电后短路电容两端放电再操作。",
        ],
        safety_notes=[
            "拆装电解电容前先断电并放电。",
        ],
        citations=["local:capacitor_filtering"],
    ),
}


_KEYWORD_TO_CONCEPT: tuple[tuple[tuple[str, ...], str], ...] = (
    # Order matters: more specific phrases first.
    (("rc 时间", "时间常数", "rc time", "充放电"), "rc_time_constant"),
    (("分压", "voltage divider"), "voltage_divider"),
    (("电容滤波", "去耦", "滤波电容", "decoupling"), "capacitor_filtering"),
    (("led", "限流电阻", "发光二极管"), "led_current_limit"),
    (("欧姆", "ohm", "v=ir", "v = ir"), "ohms_law"),
    (("面包板", "导通规则", "breadboard"), "breadboard_basics"),
)


def lookup_concept(query: str) -> ConceptPack | None:
    """Return the first matching ConceptPack by simple keyword scan, or None."""

    if not query:
        return None
    msg = query.lower()
    for phrases, concept_id in _KEYWORD_TO_CONCEPT:
        if any(phrase in msg for phrase in phrases):
            return CONCEPT_LIBRARY.get(concept_id)
    return None


def get_concept(concept_id: str) -> ConceptPack | None:
    return CONCEPT_LIBRARY.get(concept_id)
