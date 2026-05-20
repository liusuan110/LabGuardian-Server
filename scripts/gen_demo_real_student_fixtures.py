"""Phase D · 一次性生成三个新参考电路的"correct-wiring"实测 fixture。

仿照 ``tests/fixtures/real_student/inverting_amp_correct_v1.json`` 的 runtime_scene
netlist_v2 格式。每个 fixture 描述一块"学生按参考图正确接线"的面包板（与对应
reference 拓扑等价），用作 GNN 的 no-false-positive gate —— 所有 observed
(port, net) 边都应被打分 p_correct > 0.5。

输出：
    tests/fixtures/real_student/<name>_correct_v1.json          # runtime_scene netlist
    tests/fixtures/real_student/<name>_correct_v1.expected.json # gate spec

运行：
    .venv/bin/python scripts/gen_demo_real_student_fixtures.py
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "real_student"

# ---------------------------------------------------------------------------
# 拓扑 DSL
# ---------------------------------------------------------------------------

# (canonical_net_name, role, role_label, power_role)
# role ∈ {"input","output","signal","power","ground"};
# power_role ∈ {"VCC","VEE","GND",""}（仅当 role ∈ {power, ground} 时非空）
NET_ROLES: dict[str, tuple[str, str, str]] = {
    "VCC":  ("power",  "VCC",  "VCC"),
    "VEE":  ("power",  "VEE",  "VEE"),
    "GND":  ("ground", "GND",  "GND"),
    "VIN":  ("input",  "UI1",  ""),
    "UI1":  ("input",  "UI1",  ""),
    "UI2":  ("input",  "UI2",  ""),
    "VI1":  ("input",  "UI1",  ""),
    "VI2":  ("input",  "UI2",  ""),
    "VOUT": ("output", "UO1",  ""),
    "UO1":  ("output", "UO1",  ""),
    "UO2":  ("output", "UO2",  ""),
}
# 其余 net（INV / V_P / E1 / E2 / TAIL / VB3 / VE3 ...）默认 signal / 无 role_label。


def _net_meta(name: str) -> tuple[str, str, str]:
    return NET_ROLES.get(name, ("signal", "", ""))


# pin DSL：(pin_name, canonical_net | None, hole_id, node_id)
# canonical_net=None 表示 NC pin（electrical_net_id 设为 null）。
PinT = tuple[str, str | None, str, str]
# component DSL：(comp_id, component_type, part_subtype, package_type, polarity,
#                 orientation, [pin], confidence, symmetry_group)
CompT = dict


def make_resistor(comp_id: str, p1: tuple[str, str, str], p2: tuple[str, str, str], *, conf: float = 0.88, orient: int = 0) -> CompT:
    """p1/p2 = (canonical_net, hole_id, node_id)."""
    return {
        "component_id": comp_id,
        "component_type": "Resistor",
        "part_subtype": "",
        "package_type": "axial_2pin",
        "polarity": "none",
        "orientation": orient,
        "confidence": conf,
        "symmetry_group": [["pin1", "pin2"]],
        "pins": [
            ("pin1", *p1),
            ("pin2", *p2),
        ],
    }


def make_capacitor(comp_id: str, p1: tuple[str, str, str], p2: tuple[str, str, str], *, conf: float = 0.86, orient: int = 0) -> CompT:
    return {
        "component_id": comp_id,
        "component_type": "Capacitor",
        "part_subtype": "",
        "package_type": "ceramic_2pin",
        "polarity": "none",
        "orientation": orient,
        "confidence": conf,
        "symmetry_group": [["pin1", "pin2"]],
        "pins": [
            ("pin1", *p1),
            ("pin2", *p2),
        ],
    }


def make_wire(comp_id: str, p1: tuple[str, str, str], p2: tuple[str, str, str], *, conf: float = 0.90, orient: int = 0) -> CompT:
    return {
        "component_id": comp_id,
        "component_type": "Wire",
        "part_subtype": "",
        "package_type": "jumper_wire_2pin",
        "polarity": "none",
        "orientation": orient,
        "confidence": conf,
        "symmetry_group": [["pin1", "pin2"]],
        "pins": [
            ("pin1", *p1),
            ("pin2", *p2),
        ],
    }


def make_transistor(comp_id: str, b: tuple[str, str, str], c: tuple[str, str, str], e: tuple[str, str, str], *, conf: float = 0.85, orient: int = 0) -> CompT:
    """b/c/e = (canonical_net, hole_id, node_id) for base/collector/emitter."""
    return {
        "component_id": comp_id,
        "component_type": "Transistor",
        "part_subtype": "9013",
        "package_type": "to92_3pin",
        "polarity": "directional",
        "orientation": orient,
        "confidence": conf,
        "symmetry_group": [],
        "pins": [
            ("base", *b),
            ("collector", *c),
            ("emitter", *e),
        ],
    }


def make_potentiometer(comp_id: str, ta: tuple[str, str, str], tb: tuple[str, str, str], wp: tuple[str, str, str], *, conf: float = 0.83, orient: int = 0) -> CompT:
    return {
        "component_id": comp_id,
        "component_type": "Potentiometer",
        "part_subtype": "",
        "package_type": "trim_pot_3pin",
        "polarity": "directional",
        "orientation": orient,
        "confidence": conf,
        "symmetry_group": [["terminal_a", "terminal_b"]],
        "pins": [
            ("terminal_a", *ta),
            ("terminal_b", *tb),
            ("wiper", *wp),
        ],
    }


def make_ic_ua741(comp_id: str, pin_specs: dict[str, tuple[str | None, str, str]], *, conf: float = 0.94) -> CompT:
    """pin_specs = {"1": (None, hole, node), "2": ("INV", hole, node), ...}.

    None 在 canonical_net 位置 → NC pin。
    """
    pins: list[PinT] = []
    for k in ("1", "2", "3", "4", "5", "6", "7", "8"):
        net, hole, node = pin_specs[k]
        pins.append((k, net, hole, node))
    return {
        "component_id": comp_id,
        "component_type": "IC",
        "part_subtype": "UA741",
        "package_type": "dip8",
        "polarity": "directional",
        "orientation": 0,
        "confidence": conf,
        "symmetry_group": [],
        "pins": pins,
    }


# ---------------------------------------------------------------------------
# 把 DSL 落成 runtime_scene dict
# ---------------------------------------------------------------------------

def realize(scene_id: str, comps_dsl: Iterable[CompT]) -> dict:
    """把 DSL components（pins 里直接写 canonical_net 名）展开为 runtime_scene。

    - 自动分配 NET_xxx ID（按首次出现顺序，跳过 None=NC）
    - 自动汇总 nets[].member_hole_ids / member_node_ids / aliases / role / role_label
    - 自动构建 node_index{}
    """
    comps_dsl = list(comps_dsl)
    # 1. 分配 NET_xxx ID
    canon_to_netid: dict[str, str] = {}
    for c in comps_dsl:
        for _pn, canon, _h, _n in c["pins"]:
            if canon is None or canon in canon_to_netid:
                continue
            canon_to_netid[canon] = f"NET_{len(canon_to_netid):03d}"

    # 2. 汇总每条 net 的 member_hole_ids / member_node_ids
    net_holes: dict[str, list[str]] = {nid: [] for nid in canon_to_netid.values()}
    net_nodes: dict[str, list[str]] = {nid: [] for nid in canon_to_netid.values()}
    node_to_holes: dict[str, list[str]] = {}
    for c in comps_dsl:
        for _pn, canon, hole, node in c["pins"]:
            if canon is None:
                # NC pin 仍占据 hole / node，但不进 net
                node_to_holes.setdefault(node, [])
                if hole not in node_to_holes[node]:
                    node_to_holes[node].append(hole)
                continue
            nid = canon_to_netid[canon]
            if hole not in net_holes[nid]:
                net_holes[nid].append(hole)
            if node not in net_nodes[nid]:
                net_nodes[nid].append(node)
            node_to_holes.setdefault(node, [])
            if hole not in node_to_holes[node]:
                node_to_holes[node].append(hole)

    # 3. 构造 components[] —— 把 pin 元组展平
    comps_out: list[dict] = []
    for c in comps_dsl:
        pins_out = []
        for idx, (pn, canon, hole, node) in enumerate(c["pins"]):
            net_id = canon_to_netid[canon] if canon is not None else None
            pins_out.append({
                "hole_id": hole,
                "confidence": c["confidence"],
                "electrical_net_id": net_id,
                "pin_id": idx + 1,
                "pin_name": pn,
                "electrical_node_id": node,
            })
        comp_out = {
            "pins": pins_out,
            "confidence": c["confidence"],
            "component_id": c["component_id"],
            "package_type": c["package_type"],
            "component_type": c["component_type"],
            "orientation": c["orientation"],
            "part_subtype": c["part_subtype"],
            "polarity": c["polarity"],
            "symmetry_group": c["symmetry_group"],
        }
        comps_out.append(comp_out)

    # 4. 构造 nets[]
    nets_out: list[dict] = []
    for canon, nid in canon_to_netid.items():
        role, role_label, power_role = _net_meta(canon)
        n: dict = {
            "labels": [],
            "power_role": power_role,
            "aliases": ([canon] if canon != nid else []) + [nid],
            "electrical_net_id": nid,
            "canonical_name": canon if canon != nid else nid,
            "member_node_ids": net_nodes[nid],
            "member_hole_ids": net_holes[nid],
        }
        if role_label:
            n["role_label"] = role_label
        if role not in ("signal",):
            n["role"] = role
            n["manual_role"] = role
            n["role_source"] = "port_annotation"
        nets_out.append(n)

    # 5. node_index{}
    node_index = {nd: sorted(set(holes), key=str) for nd, holes in node_to_holes.items()}

    return {
        "scene_id": scene_id,
        "board_schema_id": "breadboard_legacy_v1",
        "components": comps_out,
        "nets": nets_out,
        "node_index": node_index,
    }


# ---------------------------------------------------------------------------
# 三个电路的拓扑定义
# ---------------------------------------------------------------------------

def topology_diff_amp() -> list[CompT]:
    """图 1：三极管长尾差动放大器（恒流源偏置）。

    Layout（IC-style 行号约定 ROW_NN_L/R + 电源轨 TRACK_*）：
      - VT1 在 ROW_10（base=F10, collector=G10, emitter=F11）
      - VT2 在 ROW_14（base=F14, collector=G14, emitter=F15）
      - VT3 在 ROW_22（base=F22, collector=F23, emitter=F24）
      - Rc1 / Rc2 拉 VCC ↔ UO1/UO2
      - Rp 调零电位器（三端：terminal_a=E1, terminal_b=E2, wiper=TAIL）
      - R1 / R2 偏置分压；R_E 发射极降压
    """
    return [
        # VT1
        make_transistor("VT1",
            b=("UI1", "I10", "ROW_10_R"),
            c=("UO1", "I9",  "ROW_9_R"),
            e=("E1",  "I11", "ROW_11_R")),
        # VT2
        make_transistor("VT2",
            b=("UI2", "I14", "ROW_14_R"),
            c=("UO2", "I13", "ROW_13_R"),
            e=("E2",  "I15", "ROW_15_R")),
        # VT3 (current source)
        make_transistor("VT3",
            b=("VB3",  "I22", "ROW_22_R"),
            c=("TAIL", "I23", "ROW_23_R"),
            e=("VE3",  "I24", "ROW_24_R")),
        # Rc1: VCC → UO1
        make_resistor("Rc1",
            p1=("VCC", "LP9",  "TRACK_LP_SEG1"),
            p2=("UO1", "J9",   "ROW_9_R"), orient=0),
        # Rc2: VCC → UO2
        make_resistor("Rc2",
            p1=("VCC", "LP13", "TRACK_LP_SEG2"),
            p2=("UO2", "J13",  "ROW_13_R"), orient=0),
        # Rp potentiometer
        make_potentiometer("Rp",
            ta=("E1",   "J11", "ROW_11_R"),
            tb=("E2",   "J15", "ROW_15_R"),
            wp=("TAIL", "J23", "ROW_23_R")),
        # R1: GND ↔ VB3
        make_resistor("R1",
            p1=("GND", "RN22", "TRACK_RN_SEG1"),
            p2=("VB3", "J22",  "ROW_22_R"), orient=0),
        # R2: VB3 ↔ VEE
        make_resistor("R2",
            p1=("VB3", "H22",  "ROW_22_R"),
            p2=("VEE", "LN22", "TRACK_LN_SEG1"), orient=90),
        # R_E: VE3 ↔ VEE
        make_resistor("R_E",
            p1=("VE3", "J24",  "ROW_24_R"),
            p2=("VEE", "LN24", "TRACK_LN_SEG2"), orient=90),
        # 同 net 跨接跳线（学生真实板常见）—— VCC 顶轨两段间桥接
        make_wire("W1",
            p1=("VCC", "LP9",  "TRACK_LP_SEG1"),
            p2=("VCC", "LP13", "TRACK_LP_SEG2"), orient=0),
        # GND 输入 ↔ R1.pin1 同轨补线
        make_wire("W2",
            p1=("GND", "A22",  "ROW_22_L"),
            p2=("GND", "RN22", "TRACK_RN_SEG1"), orient=0),
    ]


def topology_inverting_lpf() -> list[CompT]:
    """图 2：UA741 反相一阶 LPF（C1 与 Rf 并联）。"""
    ic_pins = {
        "1": (None,  "F19", "ROW_19_R"),
        "2": ("INV", "F20", "ROW_20_R"),
        "3": ("V_P", "F21", "ROW_21_R"),
        "4": ("VEE", "F22", "ROW_22_R"),
        "5": (None,  "E22", "ROW_22_L"),
        "6": ("VOUT","D21", "ROW_21_L"),
        "7": ("VCC", "E20", "ROW_20_L"),
        "8": (None,  "E19", "ROW_19_L"),
    }
    return [
        make_ic_ua741("IC1", ic_pins),
        # R1: VIN → INV
        make_resistor("R1",
            p1=("VIN", "I16", "ROW_16_R"),
            p2=("INV", "H20", "ROW_20_R"), orient=0),
        # Rf: INV → VOUT
        make_resistor("Rf",
            p1=("INV", "I20", "ROW_20_R"),
            p2=("VOUT","H24", "ROW_24_R"), orient=0),
        # C1: INV → VOUT (parallel with Rf)
        make_capacitor("C1",
            p1=("INV", "J20", "ROW_20_R"),
            p2=("VOUT","J24", "ROW_24_R"), orient=0),
        # Rp: V_P → GND
        make_resistor("Rp",
            p1=("V_P", "J21", "ROW_21_R"),
            p2=("GND", "RN16","TRACK_RN_SEG1"), orient=90),
        # 跳线：VCC 接 IC1.pin7
        make_wire("W1",
            p1=("VCC", "LN20","TRACK_LN_SEG1"),
            p2=("VCC", "D20", "ROW_20_L"), orient=90),
        # 跳线：VEE 接 IC1.pin4
        make_wire("W2",
            p1=("VEE", "LP22","TRACK_LP_SEG1"),
            p2=("VEE", "J22", "ROW_22_R"), orient=90),
        # 跳线：VOUT 同轨 IC1.pin6 ↔ Rf.pin2 间桥接
        make_wire("W3",
            p1=("VOUT","C21", "ROW_21_L"),
            p2=("VOUT","H24", "ROW_24_R"), orient=90),
    ]


# ---------------------------------------------------------------------------
# WRONG-WIRING topologies (positive-detection gates)
# ---------------------------------------------------------------------------
# 每个 _wrong 版本是对应 _correct 版本的微小扰动 —— 注入 1-2 处典型学生错接，
# 用作 "GNN 能 flag 错误" 的 positive-detection gate。
# 错接选择对应 v4 暴露的 3 类 OOD 盲区，同时充当 v5 改进的量化基准。

def topology_diff_amp_wrong() -> list[CompT]:
    """图 1 错接版：**VT1.base ↔ VT2.base 互换**（学生把差分对基极接反）。

    错接边（GNN 应 flag）：
      - VT1.base → UI2  (cur_port:VT1.base → cur_net:NET_xxx[UI2])
      - VT2.base → UI1  (cur_port:VT2.base → cur_net:NET_xxx[UI1])
    其余拓扑同 _correct，可作为 SwapDiffPairBases perturbation 的金标准 case。
    """
    comps = topology_diff_amp()
    for c in comps:
        if c["component_id"] == "VT1":
            # base pin 改接 UI2
            for i, (pn, canon, hole, node) in enumerate(c["pins"]):
                if pn == "base":
                    c["pins"][i] = (pn, "UI2", hole, node)
        elif c["component_id"] == "VT2":
            for i, (pn, canon, hole, node) in enumerate(c["pins"]):
                if pn == "base":
                    c["pins"][i] = (pn, "UI1", hole, node)
    return comps


def topology_inverting_lpf_wrong() -> list[CompT]:
    """图 2 错接版：**C1.pin1 接到 V_P 而非 INV**（学生把反馈电容接错了输入引脚）。

    错接边（GNN 应 flag）：
      - C1.pin1 → V_P  （原应为 C1.pin1 → INV）
    导致 op-amp 同相输入挂了 C 到 VOUT，反相节点丢了高频反馈 —— 电路实测会
    自激或畸变。
    """
    comps = topology_inverting_lpf()
    for c in comps:
        if c["component_id"] == "C1":
            for i, (pn, canon, hole, node) in enumerate(c["pins"]):
                if pn == "pin1":
                    # 把 C1.pin1 从 INV 改到 V_P；同时挪个 hole 让 layout 看起来真实
                    c["pins"][i] = (pn, "V_P", "K21", "ROW_21_R")
    return comps


def topology_summing_wrong() -> list[CompT]:
    """图 3 错接版：**R2.pin2 接到 V_P 而非 INV**（学生把第二路求和电阻接错引脚）。

    错接边（GNN 应 flag）：
      - R2.pin2 → V_P  （原应为 R2.pin2 → INV）
    导致 VI2 信号不再进入反相求和节点，求和功能失效 —— 输出只反映 VI1 单输入。
    """
    comps = topology_summing()
    for c in comps:
        if c["component_id"] == "R2":
            for i, (pn, canon, hole, node) in enumerate(c["pins"]):
                if pn == "pin2":
                    c["pins"][i] = (pn, "V_P", "K21", "ROW_21_R")
    return comps


def topology_summing() -> list[CompT]:
    """图 3：UA741 两输入反相加法器（VI1=R11/R12 分压器, VI2=外部信号）。"""
    ic_pins = {
        "1": (None,  "F19", "ROW_19_R"),
        "2": ("INV", "F20", "ROW_20_R"),
        "3": ("V_P", "F21", "ROW_21_R"),
        "4": ("VEE", "F22", "ROW_22_R"),
        "5": (None,  "E22", "ROW_22_L"),
        "6": ("VOUT","D21", "ROW_21_L"),
        "7": ("VCC", "E20", "ROW_20_L"),
        "8": (None,  "E19", "ROW_19_L"),
    }
    return [
        make_ic_ua741("IC1", ic_pins),
        # R1: VI1 → INV
        make_resistor("R1",
            p1=("VI1", "I12", "ROW_12_R"),
            p2=("INV", "H20", "ROW_20_R"), orient=0),
        # R2: VI2 → INV
        make_resistor("R2",
            p1=("VI2", "I16", "ROW_16_R"),
            p2=("INV", "J20", "ROW_20_R"), orient=0),
        # Rf: INV → VOUT
        make_resistor("Rf",
            p1=("INV", "I20", "ROW_20_R"),
            p2=("VOUT","H24", "ROW_24_R"), orient=0),
        # Rp: V_P → GND
        make_resistor("Rp",
            p1=("V_P", "J21", "ROW_21_R"),
            p2=("GND", "RN21","TRACK_RN_SEG1"), orient=90),
        # R11: VCC → VI1
        make_resistor("R11",
            p1=("VCC", "LN12","TRACK_LN_SEG1"),
            p2=("VI1", "H12", "ROW_12_R"), orient=90),
        # R12: VI1 → GND
        make_resistor("R12",
            p1=("VI1", "J12", "ROW_12_R"),
            p2=("GND", "RN12","TRACK_RN_SEG2"), orient=90),
        # 跳线：VCC 同轨补线
        make_wire("W1",
            p1=("VCC", "LN20","TRACK_LN_SEG2"),
            p2=("VCC", "D20", "ROW_20_L"), orient=0),
        # 跳线：VEE 接 IC1.pin4
        make_wire("W2",
            p1=("VEE", "LP22","TRACK_LP_SEG1"),
            p2=("VEE", "J22", "ROW_22_R"), orient=90),
        # 跳线：VOUT 同轨 IC1.pin6 ↔ Rf.pin2
        make_wire("W3",
            p1=("VOUT","C21", "ROW_21_L"),
            p2=("VOUT","H24", "ROW_24_R"), orient=0),
    ]


# ---------------------------------------------------------------------------
# Expected-gate spec 模板
# ---------------------------------------------------------------------------

def make_expected(fixture_id: str, reference_id: str, topology_desc: str, edge_count: int, wire_count: int, ic_count: int, res_count: int, *, extra_count_lines: dict[str, int] | None = None) -> dict:
    breakdown = {
        "wire_edge_count": wire_count,
        "ic_edge_count": ic_count,
        "resistor_edge_count": res_count,
    }
    if extra_count_lines:
        breakdown.update(extra_count_lines)
    return {
        "fixture_id": fixture_id,
        "source": f"synthetic runtime_scene fixture generated by scripts/gen_demo_real_student_fixtures.py (2026-05-20)",
        "reference_id": reference_id,
        "topology": topology_desc,
        "rule_verdict_expected": {
            "logic_correct": True,
            "comment": "合成 fixture，拓扑严格匹配 reference。规则路径应判 logic_correct=True。"
                       " 如规则路径报 REF-MISMATCH（例如双电源 vs 单电源 ref），属于 ref 侧问题，与 GNN gate 无关。",
        },
        "gnn_verdict_expected": {
            "all_observed_edges_correct": True,
            "n_observed_edges": edge_count,
            "p_correct_threshold": 0.5,
            **breakdown,
            "comment": "全部 observed (port, net) 边都属于正确拓扑。v4+ ckpt 应满足"
                       " p_correct > 0.5（理想 > 0.7）；任何 < 0.5 视为 false-positive。",
        },
        "stage4_gate": {
            "min_p_correct_all_edges": 0.5,
            "target_p_correct_all_edges": 0.7,
            "disagreement_with_rule_expected": False,
            "suspicious_edges_expected": 0,
        },
    }


def make_expected_wrong(
    fixture_id: str,
    reference_id: str,
    topology_desc: str,
    edge_count: int,
    wire_count: int,
    ic_count: int,
    res_count: int,
    *,
    error_description: str,
    suspicious_edges: list[dict],
    extra_count_lines: dict[str, int] | None = None,
) -> dict:
    """Gate spec for *wrong-wiring* fixtures (positive-detection gates).

    suspicious_edges: 例如
      [{"port": "VT1.base", "wrong_net": "UI2", "correct_net": "UI1",
        "rationale": "differential pair base swap"}, ...]
    """
    breakdown = {
        "wire_edge_count": wire_count,
        "ic_edge_count": ic_count,
        "resistor_edge_count": res_count,
    }
    if extra_count_lines:
        breakdown.update(extra_count_lines)
    return {
        "fixture_id": fixture_id,
        "source": "synthetic runtime_scene fixture generated by scripts/gen_demo_real_student_fixtures.py (2026-05-20)",
        "reference_id": reference_id,
        "topology": topology_desc,
        "injected_error": error_description,
        "rule_verdict_expected": {
            "logic_correct": False,
            "comment": "fixture 注入了 1-2 处错接，规则路径应能识别为 CRITICAL_*  级别不匹配。"
                       " 如规则路径未捕获（语义未覆盖），GNN gate 必须独立 flag —— 这是该 fixture 的核心价值。",
        },
        "gnn_verdict_expected": {
            "all_observed_edges_correct": False,
            "n_observed_edges": edge_count,
            "p_correct_threshold": 0.5,
            "suspicious_edges_expected": len(suspicious_edges),
            **breakdown,
            "comment": "Stage 4 GNN gate (positive-detection)：v5 应在 suspicious_edges 列表中"
                       " 每条边给出 p_correct < 0.5（最低门槛）或 < 0.3（target）。"
                       " 其余 observed 边应维持 p > 0.5（防止 collateral OOD）。",
        },
        "stage4_gate": {
            "suspicious_edges_expected": len(suspicious_edges),
            "min_p_correct_collateral": 0.5,
            "max_p_correct_on_errors": 0.5,
            "target_p_correct_on_errors": 0.3,
            "disagreement_with_rule_expected": False,
            "edges_to_flag": suspicious_edges,
        },
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # build configurations: (slug, reference_id, topology_fn, topology_desc, wrong_spec | None)
    # wrong_spec: None → 只生成 _correct_v1；否则 dict 描述错接注入。
    configs = [
        ("bjt_diff_amp",
         "test_bjt_diff_amp_v1",
         topology_diff_amp,
         "BJT differential amplifier with current-source biasing (VT1/VT2 diff pair + VT3 mirror, Rp 调零电位器), dual supply ±Ucc",
         {
            "topo_fn": topology_diff_amp_wrong,
            "topo_desc_suffix": " · **WRONG**: VT1.base ↔ VT2.base swapped (差分对基极接反)",
            "error_description": "学生把 VT1 和 VT2 的基极接到了对调的输入端：VT1.base 接到了 UI2（应为 UI1），VT2.base 接到了 UI1（应为 UI2）。这是差分对常见错接，电路实测会输出反相结果。",
            "suspicious_edges": [
                {"port": "VT1.base", "wrong_net_canonical": "UI2", "correct_net_canonical": "UI1",
                 "rationale": "differential pair base swap — VT1 应接 UI1"},
                {"port": "VT2.base", "wrong_net_canonical": "UI1", "correct_net_canonical": "UI2",
                 "rationale": "differential pair base swap — VT2 应接 UI2"},
            ],
         }),
        ("opamp_inverting_lpf",
         "test_opamp_inverting_lpf_v1",
         topology_inverting_lpf,
         "UA741 inverting active LPF · Rf ∥ C1 feedback, dual supply (VCC=NET pin7, VEE=NET pin4), 3 jumper wires bridging power rails",
         {
            "topo_fn": topology_inverting_lpf_wrong,
            "topo_desc_suffix": " · **WRONG**: C1.pin1 mis-routed from INV to V_P",
            "error_description": "学生把反馈电容 C1 的一端接到了 op-amp 同相输入 (V_P)，而非反相输入 (INV)。结果是 C 与 Rp 形成同相端 RC，反相节点丢了高频反馈 → 电路会自激或畸变。",
            "suspicious_edges": [
                {"port": "C1.pin1", "wrong_net_canonical": "V_P", "correct_net_canonical": "INV",
                 "rationale": "Cf feedback capacitor mis-connected to non-inverting input"},
            ],
         }),
        ("opamp_summing",
         "test_opamp_summing_v1",
         topology_summing,
         "UA741 two-input inverting summing amp · VI1 from R11/R12 divider off VCC, VI2 from external signal, Rf feedback, Rp bias-comp, dual supply, 3 jumper wires",
         {
            "topo_fn": topology_summing_wrong,
            "topo_desc_suffix": " · **WRONG**: R2.pin2 mis-routed from INV to V_P",
            "error_description": "学生把第二路求和电阻 R2 的输出端接到了 op-amp 同相输入 (V_P)，而非反相求和节点 (INV)。结果是 VI2 信号不再进入求和节点 → 求和功能失效，输出只反映 VI1 单输入。",
            "suspicious_edges": [
                {"port": "R2.pin2", "wrong_net_canonical": "V_P", "correct_net_canonical": "INV",
                 "rationale": "summing resistor R2 mis-connected to non-inverting input"},
            ],
         }),
    ]

    for slug, ref_id, topo_fn, topo_desc, wrong_spec in configs:
        comps_dsl = topo_fn()
        runtime_scene = realize(f"runtime_scene_{slug}", comps_dsl)

        # 落盘 runtime_scene
        out_json = OUT_DIR / f"{slug}_correct_v1.json"
        out_json.write_text(
            json.dumps(runtime_scene, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 数边数（每个 pin 接 net 计一条 observed 边；NC pin 不计）
        edge_count = 0
        wire_edges = ic_edges = res_edges = trans_edges = cap_edges = pot_edges = 0
        for c in runtime_scene["components"]:
            for p in c["pins"]:
                if p["electrical_net_id"] is None:
                    continue
                edge_count += 1
                t = c["component_type"]
                if t == "Wire":
                    wire_edges += 1
                elif t == "IC":
                    ic_edges += 1
                elif t == "Resistor":
                    res_edges += 1
                elif t == "Transistor":
                    trans_edges += 1
                elif t == "Capacitor":
                    cap_edges += 1
                elif t == "Potentiometer":
                    pot_edges += 1

        extra: dict[str, int] = {}
        if trans_edges:
            extra["transistor_edge_count"] = trans_edges
        if cap_edges:
            extra["capacitor_edge_count"] = cap_edges
        if pot_edges:
            extra["potentiometer_edge_count"] = pot_edges

        expected = make_expected(
            fixture_id=f"{slug}_correct_v1",
            reference_id=ref_id,
            topology_desc=topo_desc,
            edge_count=edge_count,
            wire_count=wire_edges,
            ic_count=ic_edges,
            res_count=res_edges,
            extra_count_lines=extra,
        )
        out_exp = OUT_DIR / f"{slug}_correct_v1.expected.json"
        out_exp.write_text(
            json.dumps(expected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"✓ {slug:25s}  comps={len(runtime_scene['components']):2d}  nets={len(runtime_scene['nets']):2d}  edges={edge_count:3d}  → {out_json.name}")

        # ---- WRONG-WIRING variant ----
        if wrong_spec is None:
            continue
        wrong_comps = wrong_spec["topo_fn"]()
        wrong_scene = realize(f"runtime_scene_{slug}_wrong", wrong_comps)
        out_wjson = OUT_DIR / f"{slug}_wrong_v1.json"
        out_wjson.write_text(
            json.dumps(wrong_scene, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 重新统计 wrong 版本的边数（注入错接不改变总边数，但保险起见重数）
        w_edge_count = 0
        w_wire = w_ic = w_res = w_trans = w_cap = w_pot = 0
        for c in wrong_scene["components"]:
            for p in c["pins"]:
                if p["electrical_net_id"] is None:
                    continue
                w_edge_count += 1
                t = c["component_type"]
                if t == "Wire": w_wire += 1
                elif t == "IC": w_ic += 1
                elif t == "Resistor": w_res += 1
                elif t == "Transistor": w_trans += 1
                elif t == "Capacitor": w_cap += 1
                elif t == "Potentiometer": w_pot += 1
        w_extra: dict[str, int] = {}
        if w_trans: w_extra["transistor_edge_count"] = w_trans
        if w_cap:   w_extra["capacitor_edge_count"]  = w_cap
        if w_pot:   w_extra["potentiometer_edge_count"] = w_pot

        wrong_expected = make_expected_wrong(
            fixture_id=f"{slug}_wrong_v1",
            reference_id=ref_id,
            topology_desc=topo_desc + wrong_spec["topo_desc_suffix"],
            edge_count=w_edge_count,
            wire_count=w_wire,
            ic_count=w_ic,
            res_count=w_res,
            error_description=wrong_spec["error_description"],
            suspicious_edges=wrong_spec["suspicious_edges"],
            extra_count_lines=w_extra,
        )
        out_wexp = OUT_DIR / f"{slug}_wrong_v1.expected.json"
        out_wexp.write_text(
            json.dumps(wrong_expected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  ⚠ {slug + '_wrong':25s}  comps={len(wrong_scene['components']):2d}  nets={len(wrong_scene['nets']):2d}  edges={w_edge_count:3d}  injected_errors={len(wrong_spec['suspicious_edges'])}  → {out_wjson.name}")

    print(f"\n✓ wrote 12 files (6 correct + 6 wrong) to {OUT_DIR.relative_to(Path.cwd())}/")


if __name__ == "__main__":
    main()
