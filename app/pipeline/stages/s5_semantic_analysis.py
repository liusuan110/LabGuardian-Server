"""
Stage 5: 电路语义分析

输入 S3/S4 之后的可信 netlist_v2，输出面向教学纠错的电路类型、角色、
结构错误和下一步修正建议。第一版保持确定性规则，避免把电气判断交给 LLM。
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from app.domain.circuit import norm_component_type


POWER_ROLE_VCC = "VCC"
POWER_ROLE_GND = "GND"
WIRE_TYPES = {"Wire"}
CAPACITOR_TYPES = {"Capacitor", "CapacitorCeramic", "CapacitorElectrolytic"}


@dataclass(frozen=True)
class SemanticPin:
    component_id: str
    component_type: str
    pin_name: str
    hole_id: str
    electrical_node_id: str
    electrical_net_id: str


@dataclass(frozen=True)
class SemanticComponent:
    component_id: str
    component_type: str
    pins: tuple[SemanticPin, ...]

    @property
    def net_ids(self) -> list[str]:
        return [pin.electrical_net_id for pin in self.pins if pin.electrical_net_id]

    @property
    def unique_net_ids(self) -> list[str]:
        return list(dict.fromkeys(self.net_ids))


def run_semantic_analysis(
    netlist_v2: dict[str, Any] | None,
    *,
    topology_graph: dict[str, Any] | None = None,
    reference_circuit: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """从 corrected/current netlist_v2 生成电路语义诊断。"""
    t0 = time.time()
    netlist = netlist_v2 or {}
    components = _extract_components(netlist)
    net_roles = _extract_net_roles(netlist)
    net_display_names = _extract_net_display_names(netlist)
    net_pins = _index_pins_by_net(components)

    wiring_errors = _diagnose_wiring_errors(components, net_roles, net_pins, net_display_names)
    template = _match_templates(components, net_roles, wiring_errors)
    recognized_roles = template["recognized_roles"]
    suggested_pin_moves = _build_suggested_pin_moves(template, wiring_errors, net_roles)
    student_hint = _build_student_hint(template, wiring_errors, suggested_pin_moves)

    duration_ms = (time.time() - t0) * 1000
    return {
        "version": "semantic_analysis_v1",
        "circuit_type_guess": template["circuit_type_guess"],
        "recognized_roles": recognized_roles,
        "matched_template": template["matched_template"],
        "wiring_errors": wiring_errors,
        "suggested_pin_moves": suggested_pin_moves,
        "student_hint": student_hint,
        "summary": {
            "component_count": len(components),
            "net_count": len(net_roles),
            "non_wire_component_count": sum(
                1 for comp in components if comp.component_type not in WIRE_TYPES
            ),
            "wiring_error_count": len(wiring_errors),
            "suggested_pin_move_count": len(suggested_pin_moves),
            "topology_node_count": len((topology_graph or {}).get("nodes", [])),
            "topology_edge_count": len((topology_graph or {}).get("links", [])),
            "reference_circuit_present": bool(reference_circuit),
        },
        "duration_ms": duration_ms,
    }


def _extract_components(netlist: dict[str, Any]) -> list[SemanticComponent]:
    out: list[SemanticComponent] = []
    for comp in netlist.get("components") or []:
        component_id = str(comp.get("component_id") or "UNKNOWN")
        component_type = norm_component_type(str(comp.get("component_type") or comp.get("type") or "UNKNOWN"))
        pins: list[SemanticPin] = []
        for idx, pin in enumerate(comp.get("pins") or [], start=1):
            pins.append(
                SemanticPin(
                    component_id=component_id,
                    component_type=component_type,
                    pin_name=str(pin.get("pin_name") or f"pin{idx}"),
                    hole_id=str(pin.get("hole_id") or ""),
                    electrical_node_id=str(pin.get("electrical_node_id") or ""),
                    electrical_net_id=str(pin.get("electrical_net_id") or ""),
                )
            )
        out.append(
            SemanticComponent(
                component_id=component_id,
                component_type=component_type,
                pins=tuple(pins),
            )
        )
    return out


def _extract_net_roles(netlist: dict[str, Any]) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = {}
    for net in netlist.get("nets") or []:
        net_id = str(net.get("electrical_net_id") or net.get("net_id") or "")
        if not net_id:
            continue
        current: set[str] = set()
        explicit_role = str(net.get("power_role") or "").upper()
        if explicit_role in {POWER_ROLE_VCC, POWER_ROLE_GND}:
            current.add(explicit_role)
        for label in [net.get("role_label"), *list(net.get("aliases") or []), *list(net.get("merged_role_labels") or [])]:
            normalized = str(label or "").upper()
            if normalized in {POWER_ROLE_VCC, POWER_ROLE_GND}:
                current.add(normalized)
        for item in list(net.get("member_node_ids") or []) + list(net.get("member_hole_ids") or []):
            inferred = _infer_power_role(str(item))
            if inferred:
                current.add(inferred)
        roles[net_id] = current
    return roles


def _extract_net_display_names(netlist: dict[str, Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for net in netlist.get("nets") or []:
        net_id = str(net.get("electrical_net_id") or net.get("net_id") or "")
        if not net_id:
            continue
        names[net_id] = str(
            net.get("canonical_name")
            or net.get("role_label")
            or net.get("power_role")
            or net_id
        )
    return names


def _infer_power_role(value: str) -> str | None:
    text = value.upper()
    if not text:
        return None
    if text.startswith(("LN", "RN")):
        return POWER_ROLE_GND
    if text.startswith(("LP", "RP")):
        return POWER_ROLE_VCC
    if any(token in text for token in ("GND", "VSS", "PWR_MINUS", "TRACK_LN", "TRACK_RN")):
        return POWER_ROLE_GND
    if any(token in text for token in ("VCC", "VDD", "PWR_PLUS", "TRACK_LP", "TRACK_RP")):
        return POWER_ROLE_VCC
    return None


def _index_pins_by_net(components: list[SemanticComponent]) -> dict[str, list[SemanticPin]]:
    net_pins: dict[str, list[SemanticPin]] = defaultdict(list)
    for comp in components:
        if comp.component_type in WIRE_TYPES:
            continue
        for pin in comp.pins:
            if pin.electrical_net_id:
                net_pins[pin.electrical_net_id].append(pin)
    return dict(net_pins)


def _diagnose_wiring_errors(
    components: list[SemanticComponent],
    net_roles: dict[str, set[str]],
    net_pins: dict[str, list[SemanticPin]],
    net_display_names: dict[str, str],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    for net_id, roles in net_roles.items():
        net_name = net_display_names.get(net_id, net_id)
        if {POWER_ROLE_VCC, POWER_ROLE_GND}.issubset(roles):
            errors.append(
                {
                    "error_code": "POWER_GND_SHORT",
                    "severity": "danger",
                    "message": f"{net_name}: VCC 与 GND 出现在同一电气网络，疑似电源地短路",
                    "net_id": net_name,
                    "source_net_id": net_id,
                    "expected": "VCC and GND separated",
                    "actual": sorted(roles),
                }
            )

    for comp in components:
        if comp.component_type in WIRE_TYPES or len(comp.pins) < 2:
            continue
        unique_nets = [net_id for net_id in comp.unique_net_ids if net_id]
        if len(unique_nets) == 1 and len(comp.net_ids) >= 2:
            net_name = net_display_names.get(unique_nets[0], unique_nets[0])
            errors.append(
                {
                    "error_code": "COMPONENT_SHORTED_SAME_NET",
                    "severity": "error",
                    "message": f"{comp.component_id}: {comp.component_type} 两个引脚落在同一电气网络，元件被短接",
                    "component_id": comp.component_id,
                    "component_type": comp.component_type,
                    "current_net_id": net_name,
                    "source_net_id": unique_nets[0],
                    "current_hole_id": [pin.hole_id for pin in comp.pins],
                    "expected": "component pins on different nets",
                    "actual": net_name,
                }
            )

    for net_id, pins in net_pins.items():
        if net_roles.get(net_id):
            continue
        if len(pins) == 1:
            pin = pins[0]
            # 单端信号节点经常是实验里的 VIN/VOUT 测试点，尤其是电阻端点。
            # S5 只对更可能代表接线遗漏的器件端点给出悬空告警。
            if pin.component_type == "Resistor":
                continue
            net_name = net_display_names.get(net_id, net_id)
            errors.append(
                {
                    "error_code": "FLOATING_PIN",
                    "severity": "warning",
                    "message": f"{pin.component_id}.{pin.pin_name}({pin.hole_id}) 所在网络只有一个元件引脚，可能悬空",
                    "component_id": pin.component_id,
                    "component_type": pin.component_type,
                    "pin_name": pin.pin_name,
                    "current_net_id": net_name,
                    "source_net_id": net_id,
                    "current_hole_id": pin.hole_id,
                    "expected": "signal net connected to at least two non-wire pins or a power rail",
                    "actual": "single_pin_net",
                }
            )

    for comp in components:
        if comp.component_type != "IC":
            continue
        touched_roles = set().union(*(net_roles.get(net_id, set()) for net_id in comp.unique_net_ids))
        if POWER_ROLE_VCC not in touched_roles or POWER_ROLE_GND not in touched_roles:
            errors.append(
                {
                    "error_code": "IC_POWER_RAIL_MISSING",
                    "severity": "warning",
                    "message": f"{comp.component_id}: IC 未同时检测到 VCC 和 GND 供电脚连接",
                    "component_id": comp.component_id,
                    "component_type": comp.component_type,
                    "current_net_id": comp.unique_net_ids,
                    "expected": ["VCC", "GND"],
                    "actual": sorted(touched_roles),
                }
            )

    return errors


def _match_templates(
    components: list[SemanticComponent],
    net_roles: dict[str, set[str]],
    wiring_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [
        _match_rc_low_pass(components, net_roles),
        _match_voltage_divider(components, net_roles),
        _match_led_series(components, net_roles),
        _match_generic_amplifier(components),
    ]
    best = max(candidates, key=lambda item: float(item["matched_template"]["score"]))
    if float(best["matched_template"]["score"]) <= 0:
        type_counts = Counter(comp.component_type for comp in components if comp.component_type not in WIRE_TYPES)
        label = "未知电路"
        if type_counts:
            label = " + ".join(f"{count}x{ctype}" for ctype, count in sorted(type_counts.items()))
        best = {
            "circuit_type_guess": {
                "template_id": "unknown",
                "label": label,
                "confidence": 0.0,
                "reasons": ["未匹配到内置教学模板"],
            },
            "matched_template": {
                "template_id": "unknown",
                "label": "未匹配",
                "score": 0.0,
                "missing": [],
                "matched_roles": {},
            },
            "recognized_roles": {},
        }

    if wiring_errors and best["circuit_type_guess"]["confidence"] > 0.2:
        best["circuit_type_guess"]["confidence"] = max(0.1, best["circuit_type_guess"]["confidence"] - 0.08)
        best["circuit_type_guess"].setdefault("reasons", []).append("存在接线错误，置信度已下调")
    return best


def _match_rc_low_pass(
    components: list[SemanticComponent],
    net_roles: dict[str, set[str]],
) -> dict[str, Any]:
    resistors = [c for c in components if c.component_type == "Resistor" and len(c.unique_net_ids) >= 2]
    capacitors = [c for c in components if c.component_type in CAPACITOR_TYPES and len(c.unique_net_ids) >= 2]
    best: dict[str, Any] | None = None

    for resistor in resistors:
        r_nets = resistor.unique_net_ids[:2]
        for capacitor in capacitors:
            c_nets = capacitor.unique_net_ids[:2]
            common = [net_id for net_id in r_nets if net_id in c_nets and not _is_power_net(net_id, net_roles)]
            if not common:
                continue
            vout = common[0]
            r_other = _other_net(r_nets, vout)
            c_other = _other_net(c_nets, vout)
            cap_to_gnd = _has_role(c_other, net_roles, POWER_ROLE_GND)
            score = 0.88 if cap_to_gnd else 0.58
            missing = [] if cap_to_gnd else ["capacitor_to_gnd"]
            candidate = _template_payload(
                template_id="rc_low_pass",
                label="一阶 RC 低通",
                score=score,
                reasons=[
                    f"{resistor.component_id} 与 {capacitor.component_id} 共享输出节点 {vout}",
                    "电容另一端已接 GND" if cap_to_gnd else "电容另一端尚未识别为 GND",
                ],
                roles={
                    "input_resistor": resistor.component_id,
                    "shunt_capacitor": capacitor.component_id,
                    "vin_node": r_other,
                    "vout_node": vout,
                    "gnd_node": c_other if cap_to_gnd else None,
                },
                missing=missing,
                extra={
                    "rc_shared_node": vout,
                    "rc_resistor_id": resistor.component_id,
                    "rc_capacitor_id": capacitor.component_id,
                    "rc_capacitor_free_net": c_other,
                },
            )
            if best is None or score > best["matched_template"]["score"]:
                best = candidate

    return best or _empty_template("rc_low_pass", "一阶 RC 低通")


def _match_voltage_divider(
    components: list[SemanticComponent],
    net_roles: dict[str, set[str]],
) -> dict[str, Any]:
    resistors = [c for c in components if c.component_type == "Resistor" and len(c.unique_net_ids) >= 2]
    best: dict[str, Any] | None = None
    for i, r_top in enumerate(resistors):
        for r_bot in resistors[i + 1:]:
            shared = [
                net_id
                for net_id in r_top.unique_net_ids
                if net_id in r_bot.unique_net_ids and not _is_power_net(net_id, net_roles)
            ]
            if not shared:
                continue
            mid = shared[0]
            top_other = _other_net(r_top.unique_net_ids[:2], mid)
            bot_other = _other_net(r_bot.unique_net_ids[:2], mid)
            roles = {_role_name(top_other, net_roles), _role_name(bot_other, net_roles)}
            has_vcc_gnd = {POWER_ROLE_VCC, POWER_ROLE_GND}.issubset(roles)
            score = 0.9 if has_vcc_gnd else 0.62
            candidate = _template_payload(
                template_id="voltage_divider",
                label="电阻分压电路",
                score=score,
                reasons=[
                    f"{r_top.component_id} 与 {r_bot.component_id} 共享分压输出节点 {mid}",
                    "两端已连接 VCC/GND" if has_vcc_gnd else "分压两端尚未完整识别为 VCC/GND",
                ],
                roles={
                    "upper_resistor": r_top.component_id,
                    "lower_resistor": r_bot.component_id,
                    "vout_node": mid,
                    "rail_a_node": top_other,
                    "rail_b_node": bot_other,
                },
                missing=[] if has_vcc_gnd else ["divider_power_rails"],
            )
            if best is None or score > best["matched_template"]["score"]:
                best = candidate
    return best or _empty_template("voltage_divider", "电阻分压电路")


def _match_led_series(
    components: list[SemanticComponent],
    net_roles: dict[str, set[str]],
) -> dict[str, Any]:
    leds = [c for c in components if c.component_type == "LED" and len(c.unique_net_ids) >= 2]
    resistors = [c for c in components if c.component_type == "Resistor" and len(c.unique_net_ids) >= 2]
    best: dict[str, Any] | None = None
    for led in leds:
        for resistor in resistors:
            shared = [net_id for net_id in led.unique_net_ids if net_id in resistor.unique_net_ids]
            if not shared:
                continue
            shared_node = shared[0]
            outer_roles = {
                _role_name(_other_net(led.unique_net_ids[:2], shared_node), net_roles),
                _role_name(_other_net(resistor.unique_net_ids[:2], shared_node), net_roles),
            }
            has_supply_loop = {POWER_ROLE_VCC, POWER_ROLE_GND}.issubset(outer_roles)
            score = 0.88 if has_supply_loop else 0.64
            candidate = _template_payload(
                template_id="led_series_resistor",
                label="LED 限流电路",
                score=score,
                reasons=[
                    f"{led.component_id} 与 {resistor.component_id} 串联共享节点 {shared_node}",
                    "外侧已形成 VCC/GND 回路" if has_supply_loop else "外侧电源回路尚不完整",
                ],
                roles={
                    "led": led.component_id,
                    "series_resistor": resistor.component_id,
                    "series_node": shared_node,
                },
                missing=[] if has_supply_loop else ["supply_return_loop"],
            )
            if best is None or score > best["matched_template"]["score"]:
                best = candidate
    return best or _empty_template("led_series_resistor", "LED 限流电路")


def _match_generic_amplifier(components: list[SemanticComponent]) -> dict[str, Any]:
    has_ic = any(comp.component_type == "IC" for comp in components)
    resistor_count = sum(1 for comp in components if comp.component_type == "Resistor")
    if not has_ic or resistor_count < 2:
        return _empty_template("op_amp_amplifier", "运放/IC 放大电路")
    return _template_payload(
        template_id="op_amp_amplifier",
        label="运放/IC 放大电路",
        score=0.35,
        reasons=["检测到 IC 和多个电阻，但第一版尚未识别具体反馈网络"],
        roles={"amplifier_ic_present": True, "resistor_count": resistor_count},
        missing=["op_amp_feedback_template_not_enabled"],
    )


def _template_payload(
    *,
    template_id: str,
    label: str,
    score: float,
    reasons: list[str],
    roles: dict[str, Any],
    missing: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "circuit_type_guess": {
            "template_id": template_id,
            "label": label,
            "confidence": round(score, 3),
            "reasons": reasons,
        },
        "matched_template": {
            "template_id": template_id,
            "label": label,
            "score": round(score, 3),
            "missing": missing,
            "matched_roles": roles,
        },
        "recognized_roles": {key: value for key, value in roles.items() if value is not None},
    }
    if extra:
        payload["matched_template"].update(extra)
    return payload


def _empty_template(template_id: str, label: str) -> dict[str, Any]:
    return _template_payload(
        template_id=template_id,
        label=label,
        score=0.0,
        reasons=[],
        roles={},
        missing=[],
    )


def _build_suggested_pin_moves(
    template: dict[str, Any],
    wiring_errors: list[dict[str, Any]],
    net_roles: dict[str, set[str]],
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    matched = template.get("matched_template") or {}

    if matched.get("template_id") == "rc_low_pass" and "capacitor_to_gnd" in matched.get("missing", []):
        suggestions.append(
            {
                "kind": "connect_to_role",
                "target_role": "GND",
                "component_id": matched.get("rc_capacitor_id"),
                "current_net_id": matched.get("rc_capacitor_free_net"),
                "message": "将电容未接输出节点的另一脚移动到 GND 电源轨，形成 RC 低通的泄放支路",
            }
        )

    for error in wiring_errors:
        if error.get("error_code") == "COMPONENT_SHORTED_SAME_NET":
            suggestions.append(
                {
                    "kind": "separate_component_pins",
                    "component_id": error.get("component_id"),
                    "current_hole_id": error.get("current_hole_id"),
                    "message": "把该元件其中一脚拖到相邻的另一条导通列/导通排，避免两个引脚落在同一网络",
                }
            )
        elif error.get("error_code") == "FLOATING_PIN":
            suggestions.append(
                {
                    "kind": "connect_floating_pin",
                    "component_id": error.get("component_id"),
                    "pin_name": error.get("pin_name"),
                    "current_hole_id": error.get("current_hole_id"),
                    "message": "该引脚所在网络只有它自己，优先检查是否应接到模板中的输入、输出或电源轨节点",
                }
            )
        elif error.get("error_code") == "POWER_GND_SHORT":
            suggestions.append(
                {
                    "kind": "separate_power_rails",
                    "current_net_id": error.get("net_id"),
                    "message": "检查连接 VCC 和 GND 的跳线或元件脚，先断开电源地短路再继续纠错",
                }
            )

    return suggestions


def _build_student_hint(
    template: dict[str, Any],
    wiring_errors: list[dict[str, Any]],
    suggested_pin_moves: list[dict[str, Any]],
) -> str:
    if wiring_errors:
        first_error = wiring_errors[0]
        first_suggestion = suggested_pin_moves[0]["message"] if suggested_pin_moves else "请先处理该错误对应的孔位连接。"
        return f"{first_error['message']}。{first_suggestion}"

    guess = template.get("circuit_type_guess") or {}
    label = guess.get("label") or "当前电路"
    confidence = float(guess.get("confidence") or 0.0)
    missing = (template.get("matched_template") or {}).get("missing") or []
    if missing and suggested_pin_moves:
        return suggested_pin_moves[0]["message"]
    if confidence >= 0.8:
        return f"当前拓扑可识别为{label}，主要电气连接关系基本成立。"
    if confidence > 0:
        return f"当前拓扑接近{label}，但仍有关键连接未完全满足模板。"
    return "当前网表尚未匹配到内置教学模板，请先确认电源轨、输入输出节点和关键元件是否连接完整。"


def _other_net(nets: list[str], net_id: str) -> str | None:
    for item in nets:
        if item != net_id:
            return item
    return None


def _has_role(net_id: str | None, net_roles: dict[str, set[str]], role: str) -> bool:
    return bool(net_id and role in net_roles.get(net_id, set()))


def _role_name(net_id: str | None, net_roles: dict[str, set[str]]) -> str:
    if not net_id:
        return ""
    roles = net_roles.get(net_id, set())
    if POWER_ROLE_VCC in roles:
        return POWER_ROLE_VCC
    if POWER_ROLE_GND in roles:
        return POWER_ROLE_GND
    return "SIGNAL"


def _is_power_net(net_id: str | None, net_roles: dict[str, set[str]]) -> bool:
    return bool(net_id and net_roles.get(net_id))
