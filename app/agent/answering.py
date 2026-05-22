from __future__ import annotations

import hashlib

from typing import Any

from app.agent.contracts import AgentIntent, ConceptPack, ContextPack, RuntimeEvidence
from app.agent.tools import ToolResult
from app.agent.verification import verify_draft_answer
from app.schemas.angnt import AngntCitation, AngntEvidence


def _classify_user_intent(user_message: str) -> str:
    """简单意图分类，用于选择回答风格。"""
    msg = (user_message or "").lower()
    if any(k in msg for k in ("元件", "组件", "有什么", "哪些", "components", "parts", "器件")):
        return "components"
    if any(k in msg for k in ("为什么", "怎么回事", "原因", "解释", "悬空", "为什么判断", "为何")):
        return "explain"
    return "general"


def _describe_components(evidence: RuntimeEvidence, context_pack: ContextPack) -> str:
    """基于 netlist_v2 和 findings 生成元件清单描述。"""
    netlist = evidence.netlist_v2 or {}
    components = netlist.get("components", [])
    if not isinstance(components, list):
        components = []

    # 收集元件信息
    component_summaries: list[str] = []
    for comp in components:
        if not isinstance(comp, dict):
            continue
        cid = comp.get("component_id") or comp.get("id") or "未知元件"
        ctype = comp.get("component_type") or comp.get("type") or ""
        desc = cid
        if ctype:
            desc = f"{cid}（{ctype}）"
        component_summaries.append(desc)

    # 如果没有 netlist 组件，尝试从 findings 中提取
    if not component_summaries:
        seen: set[str] = set()
        for finding in evidence.findings:
            cid = finding.component_id
            if cid and cid not in seen:
                seen.add(cid)
                component_summaries.append(cid)

    if not component_summaries:
        component_summaries.append("未明确识别到元件")

    # 查找潜在问题
    issue_parts: list[str] = []
    for finding in evidence.findings[:3]:
        cid = finding.component_id
        pin = finding.pin_name
        code = finding.error_code
        if code == "FLOATING_PIN" and cid and pin:
            issue_parts.append(
                f"{cid} 的 {pin} 目前被判断为可能悬空，"
                "因为它只映射到了自身或未形成有效参考连接"
            )
        elif code == "COMPONENT_SHORTED_SAME_NET" and cid:
            issue_parts.append(f"{cid} 的两端似乎落在同一导通节点上，存在短接风险")
        elif code == "POLARITY_REVERSED" and cid:
            issue_parts.append(f"{cid} 的极性方向可能需要复核")
        elif cid and pin:
            issue_parts.append(f"{cid} 的 {pin} 存在 {code} 问题")
        elif cid:
            issue_parts.append(f"{cid} 存在 {code} 问题")

    lines: list[str] = []
    if len(component_summaries) == 1 and "未明确" in component_summaries[0]:
        lines.append("当前诊断结果中未明确识别到元件。请确认图像是否清晰、元件是否完整入镜。")
    else:
        lines.append(
            f"当前诊断结果中识别到了 {len(component_summaries)} 个主要对象："
            f"{', '.join(component_summaries)}。"
        )

    if issue_parts:
        lines.append(issue_parts[0] + "。")
        lines.append("建议你检查相关引脚是否插入了正确孔位，并确认是否与目标电路中的电阻/电源/地线形成有效连接。")
    else:
        lines.append("目前暂未检测到明显的结构化连接异常，建议继续验证剩余元件和连接完整性。")

    return "".join(lines)


def _explain_issue(evidence: RuntimeEvidence, context_pack: ContextPack) -> str:
    """基于 findings 生成原因解释。"""
    if not evidence.findings:
        return "当前没有检测到明确的结构化异常。如果仍有疑问，建议对照参考电路逐项核对连接。"

    finding = evidence.findings[0]
    cid = finding.component_id
    pin = finding.pin_name
    code = finding.error_code
    expected = finding.expected
    actual = finding.actual

    parts: list[str] = []

    if code == "FLOATING_PIN" and cid and pin:
        parts.append(
            f"{cid} 的 {pin} 被判断为可能悬空，"
            f"是因为在网表或 validator 检查中，该引脚没有映射到有效的电气节点或其他元件引脚。"
        )
        if expected and actual:
            parts.append(f"期望连接为 {expected}，但实际映射为 {actual}。")
        parts.append(
            "建议你检查该引脚是否确实插入了面包板孔位，"
            "并确认跳线或元件引脚是否与目标电路中的电源、地线或信号节点连通。"
        )
    elif code == "COMPONENT_SHORTED_SAME_NET" and cid:
        parts.append(
            f"{cid} 的两端被检测到落在同一电气节点上，"
            f"这意味着该元件两端电位相同，没有起到应有的作用，相当于被短接。"
        )
        parts.append("建议你重新跨行插接该元件，确保两端位于不同的导通组。")
    elif code == "POLARITY_REVERSED" and cid:
        parts.append(
            f"{cid} 的极性方向与期望不符。"
            f"对于极性器件（如 LED、电解电容、二极管），引脚方向决定了电流是否能正确导通。"
        )
        parts.append("建议核对器件丝印、引脚长度或datasheet，确认正负极后再接入电路。")
    elif code == "NODE_MISMATCH" and cid:
        parts.append(
            f"{cid} 的连接节点与参考电路不一致。"
        )
        if expected and actual:
            parts.append(f"期望节点为 {expected}，但实际为 {actual}。")
        parts.append("建议对照参考电路的网表，确认该元件各引脚所在的电气节点是否正确。")
    else:
        parts.append(f"检测到 {code} 问题")
        if cid:
            parts[-1] += f"，涉及元件 {cid}"
            if pin:
                parts[-1] += f" 的 {pin}"
        parts[-1] += "。"
        if expected and actual:
            parts.append(f"期望值为 {expected}，实际值为 {actual}。")
        parts.append("建议对照 validator 报告和参考电路逐项排查。")

    if evidence.risk_level == "danger":
        parts.append("当前风险等级较高，建议先断电复查，再重新连接。")

    return "".join(parts)


def _build_general_diagnostic_answer(
    station_id: str,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
    user_message: str,
) -> str:
    """通用诊断摘要，不暴露原始证据。"""
    conclusion = diagnostic_conclusion(evidence=evidence, context_pack=context_pack)

    parts: list[str] = []
    parts.append(f"工位 {station_id} 的诊断结论：{conclusion}。")

    # 历史上下文摘要（追问时有用）
    if context_pack.history_summary:
        parts.append(context_pack.history_summary + "。")

    # 1-2 条关键发现，用自然语言描述
    finding_descs: list[str] = []
    for finding in evidence.findings[:2]:
        cid = finding.component_id
        pin = finding.pin_name
        code = finding.error_code
        if code == "FLOATING_PIN" and cid and pin:
            finding_descs.append(f"{cid} 的 {pin} 可能未形成有效连接")
        elif code == "COMPONENT_SHORTED_SAME_NET" and cid:
            finding_descs.append(f"{cid} 存在短接风险")
        elif code == "POLARITY_REVERSED" and cid:
            finding_descs.append(f"{cid} 的极性方向需要复核")
        elif code == "NODE_MISMATCH" and cid:
            finding_descs.append(f"{cid} 的连接节点与参考不符")
        elif cid:
            finding_descs.append(f"{cid} 存在连接异常")
        else:
            finding_descs.append("检测到连接异常")

    if finding_descs:
        parts.append("关键发现：" + ", ".join(finding_descs) + "。")

    msg = str(user_message or "").strip().lower()
    is_wiring_question = _is_wiring_question(msg)
    variant_seed = _stable_seed(
        station_id=station_id,
        evidence=evidence,
        context_pack=context_pack,
        user_message=user_message,
    )

    # 下一步建议
    suggestions: list[str] = []
    for finding in evidence.findings[:3]:
        if finding.suggested_action:
            suggestions.append(finding.suggested_action)

    if is_wiring_question and not evidence.findings:
        suggestions.extend(_wiring_steps_from_snapshot(snapshot=evidence.circuit_snapshot, limit=6))
        if not suggestions:
            suggestions.extend(
                _wiring_steps_from_reference(
                    evidence=evidence,
                    reference_label=_reference_label(evidence),
                    seed=variant_seed,
                    limit=6,
                )
            )
        suggestions.extend(_action_suggestions_from_error_codes(evidence=evidence, context_pack=context_pack, seed=variant_seed, limit=4))
        suggestions.extend(
            _wiring_generic_steps(seed=variant_seed, limit=3)
        )
    else:
        suggestions.extend(_action_suggestions_from_findings(evidence=evidence, seed=variant_seed, limit=6))
        if not evidence.findings:
            suggestions.extend(
                _action_suggestions_from_error_codes(
                    evidence=evidence,
                    context_pack=context_pack,
                    seed=variant_seed,
                    limit=4,
                )
            )

    suggestions.extend(extract_fix_steps(tool_results)[:4])
    suggestions = _dedupe_texts(suggestions)[:6]

    if not suggestions:
        suggestions.append("对照参考电路逐项核对元件和连接")

    if evidence.risk_level == "danger":
        parts.append("安全提示：当前风险等级较高，建议先断电，再优先检查短路、极性和电源轨连接情况。")
    elif evidence.risk_level == "warning":
        parts.append("建议：按诊断发现逐项排查，先检查最前面的风险原因，再核对参考电路。")
    else:
        parts.append("建议：当前风险较低，继续验证剩余元件和连接完整性即可。")

    if suggestions:
        numbered = "\n".join(f"{idx}) {item}" for idx, item in enumerate(suggestions[:5], start=1))
        parts.append(("接线建议：\n" if is_wiring_question else "具体建议：\n") + numbered)

    follow_ups = _build_follow_up_suggestions(evidence=evidence, context_pack=context_pack)
    if follow_ups:
        parts.append("追问建议：" + "；".join(follow_ups))

    return "".join(parts)


def _is_wiring_question(msg: str) -> bool:
    tokens = (
        "怎么接",
        "怎么连",
        "怎么连接",
        "如何接",
        "如何连",
        "接线",
        "连线",
        "导线",
        "跳线",
        "wire",
        "wiring",
        "jumper",
    )
    return any(t in msg for t in tokens)


def _looks_like_fix_or_wiring_question(message: str) -> bool:
    msg = str(message or "").strip().lower()
    tokens = (
        "怎么接",
        "怎么连",
        "怎么连接",
        "如何接",
        "如何连",
        "接线",
        "连线",
        "导线",
        "跳线",
        "怎么修",
        "怎么改",
        "怎么排查",
        "怎么检查",
        "怎么处理",
        "哪里错",
        "哪里不对",
        "有问题",
        "不对劲",
        "错接",
        "短路",
        "悬空",
        "反了",
        "wire",
        "wiring",
        "fix",
    )
    return any(t in msg for t in tokens)


def _extract_target_component(message: str, evidence: RuntimeEvidence) -> str:
    msg = str(message or "").strip()
    if not msg:
        return ""
    msg_lower = msg.lower()

    candidates: list[str] = []
    for comp in (evidence.netlist_v2 or {}).get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        cid = str(comp.get("component_id") or "").strip()
        if cid:
            candidates.append(cid)
    for finding in evidence.findings or []:
        cid = str(getattr(finding, "component_id", "") or "").strip()
        if cid:
            candidates.append(cid)
    for ref in evidence.evidence_refs or []:
        cid = str(getattr(ref, "component_id", "") or "").strip()
        if cid:
            candidates.append(cid)

    uniq: list[str] = []
    for cid in candidates:
        if cid not in uniq:
            uniq.append(cid)

    for cid in sorted(uniq, key=len, reverse=True):
        if cid.lower() in msg_lower:
            return cid
    return ""


def _build_component_fix_answer(
    *,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
    station_id: str,
    user_message: str,
    target_component_id: str,
) -> str:
    cid = str(target_component_id or "").strip()
    seed = _stable_seed(
        station_id=station_id,
        evidence=evidence,
        context_pack=context_pack,
        user_message=user_message,
    ) + f"|component={cid}"
    comp = _component_by_id(evidence=evidence, component_id=cid)
    comp_type = str((comp or {}).get("component_type") or (comp or {}).get("type") or "").strip()
    ref_label = _reference_label(evidence)

    related_findings = [f for f in (evidence.findings or []) if str(getattr(f, "component_id", "") or "") == cid]
    suggestions: list[str] = []
    suggestions.extend(_component_pin_check_steps(evidence=evidence, component_id=cid, seed=seed, limit=4))
    if related_findings:
        suggestions.extend(
            _action_suggestions_from_findings(
                evidence=evidence,
                seed=seed,
                limit=6,
                findings_override=related_findings,
            )
        )
    else:
        suggestions.extend(
            _choose_component_generic_steps(
                seed=seed,
                component_id=cid,
                component_type=comp_type,
                reference_label=ref_label,
            )
        )
        suggestions.extend(_action_suggestions_from_error_codes(evidence=evidence, context_pack=context_pack, seed=seed, limit=3))

    suggestions.extend(extract_fix_steps(tool_results)[:3])
    suggestions = _dedupe_texts(suggestions)[:7]

    header = f"你问的是 {cid}" + (f"（{comp_type}）" if comp_type else "") + "，我按它在当前电路里的连接关系给你一套更具体的核对步骤。"
    if ref_label:
        header += f"参考电路：{ref_label}。"

    lines: list[str] = [header]
    if suggestions:
        numbered = "\n".join(f"{idx}) {item}" for idx, item in enumerate(suggestions[:6], start=1))
        lines.append("具体建议：\n" + numbered)

    followups = _component_follow_up_questions(evidence=evidence, component_id=cid, seed=seed)
    if followups:
        lines.append("追问建议：" + "；".join(followups))
    return "".join(lines)


def _component_by_id(*, evidence: RuntimeEvidence, component_id: str) -> dict[str, Any] | None:
    for comp in (evidence.netlist_v2 or {}).get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        if str(comp.get("component_id") or "").strip() == component_id:
            return comp
    return None


def _net_power_role_map(evidence: RuntimeEvidence) -> dict[str, str]:
    out: dict[str, str] = {}
    for net in (evidence.netlist_v2 or {}).get("nets", []) or []:
        if not isinstance(net, dict):
            continue
        net_id = str(net.get("electrical_net_id") or net.get("net_id") or "").strip()
        if not net_id:
            continue
        role = str(net.get("power_role") or "").strip()
        if role:
            out[net_id] = role
    return out


def _component_pin_check_steps(*, evidence: RuntimeEvidence, component_id: str, seed: str, limit: int = 4) -> list[str]:
    comp = _component_by_id(evidence=evidence, component_id=component_id) or {}
    pins = comp.get("pins", []) if isinstance(comp, dict) else []
    if not isinstance(pins, list) or not pins:
        return [
            _choose(
                seed + "|pin|fallback",
                (
                    f"先确认 {component_id} 两端都插牢：两脚必须落在不同导通排，避免两端在同一排导致电路不起作用。",
                    f"先做最基础排查：断电后重新插紧 {component_id}，确认两端没有插在同一导通排/同一电源轨。",
                ),
            )
        ][:limit]

    role_map = _net_power_role_map(evidence)
    steps: list[str] = []
    for pin in pins[:3]:
        if not isinstance(pin, dict):
            continue
        pin_name = str(pin.get("pin_name") or pin.get("pin") or "").strip()
        net_id = str(pin.get("electrical_net_id") or "").strip()
        role = role_map.get(net_id, "")
        connections = _pin_connected_components(
            evidence=evidence,
            component_id=component_id,
            pin_name=pin_name,
            limit=4,
        )
        conn_text = "、".join(connections) if connections else ""
        if role:
            steps.append(f"核对 {component_id}.{pin_name} 是否应接到 {role}（并确认共地/供电轨没有接反）。")
        elif conn_text:
            steps.append(f"核对 {component_id}.{pin_name} 当前与 {conn_text} 同网：确认这是不是参考电路要求的连接。")
        else:
            steps.append(f"核对 {component_id}.{pin_name}：先确认插孔接触可靠，再用通断档确认它是否连到目标节点。")
        if len(steps) >= limit:
            break
    return _dedupe_texts(steps)[:limit]


def _pin_connected_components(
    *,
    evidence: RuntimeEvidence,
    component_id: str,
    pin_name: str,
    limit: int = 4,
) -> list[str]:
    comp = _component_by_id(evidence=evidence, component_id=component_id) or {}
    pins = comp.get("pins", []) if isinstance(comp, dict) else []
    net_id = ""
    for pin in pins or []:
        if not isinstance(pin, dict):
            continue
        name = str(pin.get("pin_name") or pin.get("pin") or "").strip()
        if name == pin_name:
            net_id = str(pin.get("electrical_net_id") or "").strip()
            break
    if not net_id:
        return []

    connected: list[str] = []
    for other in (evidence.netlist_v2 or {}).get("components", []) or []:
        if not isinstance(other, dict):
            continue
        other_id = str(other.get("component_id") or "").strip()
        if not other_id or other_id == component_id:
            continue
        other_type = str(other.get("component_type") or other.get("type") or "").strip().lower()
        if other_type in {"wire", "jumper", "lead", "dupont"} or other_id.lower().startswith("w"):
            continue
        for op in other.get("pins", []) or []:
            if not isinstance(op, dict):
                continue
            if str(op.get("electrical_net_id") or "").strip() == net_id:
                pin_n = str(op.get("pin_name") or op.get("pin") or "").strip()
                label = f"{other_id}.{pin_n}" if pin_n else other_id
                if label not in connected:
                    connected.append(label)
                if len(connected) >= limit:
                    return connected
    return connected


def _choose_component_generic_steps(
    *,
    seed: str,
    component_id: str,
    component_type: str,
    reference_label: str,
) -> list[str]:
    ctype = (component_type or "").lower()
    ref = reference_label or ""
    out: list[str] = []

    if ctype in {"resistor", "r"} or component_id.lower().startswith("r"):
        out.append(
            _choose(
                seed + "|cg|r|1",
                (
                    f"如果 {component_id} 是电阻：先核对阻值/型号是否符合参考电路，再确认两端确实跨在两个不同节点上。",
                    f"{component_id}（电阻）优先检查两点：阻值是否对、两脚是否跨在不同导通排（不然等效短接）。",
                ),
            )
        )
        if "rc_highpass" in ref.lower() or "高通" in ref:
            out.append(f"在 RC 高通里，电阻通常负责把输出节点下拉到地：你可以重点确认 {component_id} 是否有一端在输出节点、另一端在地。")
        if "rc_lowpass" in ref.lower() or "低通" in ref:
            out.append(f"在 RC 低通里，电阻通常串在输入到输出之间：你可以重点确认 {component_id} 是否位于输入与输出节点之间。")
    elif ctype in {"capacitor", "c"} or component_id.lower().startswith("c"):
        out.append(
            _choose(
                seed + "|cg|c|1",
                (
                    f"如果 {component_id} 是电容：先确认两端分别落在目标两个节点上（并注意是否为极性电容）。",
                    f"{component_id}（电容）先核对连接位置：它应该跨在两个节点之间，别两端插在同一导通排。",
                ),
            )
        )
        if "rc_highpass" in ref.lower() or "高通" in ref:
            out.append(f"在 RC 高通里，电容通常串联在输入与输出之间：你可以重点确认 {component_id} 是否真的串在“输入→输出”路径上。")
        if "rc_lowpass" in ref.lower() or "低通" in ref:
            out.append(f"在 RC 低通里，电容通常从输出节点并到地：你可以重点确认 {component_id} 是否有一端在输出节点、另一端在地。")
    elif ctype in {"led", "diode"} or component_id.lower().startswith("led"):
        out.append(
            _choose(
                seed + "|cg|led|1",
                (
                    f"如果 {component_id} 是 LED/二极管：先核对方向与限流电阻是否到位，再检查它两端是不是接到正确节点。",
                    f"{component_id}（LED/二极管）先处理方向问题：方向反了通常不亮；再确认串联限流电阻存在。",
                ),
            )
        )
    else:
        out.append(
            _choose(
                seed + "|cg|x|1",
                (
                    f"先把 {component_id} 的每个引脚对应到一个明确节点，再对照参考电路逐项核对是否接到该去的地方。",
                    f"针对 {component_id}：建议按“引脚→节点→连接对象”三步核对，先用通断档确认连通性，再对照参考电路确认节点。",
                ),
            )
        )

    return _dedupe_texts(out)[:4]


def _component_follow_up_questions(*, evidence: RuntimeEvidence, component_id: str, seed: str) -> list[str]:
    comp = _component_by_id(evidence=evidence, component_id=component_id) or {}
    pins = comp.get("pins", []) if isinstance(comp, dict) else []
    pin_names = [
        str(p.get("pin_name") or p.get("pin") or "").strip()
        for p in (pins or [])
        if isinstance(p, dict)
    ]
    pin_names = [p for p in pin_names if p][:3]
    out: list[str] = []
    if pin_names:
        out.append(f"{component_id} 你想确认的是哪一脚：{ ' / '.join(pin_names) }？")
    out.append(
        _choose(
            seed + "|fq|1",
            (
                f"{component_id} 两端目前分别接到了哪些对象/节点？（可以描述同一导通排上的元件或用通断档测）",
                f"{component_id} 的两脚现在各自连到哪里？你可以用通断档说一下它分别和哪些点导通。",
            ),
        )
    )
    out.append(
        _choose(
            seed + "|fq|2",
            (
                "你测到的输入/输出波形或关键节点电压是什么（通电后）？",
                "现象是什么：输出有没有变化、是否明显衰减/相位变化/不稳定？",
            ),
        )
    )
    if evidence.circuit_snapshot:
        out.append(
            _choose(
                seed + "|fq|3",
                (
                    "电路快照里对这个元件的描述是哪一句？我可以按那一句逐条对照给你检查点。",
                    "把电路快照里提到该元件的那一行贴出来，我可以更精确地指出应该连到哪。",
                ),
            )
        )
    return _dedupe_texts(out)[:4]

def _stable_seed(
    *,
    station_id: str,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    user_message: str,
) -> str:
    first_code = evidence.error_codes[0] if evidence.error_codes else ""
    first_finding = evidence.findings[0] if evidence.findings else None
    first_component = str(getattr(first_finding, "component_id", "") or "") if first_finding else ""
    first_ref = evidence.evidence_refs[0] if evidence.evidence_refs else None
    ref_label = _reference_label(evidence)
    ref_token = ""
    if first_ref:
        ref_token = ":".join(
            str(item or "")
            for item in (first_ref.ref_id, first_ref.component_id, first_ref.pin_name, first_ref.hole_id)
        )
    base = "|".join(
        [
            str(station_id or ""),
            str(first_code or ""),
            str(first_component or ""),
            str(context_pack.error_family or ""),
            str(ref_label or ""),
            str(ref_token or ""),
            str((user_message or "")[:80]),
        ]
    )
    return base


def _choose(seed: str, options: tuple[str, ...]) -> str:
    if not options:
        return ""
    payload = (seed or "").encode("utf-8", errors="ignore")
    digest = hashlib.md5(payload).hexdigest()
    idx = int(digest[:8], 16) % len(options)
    return options[idx]


def _wiring_steps_from_snapshot(*, snapshot: str, limit: int = 6) -> list[str]:
    text = str(snapshot or "").strip()
    if not text:
        return []
    raw_lines = [line.strip(" \t\r\n-•") for line in text.splitlines()]
    chunks: list[str] = []
    for line in raw_lines:
        if not line:
            continue
        for part in line.replace("。", "；").split("；"):
            p = part.strip()
            if p:
                chunks.append(p)
    steps: list[str] = []
    for chunk in chunks:
        if len(chunk) < 4:
            continue
        steps.append("按电路快照核对：" + chunk if not chunk.startswith("按") else chunk)
        if len(steps) >= limit:
            break
    return steps


def _wiring_steps_from_reference(
    *,
    evidence: RuntimeEvidence,
    reference_label: str,
    seed: str,
    limit: int = 6,
) -> list[str]:
    label = str(reference_label or "").lower()
    steps: list[str] = []
    resistor = _component_label_by_hint(evidence=evidence, hints=("resistor", "电阻", "r"))
    capacitor = _component_label_by_hint(evidence=evidence, hints=("capacitor", "电容", "c"))
    led = _component_label_by_hint(evidence=evidence, hints=("led", "发光", "d"))
    r_text = resistor or "电阻"
    c_text = capacitor or "电容"
    led_text = led or "LED/负载"

    if "rc_highpass" in label or "高通" in reference_label:
        steps.append(
            _choose(
                seed + "|hp|1",
                (
                    f"按高通思路连：输入先串联经过 {c_text}，到输出节点。",
                    f"高通连接：让信号先通过 {c_text}（串联），再到输出节点。",
                    f"先把 {c_text} 串在输入与输出节点之间，输出点取在 {c_text} 后侧。",
                ),
            )
        )
        steps.append(
            _choose(
                seed + "|hp|2",
                (
                    f"在输出节点用 {r_text} 下拉到地（输出通常取在 {r_text} 上）。",
                    f"输出节点接 {r_text} 到地，构成 RC 高通（输出点看 {r_text} 对地）。",
                    f"把 {r_text} 一端接输出节点，另一端接地，输出用“输出节点对地”测。",
                ),
            )
        )
        steps.append(
            _choose(
                seed + "|hp|3",
                (
                    "信号源 GND、面包板地、示波器地必须共地，否则输出会漂或测不准。",
                    "先把地线统一：信号源 GND 与示波器地夹都接到同一地轨/地节点。",
                ),
            )
        )
        steps.append(
            _choose(
                seed + "|hp|4",
                (
                    "示波器建议：CH1 看输入，CH2 看输出；探头地夹一定夹地，别夹到输出节点。",
                    "测量建议：CH1 输入、CH2 输出；地夹只夹地轨，避免把输出短到地。",
                ),
            )
        )
    elif "rc_lowpass" in label or "低通" in reference_label:
        steps.append(
            _choose(
                seed + "|lp|1",
                (
                    f"按低通思路连：输入先串联经过 {r_text}，到输出节点。",
                    f"低通连接：让输入先过 {r_text}（串联）再到输出点。",
                    f"先把 {r_text} 串在输入与输出节点之间，输出点取在 {r_text} 后侧。",
                ),
            )
        )
        steps.append(
            _choose(
                seed + "|lp|2",
                (
                    f"在输出节点把 {c_text} 并到地（输出通常看输出节点对地）。",
                    f"输出节点并联 {c_text} 到地，构成 RC 低通（输出点看节点对地）。",
                    f"{c_text} 一端接输出节点、另一端接地，输出用“输出节点对地”测。",
                ),
            )
        )
        steps.append(
            _choose(
                seed + "|lp|3",
                (
                    "信号源 GND 与示波器地必须接到同一地节点/地轨，保证参考一致。",
                    "共地优先：信号源地与示波器地夹都接地轨，否则波形会漂。",
                ),
            )
        )
        steps.append(
            _choose(
                seed + "|lp|4",
                (
                    "示波器：CH1 输入、CH2 输出；地夹夹地轨，避免夹到输出点造成短路。",
                    "测量：CH1 看输入、CH2 看输出；确认地夹位置正确。",
                ),
            )
        )
    elif "basic_series_resistor" in label or "串联电阻" in reference_label:
        steps.append(
            _choose(
                seed + "|sr|1",
                (
                    f"供电链路按串联来：电源正端 → {r_text} → {led_text} → 地（电源负端）。",
                    f"串联电阻连接：V+ → {r_text} → {led_text} → GND。",
                    f"先把 {r_text} 串到回路里：V+ 经 {r_text} 再到 {led_text}，最后回到地。",
                ),
            )
        )
        steps.append(
            _choose(
                seed + "|sr|2",
                (
                    f"如果包含 LED，先核对方向后再上电：方向反了通常不亮。",
                    f"极性器件要先看方向（如 LED/二极管），方向错了会导致不导通或异常。",
                ),
            )
        )
        steps.append(
            _choose(
                seed + "|sr|3",
                (
                    "上电前确认电阻值与供电电压匹配，避免电流过大。",
                    "先核对限流电阻是否到位，再上电测试。",
                ),
            )
        )

    if len(steps) > limit:
        return steps[:limit]
    return [step for step in steps if step][:limit]


def _wiring_generic_steps(*, seed: str, limit: int = 3) -> list[str]:
    pool = [
        _choose(
            seed + "|wg|1",
            (
                "先把“输入/输出/地”三点标清：信号源 GND 与示波器地必须共地，再确认输出点取在正确节点。",
                "先固定测量基准：信号源地、示波器地夹都接到地轨，然后再找输出节点。",
            ),
        ),
        _choose(
            seed + "|wg|2",
            (
                "用万用表蜂鸣档做两类确认：该通的两端是否真通；不该通的两点是否被误短接。",
                "断电后用通断档快速扫一遍：导线两端是否导通、以及电源与地之间是否意外导通。",
            ),
        ),
        _choose(
            seed + "|wg|3",
            (
                "如果面包板跨中缝插接：两脚器件务必跨到不同导通排，避免两端落在同一排导致等效短接。",
                "检查两端器件是否插在同一导通排：如果是，两端电位相同，电路不会按预期工作。",
            ),
        ),
    ]
    return [item for item in _dedupe_texts(pool) if item][:limit]


def _dedupe_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if text not in result:
            result.append(text)
    return result


def _action_suggestions_from_findings(
    *,
    evidence: RuntimeEvidence,
    seed: str,
    limit: int = 6,
    findings_override: list[Any] | None = None,
) -> list[str]:
    suggestions: list[str] = []
    source_findings = findings_override if findings_override is not None else (evidence.findings or [])
    for finding in source_findings[:4]:
        cid = str(finding.component_id or "").strip()
        pin = str(finding.pin_name or "").strip()
        code = str(finding.error_code or "").strip()
        expected = str(getattr(finding, "expected", "") or "").strip()
        actual = str(getattr(finding, "actual", "") or "").strip()

        subject = cid
        if cid and pin:
            subject = f"{cid}.{pin}"
        elif pin:
            subject = pin

        if code == "FLOATING_PIN":
            suggestions.append(
                _choose(
                    seed + f"|fp|{subject}|1",
                    (
                        f"断电后确认 {subject} 真实插入孔位，并与导通排/跳线可靠接触（避免只插半格或插在松孔）。",
                        f"先处理悬空：断电重新插紧 {subject}，并确认引脚没有插偏到隔壁孔。",
                    ),
                )
            )
            if expected:
                suggestions.append(
                    _choose(
                        seed + f"|fp|{subject}|2|{expected}",
                        (
                            f"用万用表蜂鸣档核对 {subject} 是否应与 {expected} 连通；不连通就重插/更换跳线。",
                            f"通断档验证 {subject}→{expected}：应当导通；若不导通，优先换一根跳线再测。",
                        ),
                    )
                )
            else:
                suggestions.append(
                    _choose(
                        seed + f"|fp|{subject}|3",
                        (
                            f"用万用表蜂鸣档从 {subject} 往外测连通性，确认它是否连到目标节点（电源/地/信号节点）。",
                            f"从 {subject} 起点做连通性追踪：先确认是否接到地/电源轨，再确认信号节点是否正确。",
                        ),
                    )
                )
        elif code in {"COMPONENT_SHORTED_SAME_NET", "POWER_RAIL_SHORT", "SHORT_CIRCUIT"}:
            culprit = cid or "相关元件"
            suggestions.append(
                _choose(
                    seed + f"|sc|{culprit}|1",
                    (
                        f"断电后检查 {culprit} 两端是否落在同一导通排/同一电源轨；必要时把两端跨到不同导通组。",
                        f"先排查同网：确认 {culprit} 两端没有插在同一排（否则等效短接/不起作用）。",
                    ),
                )
            )
            suggestions.append(
                _choose(
                    seed + f"|sc|{culprit}|2",
                    (
                        "先把最可疑的跳线拔掉再测短路是否消失，然后逐根加回去定位是哪一根造成短接。",
                        "用“减法”定位：断电后先移除新增跳线/器件，再逐步恢复，找出把两点短到一起的那一步。",
                    ),
                )
            )
            suggestions.append(
                _choose(
                    seed + f"|sc|{culprit}|3",
                    (
                        "断电用万用表测电源正负/电源与地之间的电阻，确认是否存在明显短路（阻值很低）。",
                        "断电后测 VCC-GND 阻值/通断：若接近短路，优先检查电源轨跨接与地夹位置。",
                    ),
                )
            )
        elif code == "POLARITY_REVERSED":
            device = cid or "极性器件"
            suggestions.append(
                _choose(
                    seed + f"|pr|{device}|1",
                    (
                        f"核对 {device} 的方向标记（丝印/缺口/长短脚），按参考电路把方向调回正确。",
                        f"先校对极性：把 {device} 的正负/阴阳极与参考一致后再上电。",
                    ),
                )
            )
            suggestions.append(
                _choose(
                    seed + f"|pr|{device}|2",
                    (
                        "如果是 LED/二极管/电解电容，确认正负极或阴阳极后再上电，避免反接损坏。",
                        "极性器件先确认方向再上电；若不确定，先断电对照 datasheet/丝印。",
                    ),
                )
            )
        elif code in {"NODE_MISMATCH", "WRONG_CONNECTION"}:
            if expected and actual:
                suggestions.append(
                    _choose(
                        seed + f"|nm|{subject}|{actual}->{expected}",
                        (
                            f"对照参考电路把 {subject} 从 {actual} 调整到 {expected}（先断电再改线）。",
                            f"按参考把 {subject} 迁回目标节点：从 {actual} 改到 {expected}，改完再复测连通性。",
                        ),
                    )
                )
            suggestions.append(
                _choose(
                    seed + f"|nm|{cid}|pins",
                    (
                        f"按“元件引脚→电气节点”逐项核对 {cid or '相关元件'} 的每个引脚，确保与参考一致。",
                        f"把 {cid or '相关元件'} 的每个引脚都对应到一个明确节点，再与参考电路逐项对齐。",
                    ),
                )
            )
        elif code in {"REFERENCE_NOT_SET", "REFERENCE_MISSING"}:
            suggestions.append(
                _choose(
                    seed + "|ref|missing",
                    (
                        "先确认系统里选择/加载的参考电路是否正确，否则所有对比结论都会偏差。",
                        "检查参考电路是否选对/加载成功，再看错接与缺件结论是否仍然存在。",
                    ),
                )
            )
        elif code in {"MISSING_COMPONENT", "COMPONENT_MISSING"}:
            suggestions.append(
                _choose(
                    seed + f"|mc|{cid}",
                    (
                        f"确认 {cid or '相关元件'} 是否缺失/未入镜/未插稳；必要时重新插紧并复拍全景图。",
                        f"先确认器件到位：{cid or '相关元件'} 是否真的插上且两脚都插牢，必要时换孔重插。",
                    ),
                )
            )
        elif code:
            if expected and actual:
                suggestions.append(
                    _choose(
                        seed + f"|gen|{code}|{subject}|{actual}->{expected}",
                        (
                            f"针对 {code}：把 {subject} 从 {actual} 调整到 {expected}（按参考电路逐项核对）。",
                            f"按 {code} 的提示把 {subject} 迁到目标连接（{expected}），调整后再复跑一次诊断验证。",
                        ),
                    )
                )
            else:
                suggestions.append(
                    _choose(
                        seed + f"|gen|{code}|{subject}|nocmp",
                        (
                            f"针对 {code}：优先核对 {subject} 的孔位与跳线连通性，再对照参考电路确认目标节点。",
                            f"{code} 先从连通性排查：确认 {subject} 插孔正确、跳线可靠，再对照参考定位目标节点。",
                        ),
                    )
                )

        if len(suggestions) >= limit:
            break

    if evidence.risk_level == "danger":
        suggestions.append(
            _choose(
                seed + "|danger|1",
                (
                    "在确认短路排除前不要上电；上电前先断电检查一遍电源轨、极性器件和可能发热元件。",
                    "风险较高：先断电排除短路/反接，再上电；必要时从电源轨开始逐段排查。",
                ),
            )
        )

    return _dedupe_texts(suggestions)[:limit]


def _action_suggestions_from_error_codes(
    *,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    seed: str,
    limit: int = 4,
) -> list[str]:
    codes = list(evidence.error_codes or [])
    if not codes:
        return []
    ref = _reference_label(evidence)
    components = _component_labels(evidence)
    comp_text = "、".join(components[:3]) if components else ""
    out: list[str] = []
    for code in codes[:3]:
        if code in {"NODE_MISMATCH", "WRONG_CONNECTION", "HOLE_MISMATCH"}:
            out.append(
                _choose(
                    seed + f"|ec|{code}|1",
                    (
                        f"先处理 {code}：对照参考电路 {ref}，按“元件引脚→节点/孔位”逐项核对（优先检查 {comp_text or '输出节点附近'}）。",
                        f"{code} 通常是错接：对照参考 {ref}，先从最关键节点（输入/输出/地）开始核对连通性。",
                    ),
                )
            )
        elif code in {"COMPONENT_MISSING", "COMPONENT_INSTANCE_MISSING", "MISSING_COMPONENT"}:
            out.append(
                _choose(
                    seed + f"|ec|{code}|2",
                    (
                        f"{code} 先做清点：把参考 {ref} 需要的器件逐个摆出来并逐个确认插入到位（{comp_text or '电阻/电容/导线'}）。",
                        f"先排除缺件：按参考 {ref} 逐个核对器件是否都有且插牢，必要时复拍全景。",
                    ),
                )
            )
        elif code in {"COMPONENT_EXTRA", "PIN_EXTRA"}:
            out.append(
                _choose(
                    seed + f"|ec|{code}|3",
                    (
                        f"{code} 表示可能多接了：先移除最近新增的跳线/器件，再逐步加回去定位多余连接来源。",
                        f"先做减法：暂时拔掉不在参考 {ref} 里的那根跳线/器件，看错误码是否消失。",
                    ),
                )
            )
        elif code in {"POWER_NODE_MISMATCH"}:
            out.append(
                _choose(
                    seed + f"|ec|{code}|4",
                    (
                        f"{code} 优先核对电源轨：VCC/GND 走线是否接到了正确电源轨，地是否与示波器/信号源共地。",
                        f"先把供电理顺：确认电源正负与地轨一致，再检查输出节点是否被误接到电源轨。",
                    ),
                )
            )
        elif code in {"POLARITY_REVERSED"}:
            out.append(
                _choose(
                    seed + f"|ec|{code}|5",
                    (
                        f"{code}：先暂停上电，逐个核对极性器件方向与参考 {ref} 是否一致。",
                        f"先处理反接风险：核对极性器件方向，确认无误再上电。",
                    ),
                )
            )
        else:
            out.append(
                _choose(
                    seed + f"|ec|{code}|x",
                    (
                        f"围绕 {code} 做验证：对照参考 {ref} 先核对关键节点，再复测连通性并重新跑一次诊断。",
                        f"{code}：先核对 {comp_text or '关键连线'} 的孔位与连通性，再对照参考 {ref} 逐项确认。",
                    ),
                )
            )
        if len(out) >= limit:
            break

    if context_pack.error_family == "short_circuit":
        out.append(
            _choose(
                seed + "|ec|family|short",
                (
                    "如果怀疑短路：断电后先测电源与地是否导通，再逐根拔跳线定位短接点。",
                    "短路类优先：先断电排除电源轨短接，再去做节点/孔位核对。",
                ),
            )
        )

    return _dedupe_texts(out)[:limit]


def _component_label_by_hint(*, evidence: RuntimeEvidence, hints: tuple[str, ...]) -> str:
    for label in _component_labels(evidence, limit=6):
        lower = label.lower()
        if any(h.lower() in lower for h in hints):
            return label.split("(")[0] if "(" in label else label
    return ""


def _component_label_from_finding(finding) -> str:
    component_id = str(getattr(finding, "component_id", "") or "").strip()
    if not component_id:
        payload = getattr(finding, "payload", {}) or {}
        actual = payload.get("component_actual") if isinstance(payload, dict) else None
        ref = payload.get("component_ref") if isinstance(payload, dict) else None
        if isinstance(actual, dict):
            component_id = str(actual.get("component_id") or "").strip()
        if not component_id and isinstance(ref, dict):
            component_id = str(ref.get("ref_id") or "").strip()
    return component_id


def _component_labels(evidence: RuntimeEvidence, limit: int = 4) -> list[str]:
    labels: list[str] = []
    components = (evidence.netlist_v2 or {}).get("components", [])
    if isinstance(components, list):
        for comp in components:
            if not isinstance(comp, dict):
                continue
            cid = str(comp.get("component_id") or comp.get("id") or "").strip()
            ctype = str(comp.get("component_type") or comp.get("type") or "").strip()
            label = f"{cid}({ctype})" if cid and ctype else cid or ctype
            if label and label not in labels:
                labels.append(label)
            if len(labels) >= limit:
                return labels

    for finding in evidence.findings:
        label = _component_label_from_finding(finding)
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def _reference_label(evidence: RuntimeEvidence) -> str:
    report = evidence.validator_report_v2 or {}
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    runtime_ref = (evidence.runtime_metadata or {}).get("reference", {})
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(runtime_ref, dict):
        runtime_ref = {}

    reference_name = str(summary.get("reference_name") or runtime_ref.get("reference_name") or "").strip()
    reference_id = str(summary.get("reference_id") or runtime_ref.get("reference_id") or "").strip()
    if reference_name and reference_id:
        return f"{reference_name}({reference_id})"
    if reference_name:
        return reference_name
    if reference_id:
        return reference_id
    if evidence.error_codes and "REFERENCE_NOT_SET" in evidence.error_codes:
        return "未设置参考电路"
    return "未提供参考电路"


def circuit_opening_sentence(evidence: RuntimeEvidence) -> str:
    """Stable first sentence for diagnostic answers after circuit recognition."""
    if evidence.error_codes:
        issue_text = f"错误码为 {'、'.join(evidence.error_codes[:4])}"
    elif evidence.diagnostics:
        issue_text = f"诊断项为 {'、'.join(evidence.diagnostics[:2])}"
    elif evidence.risk_reasons:
        issue_text = f"风险原因为 {'、'.join(evidence.risk_reasons[:2])}"
    else:
        issue_text = "暂无明确结构化错误码"
    components = _component_labels(evidence)
    component_text = "、".join(components) if components else "暂未识别到明确元件"
    if len(components) >= 4:
        component_text += "等"
    return (
        "先看这个电路本身："
        f"{issue_text}，"
        f"参考电路为 {_reference_label(evidence)}，"
        f"涉及元件为 {component_text}。"
    )


def ensure_circuit_opening(answer: str, evidence: RuntimeEvidence) -> str:
    opening = circuit_opening_sentence(evidence)
    stripped = (answer or "").lstrip()
    if stripped.startswith(opening):
        return stripped
    return opening + ("\n" + stripped if stripped else "")


def _datasheet_hits_from(tool_results: list[ToolResult]) -> tuple[ToolResult | None, list[dict[str, Any]]]:
    """Return (datasheet_tool_result, hits) when datasheet_lookup_tool produced
    chunk-shaped hits this turn. Empty list when only `local_fallback` rules
    fired or the tool wasn't called.
    """
    for result in tool_results:
        if result.tool_name != "datasheet_lookup_tool":
            continue
        payload = result.payload or {}
        provider = str(payload.get("provider") or "")
        hits = payload.get("hits") or []
        if provider in {"local_datasheet_v2", "kb_retrieval"} and isinstance(hits, list) and hits:
            return result, [h for h in hits if isinstance(h, dict)]
    return None, []


def _fallback_rules_from(tool_results: list[ToolResult]) -> tuple[ToolResult | None, list[dict[str, Any]]]:
    for result in tool_results:
        if result.tool_name != "datasheet_lookup_tool":
            continue
        payload = result.payload or {}
        if str(payload.get("provider") or "") != "local_fallback":
            continue
        rules = payload.get("structured_rules") or []
        if isinstance(rules, list) and rules:
            return result, [r for r in rules if isinstance(r, dict)]
    return None, []


def build_datasheet_answer(
    *,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
    user_message: str,
) -> str | None:
    """Render a chip-spec answer directly from datasheet_lookup_tool hits.

    Phase 5 design — chip-parameter questions do NOT go through any LLM. The
    answer is assembled deterministically from the retrieved chunks so the
    verifier's chunk_id citation requirement is trivially satisfied and the
    user sees the exact text we retrieved. Returns None when no datasheet
    evidence is available (caller falls back to the regular diagnostic
    template).
    """

    ds_result, hits = _datasheet_hits_from(tool_results)
    fb_result, rules = _fallback_rules_from(tool_results)
    if ds_result is None and fb_result is None:
        return None

    parts: list[str] = []
    if ds_result is not None and hits:
        provider = ds_result.payload.get("provider", "")
        first_doc = str(hits[0].get("document_id") or "datasheet")
        parts.append(f"已从本地 datasheet 检索到 {len(hits)} 段相关内容（{first_doc}）：")

        # Cap at top 3 hits — keep the answer scannable. Each entry quotes
        # the chunk_id so the verifier passes and the student can audit.
        for idx, hit in enumerate(hits[:3], start=1):
            title = str(hit.get("title") or hit.get("document_id") or "datasheet").strip()
            modality = str(hit.get("modality") or "text")
            page = hit.get("page")
            page_label = f"p{page}" if isinstance(page, int) else ""
            chunk_id = str(hit.get("chunk_id") or "")
            snippet = str(hit.get("snippet") or "").strip()
            if len(snippet) > 280:
                snippet = snippet[:277] + "..."
            tag = f"[{modality}]" if modality and modality != "text" else ""
            header = " ".join(p for p in (f"{idx}.", title, page_label, tag) if p)
            parts.append(header)
            if snippet:
                parts.append(snippet)
            if chunk_id:
                parts.append(f"引用：{chunk_id}")

        # Surface asset/table when present so a downstream UI can render them.
        asset_lines: list[str] = []
        for hit in hits[:3]:
            asset = hit.get("asset_path")
            if asset and isinstance(asset, str):
                asset_lines.append(f"参考资产：{asset}")
            table_html = hit.get("table_html")
            if table_html and isinstance(table_html, str):
                asset_lines.append("（命中数据包含结构化表格，详见 chunk）")
        if asset_lines:
            parts.append("")
            parts.extend(asset_lines[:3])

        provider_label = "本地结构化 datasheet" if provider == "local_datasheet_v2" else "PDF 知识库"
        parts.append("")
        parts.append(f"知识来源：{provider_label}（无 LLM 合成，直接呈现检索片段）。")

    if fb_result is not None and rules and not hits:
        # Rule-based fallback path — no chunks, just deterministic safety /
        # pin rules with rule_id citations the verifier accepts.
        comp_type = str(fb_result.payload.get("component_type") or "未知元件")
        parts.append(f"未找到该器件的具体 datasheet，按 {comp_type} 通用规则回答：")
        for idx, rule in enumerate(rules[:5], start=1):
            text = str(rule.get("text") or "").strip()
            rule_id = str(rule.get("rule_id") or "")
            line = f"{idx}. {text}" if text else f"{idx}. ({rule_id})"
            if rule_id and rule_id not in line:
                line += f"（依据 {rule_id}）"
            parts.append(line)
        parts.append("")
        parts.append("知识来源：本地通用规则库（无 datasheet 命中，无 LLM 合成）。")

    return "\n".join(parts).strip() or None


def build_diagnostic_template_answer(
    *,
    station_id: str,
    query: str,
    user_message: str,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
) -> str:
    """基于用户意图生成自然语言回答，不暴露 system prompt / raw JSON。"""
    # Datasheet path: if the SemanticRouter enabled datasheet_lookup_tool and
    # the tool produced chunk hits or rule fallback, answer directly from the
    # retrieved evidence — no LLM. This is the only "no-LLM" answer path; the
    # diagnostic templates below still drive everything else.
    datasheet_answer = build_datasheet_answer(
        evidence=evidence,
        context_pack=context_pack,
        tool_results=tool_results,
        user_message=user_message or query,
    )
    if datasheet_answer:
        return _with_diagnostic_anchors(datasheet_answer, evidence)

    circuit_answer = build_circuit_kb_answer(
        user_message=user_message or query,
        tool_results=tool_results,
        evidence=evidence,
    )
    if circuit_answer:
        return _with_diagnostic_anchors(circuit_answer, evidence)

    message = user_message or query
    target_component = _extract_target_component(message, evidence)
    if target_component and _looks_like_fix_or_wiring_question(message):
        answer = _with_diagnostic_anchors(
            _build_component_fix_answer(
                evidence=evidence,
                context_pack=context_pack,
                tool_results=tool_results,
                station_id=station_id,
                user_message=message,
                target_component_id=target_component,
            ),
            evidence,
        )
        return ensure_circuit_opening(answer, evidence)

    intent = _classify_user_intent(message)

    if intent == "components":
        answer = _with_diagnostic_anchors(
            _describe_components(evidence, context_pack),
            evidence,
        )
        return ensure_circuit_opening(answer, evidence)
    if intent == "explain":
        answer = _with_diagnostic_anchors(_explain_issue(evidence, context_pack), evidence)
        return ensure_circuit_opening(answer, evidence)

    answer = _with_diagnostic_anchors(
        _build_general_diagnostic_answer(station_id, evidence, context_pack, tool_results, message),
        evidence,
    )
    return ensure_circuit_opening(answer, evidence)


def build_circuit_kb_llm_answer(
    *,
    user_message: str,
    circuit: dict[str, Any],
    evidence: RuntimeEvidence | None = None,
) -> str | None:
    """Generate a natural-language answer via Ollama from circuit JSON context.

    Returns ``None`` when Ollama is unreachable or returns empty content,
    so callers can fall back to the deterministic template answer.
    """
    import logging

    import httpx

    from app.core.config import settings

    logger = logging.getLogger(__name__)

    circuit_text = _format_circuit_for_llm(circuit, evidence)

    system_prompt = (
        "你是模电实验教学助手。你的任务是根据本地电路知识库中的信息回答学生问题。"
        "回答要求："
        "1. 紧扣提供的电路信息，元件编号、公式、参数必须与知识库一致，禁止编造。"
        "2. 优先正面回答学生的问题，再补充相关知识点或注意事项。"
        "3. 用简洁的中文，避免冗余铺垫。"
        "4. 如果涉及故障排查，结合常见故障列表给出具体排查方向。"
        "5. 如果电路信息不足以回答该问题，请诚实告知，不要猜测。"
    )

    prompt = (
        f"【电路信息】\n{circuit_text}\n\n"
        f"【学生问题】\n{user_message}\n\n"
        f"请根据上面的电路信息回答学生的问题。"
    )

    try:
        payload = {
            "model": settings.AGENT_LLM_OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "keep_alive": "30m",
            "options": {
                "temperature": 0.3,
                "num_predict": 1024,
            },
        }
        base_url = settings.AGENT_LLM_OLLAMA_BASE_URL.rstrip("/")
        endpoint = f"{base_url}/api/chat"
        timeout = max(10.0, float(settings.AGENT_LLM_OLLAMA_TIMEOUT_S))

        with httpx.Client(timeout=timeout, trust_env=False) as client:
            resp = client.post(endpoint, json=payload)
            resp.raise_for_status()
            body = resp.json()
    except Exception as exc:
        logger.warning("Ollama circuit KB answer generation failed: %s", exc)
        return None

    if not isinstance(body, dict):
        return None
    message = body.get("message")
    if not isinstance(message, dict):
        return None
    content = str(message.get("content") or "").strip()
    return content or None


def _format_circuit_for_llm(
    circuit: dict[str, Any],
    evidence: RuntimeEvidence | None,
) -> str:
    """Render circuit JSON into a compact Chinese text block for the LLM."""
    parts: list[str] = []

    name = str(circuit.get("name") or "")
    category = str(circuit.get("category") or "")
    subcategory = str(circuit.get("subcategory") or "")
    if name:
        label = name
        if category:
            label += f"（{category}"
            if subcategory:
                label += f" / {subcategory}"
            label += "）"
        parts.append(f"电路名称：{label}")

    summary = str(circuit.get("summary") or "").strip()
    if summary:
        parts.append(f"概述：{summary}")

    components = circuit.get("components")
    if isinstance(components, list) and components:
        comp_lines: list[str] = []
        for comp in components:
            if not isinstance(comp, dict):
                continue
            ref = comp.get("ref", "")
            ctype = comp.get("type", "")
            role = comp.get("role", "")
            purpose = comp.get("purpose", "")
            line = f"  {ref}（{ctype}）：{role}"
            if purpose:
                line += f"，{purpose}"
            comp_lines.append(line)
        if comp_lines:
            parts.append("关键元件：\n" + "\n".join(comp_lines))

    connections = circuit.get("connections")
    if isinstance(connections, list) and connections:
        conn_lines: list[str] = []
        for conn in connections[:12]:
            if isinstance(conn, dict):
                frm = conn.get("from", "")
                to = conn.get("to", "")
                conn_lines.append(f"  {frm} → {to}")
        if conn_lines:
            parts.append("连接关系：\n" + "\n".join(conn_lines))

    analysis = circuit.get("analysis")
    if isinstance(analysis, dict) and analysis:
        analysis_lines: list[str] = []
        for key, val in analysis.items():
            if isinstance(val, str) and len(val) < 150:
                analysis_lines.append(f"  {key}: {val}")
        if analysis_lines:
            parts.append("电路分析公式：\n" + "\n".join(analysis_lines[:8]))

    faults = circuit.get("common_faults")
    if isinstance(faults, list) and faults:
        fault_lines: list[str] = []
        for f in faults[:6]:
            if isinstance(f, dict):
                fault_name = f.get("fault", "")
                consequence = f.get("consequence", "")
                fault_lines.append(f"  · {fault_name}：{consequence}")
        if fault_lines:
            parts.append("常见故障：\n" + "\n".join(fault_lines))

    teaching = circuit.get("teaching_points")
    if isinstance(teaching, list) and teaching:
        tp_lines: list[str] = []
        for tp in teaching[:6]:
            tp_lines.append(f"  · {tp}")
        if tp_lines:
            parts.append("教学要点：\n" + "\n".join(tp_lines))

    annotations = circuit.get("image_annotations")
    if isinstance(annotations, dict):
        ann_parts: list[str] = []
        visible = annotations.get("visible_components")
        if isinstance(visible, list) and visible:
            ann_parts.append(f"  图上可见元件：{'、'.join(str(v) for v in visible)}")
        notes = annotations.get("notes")
        if isinstance(notes, list) and notes:
            for note in notes:
                ann_parts.append(f"  备注：{note}")
        if ann_parts:
            parts.append("图片标注信息：\n" + "\n".join(ann_parts))

    image = str(circuit.get("image") or "").strip()
    if image:
        parts.append(f"参考电路图路径：{image}")

    current_hint = _current_diagnostic_hint_for_circuit(evidence, circuit)
    if current_hint:
        parts.append(f"当前诊断上下文：{current_hint}")

    return "\n\n".join(parts)


def build_circuit_kb_answer(
    *,
    user_message: str,
    tool_results: list[ToolResult],
    evidence: RuntimeEvidence | None = None,
) -> str | None:
    """Render a direct answer from ``circuit_lookup_tool`` hits.

    This handles schematic/theory questions inside the diagnostic graph.  It
    keeps the current validator context from drowning out questions such as
    "差分电路一共需要几个电阻".
    """

    result = next(
        (
            item
            for item in tool_results
            if item.tool_name == "circuit_lookup_tool"
            and item.status == "ok"
            and isinstance(item.payload.get("circuits"), list)
            and item.payload.get("circuits")
        ),
        None,
    )
    if result is None:
        return None

    circuit = result.payload["circuits"][0]
    question = (user_message or "").strip()
    question_lower = question.lower()
    components = circuit.get("components", []) if isinstance(circuit, dict) else []
    resistors = _resistive_components(components)

    lines: list[str] = []
    if _asks_component_count(question_lower) and resistors:
        labels = [str(item.get("ref") or "").strip() for item in resistors if item.get("ref")]
        lines.append(
            f"按本地典型电路知识库里的「{circuit.get('name', '')}」这张图，"
            f"电阻类元件一共是 {len(resistors)} 个：{_join_cn(labels)}。"
        )
        if circuit.get("circuit_id") == "differential_amplifier":
            lines.append(
                "其中 RP 是发射极平衡电位器/可调电阻；如果按当前逻辑参考把 RP 拆成 "
                "RP_LEFT 和 RP_RIGHT 两段等效电阻，则网表里的电阻对象会变成 7 个："
                "RC1、RC2、RP_LEFT、RP_RIGHT、R1、R2、RE。"
            )
    else:
        lines.append(f"根据本地电路知识库，「{circuit.get('name', '')}」：")
        summary = str(circuit.get("summary") or "").strip()
        if summary:
            lines.append(summary)
        if components:
            comp_lines = []
            for comp in components[:8]:
                ref = comp.get("ref", "")
                ctype = comp.get("type", "")
                role = comp.get("role", "")
                value = comp.get("value", "")
                value_text = f"（{value}）" if value else ""
                comp_lines.append(f"{ref}：{ctype}{value_text}，{role}")
            if comp_lines:
                lines.append("关键元件：" + "；".join(comp_lines) + "。")

    current_hint = _current_diagnostic_hint_for_circuit(evidence, circuit)
    if current_hint:
        lines.append(current_hint)

    image = str(circuit.get("image") or "").strip()
    if image:
        lines.append(f"参考电路图：{image}")
    matched = circuit.get("matched_features") or []
    if matched:
        lines.append(f"检索命中依据：{_join_cn([str(item) for item in matched[:3]])}。")
    lines.append("知识来源：本地电路知识库（circuit_kb）。")
    return "\n\n".join(line for line in lines if line).strip()


def _asks_component_count(message: str) -> bool:
    return any(
        word in message
        for word in (
            "几个",
            "多少",
            "一共",
            "需要",
            "有哪些",
            "哪几个",
            "数量",
        )
    )


def _resistive_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for comp in components:
        if not isinstance(comp, dict):
            continue
        text = " ".join(
            str(comp.get(key) or "")
            for key in ("ref", "type", "role", "purpose")
        ).lower()
        if any(token in text for token in ("电阻", "resistor", "potentiometer", "可调")):
            out.append(comp)
    return out


def _join_cn(items: list[str]) -> str:
    clean = [item for item in items if item]
    if not clean:
        return ""
    return "、".join(clean)


def _current_diagnostic_hint_for_circuit(
    evidence: RuntimeEvidence | None,
    circuit: dict[str, Any],
) -> str:
    if evidence is None:
        return ""
    circuit_id = str(circuit.get("circuit_id") or "")
    if circuit_id != "differential_amplifier":
        return ""
    refs: list[str] = []
    for finding in evidence.findings:
        payload = finding.payload if isinstance(finding.payload, dict) else {}
        ref = ""
        expected = finding.expected
        if isinstance(expected, dict):
            ref = str(expected.get("ref_id") or expected.get("component_id") or "")
        if not ref and isinstance(payload.get("expected"), dict):
            ref = str(payload["expected"].get("ref_id") or "")
        if not ref:
            ref = finding.component_id
        if ref in {"RC1", "RC2", "RP", "RP_LEFT", "RP_RIGHT", "R1", "R2", "RE"}:
            refs.append(ref)
    refs = sorted(set(refs))
    if refs:
        return f"结合当前诊断，优先核对这些电阻相关项：{_join_cn(refs)}。"
    if evidence.error_codes:
        return f"结合当前诊断，当前仍有 {evidence.error_codes[0]} 等错误码，回答上面的数量后再回到缺件/接线项逐个核对。"
    return ""


def _with_diagnostic_anchors(answer: str, evidence: RuntimeEvidence) -> str:
    anchored = answer
    if evidence.error_codes and not any(code in anchored for code in evidence.error_codes):
        anchored += f"\n校验依据：{evidence.error_codes[0]}。"
    if evidence.evidence_refs and not _mentions_any_runtime_ref(evidence, anchored):
        anchored += f"\n证据引用：{_first_ref_text(evidence)}。"
    return anchored


def repair_diagnostic_answer(
    *,
    draft_answer: str,
    evidence: RuntimeEvidence,
    verification_issues: list[str],
) -> str:
    """修复回答，仅补充 verifier 要求的可审计诊断锚点和安全提示。"""
    repaired = _with_diagnostic_anchors(draft_answer, evidence)
    if evidence.risk_level == "danger" and not any(
        word in repaired for word in ("断电", "电源", "短路")
    ):
        repaired += "\n安全提示：请先断电，再复查电源轨和短路风险。"
    if (
        evidence.ambiguous_pin_count
        or evidence.fallback_pin_count
        or evidence.snap_conflict_count
        or evidence.low_confidence_component_count
    ) and not any(
        hint in repaired
        for hint in ("复拍", "重新拍照", "人工确认", "识别置信度", "孔位识别")
    ):
        repaired += "\n提示：当前孔位识别置信度较低，建议复拍或人工确认引脚孔位。"
    return repaired


def build_verified_diagnostic_answer(
    *,
    station_id: str,
    query: str,
    user_message: str,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
) -> tuple[str, bool, list[str]]:
    draft = build_diagnostic_template_answer(
        station_id=station_id,
        query=query,
        user_message=user_message,
        evidence=evidence,
        context_pack=context_pack,
        tool_results=tool_results,
    )
    verification = verify_draft_answer(
        evidence=evidence,
        context_pack=context_pack,
        draft_answer=draft,
    )
    if verification.passed:
        return draft, True, []

    repaired = repair_diagnostic_answer(
        draft_answer=draft,
        evidence=evidence,
        verification_issues=verification.issues,
    )
    repaired_verification = verify_draft_answer(
        evidence=evidence,
        context_pack=context_pack,
        draft_answer=repaired,
    )
    return repaired, repaired_verification.passed, repaired_verification.issues


def build_diagnostic_citations(
    *,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
) -> list[AngntCitation]:
    citations = [
        AngntCitation(
            source_type="runtime_evidence",
            source_id=evidence.station_id,
            title="结构化运行时证据",
            snippet="、".join(evidence.error_codes) or evidence.risk_level,
        ),
        AngntCitation(
            source_type="context_pack",
            source_id=context_pack.pack_id,
            title="PCM 上下文包",
            snippet=context_pack.error_family,
        ),
    ]
    for result in tool_results:
        citations.append(
            AngntCitation(
                source_type="diagnostic_tool",
                source_id=result.tool_name,
                title=result.tool_name,
                snippet=result.summary[:260],
            )
        )
        if result.tool_name == "datasheet_lookup_tool":
            hits = result.payload.get("hits", [])
            if isinstance(hits, list):
                for item in hits[:3]:
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title") or item.get("filename") or "datasheet").strip()
                    snippet = str(item.get("snippet") or "").strip()[:260]
                    source_id = str(item.get("source_id") or title).strip()
                    if title:
                        citations.append(
                            AngntCitation(
                                source_type="datasheet_pdf",
                                source_id=source_id,
                                title=title,
                                snippet=snippet,
                            )
                        )
    return citations


def build_diagnostic_evidence(
    *,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
    tool_results: list[ToolResult],
    verification_passed: bool,
    verification_issues: list[str],
    graph_metrics: list[dict] | None = None,
    react_trace: list[dict] | None = None,
    react_iterations: int = 0,
    react_terminate_reason: str = "",
) -> list[AngntEvidence]:
    items = [
        AngntEvidence(
            evidence_type="runtime_evidence",
            source_id=evidence.station_id,
            summary="PCM Agent 输入证据",
            payload=evidence.model_dump(),
        ),
        AngntEvidence(
            evidence_type="context_pack",
            source_id=context_pack.pack_id,
            summary="按错误类型推送的上下文和工具",
            payload=context_pack.model_dump(),
        ),
        AngntEvidence(
            evidence_type="context_timeline",
            source_id=f"{evidence.station_id}:context_timeline",
            summary=context_pack.history_summary or "暂无历史上下文",
            payload={
                "history_facts": context_pack.history_facts,
                "history_summary": context_pack.history_summary,
            },
        ),
        AngntEvidence(
            evidence_type="tool_results",
            source_id=f"{evidence.station_id}:diagnostic_tools",
            summary="白盒诊断工具输出",
            payload={"results": [result.model_dump() for result in tool_results]},
        ),
        AngntEvidence(
            evidence_type="verification_report",
            source_id=f"{evidence.station_id}:verifier",
            summary="Reflection Node 校验结果",
            payload={
                "passed": verification_passed,
                "issues": verification_issues,
            },
        ),
    ]
    if graph_metrics:
        items.append(
            AngntEvidence(
                evidence_type="graph_metrics",
                source_id=f"{evidence.station_id}:langgraph",
                summary="PCM LangGraph 节点级指标",
                payload={"metrics": graph_metrics},
            )
        )
    if react_trace:
        terminate_reason = react_terminate_reason or "completed"
        items.append(
            AngntEvidence(
                evidence_type="react_trace",
                source_id=f"{evidence.station_id}:react",
                summary=f"ReAct {react_iterations} 轮 ({terminate_reason})",
                payload={
                    "steps": react_trace,
                    "iterations": react_iterations,
                    "terminate_reason": terminate_reason,
                },
            )
        )
    highlight_protocol = evidence.validator_report_v2.get("highlight_protocol", {})
    if highlight_protocol.get("targets"):
        items.append(
            AngntEvidence(
                evidence_type="highlight_protocol",
                source_id=f"{evidence.station_id}:highlight_protocol",
                summary="前端高亮协议",
                payload=highlight_protocol,
            )
        )
    return items


def diagnostic_conclusion(
    *,
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
) -> str:
    if context_pack.error_family == "short_circuit":
        return "检测到短路或同网风险，需要优先安全复查"
    if context_pack.error_family == "wiring_mismatch":
        return "检测到接线孔位或电气节点不匹配"
    if context_pack.error_family == "polarity_error":
        return "检测到极性方向需要复核"
    if context_pack.error_family == "missing_protection":
        return "检测到缺少保护或限流元件"
    if evidence.risk_level == "safe":
        return "当前没有明确高风险结构化错误"
    return "检测到需要进一步排查的结构化诊断项"


def extract_fix_steps(tool_results: list[ToolResult]) -> list[str]:
    steps: list[str] = []
    for result in tool_results:
        for case in result.payload.get("fault_cases", []):
            steps.extend(str(step) for step in case.get("fix_steps", []) if step)
        steps.extend(str(rule) for rule in result.payload.get("rules", []) if rule)

    deduped: list[str] = []
    for step in steps:
        if step not in deduped:
            deduped.append(step)
    return deduped


def _build_follow_up_suggestions(
    evidence: RuntimeEvidence,
    context_pack: ContextPack,
) -> list[str]:
    """基于诊断内容生成追问建议。"""
    suggestions: list[str] = []

    first_finding = evidence.findings[0] if evidence.findings else None
    if first_finding:
        cid = first_finding.component_id
        pin = first_finding.pin_name
        code = first_finding.error_code
        if code == "FLOATING_PIN" and cid and pin:
            suggestions.append(f"为什么判断 {cid} 的 {pin} 悬空？")
            suggestions.append(f"我应该如何修复 {cid} 的连接？")
        elif code == "COMPONENT_SHORTED_SAME_NET" and cid:
            suggestions.append(f"为什么 {cid} 会被短接？")
            suggestions.append(f"{cid} 应该如何正确插接？")
        elif code == "POLARITY_REVERSED" and cid:
            suggestions.append(f"{cid} 的正确极性方向是什么？")
        elif cid:
            suggestions.append(f"为什么检测到 {cid} 有问题？")

    if evidence.risk_level == "danger":
        suggestions.append("我应该先检查哪些安全事项？")
    if (
        getattr(evidence, "ambiguous_pin_count", 0)
        or getattr(evidence, "fallback_pin_count", 0)
        or getattr(evidence, "snap_conflict_count", 0)
        or getattr(evidence, "low_confidence_component_count", 0)
    ):
        suggestions.append("孔位识别可能不稳定，我需要怎么复拍才能更准确？")
        suggestions.append("你能指出最可疑的那几个孔位/导通排吗？我该怎么重点核对？")

    if evidence.error_codes:
        suggestions.append(f"这些错误码里我应该先处理哪一个：{'、'.join(evidence.error_codes[:3])}？")

    if not suggestions:
        suggestions.append("这个电路图中都有什么元件？")
        suggestions.append("当前诊断结果是否存在风险？")

    return _dedupe_texts(suggestions)[:6]


def _first_ref_text(evidence: RuntimeEvidence) -> str:
    first_ref = evidence.evidence_refs[0] if evidence.evidence_refs else None
    if not first_ref:
        return "暂无 evidence_ref"
    return (
        f"{first_ref.ref_id}"
        + (f" / {first_ref.component_id}" if first_ref.component_id else "")
        + (f".{first_ref.pin_name}" if first_ref.pin_name else "")
    )


def _mentions_any_runtime_ref(evidence: RuntimeEvidence, text: str) -> bool:
    tokens: list[str] = []
    for ref in evidence.evidence_refs:
        tokens.extend(
            value
            for value in (ref.ref_id, ref.component_id, ref.pin_name, ref.hole_id)
            if value
        )
    return any(token in text for token in tokens)


# ---------------------------------------------------------------------------
# Concept-tutor / lab-guidance answer paths (no LangGraph; deterministic).
# ---------------------------------------------------------------------------

_CONCEPT_SAFETY_TRIGGERS: tuple[str, ...] = (
    "led_current_limit",
    "capacitor_filtering",
    "ohms_law",
)


def build_concept_answer(
    *,
    question: str,
    concept: ConceptPack | None,
    evidence: RuntimeEvidence | None = None,
) -> str:
    """Generate a 6-section concept_tutor answer from a local ConceptPack.

    The answer never asserts specific holes / nets / connections of the
    current circuit. The "和当前实验的关系" section either references the
    current risk_level / error_codes at a high level, or explicitly states
    that this is generic knowledge unrelated to the current circuit.
    """

    if concept is None:
        return _generic_concept_fallback(question)

    relate = _concept_relation_to_experiment(concept, evidence)
    lines: list[str] = []
    lines.append(f"直接回答：{concept.summary}")
    if concept.key_points:
        lines.append("原理解释：" + "；".join(concept.key_points))
    if concept.formulas:
        lines.append("公式：" + "；".join(concept.formulas))
    lines.append(f"和当前实验的关系：{relate}")
    if concept.common_mistakes:
        lines.append("常见错误：" + "；".join(concept.common_mistakes))
    if concept.lab_guidance:
        lines.append("如何验证：" + "；".join(concept.lab_guidance))
    safety_notes = list(concept.safety_notes)
    if concept.concept_id in _CONCEPT_SAFETY_TRIGGERS and not any(
        word in "；".join(safety_notes) for word in ("断电", "电源", "短路")
    ):
        safety_notes.append("操作前先断电，再复查电源和短路风险。")
    if safety_notes:
        lines.append("安全提醒：" + "；".join(safety_notes))
    lines.append(f"知识来源：{concept.concept_id}")
    return "\n".join(lines)


def _generic_concept_fallback(question: str) -> str:
    """Returned only when no concept matched — never invents domain facts."""
    return (
        "直接回答：本地知识库未匹配到对应概念，建议补充关键词后再次提问。\n"
        "和当前实验的关系：这是通用问题，未与当前电路状态直接关联。\n"
        "如何验证：可以查阅教材或参考权威资料对照学习。\n"
        "安全提醒：上电或调整接线前请先断电，再复查电源与短路风险。\n"
        "知识来源：concept_not_found"
    )


def _concept_relation_to_experiment(
    concept: ConceptPack,
    evidence: RuntimeEvidence | None,
) -> str:
    if evidence is None or not (evidence.findings or evidence.error_codes):
        return "这是通用知识，与当前电路状态无直接对应。"
    family_hint = ""
    error_codes = "、".join(evidence.error_codes[:2]) if evidence.error_codes else ""
    if error_codes:
        family_hint = f"当前诊断报告中出现 {error_codes}，"
    risk_hint = f"风险等级为 {evidence.risk_level}。" if evidence.risk_level else ""
    return (
        f"{family_hint}{risk_hint}"
        "该概念可作为理解上述现象的背景知识，但具体接线请以诊断结果为准。"
    )


def build_lab_guidance_answer(
    *,
    question: str,
    concept: ConceptPack | None,
    evidence: RuntimeEvidence | None = None,
) -> str:
    """Generate a numbered step-by-step lab-guidance answer with safety hint."""
    steps: list[str] = []
    if concept is not None and concept.lab_guidance:
        steps.extend(concept.lab_guidance)
    if not steps:
        steps = [
            "断电状态下检查接线是否与原理图一致。",
            "用万用表通断挡确认怀疑短路的两点是否真的导通。",
            "通电后用电压挡逐节点验证关键节点电压。",
        ]

    safety: list[str] = []
    if concept is not None:
        safety.extend(concept.safety_notes)
    if not any(
        word in "；".join(safety) for word in ("断电", "电源", "短路")
    ):
        safety.append("先断电再操作，复查电源轨与短路风险后再上电。")

    lines: list[str] = ["实验操作步骤："]
    for idx, step in enumerate(steps, start=1):
        lines.append(f"{idx}. {step}")
    lines.append("安全提醒：" + "；".join(safety))
    if concept is not None:
        lines.append(f"知识来源：{concept.concept_id}")
    return "\n".join(lines)


def build_concept_citations(
    *,
    station_id: str,
    concept: ConceptPack | None,
    tool_results: list[ToolResult],
) -> list[AngntCitation]:
    citations: list[AngntCitation] = []
    if concept is not None:
        citations.append(
            AngntCitation(
                source_type="concept_pack",
                source_id=concept.concept_id,
                title=concept.title,
                snippet=concept.summary[:260],
            )
        )
    for result in tool_results:
        citations.append(
            AngntCitation(
                source_type="diagnostic_tool",
                source_id=result.tool_name,
                title=result.tool_name,
                snippet=result.summary[:260],
            )
        )
    if not citations:
        citations.append(
            AngntCitation(
                source_type="concept_pack",
                source_id="concept_not_found",
                title="未匹配到本地概念",
                snippet="建议补充关键词后再次提问",
            )
        )
    return citations


def build_concept_evidence(
    *,
    station_id: str,
    intent: AgentIntent,
    concept: ConceptPack | None,
    tool_results: list[ToolResult],
    verification_passed: bool,
    verification_issues: list[str],
    evidence: RuntimeEvidence | None = None,
) -> list[AngntEvidence]:
    items: list[AngntEvidence] = []
    items.append(
        AngntEvidence(
            evidence_type="intent",
            source_id=f"{station_id}:intent",
            summary=f"intent={intent}",
            payload={"intent": intent},
        )
    )
    if concept is not None:
        items.append(
            AngntEvidence(
                evidence_type="concept_pack",
                source_id=concept.concept_id,
                summary=concept.title,
                payload=concept.model_dump(),
            )
        )
    if evidence is not None:
        items.append(
            AngntEvidence(
                evidence_type="runtime_evidence",
                source_id=evidence.station_id,
                summary="PCM Agent 输入证据（仅供前端展示当前电路状态）",
                payload=evidence.model_dump(),
            )
        )
    items.append(
        AngntEvidence(
            evidence_type="tool_results",
            source_id=f"{station_id}:concept_tools",
            summary="本地概念查找工具输出",
            payload={"results": [result.model_dump() for result in tool_results]},
        )
    )
    items.append(
        AngntEvidence(
            evidence_type="verification_report",
            source_id=f"{station_id}:verifier",
            summary="Reflection Node 校验结果",
            payload={"passed": verification_passed, "issues": verification_issues},
        )
    )
    return items
