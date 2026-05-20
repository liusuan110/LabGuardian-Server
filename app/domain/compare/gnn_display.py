"""GNN advice · 显示文案注入（demo / 现场演示友好）。

把 ``GNNAdvice`` 里清一色的 ``cur_port:IC1.pin4`` / ``cur_net:NET_004``
之类内部 ID 翻译成 ``U1 · pin4 (V−电源)`` / ``GND (地)`` 这种人类可读的
display 字符串，作为 ``*_display`` 字段挂在原结构旁边（**raw ID 永远保留**，
做 traceability + frontend 高亮联动）。

数据来源（已经由 ``_enrich_result`` / ``_attach_mappings`` 写入）：

- ``summary["ref_to_current_component_mapping"]`` ``{ref_ref_id: cur_comp_id}``
- ``summary["ref_to_current_net_mapping"]``       ``{ref_net: cur_net_id}``
- ``ref_payload["nets"][*]``                       ``role`` / ``role_label`` / ``net``
- ``ref_payload["components"][*]``                 ``ref_id`` / ``type`` / ``subtype``
- 对 IC（含 subtype）：``app.domain.gnn.graph_schema.IC_PIN_MAPS`` 给出
  ``pin → PortType`` 映射，进一步翻译为中文角色

仅当 ref↔cur 对齐成功时 display 才有实际信息；对齐失败的部分回退到 raw ID
（避免编出错误的标注）。
"""

from __future__ import annotations

from typing import Any, Iterable

# 中文角色文案 — 内部内联，演示场景便于现场修改
_NET_ROLE_ZH: dict[str, str] = {
    "input": "输入",
    "output": "输出",
    "power": "电源",
    "ground": "地",
    "signal": "信号",
}

_PORT_TYPE_ZH: dict[str, str] = {
    "inverting_input": "反相输入",
    "non_inverting_input": "非反相输入",
    "output": "输出",
    "v_plus": "V+电源",
    "v_minus": "V−电源",
    "offset_null": "调零",
    "nc": "空脚",
    "power": "电源",
    "ground": "地",
    "anode": "阳极",
    "cathode": "阴极",
    "base": "基极",
    "collector": "集电极",
    "emitter": "发射极",
    "drain": "漏极",
    "gate": "栅极",
    "source": "源极",
    "wiper": "滑片",
    "terminal_a": "端子A",
    "terminal_b": "端子B",
}


def _ic_pin_role(ref_comp: dict[str, Any], pin_name: str) -> str | None:
    """For an IC ref component, look up pin's PortType via subtype map.

    Returns the ``PortType.value`` (e.g. ``"inverting_input"``) or
    None if no map available / pin not in map. Resistors / caps etc.
    return None — their pins don't carry semantic roles for display.
    """

    subtype = (ref_comp.get("subtype") or "").upper()
    if not subtype:
        return None
    try:
        from app.domain.gnn.graph_schema import IC_PIN_MAPS  # type: ignore
    except ImportError:
        return None
    pin_map = IC_PIN_MAPS.get(subtype)
    if not pin_map:
        return None
    # IC_PIN_MAPS 接受 "3" 或 "pin3" 两种写法
    return pin_map.get(pin_name) or pin_map.get(str(pin_name).lstrip("pin"))


def _net_display_for(ref_net_id: str, ref_net_info: dict[str, dict[str, Any]]) -> str:
    """格式："VOUT (输出)" / "GND (地)" / "INV (信号)"。"""

    ref_net = ref_net_info.get(ref_net_id, {})
    role = str(ref_net.get("role") or "").lower()
    role_zh = _NET_ROLE_ZH.get(role)
    label = (
        ref_net.get("role_label")
        or ref_net.get("label")
        or ref_net.get("net")
        or ref_net_id
    )
    if role_zh:
        return f"{label} ({role_zh})"
    return str(label)


def _port_display_for(
    ref_comp_id: str,
    pin_name: str,
    ref_comp_info: dict[str, dict[str, Any]],
) -> str:
    """格式："U1 · pin2 (反相输入)" / "R_f · pin1"（无角色时省略尾括号）。"""

    ref_comp = ref_comp_info.get(ref_comp_id, {})
    pin_role = _ic_pin_role(ref_comp, pin_name)
    pin_role_zh = _PORT_TYPE_ZH.get(pin_role or "")
    base = f"{ref_comp_id} · {pin_name}"
    if pin_role_zh:
        return f"{base} ({pin_role_zh})"
    return base


def build_display_maps(
    ref_payload: dict[str, Any],
    cur_netlist_v2: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """构造 ``cur_net_full_id → display`` 和 ``cur_port_full_id → display``
    两张查表。两张表 key 都用 GNN 输出里那套带前缀的全 ID
    （``cur_net:<id>`` / ``cur_port:<comp>.<pin>``），调用方直接 dict.get 即可。
    """

    ref_to_cur_net = summary.get("ref_to_current_net_mapping") or {}
    ref_to_cur_comp = summary.get("ref_to_current_component_mapping") or {}

    # 反向映射
    cur_to_ref_net = {v: k for k, v in ref_to_cur_net.items() if v}
    cur_to_ref_comp = {v: k for k, v in ref_to_cur_comp.items() if v}

    # ref 侧索引
    ref_net_info: dict[str, dict[str, Any]] = {
        n.get("net"): n for n in ref_payload.get("nets", []) or [] if n.get("net")
    }
    ref_comp_info: dict[str, dict[str, Any]] = {
        c.get("ref_id"): c for c in ref_payload.get("components", []) or [] if c.get("ref_id")
    }

    # 1) net 显示表
    net_display: dict[str, str] = {}
    for cur_net_id, ref_net_id in cur_to_ref_net.items():
        net_display[f"cur_net:{cur_net_id}"] = _net_display_for(ref_net_id, ref_net_info)

    # 2) port 显示表 —— 遍历 cur 的所有 pin
    port_display: dict[str, str] = {}
    for cur_comp in cur_netlist_v2.get("components", []) or []:
        cur_comp_id = cur_comp.get("component_id")
        if not cur_comp_id:
            continue
        ref_comp_id = cur_to_ref_comp.get(cur_comp_id)
        if not ref_comp_id:
            # 未对齐的 cur 元件（多余 / 学生 wire 等）：用 cur 自己的 id
            # 显示 "<cur_comp_id> · <pin>" 不带角色信息
            for pin in cur_comp.get("pins", []) or []:
                pin_name = pin.get("pin_name")
                if not pin_name:
                    continue
                port_display[f"cur_port:{cur_comp_id}.{pin_name}"] = (
                    f"{cur_comp_id} · {pin_name}"
                )
            continue
        for pin in cur_comp.get("pins", []) or []:
            pin_name = pin.get("pin_name")
            if not pin_name:
                continue
            port_display[f"cur_port:{cur_comp_id}.{pin_name}"] = _port_display_for(
                ref_comp_id, pin_name, ref_comp_info
            )

    return net_display, port_display


def _resolve_one(table: dict[str, str], full_id: str) -> str:
    """Fall back to raw id when display missing — better to show
    ``NET_007`` than to silently drop the value."""

    return table.get(full_id, full_id)


def _resolve_many(table: dict[str, str], full_ids: Iterable[str]) -> list[str]:
    return [_resolve_one(table, fid) for fid in full_ids]


def enrich_advice_with_display(
    advice_dict: dict[str, Any],
    ref_payload: dict[str, Any],
    cur_netlist_v2: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Return a new advice dict with ``*_display`` fields added to every
    user-facing structure. Raw IDs preserved everywhere (frontend may
    need them for high-lighting / structured matching)."""

    net_display, port_display = build_display_maps(ref_payload, cur_netlist_v2, summary)

    out = dict(advice_dict)

    # edge_predictions
    new_edges: list[dict[str, Any]] = []
    for ep in out.get("edge_predictions", []) or []:
        ep2 = dict(ep)
        edge = ep2.get("edge") or []
        if len(edge) == 2:
            ep2["edge_display"] = [
                _resolve_one(port_display, str(edge[0])),
                _resolve_one(net_display, str(edge[1])),
            ]
        new_edges.append(ep2)
    if new_edges:
        out["edge_predictions"] = new_edges

    # hotspots
    new_hotspots: list[dict[str, Any]] = []
    for hs in out.get("hotspots", []) or []:
        hs2 = dict(hs)
        node = hs2.get("node")
        if isinstance(node, str):
            hs2["node_display"] = _resolve_one(port_display, node)
        new_hotspots.append(hs2)
    if new_hotspots:
        out["hotspots"] = new_hotspots

    # suggested_targets
    new_targets: list[dict[str, Any]] = []
    for t in out.get("suggested_targets", []) or []:
        t2 = dict(t)
        port = t2.get("port")
        if isinstance(port, str):
            t2["port_display"] = _resolve_one(port_display, port)
        t2["current_nets_display"] = _resolve_many(
            net_display, t2.get("current_nets", []) or []
        )
        new_cands: list[dict[str, Any]] = []
        for c in t2.get("candidates", []) or []:
            c2 = dict(c)
            net = c2.get("net")
            if isinstance(net, str):
                c2["net_display"] = _resolve_one(net_display, net)
            new_cands.append(c2)
        t2["candidates"] = new_cands
        new_targets.append(t2)
    if new_targets:
        out["suggested_targets"] = new_targets

    return out


def enrich_suspicious_edges_for_warning(
    suspicious_actual: list[dict[str, Any]],
    advice_dict: dict[str, Any],
    ref_payload: dict[str, Any],
    cur_netlist_v2: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """同样思路，给 R2 warning 的 ``actual.gnn_suspicious_edges`` 每条
    加 ``edge_display`` 以及内嵌 ``suggested_targets[*].net_display``。

    这里 advice_dict 必须是**已经 enriched** 过的（即 candidates 里有
    net_display 了），调用方负责顺序。
    """

    net_display, port_display = build_display_maps(ref_payload, cur_netlist_v2, summary)

    out: list[dict[str, Any]] = []
    for e in suspicious_actual:
        e2 = dict(e)
        edge = e2.get("edge") or []
        if len(edge) == 2:
            e2["edge_display"] = [
                _resolve_one(port_display, str(edge[0])),
                _resolve_one(net_display, str(edge[1])),
            ]
        new_cands: list[dict[str, Any]] = []
        for c in e2.get("suggested_targets", []) or []:
            c2 = dict(c)
            net = c2.get("net")
            if isinstance(net, str):
                c2["net_display"] = _resolve_one(net_display, net)
            new_cands.append(c2)
        e2["suggested_targets"] = new_cands
        out.append(e2)
    return out


__all__ = [
    "build_display_maps",
    "enrich_advice_with_display",
    "enrich_suspicious_edges_for_warning",
]
