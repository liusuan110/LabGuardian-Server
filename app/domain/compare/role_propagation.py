"""Phase E · S2 — 通过 ComponentAlignment 把 ref 的 net 语义传播到 cur。

替换 ``role_inference._apply_role_inferences_to_netlist`` 的路径（那条路径
依赖 ``GraphMatcher.is_isomorphic()`` 在生产几乎永远返回 None；本模块基于
fuzzy 组件对齐，对部分缺组件 / 多余跳线 / 跨拓扑变体都稳定）。

**Wire 特殊性（项目根本约束 · 用户明示）**：

跳线在本项目里**不是常规元件**，是 net 的物理延伸：
- Wire pin **绝不参与投票** —— wire 没语义身份，只继承所在 net 的 canonical_name
- 投票池只来自非 wire 元件的 pin（Resistor / Capacitor / Transistor / IC / Pot）
- 投票完后，该 net 上所有 wire 元件的 pin 自动继承 net 的 canonical_name

**保护规则（优先级）**：

```
manual_role / port_annotation  (用户显式标的)         ← 永不覆盖
power_role (hole-id 启发式标的 VCC/GND/VEE)            ← 永不覆盖
inferred_from_reference / default_signal / 空           ← 可覆盖
```

这样保证：
- 学生在 UI 标的 IC pin（INV/VOUT 等）不会被错误传播覆盖
- hole-id 启发式（LN/RN/LP/RP）已稳健地标出的 VCC/GND/VEE 不会被
  ref 的不同 supply convention（如 ref 单电源 pin4=GND, cur 双电源
  pin4=VEE）错误覆盖

**输入 / 输出**：

- 输入：``ref_hcg``, ``cur_hcg``, :class:`ComponentAlignment` (来自
  ``alignment_fuzzy.align_components_by_signature``), ``cur_netlist_v2``
  (mutated in-place — 跟现有 ``_apply_role_inferences_to_netlist`` 同惯例)
- 输出：``list[dict]`` 应用记录（用于上层观测性 / report.details）
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from app.domain.gnn.alignment import ComponentAlignment
from app.domain.gnn.alignment_fuzzy import WIRE_CTYPES, _comp_pin_to_net
from app.domain.gnn.hetero_circuit import HeteroCircuitGraph

# Role sources that should NEVER be overwritten by alignment-derived inference.
# These represent stronger evidence than fuzzy alignment (user explicit input
# or topology-independent hole-ID heuristic).
PROTECTED_ROLE_SOURCES: frozenset[str] = frozenset({
    "manual_role",      # set via apply_net_role_assignments (user-facing)
    "port_annotation",  # set via port pin labeling in UI
    "power_role",       # set via hole-id heuristic on LN/RN/LP/RP rails
    "role_label",       # set via DSL when role_label is explicit
    "explicit_role",    # set via DSL when role is explicit
})

# Source string written to net["role_source"] when propagation succeeds.
SOURCE_PROPAGATION: str = "inferred_from_component_alignment"

# Default majority threshold for vote resolution.
DEFAULT_VOTE_THRESHOLD: float = 0.5


def _is_protected_role_source(net: dict[str, Any]) -> bool:
    """Return True if this cur net's existing labeling should NOT be overwritten.

    Protection rules (any one triggers protection):

    1. ``role_source`` is in :data:`PROTECTED_ROLE_SOURCES` (manual_role,
       port_annotation, power_role, role_label, explicit_role).
    2. ``power_role`` is set — hole-id heuristic is strongest evidence for
       VCC/GND/VEE; ref's supply convention is allowed to differ.
    3. ``role_label`` OR ``canonical_name`` is set to a non-empty value —
       this is explicit user intent in the cur netlist payload, regardless
       of whether ``role_source`` was filled in. Protects upstream data
       sources / synthetic fixtures that set role_label but don't bother
       to populate role_source.
    """
    source = str(net.get("role_source") or "")
    if source in PROTECTED_ROLE_SOURCES:
        return True
    if (net.get("power_role") or "").strip():
        return True
    if (net.get("role_label") or "").strip():
        return True
    # Explicit ``role`` field (even ``"signal"``) signals user intent —
    # if the rule comparator should detect a ref↔cur role mismatch, the
    # propagator MUST NOT silently "fix" it by overwriting role. Only
    # nets that genuinely lack a role field (truly anonymous) are
    # eligible for role propagation.
    if (net.get("role") or "").strip():
        return True
    # canonical_name protects only when it is **semantically distinct** from
    # the raw electrical_net_id. Many upstream stages default canonical_name
    # to ``electrical_net_id`` as a placeholder fallback (e.g. ``"NET_002"``);
    # treating that as protected would block all useful propagation.
    canonical = (net.get("canonical_name") or "").strip()
    net_id = str(net.get("electrical_net_id") or net.get("net_id") or "")
    if canonical and canonical != net_id:
        return True
    return False


def propagate_canonical_via_alignment(
    ref_hcg: HeteroCircuitGraph,
    cur_hcg: HeteroCircuitGraph,
    alignment: ComponentAlignment,
    cur_netlist_v2: dict[str, Any],
    *,
    vote_threshold: float = DEFAULT_VOTE_THRESHOLD,
    wire_ctypes: frozenset[str] = WIRE_CTYPES,
) -> list[dict[str, Any]]:
    """Propagate ref net canonical names / roles to cur via component alignment.

    Mutates ``cur_netlist_v2["nets"][...]`` in place (matches the existing
    ``_apply_role_inferences_to_netlist`` convention).

    Algorithm:
        1. For each (ref_comp, cur_comp) in alignment (excluding Wire components),
           walk their pin-net mappings. Each matched (pin_key) yields a vote:
           ``cur_net.source_id ← (ref_net.canonical_name, ref_net.role, ref_net.role_label)``
        2. For each cur net, resolve via majority vote. Below threshold → skip.
        3. Skip nets where existing role_source is in PROTECTED_ROLE_SOURCES.
        4. Apply: mutate canonical_name / role / role_label / aliases /
           role_source / inferred_confidence on the cur net dict.

    Args:
        ref_hcg: reference HCG (from ``build_from_logical_reference``).
        cur_hcg: current HCG (from ``build_from_netlist_v2``).
        alignment: component alignment (from
            ``align_components_by_signature``).
        cur_netlist_v2: the original cur netlist_v2 dict (mutated in place).
        vote_threshold: majority threshold for accepting a vote (default 0.5).
        wire_ctypes: ctypes treated as wires (excluded from voting).

    Returns:
        List of applied records, each containing::

            {
                "current_net": str,
                "canonical_name": str,
                "role": str,
                "role_label": str,
                "reference_net": str,
                "confidence": float,
                "source": "inferred_from_component_alignment",
            }
    """
    # 1. Build votes ─────────────────────────────────────────────────────────
    # cur_net_sid → Counter({(canonical_name, role, role_label, ref_net_sid) → count})
    votes: dict[str, Counter[tuple[str, str, str, str]]] = defaultdict(Counter)
    ref_comp_by_sid = {c.source_id: c for c in ref_hcg.components.values()}
    cur_comp_by_sid = {c.source_id: c for c in cur_hcg.components.values()}

    for ref_sid, cur_sid in alignment.ref_to_cur_component.items():
        ref_comp = ref_comp_by_sid.get(ref_sid)
        cur_comp = cur_comp_by_sid.get(cur_sid)
        if ref_comp is None or cur_comp is None:
            continue
        # **Wire 特殊性**: wire pins don't vote (no semantic identity)
        if ref_comp.ctype in wire_ctypes or cur_comp.ctype in wire_ctypes:
            continue
        ref_pin_to_net = _comp_pin_to_net(ref_hcg, ref_comp)
        cur_pin_to_net = _comp_pin_to_net(cur_hcg, cur_comp)
        for pin_key, ref_net_sid in ref_pin_to_net.items():
            cur_net_sid = cur_pin_to_net.get(pin_key)
            if cur_net_sid is None:
                continue
            ref_net_node = ref_hcg.nets.get(f"ref_net:{ref_net_sid}")
            if ref_net_node is None:
                continue
            # Canonical name = ref's role_label if present, else its source_id
            canonical_name = ref_net_node.role_label or ref_net_node.source_id
            vote_key = (
                canonical_name,
                ref_net_node.role,
                ref_net_node.role_label or "",
                ref_net_node.source_id,
            )
            votes[cur_net_sid][vote_key] += 1

    # 2. Resolve + apply with protection rules ────────────────────────────────
    applied: list[dict[str, Any]] = []
    cur_nets = cur_netlist_v2.get("nets")
    if not isinstance(cur_nets, list):
        return applied

    for net in cur_nets:
        if not isinstance(net, dict):
            continue
        net_id = str(net.get("electrical_net_id") or net.get("net_id") or "")
        if not net_id or net_id not in votes:
            continue
        if _is_protected_role_source(net):
            continue
        vote_counter = votes[net_id]
        if not vote_counter:
            continue
        (winner_canonical, winner_role, winner_label, winner_ref_net), winner_count = (
            vote_counter.most_common(1)[0]
        )
        total = sum(vote_counter.values())
        confidence = winner_count / total
        if confidence < vote_threshold:
            continue

        # Apply mutations (mirror the field set written by the legacy
        # _apply_role_inferences_to_netlist so downstream consumers don't
        # need to know which path produced the labeling).
        net["canonical_name"] = winner_canonical
        net["role"] = winner_role
        net["role_label"] = winner_label
        aliases = list(net.get("aliases") or [])
        net["aliases"] = list(dict.fromkeys(
            [winner_canonical, net_id, *aliases]
        ))
        net["role_source"] = SOURCE_PROPAGATION
        net["inferred_reference_net"] = winner_ref_net
        net["inferred_confidence"] = round(confidence, 4)
        # Mirror power_role for the consumers that key on it (e.g. s5
        # _extract_net_roles, _diagnose_wiring_errors).
        if winner_role == "power":
            net["power_role"] = winner_canonical if winner_canonical in {"VCC", "VEE", "VDD", "VSS"} else "VCC"
        elif winner_role == "ground":
            net["power_role"] = "GND"

        applied.append({
            "current_net": net_id,
            "canonical_name": winner_canonical,
            "role": winner_role,
            "role_label": winner_label,
            "reference_net": winner_ref_net,
            "confidence": round(confidence, 4),
            "source": SOURCE_PROPAGATION,
        })

    return applied


__all__ = [
    "propagate_canonical_via_alignment",
    "PROTECTED_ROLE_SOURCES",
    "SOURCE_PROPAGATION",
    "DEFAULT_VOTE_THRESHOLD",
]
