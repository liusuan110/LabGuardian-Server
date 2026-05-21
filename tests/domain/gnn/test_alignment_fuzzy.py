"""Phase E · S1 — fuzzy 组件对齐器单测。

固化以下契约（按"wire 是特殊角色"的根本设计原则）：
- 同 net wire / 缺组件 / 多余 wire 都不影响真实元件对齐
- IC pin_key 跨数据源规范化（ref ``"2"`` ↔ cur ``"pin2"``）
- 跨 net wire 被识别并 union，记录到 ``notes['wire_collapsed_groups']``
- 子类型差异（UA741 vs LM358）通过 bucket key 隔离，不跨 IC 对齐
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.domain.gnn.alignment_fuzzy import (
    DEFAULT_MAX_MATCH_COST,
    WIRE_CTYPES,
    _bucket_components,
    _build_wire_collapsed_net_union,
    _canonical_pin_key,
    _comp_pin_to_net,
    align_components_by_signature,
)
from app.domain.gnn.port_graph import (
    build_from_logical_reference,
    build_from_netlist_v2,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
REF_INVERTING = FIXTURES / "references" / "test_opamp_inverting_v1.json"
CUR_INVERTING = FIXTURES / "real_student" / "inverting_amp_correct_v1.json"
REF_DIFF_AMP = FIXTURES / "references" / "test_bjt_diff_amp_v1.json"
CUR_DIFF_AMP_CORRECT = FIXTURES / "real_student" / "bjt_diff_amp_correct_v1.json"


def _load_ref(path: Path, subtypes: dict[str, str] | None = None):
    return build_from_logical_reference(
        json.loads(path.read_text(encoding="utf-8")),
        extra_subtypes_by_source_id=subtypes or {},
    )


def _load_cur(path: Path):
    return build_from_netlist_v2(json.loads(path.read_text(encoding="utf-8")))


def _expected_inverting_align() -> dict[str, str]:
    """ref → cur (verified manually against the fixtures)."""
    return {"U1": "IC1", "R_in": "R3", "R_f": "R1", "R_p": "R2"}


# ---------------------------------------------------------------------------
# _canonical_pin_key — IC pin normalization
# ---------------------------------------------------------------------------


def test_canonical_pin_key_strips_pin_prefix_for_ic() -> None:
    """IC pins: ref uses "2", cur uses "pin2" — must normalize to same key."""
    assert _canonical_pin_key("IC", "2") == "2"
    assert _canonical_pin_key("IC", "pin2") == "2"
    assert _canonical_pin_key("IC", "Pin7") == "7"
    assert _canonical_pin_key("IC", "PIN8") == "8"


def test_canonical_pin_key_passthrough_for_non_ic() -> None:
    """Non-IC components already use canonical keys on both paths."""
    assert _canonical_pin_key("Resistor", "pin1") == "pin1"
    assert _canonical_pin_key("Transistor", "base") == "base"
    assert _canonical_pin_key("Potentiometer", "wiper") == "wiper"
    assert _canonical_pin_key("LED", "anode") == "anode"


# ---------------------------------------------------------------------------
# Bucketing — Wire excluded; IC subtype separates buckets
# ---------------------------------------------------------------------------


def test_bucket_components_excludes_wire() -> None:
    """**Wire 特殊性**：Wire 绝不进 component 对齐池。"""
    cur = _load_cur(CUR_INVERTING)
    buckets = _bucket_components(cur)
    all_comps = [c.source_id for bucket in buckets.values() for c in bucket]
    # cur has W1, W2, W3 wires — none should appear in buckets
    assert not any(c.startswith("W") for c in all_comps), (
        f"Wire components leaked into alignment buckets: {all_comps}"
    )
    # The 4 real components (IC1 + 3 resistors) should all be bucketed
    assert "IC1" in all_comps
    assert {"R1", "R2", "R3"} <= set(all_comps)


def test_bucket_keys_include_ic_subtype() -> None:
    """IC subtype must factor into bucket key — prevents UA741 ↔ LM358 cross-match."""
    ref = _load_ref(REF_INVERTING, subtypes={"U1": "UA741"})
    buckets = _bucket_components(ref)
    ic_keys = [k for k in buckets if k[0] == "IC"]
    assert ic_keys, "should have an IC bucket"
    # Each IC bucket key carries the subtype as 2nd element, normalized upper
    for key in ic_keys:
        assert key[1] == "UA741", f"IC bucket key should carry subtype: {key}"


# ---------------------------------------------------------------------------
# _build_wire_collapsed_net_union
# ---------------------------------------------------------------------------


def test_same_net_wires_yield_no_union() -> None:
    """**Wire 特殊性 #1**：同 net wire 是 net 物理延伸，不产生 union。

    cur 的 W1/W2/W3 三根 wire 两端都在同一 net 上。``wire_collapsed_groups``
    应该为空列表（无 net 被合并）。
    """
    cur = _load_cur(CUR_INVERTING)
    net_union = _build_wire_collapsed_net_union(cur)
    # All nets should map to themselves (no actual unioning happened)
    for net_sid, repr_sid in net_union.items():
        assert net_sid == repr_sid, (
            f"same-net wire incorrectly unioned: {net_sid} → {repr_sid}"
        )


def test_cross_net_wire_unions_endpoint_nets() -> None:
    """**Wire 特殊性 #2**：跨 net wire (短路) 应触发 union。"""
    cur_payload = json.loads(CUR_INVERTING.read_text(encoding="utf-8"))
    # Inject a cross-net wire connecting VCC (NET_001) and GND (NET_004)
    bridge = copy.deepcopy(cur_payload["components"][1])  # clone a wire shell
    bridge["component_id"] = "W_bad"
    bridge["pins"][0]["electrical_net_id"] = "NET_001"
    bridge["pins"][1]["electrical_net_id"] = "NET_004"
    cur_payload["components"].append(bridge)

    cur = build_from_netlist_v2(cur_payload)
    net_union = _build_wire_collapsed_net_union(cur)
    # NET_001 and NET_004 should share the same representative
    assert net_union["NET_001"] == net_union["NET_004"], (
        f"cross-net wire did not union: NET_001→{net_union['NET_001']} "
        f"NET_004→{net_union['NET_004']}"
    )


# ---------------------------------------------------------------------------
# _comp_pin_to_net — uses canonical pin_key
# ---------------------------------------------------------------------------


def test_comp_pin_to_net_uses_canonical_keys() -> None:
    """IC pins on ref vs cur should produce the same key set after normalization."""
    ref = _load_ref(REF_INVERTING, subtypes={"U1": "UA741"})
    cur = _load_cur(CUR_INVERTING)
    u1 = next(c for c in ref.components.values() if c.source_id == "U1")
    ic1 = next(c for c in cur.components.values() if c.source_id == "IC1")
    ref_pins = _comp_pin_to_net(ref, u1)
    cur_pins = _comp_pin_to_net(cur, ic1)
    # Both should yield numeric keys "2" / "3" / "4" / "6" / "7"
    assert set(ref_pins) == set(cur_pins), (
        f"ref keys {set(ref_pins)} vs cur keys {set(cur_pins)} — normalization failed"
    )
    assert set(ref_pins) >= {"2", "3", "4", "6", "7"}


# ---------------------------------------------------------------------------
# align_components_by_signature — 4 wire scenarios + 4 e2e scenarios
# ---------------------------------------------------------------------------


def test_align_baseline_wire_aware() -> None:
    """场景 0 · baseline: cur 有 3 根同 net wire, 4/4 真实元件应对齐。"""
    ref = _load_ref(REF_INVERTING, subtypes={"U1": "UA741"})
    cur = _load_cur(CUR_INVERTING)
    align = align_components_by_signature(ref, cur)
    assert align.ref_to_cur_component == _expected_inverting_align()
    assert align.notes["unmatched_ref_components"] == []
    assert align.notes["unmatched_cur_components"] == []
    # Same-net wires should produce no union groups
    assert align.notes["wire_collapsed_groups"] == []


def test_align_handles_missing_component() -> None:
    """场景 1 · 缺 R_p: 剩余 3/3 对齐, R_p 进 unmatched_ref。"""
    cur_payload = json.loads(CUR_INVERTING.read_text(encoding="utf-8"))
    cur_payload["components"] = [
        c for c in cur_payload["components"] if c["component_id"] != "R2"
    ]
    align = align_components_by_signature(
        _load_ref(REF_INVERTING, subtypes={"U1": "UA741"}),
        build_from_netlist_v2(cur_payload),
    )
    assert align.ref_to_cur_component == {
        "U1": "IC1", "R_in": "R3", "R_f": "R1",
    }
    assert "R_p" in align.notes["unmatched_ref_components"]


def test_align_unaffected_by_dropping_same_net_wire() -> None:
    """场景 2 · 删 W2: wire 不入桶, 结果与 baseline 完全一致。"""
    cur_payload = json.loads(CUR_INVERTING.read_text(encoding="utf-8"))
    cur_payload["components"] = [
        c for c in cur_payload["components"] if c["component_id"] != "W2"
    ]
    align = align_components_by_signature(
        _load_ref(REF_INVERTING, subtypes={"U1": "UA741"}),
        build_from_netlist_v2(cur_payload),
    )
    assert align.ref_to_cur_component == _expected_inverting_align()


def test_align_unaffected_by_extra_same_net_wire() -> None:
    """场景 3 · 多 1 根同 net wire: 结果应与 baseline 一致 (wire 不计入对齐)。"""
    cur_payload = json.loads(CUR_INVERTING.read_text(encoding="utf-8"))
    extra = copy.deepcopy(cur_payload["components"][1])
    extra["component_id"] = "W_extra"
    extra["pins"][0]["hole_id"] = "A30"
    extra["pins"][0]["electrical_node_id"] = "ROW_30_L"
    extra["pins"][1]["hole_id"] = "B30"
    extra["pins"][1]["electrical_node_id"] = "ROW_30_L"
    cur_payload["components"].append(extra)
    align = align_components_by_signature(
        _load_ref(REF_INVERTING, subtypes={"U1": "UA741"}),
        build_from_netlist_v2(cur_payload),
    )
    assert align.ref_to_cur_component == _expected_inverting_align()
    # Extra wire should not appear in unmatched_cur (wires aren't part of pool)
    assert "W_extra" not in align.notes["unmatched_cur_components"]


def test_align_records_cross_net_wire_groups() -> None:
    """场景 4 · 跨 net wire: 在 notes 记录 union 信息以便上层告警。"""
    cur_payload = json.loads(CUR_INVERTING.read_text(encoding="utf-8"))
    bridge = copy.deepcopy(cur_payload["components"][1])
    bridge["component_id"] = "W_bad"
    bridge["pins"][0]["electrical_net_id"] = "NET_001"
    bridge["pins"][1]["electrical_net_id"] = "NET_004"
    cur_payload["components"].append(bridge)

    align = align_components_by_signature(
        _load_ref(REF_INVERTING, subtypes={"U1": "UA741"}),
        build_from_netlist_v2(cur_payload),
    )
    # Cross-net wire should produce a non-empty group containing both nets
    groups = align.notes["wire_collapsed_groups"]
    assert any(
        {"NET_001", "NET_004"} <= set(group) for group in groups
    ), f"cross-net wire group missing from notes: {groups}"


# ---------------------------------------------------------------------------
# Net alignment derivation (multi-vote)
# ---------------------------------------------------------------------------


def test_net_alignment_derived_from_aligned_components() -> None:
    """每个 ref net 应根据已对齐元件的 pin-net 关系投票得到 cur net。"""
    align = align_components_by_signature(
        _load_ref(REF_INVERTING, subtypes={"U1": "UA741"}),
        _load_cur(CUR_INVERTING),
    )
    # Critical: INV / VOUT / VIN / V_P / VCC are all derivable from
    # the U1 + Resistors alignment
    assert align.ref_to_cur_net.get("INV") == "NET_002"
    assert align.ref_to_cur_net.get("VOUT") == "NET_007"
    assert align.ref_to_cur_net.get("VIN") == "NET_005"
    assert align.ref_to_cur_net.get("V_P") == "NET_008"
    assert align.ref_to_cur_net.get("VCC") == "NET_001"


# ---------------------------------------------------------------------------
# Diff amp (3 transistors + Potentiometer) — exercises richer ctype mix
# ---------------------------------------------------------------------------


def test_align_diff_amp_with_transistors_and_pot() -> None:
    """差分放大器: 9 真实元件 (3 BJT + 5 R + 1 Pot) + 2 wire。

    用 _correct_v1 fixture（合成 runtime_scene，含 W1/W2 同 net wire）
    验证 Hungarian 在更复杂拓扑下仍稳定。
    """
    ref = _load_ref(REF_DIFF_AMP)
    cur = _load_cur(CUR_DIFF_AMP_CORRECT)
    align = align_components_by_signature(ref, cur)
    # All 9 ref components should align (4 buckets: Transistor×3, Resistor×5,
    # Potentiometer×1, all bucket sizes match between ref and cur).
    expected_ref_comps = {
        "VT1", "VT2", "VT3", "Rc1", "Rc2", "R1", "R2", "R_E", "Rp",
    }
    assert set(align.ref_to_cur_component) == expected_ref_comps, (
        f"diff amp alignment incomplete: {align.ref_to_cur_component}"
    )
    assert align.notes["unmatched_ref_components"] == []
    # Cur should also have no unmatched non-wire components
    assert align.notes["unmatched_cur_components"] == []


def test_align_diff_amp_transistor_base_correctly_routed() -> None:
    """差分对 base 必须对齐到正确的输入 (UI1/UI2)。"""
    ref = _load_ref(REF_DIFF_AMP)
    cur = _load_cur(CUR_DIFF_AMP_CORRECT)
    align = align_components_by_signature(ref, cur)
    # VT1.base → UI1, VT2.base → UI2 (UI1 ≠ UI2 in cur)
    assert align.ref_to_cur_net.get("UI1") != align.ref_to_cur_net.get("UI2"), (
        "diff amp differential inputs got collapsed to same cur net"
    )


# ---------------------------------------------------------------------------
# ComponentAlignment contract preservation
# ---------------------------------------------------------------------------


def test_alignment_notes_carry_diagnostics() -> None:
    """notes 字段必须含可观测性诊断信息（上层 UI / log 会读这些字段）。"""
    align = align_components_by_signature(
        _load_ref(REF_INVERTING, subtypes={"U1": "UA741"}),
        _load_cur(CUR_INVERTING),
    )
    assert align.notes["constructor"] == "align_components_by_signature"
    assert "unmatched_ref_components" in align.notes
    assert "unmatched_cur_components" in align.notes
    assert "wire_collapsed_groups" in align.notes
    assert "match_costs" in align.notes
    assert "wire_ctypes" in align.notes
    # match_costs values should all be ≤ DEFAULT_MAX_MATCH_COST
    for pair_key, cost in align.notes["match_costs"].items():
        assert cost <= DEFAULT_MAX_MATCH_COST, (
            f"impossible: matched {pair_key} above cutoff cost {cost}"
        )


def test_alignment_reverse_caches_populated() -> None:
    """ComponentAlignment.__post_init__ 应自动建立反向缓存。"""
    align = align_components_by_signature(
        _load_ref(REF_INVERTING, subtypes={"U1": "UA741"}),
        _load_cur(CUR_INVERTING),
    )
    # ref_to_cur_component → cur_to_ref_component
    for ref_sid, cur_sid in align.ref_to_cur_component.items():
        assert align.cur_to_ref_component.get(cur_sid) == ref_sid
    # ref_to_cur_net → cur_to_ref_net
    for ref_sid, cur_sid in align.ref_to_cur_net.items():
        assert align.cur_to_ref_net.get(cur_sid) == ref_sid


def test_empty_hcg_pair_returns_empty_alignment() -> None:
    """边缘情况：cur 完全空（无组件）应安全返回空对齐而不崩溃。"""
    from app.domain.gnn.hetero_circuit import HeteroCircuitGraph

    ref = _load_ref(REF_INVERTING, subtypes={"U1": "UA741"})
    bare_cur = HeteroCircuitGraph(side="cur")  # 全空
    align = align_components_by_signature(ref, bare_cur)
    assert align.ref_to_cur_component == {}
    assert align.ref_to_cur_net == {}
    # All ref comps land in unmatched
    assert set(align.notes["unmatched_ref_components"]) == {
        "U1", "R_in", "R_f", "R_p",
    }
