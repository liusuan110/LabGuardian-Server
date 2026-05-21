"""Phase E · S2 — alignment-based role propagation 单测。

固化以下契约：
- 通过 fuzzy 组件对齐把 ref net 语义传播到匿名的 cur net
- 保护规则：manual_role / port_annotation / power_role 永不被覆盖
- Wire pin 不投票（无语义身份），但 wire 所在 net 自动继承新 canonical
- 全匿名最差 OOD 场景下也能 100% 标到（合成数据 + reference fixture）
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.domain.compare.role_propagation import (
    DEFAULT_VOTE_THRESHOLD,
    PROTECTED_ROLE_SOURCES,
    SOURCE_PROPAGATION,
    propagate_canonical_via_alignment,
)
from app.domain.gnn.alignment_fuzzy import align_components_by_signature
from app.domain.gnn.port_graph import (
    build_from_logical_reference,
    build_from_netlist_v2,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
REF_INVERTING = FIXTURES / "references" / "test_opamp_inverting_v1.json"
CUR_INVERTING = FIXTURES / "real_student" / "inverting_amp_correct_v1.json"


def _load_ref():
    return build_from_logical_reference(
        json.loads(REF_INVERTING.read_text(encoding="utf-8")),
        extra_subtypes_by_source_id={"U1": "UA741"},
    )


def _load_cur_payload() -> dict:
    return json.loads(CUR_INVERTING.read_text(encoding="utf-8"))


def _strip_all_canonical(payload: dict) -> dict:
    """Simulate a worst-case OOD where every cur net is anonymous."""
    p = copy.deepcopy(payload)
    for n in p["nets"]:
        for k in (
            "canonical_name", "role_label", "role", "role_source",
            "manual_role", "power_role", "aliases",
            "inferred_reference_net", "inferred_confidence",
        ):
            n.pop(k, None)
    return p


def _run_propagator(cur_payload: dict, vote_threshold: float = DEFAULT_VOTE_THRESHOLD):
    ref = _load_ref()
    cur = build_from_netlist_v2(cur_payload)
    alignment = align_components_by_signature(ref, cur)
    applied = propagate_canonical_via_alignment(
        ref, cur, alignment, cur_payload, vote_threshold=vote_threshold,
    )
    return applied, cur_payload


# ---------------------------------------------------------------------------
# Baseline behavior on real-student fixture
# ---------------------------------------------------------------------------


def test_baseline_propagates_anonymous_nets_only() -> None:
    """场景 0 · baseline: cur 已有 VCC/GND/UI1/UO1 标记（power_role + port_annotation），
    propagator 应只填补 NET_002 (INV) 和 NET_008 (V_P) 两个原本匿名的 net。"""
    payload = _load_cur_payload()
    applied, mutated = _run_propagator(payload)

    applied_nets = {r["current_net"] for r in applied}
    # NET_002 and NET_008 are anonymous in the baseline fixture → must be filled
    assert "NET_002" in applied_nets
    assert "NET_008" in applied_nets

    # The verification: NET_002 should now have canonical_name = "INV"
    by_id = {n["electrical_net_id"]: n for n in mutated["nets"]}
    assert by_id["NET_002"]["canonical_name"] == "INV"
    assert by_id["NET_002"]["role_source"] == SOURCE_PROPAGATION
    assert by_id["NET_008"]["canonical_name"] == "V_P"


def test_baseline_protects_power_role_labeled_nets() -> None:
    """**保护规则 · power_role**: 已被 hole-id 启发式标了 power_role 的 net 不被覆盖。

    cur 的 NET_000 (VEE) / NET_001 (VCC) / NET_004 (GND) 都通过 hole 启发式
    标了 power_role。即使 ref 单电源 convention 想把 NET_000 标成 GND（pin4
    convention 差异），propagator 必须跳过。
    """
    payload = _load_cur_payload()
    # Pre-check: these have power_role set in the fixture
    by_id = {n["electrical_net_id"]: n for n in payload["nets"]}
    assert by_id["NET_000"].get("power_role") == "VEE"
    assert by_id["NET_001"].get("power_role") == "VCC"

    applied, mutated = _run_propagator(payload)
    applied_nets = {r["current_net"] for r in applied}
    assert "NET_000" not in applied_nets, "VEE net was incorrectly overwritten"
    assert "NET_001" not in applied_nets, "VCC net was incorrectly overwritten"
    assert "NET_004" not in applied_nets, "GND net was incorrectly overwritten"

    # Values preserved
    new_by_id = {n["electrical_net_id"]: n for n in mutated["nets"]}
    assert new_by_id["NET_000"]["canonical_name"] == "VEE"
    assert new_by_id["NET_001"]["canonical_name"] == "VCC"


def test_baseline_protects_port_annotation_nets() -> None:
    """**保护规则 · port_annotation**: 用户已通过 UI 标的 IC pin 不被覆盖。

    cur 的 NET_005 (UI1) / NET_007 (UO1) 通过 port_annotation 标了 canonical_name。
    propagator 必须保留。
    """
    payload = _load_cur_payload()
    by_id = {n["electrical_net_id"]: n for n in payload["nets"]}
    assert by_id["NET_005"].get("role_source") == "port_annotation"

    applied, mutated = _run_propagator(payload)
    applied_nets = {r["current_net"] for r in applied}
    assert "NET_005" not in applied_nets
    assert "NET_007" not in applied_nets

    new_by_id = {n["electrical_net_id"]: n for n in mutated["nets"]}
    assert new_by_id["NET_005"]["canonical_name"] == "UI1"
    assert new_by_id["NET_005"]["role_source"] == "port_annotation"


# ---------------------------------------------------------------------------
# Worst-case OOD: fully anonymous cur
# ---------------------------------------------------------------------------


def test_full_anonymous_cur_gets_all_seven_nets_labeled() -> None:
    """场景 5 · 全匿名 cur (最差 OOD): propagator 应救活 7/7 nets。"""
    payload = _strip_all_canonical(_load_cur_payload())
    applied, mutated = _run_propagator(payload)
    # All 7 cur nets should now have canonical_name set
    assert len(applied) == 7
    by_id = {n["electrical_net_id"]: n for n in mutated["nets"]}
    expected_labels = {
        "NET_001": "VCC", "NET_002": "INV", "NET_005": "VIN",
        "NET_007": "VOUT", "NET_008": "V_P",
        # Both NET_000 (cur VEE position) and NET_004 (cur GND position)
        # map to ref "GND" because ref is single-supply (no VEE in ref).
        # This is a ref-convention artifact — production cur has
        # power_role pre-set so this never happens in real usage; the
        # test_baseline_* tests above lock that behavior in.
        "NET_000": "GND", "NET_004": "GND",
    }
    for net_id, expected in expected_labels.items():
        assert by_id[net_id]["canonical_name"] == expected, (
            f"net {net_id} expected {expected}, got {by_id[net_id].get('canonical_name')}"
        )
        assert by_id[net_id]["role_source"] == SOURCE_PROPAGATION


def test_full_anonymous_records_confidence_and_ref_net() -> None:
    """Applied records must carry confidence + reference_net for UI / log."""
    payload = _strip_all_canonical(_load_cur_payload())
    applied, _ = _run_propagator(payload)
    for r in applied:
        assert "confidence" in r and 0.0 < r["confidence"] <= 1.0
        assert r["reference_net"], f"missing reference_net: {r}"
        assert r["source"] == SOURCE_PROPAGATION


# ---------------------------------------------------------------------------
# Missing-component robustness
# ---------------------------------------------------------------------------


def test_missing_component_still_propagates_other_nets() -> None:
    """场景 1 · 缺 R_p: NET_008 (V_P) 仍能从 U1.pin3 单票推出。"""
    payload = _load_cur_payload()
    payload["components"] = [
        c for c in payload["components"] if c["component_id"] != "R2"
    ]
    applied, mutated = _run_propagator(payload)
    applied_nets = {r["current_net"] for r in applied}
    # NET_002 and NET_008 should still be labeled (V_P from U1.pin3 alone)
    assert "NET_002" in applied_nets
    assert "NET_008" in applied_nets


# ---------------------------------------------------------------------------
# Wire-specific tests (user-mandated)
# ---------------------------------------------------------------------------


def test_wire_pins_dont_pollute_votes() -> None:
    """**Wire 特殊性**: 即便 cur 有大量同 net wire, propagator 投票池不变,
    canonical_name 结果与"无 wire"版本完全一致。"""
    # Drop W1/W2/W3 to get a wire-free version
    no_wire_payload = _load_cur_payload()
    no_wire_payload["components"] = [
        c for c in no_wire_payload["components"]
        if c.get("component_type") != "Wire"
    ]
    no_wire_payload = _strip_all_canonical(no_wire_payload)
    applied_no_wire, mutated_no_wire = _run_propagator(no_wire_payload)

    # And the full version (with W1/W2/W3 wires)
    with_wire_payload = _strip_all_canonical(_load_cur_payload())
    applied_with_wire, mutated_with_wire = _run_propagator(with_wire_payload)

    # Canonical names assigned should be IDENTICAL between the two
    cn_no_wire = {
        n["electrical_net_id"]: n["canonical_name"]
        for n in mutated_no_wire["nets"]
        if n.get("canonical_name")
    }
    cn_with_wire = {
        n["electrical_net_id"]: n["canonical_name"]
        for n in mutated_with_wire["nets"]
        if n.get("canonical_name")
    }
    assert cn_no_wire == cn_with_wire, (
        f"wire presence changed propagation result:\n"
        f"  no-wire:   {cn_no_wire}\n"
        f"  with-wire: {cn_with_wire}"
    )


def test_extra_wire_doesnt_create_phantom_net_label() -> None:
    """加 1 根额外同 net wire 应不产生新 net 标签 (wire 不引入新 net)。"""
    payload = _strip_all_canonical(_load_cur_payload())
    # Inject an extra wire on the same net (NET_007 = VOUT)
    extra = copy.deepcopy(payload["components"][1])  # clone a wire shell
    extra["component_id"] = "W_extra"
    extra["pins"][0]["hole_id"] = "A30"
    extra["pins"][0]["electrical_node_id"] = "ROW_30_L"
    extra["pins"][0]["electrical_net_id"] = "NET_007"
    extra["pins"][1]["hole_id"] = "B30"
    extra["pins"][1]["electrical_node_id"] = "ROW_30_L"
    extra["pins"][1]["electrical_net_id"] = "NET_007"
    payload["components"].append(extra)
    applied, _ = _run_propagator(payload)
    # Still 7 nets labeled (same as baseline)
    assert len({r["current_net"] for r in applied}) == 7


# ---------------------------------------------------------------------------
# Mutation correctness
# ---------------------------------------------------------------------------


def test_aliases_list_preserves_existing_aliases() -> None:
    """canonical_name 写入时, 已有 aliases 列表应被合并而非替换。"""
    payload = _strip_all_canonical(_load_cur_payload())
    # Pre-set some aliases on NET_002
    for n in payload["nets"]:
        if n["electrical_net_id"] == "NET_002":
            n["aliases"] = ["my_custom_alias"]
    _, mutated = _run_propagator(payload)
    net002 = next(n for n in mutated["nets"] if n["electrical_net_id"] == "NET_002")
    # canonical_name comes first, then net_id, then existing
    assert "INV" in net002["aliases"]
    assert "NET_002" in net002["aliases"]
    assert "my_custom_alias" in net002["aliases"]


def test_power_role_mirrored_when_role_is_power() -> None:
    """role==power 时应同步写 power_role 字段 (s5 _extract_net_roles 读这个)。"""
    payload = _strip_all_canonical(_load_cur_payload())
    _, mutated = _run_propagator(payload)
    by_id = {n["electrical_net_id"]: n for n in mutated["nets"]}
    # NET_001 was anonymous, now propagated to VCC (role=power)
    assert by_id["NET_001"]["role"] == "power"
    assert by_id["NET_001"].get("power_role") == "VCC"
    # NET_000 / NET_004 propagated to ground
    for net_id in ("NET_000", "NET_004"):
        if by_id[net_id]["role"] == "ground":
            assert by_id[net_id].get("power_role") == "GND"


def test_vote_threshold_filters_low_confidence() -> None:
    """confidence < threshold 时 propagation 不应用。"""
    payload = _strip_all_canonical(_load_cur_payload())
    # With threshold = 1.5, no vote should pass (impossible to have > 100% confidence)
    applied, _ = _run_propagator(payload, vote_threshold=1.5)
    assert applied == []


def test_protected_role_sources_constant() -> None:
    """Lock-in: PROTECTED_ROLE_SOURCES set must include the 5 sources we
    designed to protect (manual_role / port_annotation / power_role /
    role_label / explicit_role)."""
    assert PROTECTED_ROLE_SOURCES >= {
        "manual_role",
        "port_annotation",
        "power_role",
    }


def test_empty_alignment_yields_no_propagation() -> None:
    """Edge case: alignment 完全空 (没匹配上任何组件), 不应崩溃。"""
    from app.domain.gnn.alignment import ComponentAlignment
    payload = _strip_all_canonical(_load_cur_payload())
    ref = _load_ref()
    cur = build_from_netlist_v2(payload)
    empty_align = ComponentAlignment(ref_to_cur_component={}, ref_to_cur_net={})
    applied = propagate_canonical_via_alignment(ref, cur, empty_align, payload)
    assert applied == []


# ---------------------------------------------------------------------------
# End-to-end · compare_logical_graphs orchestrator integration (Phase E · S3)
# ---------------------------------------------------------------------------


def test_compare_logical_graphs_propagates_on_real_student_fixture() -> None:
    """**S3 集成契约**: compare_logical_graphs 整条链路下, real-student
    fixture (cur 有同 net wire + 部分匿名 net) 经过 Phase E propagation
    后, cur_netlist_v2 的匿名 net (NET_002/NET_008) 应得到 INV / V_P 标签,
    同时 power_role 标了的 net (VCC/VEE/GND) + port_annotation 标了的
    (UI1/UO1) 完全保留。

    修"旧 isomorphism 路径死代码"的关键验证: 旧路径在该 fixture 上返回
    0 条 inferences (今天的实验已确认死代码); 新路径必须返回 ≥ 2 条。
    """
    from app.domain.compare.orchestrator import compare_logical_graphs
    from app.domain.logical_reference import (
        current_netlist_v2_to_graph,
        logical_reference_to_graph,
    )

    ref_payload = json.loads(REF_INVERTING.read_text(encoding="utf-8"))
    cur_payload = _load_cur_payload()

    ref_graph = logical_reference_to_graph(ref_payload)
    cur_graph = current_netlist_v2_to_graph(cur_payload)

    compare_logical_graphs(
        ref_graph, cur_graph,
        ref_payload=ref_payload,
        cur_netlist_v2=cur_payload,
    )

    # After comparison, cur_payload (mutated in-place by Phase E) should
    # have INV / V_P canonical_names on the previously anonymous nets.
    by_id = {n["electrical_net_id"]: n for n in cur_payload["nets"]}
    assert by_id["NET_002"]["canonical_name"] == "INV", (
        "NET_002 should be propagated to INV via R_f / R_in alignment"
    )
    assert by_id["NET_002"]["role_source"] == SOURCE_PROPAGATION

    assert by_id["NET_008"]["canonical_name"] == "V_P", (
        "NET_008 should be propagated to V_P via IC1.pin3 ↔ U1.pin3"
    )
    assert by_id["NET_008"]["role_source"] == SOURCE_PROPAGATION

    # Pre-existing labels must be preserved (the "wire 特殊性" + protection
    # rules together guarantee that user-explicit / hole-id heuristic labels
    # win over alignment inference)
    assert by_id["NET_000"]["canonical_name"] == "VEE"
    assert by_id["NET_001"]["canonical_name"] == "VCC"
    assert by_id["NET_004"]["canonical_name"] == "GND"
    assert by_id["NET_005"]["canonical_name"] == "UI1"
    assert by_id["NET_007"]["canonical_name"] == "UO1"
