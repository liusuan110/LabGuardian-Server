"""P0.8 · build_seal_samples + assert_observed_edges_covered tests.

Covers all 10 critical concerns from plan §附录 A.8:
- TaskType + field correctness (4)
- LabelSource one-by-one (10)
- SealSampleGroup with floating + wrong_redirect query_origin (6)
- LabelStats accuracy (5)
- Behavior / edge cases (3)
- Performance budget
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.gnn import (
    LabelBuildResult,
    LabelSource,
    TaskType,
    alignment_from_dicts,
    assert_observed_edges_covered,
    build_from_logical_reference,
    build_seal_samples,
    identity_alignment,
)
from app.domain.gnn.port_graph import build_hetero_circuit_graph

from .conftest import hcg_to_cur_nx

FIXTURE_RC = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_rc_v1.json"
)
FIXTURE_OPAMP = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "references"
    / "test_opamp_buffer_v1.json"
)


# ---------------------------------------------------------------------------
# Helpers — ref + cur HCG synthesis
# ---------------------------------------------------------------------------


def _build_ref(path: Path):
    return build_from_logical_reference(json.loads(path.read_text()))


def _build_cur(ref, perturbations=None, subtype_by_source_id=None):
    g = hcg_to_cur_nx(ref, perturbations=perturbations)
    return build_hetero_circuit_graph(
        g, side="cur", subtype_by_source_id=subtype_by_source_id
    )


# UA741 buffer always needs subtype propagation for FORBIDDEN/OPTIONAL semantics.
_UA741_SUBTYPES = {"U1": "UA741"}


def _perfect_pair(path: Path = FIXTURE_RC):
    ref = _build_ref(path)
    cur = _build_cur(ref)
    return ref, cur, identity_alignment(ref, cur)


# ---------------------------------------------------------------------------
# TaskType + field correctness (4)
# ---------------------------------------------------------------------------


def test_every_sample_has_known_task_type() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align)
    known = {t.value for t in TaskType}
    for s in result.samples:
        assert s.task_type in known, s


def test_wrong_edge_candidate_matches_subgraph_anchors() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align)
    for s in result.samples:
        if s.task_type != TaskType.WRONG_EDGE.value:
            continue
        assert s.candidate_edge == (
            s.subgraph.target_port_id,
            s.subgraph.target_net_id,
        ), s


def test_ref_present_expected_equals_candidate() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align)
    ref_present = [
        s for s in result.samples
        if s.label_source == LabelSource.REF_PRESENT.value
    ]
    assert ref_present, "expected at least one REF_PRESENT in identity pair"
    for s in ref_present:
        assert s.expected_edge == s.candidate_edge


def test_ref_symmetric_swap_expected_points_to_canonical() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align)
    sym_samples = [
        s for s in result.samples
        if s.label_source == LabelSource.REF_SYMMETRIC_SWAP.value
    ]
    assert sym_samples, "RC fixture has R/C with sym-class pin1/pin2"
    for s in sym_samples:
        assert s.is_symmetric_equivalent is True
        # expected points to the canonical ref-mapped port, not the sibling
        assert s.expected_edge != s.candidate_edge
        assert s.label == 1


# ---------------------------------------------------------------------------
# LabelSource — one-by-one coverage (10)
# ---------------------------------------------------------------------------


def test_ref_present_on_perfect_copy() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align)
    assert result.stats.by_source[LabelSource.REF_PRESENT.value] > 0


def test_ref_absent_required_when_edge_dropped() -> None:
    ref = _build_ref(FIXTURE_RC)
    cur = _build_cur(ref, perturbations=[("drop_edge", "pin1", "VIN")])
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    # Either WRONG_EDGE positive with REF_ABSENT_REQUIRED, or MISSING_EDGE
    # group's correct sample (also REF_ABSENT_REQUIRED). Both emit it.
    assert result.stats.by_source[LabelSource.REF_ABSENT_REQUIRED.value] >= 1


def test_ref_symmetric_swap_on_resistor_pin_swap() -> None:
    ref = _build_ref(FIXTURE_RC)
    cur = _build_cur(
        ref, perturbations=[("swap_pins", "R1", "pin1", "pin2")]
    )
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    # Sym swap is allowed for R.pin1↔pin2 — both directions get label=1
    assert result.stats.by_source[LabelSource.REF_SYMMETRIC_SWAP.value] > 0


def test_wrong_observed_strong_negative() -> None:
    """KEY TEST: when cur has an extra wrong edge (R1.pin1 → GND instead of
    VIN), label_builder MUST emit a label=0 WRONG_OBSERVED sample for that
    exact (port, net) — not rely on NEGATIVE_RANDOM."""

    ref = _build_ref(FIXTURE_RC)
    # Replace R1.pin1's VIN connection with GND (move it to the wrong net)
    cur = _build_cur(
        ref,
        perturbations=[
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ],
    )
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)

    wrong_obs_samples = [
        s for s in result.samples
        if s.label_source == LabelSource.WRONG_OBSERVED.value
    ]
    assert wrong_obs_samples, "WRONG_OBSERVED must fire when cur has wrong edge"
    # Find the specific wrong-net sample
    target = [
        s for s in wrong_obs_samples
        if s.candidate_edge == ("cur_port:R1.pin1", "cur_net:GND")
    ]
    assert target, "specific (R1.pin1, GND) wrong edge must be labeled"
    assert target[0].label == 0
    assert target[0].task_type == TaskType.WRONG_EDGE.value
    # expected_edge should point to the correct net (VIN)
    assert target[0].expected_edge == ("cur_port:R1.pin1", "cur_net:VIN")


def test_wrong_observed_coverage_invariant_holds() -> None:
    """Every observed cur edge that isn't ref-correct MUST have a WRONG_EDGE
    negative sample — no silent gaps."""

    ref = _build_ref(FIXTURE_RC)
    cur = _build_cur(
        ref,
        perturbations=[
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ],
    )
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    # Should not raise
    assert_observed_edges_covered(result, cur, ref, align)


def test_same_net_wire_labeled_as_positive_y1() -> None:
    """**Stage 2 contract** — 当 cur 里某个 Wire 组件的全部 pin 落到同一个
    net 时，label_builder 必须把这些 edges 标成 WIRE_SAME_NET_POSITIVE
    y=1（不是默认的 WRONG_OBSERVED y=0）。

    用 `insert_same_net_wire` perturbation 把 wire 注入到反相放大器 ref。
    """

    from app.domain.gnn.perturbation import apply_perturbation

    ref = _build_ref(
        Path(__file__).resolve().parents[2]
        / "fixtures" / "references" / "test_opamp_inverting_v1.json"
    )
    p = apply_perturbation("insert_same_net_wire", ref, seed=11)
    result = build_seal_samples(ref, p.cur_hcg, p.alignment)

    wire_pos = [
        s for s in result.samples
        if s.label_source == LabelSource.WIRE_SAME_NET_POSITIVE.value
    ]
    assert wire_pos, "无 WIRE_SAME_NET_POSITIVE 样本 — Stage 2 自描述规则未生效"

    # 期望: 每个 intentional wire 贡献 2 条边 (pin1 + pin2)
    expected_count = 2 * len(p.notes["intentional_wires"])
    assert len(wire_pos) == expected_count, (
        f"WIRE_SAME_NET_POSITIVE 数量对不上: {len(wire_pos)} vs {expected_count}"
    )

    # 每条都必须是 y=1 / WRONG_EDGE / 两端 net 一致 (candidate==expected)
    for s in wire_pos:
        assert s.label == 1
        assert s.task_type == TaskType.WRONG_EDGE.value
        assert s.candidate_edge == s.expected_edge, (
            "same-net wire 的 expected_edge 应该自指 (两端 net 相同)"
        )


def test_cross_net_wire_remains_wrong_observed_y0() -> None:
    """**Stage 2 反例不变性** — `extra_wire_bridge`（两端不同 net）仍必须
    标 WRONG_OBSERVED y=0，**绝不**能因为是 Wire 就被误识别成 positive。"""

    from app.domain.gnn.perturbation import apply_perturbation

    ref = _build_ref(
        Path(__file__).resolve().parents[2]
        / "fixtures" / "references" / "test_voltage_divider_v1.json"
    )
    p = apply_perturbation("extra_wire_bridge", ref, seed=1)
    if p.perturbation_chain == ("identity",):
        pytest.skip("divider 上没有可桥的网络对")
    result = build_seal_samples(ref, p.cur_hcg, p.alignment)

    # 只看 wire 真正的 observed edges (在 cur_hcg.edges 里的)，不要把
    # negative_random 给 wire 的 port 采样的非边样本算进来。
    wire_observed_edges = {
        (e.src_port_id, e.dst_net_id)
        for e in p.cur_hcg.edges
        if cur_port_belongs_to_wire(p.cur_hcg, e.src_port_id, p.notes["wire_id"])
    }
    assert len(wire_observed_edges) == 2, (
        f"extra_wire_bridge 应该注入 2 条 observed edges，实际 {len(wire_observed_edges)}"
    )

    wire_observed_samples = [
        s for s in result.samples
        if s.candidate_edge in wire_observed_edges
    ]
    assert wire_observed_samples, "wire 真实 observed edges 必须被 Step 2.5 采到"
    # 全部应该是 WRONG_OBSERVED，绝不能是 WIRE_SAME_NET_POSITIVE
    for s in wire_observed_samples:
        assert s.label_source == LabelSource.WRONG_OBSERVED.value, (
            f"cross-net wire 误判: {s.label_source} (期望 wrong_observed)\n"
            f"candidate_edge={s.candidate_edge}"
        )
        assert s.label == 0
    # 同时确保整个结果里没有 WIRE_SAME_NET_POSITIVE
    wire_pos = [
        s for s in result.samples
        if s.label_source == LabelSource.WIRE_SAME_NET_POSITIVE.value
    ]
    assert not wire_pos, (
        f"extra_wire_bridge 不应产生 WIRE_SAME_NET_POSITIVE，但有 {len(wire_pos)} 条"
    )


def test_wire_with_single_pin_not_marked_positive() -> None:
    """边界 case：Wire 只有一条边（degenerate）→ 不构成"同 net 延长线"模式
    → 不被识别为 WIRE_SAME_NET_POSITIVE，回退到普通逻辑。"""

    import networkx as nx
    from app.domain.gnn.port_graph import build_hetero_circuit_graph

    ref = _build_ref(FIXTURE_RC)
    # 在 cur 里手动塞一个**只有 pin1 接 net** 的 Wire
    cur_g = hcg_to_cur_nx(ref)
    cur_g.add_node("cur_comp:LooseWire", kind="comp", ctype="Wire", source_id="LooseWire")
    cur_g.add_edge(
        "cur_comp:LooseWire", "cur_net:VIN",
        kind="port_net", pin="pin1", pin_role="pin1",
    )
    cur = build_hetero_circuit_graph(cur_g, side="cur")

    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    # 唯一这条 wire 边不应被打成 WIRE_SAME_NET_POSITIVE（因为只有 1 pin
    # 不构成 same-net 模式）。它应该作为 WRONG_OBSERVED 处理。
    loose_samples = [
        s for s in result.samples
        if s.candidate_edge[0] == "cur_port:LooseWire.pin1"
    ]
    if loose_samples:  # 取决于 OPTIONAL/FORBIDDEN 处理，可能被跳过
        for s in loose_samples:
            assert s.label_source != LabelSource.WIRE_SAME_NET_POSITIVE.value, (
                f"单 pin wire 被误识别为 same-net positive: {s}"
            )


def cur_port_belongs_to_wire(cur_hcg, port_node_id: str, wire_sid: str) -> bool:
    """辅助：判断某 port 是否属于指定 source_id 的 wire 组件。"""

    if port_node_id not in cur_hcg.ports:
        return False
    parent_id = cur_hcg.ports[port_node_id].parent_component_id
    if parent_id not in cur_hcg.components:
        return False
    return cur_hcg.components[parent_id].source_id == wire_sid


def test_forbidden_violated_when_pin8_wired() -> None:
    ref = _build_ref(FIXTURE_OPAMP)
    # UA741 fixture has pin 8 = NC = FORBIDDEN. Add a wrong edge to it.
    cur_g = hcg_to_cur_nx(ref)
    cur_g.add_edge(
        "cur_comp:U1",
        "cur_net:GND",
        pin="8",
        pin_role="8",
        comp_type="IC",
    )
    cur = build_hetero_circuit_graph(
        cur_g, side="cur", subtype_by_source_id={"U1": "UA741"}
    )
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    assert result.stats.by_source[LabelSource.FORBIDDEN_VIOLATED.value] >= 1
    fv = [
        s for s in result.samples
        if s.label_source == LabelSource.FORBIDDEN_VIOLATED.value
    ]
    assert fv[0].label == 0


def test_forbidden_negative_count_matches_kwarg() -> None:
    """Each floating FORBIDDEN pin should get exactly N synthetic negatives,
    where N = forbidden_negative_samples (default 4)."""

    ref = _build_ref(FIXTURE_OPAMP)
    cur = _build_cur(ref, subtype_by_source_id=_UA741_SUBTYPES)
    align = identity_alignment(ref, cur)
    # UA741 has 1 FORBIDDEN pin (pin 8), 4 nets total.
    result = build_seal_samples(
        ref, cur, align, forbidden_negative_samples=3
    )
    # 3 FORBIDDEN_NEGATIVE per FORBIDDEN pin × 1 pin = 3
    assert result.stats.by_source[LabelSource.FORBIDDEN_NEGATIVE.value] == 3


def test_default_forbidden_negative_samples_is_4_capped_by_nets() -> None:
    ref = _build_ref(FIXTURE_OPAMP)
    cur = _build_cur(ref, subtype_by_source_id=_UA741_SUBTYPES)
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    # UA741 buffer has 4 nets; FORBIDDEN pin 8 has 0 already-paired nets.
    # min(4, 4) = 4 → 4 FORBIDDEN_NEGATIVE
    assert result.stats.by_source[LabelSource.FORBIDDEN_NEGATIVE.value] == 4


def test_negative_random_targets_required_only_and_avoids_correct() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align, negatives_per_positive=2.0)
    rand_negs = [
        s for s in result.samples
        if s.label_source == LabelSource.NEGATIVE_RANDOM.value
    ]
    for s in rand_negs:
        port_id = s.candidate_edge[0]
        port = cur.ports[port_id]
        assert port.connection_policy == "required", s
    # No random negative should duplicate a WRONG_OBSERVED (port, net)
    wrong_obs_pairs = {
        s.candidate_edge for s in result.samples
        if s.label_source == LabelSource.WRONG_OBSERVED.value
    }
    rand_neg_pairs = {s.candidate_edge for s in rand_negs}
    assert not (wrong_obs_pairs & rand_neg_pairs)


def test_negative_hard_slot_exists_but_unused_in_p08() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align)
    assert result.stats.by_source[LabelSource.NEGATIVE_HARD.value] == 0
    # The enum value exists
    assert LabelSource.NEGATIVE_HARD.value == "negative_hard"


# ---------------------------------------------------------------------------
# SealSampleGroup — floating + wrong_redirect (6)
# ---------------------------------------------------------------------------


def test_missing_edge_group_for_floating_required_port() -> None:
    ref = _build_ref(FIXTURE_RC)
    # Drop one edge → R1.pin1 becomes floating
    cur = _build_cur(ref, perturbations=[("drop_edge", "pin1", "VIN")])
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    floating_groups = [
        g for g in result.groups if g.query_origin == "floating"
    ]
    assert floating_groups, "floating REQUIRED port must spawn a group"
    g = floating_groups[0]
    assert g.task_type == TaskType.MISSING_EDGE.value
    assert g.query_port_id == "cur_port:R1.pin1"
    # correct_index points to a label=1 sample
    assert g.correct_index is not None
    correct = result.samples[g.sample_indices[g.correct_index]]
    assert correct.label == 1


def test_missing_edge_group_for_wrong_redirect() -> None:
    ref = _build_ref(FIXTURE_RC)
    # R1.pin1 connected to GND instead of VIN
    cur = _build_cur(
        ref,
        perturbations=[
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ],
    )
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    wr_groups = [
        g for g in result.groups if g.query_origin == "wrong_redirect"
    ]
    assert wr_groups, "wrong_redirect group must spawn for misconnected REQUIRED port"
    g = wr_groups[0]
    assert g.query_port_id == "cur_port:R1.pin1"
    # Group must include both current-wrong net (GND) and the correct net (VIN)
    candidate_nets = {
        result.samples[idx].candidate_edge[1] for idx in g.sample_indices
    }
    assert "cur_net:VIN" in candidate_nets
    assert "cur_net:GND" in candidate_nets


def test_group_sample_indices_point_to_missing_edge_samples() -> None:
    ref = _build_ref(FIXTURE_RC)
    cur = _build_cur(ref, perturbations=[("drop_edge", "pin1", "VIN")])
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    for g in result.groups:
        for idx in g.sample_indices:
            s = result.samples[idx]
            assert s.task_type == g.task_type
            assert s.group_id == g.group_id


def test_group_correct_index_label_invariant() -> None:
    ref = _build_ref(FIXTURE_RC)
    cur = _build_cur(ref, perturbations=[("drop_edge", "pin1", "VIN")])
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    for g in result.groups:
        if g.correct_index is None:
            continue
        assert (
            result.samples[g.sample_indices[g.correct_index]].label == 1
        )


def test_missing_edge_group_size_caps_candidates() -> None:
    ref = _build_ref(FIXTURE_RC)
    cur = _build_cur(ref, perturbations=[("drop_edge", "pin1", "VIN")])
    align = identity_alignment(ref, cur)
    result = build_seal_samples(
        ref, cur, align, missing_edge_group_size=3
    )
    for g in result.groups:
        assert len(g.sample_indices) <= 3


def test_perfect_pair_yields_no_missing_groups() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align)
    assert result.groups == ()


# ---------------------------------------------------------------------------
# LabelStats accuracy (5)
# ---------------------------------------------------------------------------


def test_stats_by_source_matches_actual() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align)
    for src_value, count in result.stats.by_source.items():
        actual = sum(1 for s in result.samples if s.label_source == src_value)
        assert actual == count, src_value


def test_stats_by_task_type_matches_actual() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align)
    for task_value, count in result.stats.by_task_type.items():
        actual = sum(1 for s in result.samples if s.task_type == task_value)
        assert actual == count


def test_pos_neg_ratio_formula() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align)
    assert result.stats.pos_neg_ratio == pytest.approx(
        result.stats.n_positives / max(1, result.stats.n_negatives)
    )


def test_stats_missing_component_counts() -> None:
    ref = _build_ref(FIXTURE_RC)
    cur = _build_cur(ref, perturbations=[("remove_component", "C1")])
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    # C1 had 2 edges in ref (pin1↔VC, pin2↔GND). Step 1 iterates ref.edges,
    # and Step 2 too — so missing component is counted multiple times.
    assert result.stats.n_skipped_missing_component >= 1


def test_stats_unique_ports_and_nets_covered() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align)
    actual_ports = {s.candidate_edge[0] for s in result.samples}
    actual_nets = {s.candidate_edge[1] for s in result.samples}
    assert result.stats.n_unique_ports_covered == len(actual_ports)
    assert result.stats.n_unique_nets_covered == len(actual_nets)


# ---------------------------------------------------------------------------
# Behavior / edge cases (3)
# ---------------------------------------------------------------------------


def test_same_seed_reproducible() -> None:
    ref = _build_ref(FIXTURE_RC)
    cur = _build_cur(
        ref,
        perturbations=[
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ],
    )
    align = identity_alignment(ref, cur)
    r1 = build_seal_samples(ref, cur, align, seed=42, negatives_per_positive=3.0)
    r2 = build_seal_samples(ref, cur, align, seed=42, negatives_per_positive=3.0)
    assert len(r1.samples) == len(r2.samples)
    assert r1.stats == r2.stats
    for a, b in zip(r1.samples, r2.samples):
        assert a.candidate_edge == b.candidate_edge
        assert a.label == b.label
        assert a.label_source == b.label_source


def test_optional_pin_default_excluded() -> None:
    ref = _build_ref(FIXTURE_OPAMP)
    cur = _build_cur(ref, subtype_by_source_id=_UA741_SUBTYPES)
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    # OPTIONAL pins on UA741 (1 & 5) should not appear in samples
    sample_ports = {s.candidate_edge[0] for s in result.samples}
    assert "cur_port:U1.1" not in sample_ports
    assert "cur_port:U1.5" not in sample_ports


def test_negatives_per_positive_ratio_monotonic() -> None:
    """Higher ``negatives_per_positive`` produces ≥ same WRONG_EDGE negatives,
    bounded by candidate pool exhaustion."""

    ref, cur, align = _perfect_pair()
    r1 = build_seal_samples(ref, cur, align, negatives_per_positive=1.0)
    r2 = build_seal_samples(ref, cur, align, negatives_per_positive=3.0)
    n1 = sum(
        1
        for s in r1.samples
        if s.task_type == TaskType.WRONG_EDGE.value and s.label == 0
    )
    n2 = sum(
        1
        for s in r2.samples
        if s.task_type == TaskType.WRONG_EDGE.value and s.label == 0
    )
    assert n2 >= n1


def test_negatives_capped_by_candidate_pool() -> None:
    """When the random-negative pool is small, builder returns whatever the
    pool allows rather than throwing."""

    ref, cur, align = _perfect_pair()
    # RC fixture: 4 REQUIRED ports × 3 nets = 12 (port, net) pairs.
    # 8 of those are sym-aware correct (positives) → only 4 negatives possible.
    # Even at negatives_per_positive=100 we should not get more than 4.
    result = build_seal_samples(
        ref, cur, align, negatives_per_positive=100.0
    )
    we_neg = sum(
        1
        for s in result.samples
        if s.task_type == TaskType.WRONG_EDGE.value and s.label == 0
    )
    assert we_neg <= 4


# ---------------------------------------------------------------------------
# WRONG_EDGE vs MISSING_EDGE task type separation
# ---------------------------------------------------------------------------


def test_wrong_observed_is_wrong_edge_task() -> None:
    ref = _build_ref(FIXTURE_RC)
    cur = _build_cur(
        ref,
        perturbations=[
            ("drop_edge", "pin1", "VIN"),
            ("add_edge", "R1", "pin1", "GND", "pin1"),
        ],
    )
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    for s in result.samples:
        if s.label_source == LabelSource.WRONG_OBSERVED.value:
            assert s.task_type == TaskType.WRONG_EDGE.value


# ---------------------------------------------------------------------------
# Hard-negative kwarg refuses (NotImplementedError)
# ---------------------------------------------------------------------------


def test_hard_negative_mining_kwarg_raises() -> None:
    ref, cur, align = _perfect_pair()
    with pytest.raises(NotImplementedError):
        build_seal_samples(ref, cur, align, enable_hard_negative_mining=True)


# ---------------------------------------------------------------------------
# Renamed component / net via alignment_from_dicts
# ---------------------------------------------------------------------------


def test_alignment_from_dicts_with_rename_yields_correct_labels() -> None:
    ref = _build_ref(FIXTURE_RC)
    cur = _build_cur(
        ref,
        perturbations=[
            ("rename_component", "R1", "U_R_3"),
            ("rename_net", "VIN", "n_07"),
        ],
    )
    align = alignment_from_dicts(
        ref,
        cur,
        component_map={"R1": "U_R_3", "C1": "C1"},
        net_map={"VIN": "n_07", "VC": "VC", "GND": "GND"},
    )
    result = build_seal_samples(ref, cur, align)
    # Should have REF_PRESENT for the renamed port→net mapping
    found = any(
        s.candidate_edge == ("cur_port:U_R_3.pin1", "cur_net:n_07")
        and s.label == 1
        for s in result.samples
    )
    assert found


# ---------------------------------------------------------------------------
# Performance — < 80 ms on UA741 buffer
# ---------------------------------------------------------------------------


def test_performance_ua741_under_80ms() -> None:
    import time

    ref = _build_ref(FIXTURE_OPAMP)
    cur = _build_cur(ref, subtype_by_source_id=_UA741_SUBTYPES)
    align = identity_alignment(ref, cur)
    # warm
    for _ in range(3):
        build_seal_samples(ref, cur, align)
    t0 = time.perf_counter()
    result = build_seal_samples(ref, cur, align)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert isinstance(result, LabelBuildResult)
    assert elapsed_ms < 80.0, f"build took {elapsed_ms:.2f} ms (budget 80)"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


def test_seal_sample_is_frozen() -> None:
    ref, cur, align = _perfect_pair()
    result = build_seal_samples(ref, cur, align)
    if not result.samples:
        pytest.skip("no samples")
    with pytest.raises((AttributeError, TypeError)):
        result.samples[0].label = 99  # type: ignore[misc]


def test_seal_sample_group_is_frozen() -> None:
    ref = _build_ref(FIXTURE_RC)
    cur = _build_cur(ref, perturbations=[("drop_edge", "pin1", "VIN")])
    align = identity_alignment(ref, cur)
    result = build_seal_samples(ref, cur, align)
    if not result.groups:
        pytest.skip("no groups")
    with pytest.raises((AttributeError, TypeError)):
        result.groups[0].correct_index = 99  # type: ignore[misc]
