"""Stage 0 · 金标准真实学生样本回归测试 (plan retrain Stage 4 gate).

把一份**已知拓扑正确**的真实学生网表固化为黄金样本，绑定一条强 assertion：
GNN advisor 必须对该样本里**每一条 observed (port, net) edge** 给出
``p_correct > 0.5``。这是 Stage 4 重训（v4 ckpt）是否成功的最终门槛之一。

Fixture: ``tests/fixtures/real_student/inverting_amp_correct_v1.json``
- 拓扑: UA741 反相放大器 (R_in + R_f + R_p bias-compensation, 双电源)
- 来源: 真实学生面包板拍照 → pipeline → netlist_v2 (2026-05-20)
- 特征: 含 3 个物理跳线 W1/W2/W3，每个 wire 两端 pin 在拓扑解析后落到同一 net
       —— 这是当前 v3 ckpt OOD 失效的核心形态 (训练分布里 wire 永远是 negative)

期望 (见 ``inverting_amp_correct_v1.expected.json``):
- 规则路径: ``logic_correct=True`` (wire-collapse 后拓扑与 ref 等价)
- GNN: 17/17 边 ``p_correct > 0.5``、no R2 warning、no suspicious edges

当前状态:
- v3 ckpt: 7/17 边 fail (6 wire + 1 IC1.pin4 旁噪)
- v4 ckpt (Stage 4 重训后): 期望 17/17 通过

测试用 ``@pytest.mark.xfail(strict=False)`` 标记 —— 当前 v3 fail 时不让 CI 红，
但一旦 v4 落地通过，pytest 会显示 XPASSED 提示我们移除标记。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from app.domain.gnn import GNNAdvisor  # noqa: E402
from app.domain.gnn.port_graph import (  # noqa: E402
    build_from_logical_reference,
    build_from_netlist_v2,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "real_student"
REF_DIR = REPO_ROOT / "tests" / "fixtures" / "references"

GOLDEN_NETLIST = GOLDEN_DIR / "inverting_amp_correct_v1.json"
GOLDEN_EXPECTED = GOLDEN_DIR / "inverting_amp_correct_v1.expected.json"


def _load_golden() -> tuple[dict, dict, dict]:
    cur = json.loads(GOLDEN_NETLIST.read_text(encoding="utf-8"))
    expected = json.loads(GOLDEN_EXPECTED.read_text(encoding="utf-8"))
    ref = json.loads(
        (REF_DIR / f"{expected['reference_id']}.json").read_text(encoding="utf-8")
    )
    return ref, cur, expected


def test_golden_fixture_structure_intact() -> None:
    """精简版 fixture 必须保留 GNN/规则路径需要的全部字段。"""

    cur = json.loads(GOLDEN_NETLIST.read_text(encoding="utf-8"))
    assert cur["components"], "components must be present"
    assert cur["nets"], "nets must be present"

    # IC1 必须带 part_subtype=UA741 (否则 IC pin map 不命中, 退回 generic)
    ic1 = next(c for c in cur["components"] if c["component_id"] == "IC1")
    assert ic1.get("part_subtype") == "UA741"

    # 7 个 component, 7 个 net (3 wire + 1 IC + 3 resistor; 7 nets)
    assert len(cur["components"]) == 7
    assert len(cur["nets"]) == 7

    # 3 个 wire 每个两端必须是同一个 net (这是测试中"wire-positive"形态的关键)
    wires = [c for c in cur["components"] if c["component_type"] == "Wire"]
    assert len(wires) == 3
    for w in wires:
        nets = {p["electrical_net_id"] for p in w["pins"]}
        assert len(nets) == 1, (
            f"{w['component_id']} 的两端 net 不同: {nets} — "
            "golden fixture 应只包含 same-net wire (训练 OOD case)"
        )


def test_golden_fixture_rule_path_matches_expected() -> None:
    """断言规则结论与 expected.json 中声明的当前状态一致。

    **注意 ref-mismatch**：当前 ref (``test_opamp_inverting_v1.json``) 是单
    电源版，cur 是双电源真实学生板。所以当前 expected.rule_verdict_expected
    .logic_correct = False。升级 ref 为双电源版后，更新 expected.json 把
    logic_correct 改为 True，此测试会自动随之转向。
    """

    from app.domain.compare.orchestrator import compare_logical_graphs
    from app.domain.logical_reference import (
        current_netlist_v2_to_graph,
        logical_reference_to_graph,
    )

    ref, cur, expected = _load_golden()
    ref_g = logical_reference_to_graph(ref)
    cur_g = current_netlist_v2_to_graph(cur)
    result = compare_logical_graphs(
        ref_g, cur_g, ref_payload=ref, cur_netlist_v2=cur,
    )
    expected_correct = expected["rule_verdict_expected"]["logic_correct"]
    assert result["logic_correct"] is expected_correct, (
        f"rule 判定与 expected.json 不符: logic_correct={result['logic_correct']} "
        f"vs expected {expected_correct}\n"
        f"match_type={result.get('details', {}).get('match_type')}"
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Stage 4 gate · 当前 v3 ckpt 在该样本上 7/17 wire-related 边 fail "
        "(已知 wire OOD)。Stage 4 v4 重训后此测试必须 XPASS，届时移除该标记。"
    ),
)
def test_golden_gnn_all_observed_edges_score_above_threshold() -> None:
    """**Stage 4 重训 gate** — 每条 observed edge 必须 p_correct > 0.5.

    这是判断 v4 ckpt 是否解决 wire-OOD 问题的最终门槛。失败列表会打印出来
    便于训练时定位还有哪些 case 没救活。"""

    ref, cur, expected = _load_golden()
    threshold = expected["stage4_gate"]["min_p_correct_all_edges"]
    expected_total = expected["gnn_verdict_expected"]["n_observed_edges"]

    ref_hcg = build_from_logical_reference(ref)
    cur_hcg = build_from_netlist_v2(cur)
    advisor = GNNAdvisor.get()
    advice = advisor.advise(ref_hcg, cur_hcg)
    assert advice is not None, "advisor returned None"
    assert advice.n_edges_scored == expected_total, (
        f"observed edge 数对不上: actual {advice.n_edges_scored} "
        f"vs expected {expected_total}"
    )

    failures = [
        ep for ep in advice.edge_predictions
        if float(ep["p_correct"]) < threshold
    ]
    if failures:
        msg_lines = [
            f"{len(failures)}/{advice.n_edges_scored} edges below "
            f"p_correct={threshold}:"
        ]
        for ep in sorted(failures, key=lambda e: float(e["p_correct"])):
            msg_lines.append(
                f"  p={float(ep['p_correct']):.4f}  "
                f"{ep['edge'][0]:35} → {ep['edge'][1]}"
            )
        pytest.fail("\n".join(msg_lines))


@pytest.mark.skip(
    reason=(
        "R2 副 gate 当前会 tautologically pass —— 规则因 ref-mismatch 判 fail，"
        "R2 只在 rule_pass + GNN 反对时触发。等 ref 升级双电源后 rule 会 pass，"
        "届时此测试再启用才有意义。"
    ),
)
def test_golden_gnn_no_r2_warning_emitted() -> None:
    """**Stage 4 副 gate** — 端到端 orchestrator 在该样本上不应该 emit
    R2 disagreement warning (因为电路完全正确)."""

    from app.domain.compare.orchestrator import compare_logical_graphs
    from app.domain.logical_reference import (
        current_netlist_v2_to_graph,
        logical_reference_to_graph,
    )

    ref, cur, _ = _load_golden()
    ref_g = logical_reference_to_graph(ref)
    cur_g = current_netlist_v2_to_graph(cur)
    result = compare_logical_graphs(
        ref_g, cur_g, ref_payload=ref, cur_netlist_v2=cur,
    )
    warnings = [
        item for item in (result.get("items") or [])
        if item.get("error_code") == "WARN_GNN_DISAGREES_WITH_RULE"
    ]
    assert not warnings, (
        f"Stage 4 期望无 R2 warning，但实际 emit 了 {len(warnings)} 条:\n"
        + "\n".join(
            f"  {w.get('message', '')[:120]}" for w in warnings
        )
    )
