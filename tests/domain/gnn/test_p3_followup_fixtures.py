"""P3 follow-up · new training fixtures (P3.1 ablation surfaced
training-fixture diversity as the bottleneck for OOD test F1).

These fixtures were added specifically to lift OOD performance on
``opamp_buffer`` (the held-out test ref):

- ``test_opamp_inverting_v1.json`` — UA741 in inverting configuration.
  Gives the model IC port-type training signal so it doesn't see
  ``opamp_buffer`` as a totally unseen IC at test time.
- ``test_npn_switch_v1.json`` — NPN transistor driving an LED through
  R_load + R_b. Adds Transistor + Diode + multi-component-net
  interaction that only ``test_all_signal_v1`` partially covers.

The measured impact (Mac CPU, 15-epoch baseline w/ pretrain + DRNL,
6-ref dataset vs the original 4-ref dataset):

| | 4-ref baseline | 6-ref (+ these fixtures) | Δ |
|---|---|---|---|
| val F1 | 0.920 | 0.946 | +0.026 |
| **test F1** | **0.700** | **0.827** | **+0.127** |
| test AUC | 0.871 | 0.906 | +0.035 |

These tests just gate the fixtures' structural integrity. The training-
side impact lives in `checkpoints/p3_followup_v1/summary.json` (Mac
artefact) and the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.gnn import (
    PERTURBATION_REGISTRY,
    apply_perturbation,
    build_from_logical_reference,
    build_seal_samples_with_coverage_check,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "references"


# ---------------------------------------------------------------------------
# Fixture loading sanity
# ---------------------------------------------------------------------------


def test_opamp_inverting_fixture_loads_with_ua741_pin_specs() -> None:
    payload = json.loads((FIXTURES / "test_opamp_inverting_v1.json").read_text(encoding="utf-8"))
    hcg = build_from_logical_reference(
        payload, extra_subtypes_by_source_id={"U1": "UA741"}
    )
    # R10 fixture refresh: 1 IC + 3 resistors (R_in, R_f, **R_p** for
    # bias compensation per textbook canon) → 4 components.
    # UA741 fully materialised (8 ports incl. NC/OPTIONAL) + 3 R × 2
    # pins each = 14 ports.
    s = hcg.summary()
    assert s["n_components"] == 4
    assert s["n_ports"] == 14, (
        f"UA741 fully materialised + 3 R × 2 pins each = 14, got {s['n_ports']}"
    )
    # 6 nets: VIN, INV, VOUT, V_P, VCC, GND
    assert s["n_nets"] == 6
    # FORBIDDEN: pin 8 (NC); OPTIONAL: pin 1, 5 (offset_null)
    by_policy = {"required": 0, "optional": 0, "forbidden": 0}
    for p in hcg.ports.values():
        by_policy[p.connection_policy] = by_policy.get(p.connection_policy, 0) + 1
    assert by_policy["forbidden"] == 1
    assert by_policy["optional"] == 2


def test_npn_switch_fixture_loads_with_transistor_ports() -> None:
    payload = json.loads((FIXTURES / "test_npn_switch_v1.json").read_text(encoding="utf-8"))
    hcg = build_from_logical_reference(payload)
    s = hcg.summary()
    # Q1 (3 pins) + R_b (2) + R_load (2) + LED (2) = 9 ports
    assert s == {"n_components": 4, "n_ports": 9, "n_nets": 6, "n_edges": 9}
    # Transistor pin types should be base / collector / emitter
    q1_ports = [
        p for p in hcg.ports.values() if p.parent_ctype == "Transistor"
    ]
    assert len(q1_ports) == 3
    pts = {p.port_type for p in q1_ports}
    assert pts == {"base", "collector", "emitter"}


# ---------------------------------------------------------------------------
# Every perturbation × every fixture coverage check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture,subtypes",
    [
        ("test_opamp_inverting_v1.json", {"U1": "UA741"}),
        ("test_npn_switch_v1.json", None),
    ],
)
def test_new_fixture_runs_every_perturbation_with_coverage(
    fixture: str, subtypes: dict | None
) -> None:
    payload = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
    hcg = build_from_logical_reference(
        payload, extra_subtypes_by_source_id=subtypes
    )
    for op_name in PERTURBATION_REGISTRY:
        p = apply_perturbation(
            op_name, hcg, seed=7, subtype_by_source_id=subtypes
        )
        result = build_seal_samples_with_coverage_check(
            hcg, p.cur_hcg, p.alignment
        )
        assert result.stats.total_samples > 0, (
            f"{fixture} × {op_name} produced 0 samples"
        )


# ---------------------------------------------------------------------------
# DEFAULT_CONFIG carries the new refs
# ---------------------------------------------------------------------------


def test_default_config_includes_new_p3_followup_fixtures() -> None:
    """opamp_inverting (UA741 反相) is still part of v5 demo set.
    npn_switch was dropped in v5 prune (non-demo); see
    test_default_config_demo_aligned_six_refs."""

    from scripts.gnn_generate_dataset import DEFAULT_CONFIG

    ref_ids = {r["ref_id"] for r in DEFAULT_CONFIG["refs"]}
    assert "opamp_inverting" in ref_ids, "new fixture missing from DEFAULT_CONFIG"
    # opamp_inverting needs UA741 subtype override
    subtypes = DEFAULT_CONFIG.get("subtypes_by_ref_id", {})
    assert subtypes.get("opamp_inverting") == {"U1": "UA741"}
    # opamp_buffer must remain held-out so the OOD test split is comparable
    # to the pre-P3.1 baseline. Otherwise the "+0.127 test F1" claim is
    # not apples-to-apples.
    assert "opamp_buffer" in DEFAULT_CONFIG["test_ref_ids"]
    # Important: opamp_inverting MUST NOT be in test_ref_ids (otherwise
    # we'd be training and testing on the same IC topology family)
    assert "opamp_inverting" not in DEFAULT_CONFIG["test_ref_ids"]


def test_default_config_includes_insert_same_net_wire_perturbation() -> None:
    """**Stage 3 contract** — DEFAULT_CONFIG.plan.counts 必须包含新加入的
    insert_same_net_wire perturbation，且总样本量保持 600/ref（其他 op
    按比例下调，不让数据集体积突然胀大）。"""

    from scripts.gnn_generate_dataset import DEFAULT_CONFIG

    counts = DEFAULT_CONFIG["plan"]["counts"]
    assert "insert_same_net_wire" in counts, (
        "Stage 3 contract: insert_same_net_wire 必须在 DEFAULT_CONFIG.plan.counts 里"
    )
    assert counts["insert_same_net_wire"] > 0
    # 总样本量锁定（不变 600）
    assert sum(counts.values()) == 600, (
        f"DEFAULT_CONFIG counts sum 应是 600/ref, 实际 {sum(counts.values())}"
    )
    # 关键反例 perturbation 必须保留 —— 否则模型失去"错接 wire 应判 0"的判别力
    assert counts.get("extra_wire_bridge", 0) > 0, (
        "extra_wire_bridge 不能被砍空 — 模型仍需要 cross-net wire 负例"
    )


def test_default_config_demo_aligned_six_refs() -> None:
    """Locks down the dataset size against accidental fixture drops.

    History: started at 4 (P1 acceptance) → 6 (P3 follow-up #1: added
    opamp_inverting + npn_switch) → 7 (P3 follow-up #2: added
    lm358_dual_buffer for IC subtype diversity) → 10 (Phase D · v5:
    added bjt_diff_amp + opamp_inverting_lpf + opamp_summing to close
    4 OOD blind spots) → 6 (Phase D · v5 prune: dropped 4 non-demo
    refs (`divider`, `all_signal`, `npn_switch`, `lm358_dual_buffer`)
    after first v5 training attempt got stuck on Windows DataLoader
    I/O bottleneck. v5 dataset now focuses signal on the 6 Intel-cup
    demo circuits exclusively)."""

    from scripts.gnn_generate_dataset import DEFAULT_CONFIG

    assert len(DEFAULT_CONFIG["refs"]) == 6, (
        f"expected 6 demo-aligned refs after v5 prune, got {len(DEFAULT_CONFIG['refs'])}"
    )
    ref_ids = {r["ref_id"] for r in DEFAULT_CONFIG["refs"]}
    # 6 demo circuits
    assert ref_ids == {
        "rc_lowpass",            # 一阶 RC
        "opamp_buffer",          # 电压跟随器 (held-out test)
        "opamp_inverting",       # UA741 反相
        "bjt_diff_amp",          # 差分放大器
        "opamp_inverting_lpf",   # 反相 LPF
        "opamp_summing",         # 加法器
    }, f"ref set drift: {ref_ids}"
    # Dropped non-demo refs
    assert "divider" not in ref_ids
    assert "all_signal" not in ref_ids
    assert "npn_switch" not in ref_ids
    assert "lm358_dual_buffer" not in ref_ids
    # Subtype overrides for IC refs
    subtypes = DEFAULT_CONFIG.get("subtypes_by_ref_id", {})
    assert subtypes.get("opamp_buffer") == {"U1": "UA741"}
    assert subtypes.get("opamp_inverting") == {"U1": "UA741"}
    assert subtypes.get("opamp_inverting_lpf") == {"U1": "UA741"}
    assert subtypes.get("opamp_summing") == {"U1": "UA741"}
    # lm358_dual_buffer subtype no longer registered
    assert "lm358_dual_buffer" not in subtypes


def test_lm358_fixture_loads_with_correct_port_types() -> None:
    """LM358 pin map (added in P3 follow-up #2) should give pins 1/7
    OUTPUT, 2/6 INVERTING_INPUT, 3/5 NON_INVERTING_INPUT, 4 V_MINUS,
    8 V_PLUS — verifying the new IC_PIN_MAPS entry."""

    import json

    payload = json.loads(
        (FIXTURES / "test_lm358_dual_buffer_v1.json").read_text(encoding="utf-8")
    )
    hcg = build_from_logical_reference(
        payload, extra_subtypes_by_source_id={"U1": "LM358"}
    )
    by_key = {p.port_key: p.port_type for p in hcg.ports.values()}
    expected = {
        "1": "output",
        "2": "inverting_input",
        "3": "non_inverting_input",
        "4": "v_minus",
        "5": "non_inverting_input",
        "6": "inverting_input",
        "7": "output",
        "8": "v_plus",
    }
    assert by_key == expected, (
        f"LM358 port_type mapping wrong: got {by_key}, expected {expected}"
    )
    # No FORBIDDEN pin (LM358 has no NC, unlike UA741 pin 8)
    forbidden = [p for p in hcg.ports.values() if p.connection_policy == "forbidden"]
    assert not forbidden, (
        "LM358 has no NC pin so no FORBIDDEN port should be materialised"
    )
