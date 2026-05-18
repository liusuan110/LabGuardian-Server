"""GNN 模块 · P5 离线评估器（plan §九 P5 / §八 metric table）。

把训练好的 :class:`GNNAdvisor` 与现有规则比较器 :func:`compare_logical_graphs`
端到端地跑一遍（按 plan §五 的 dataset 划分），输出 plan §八 表中的硬指标：

* **rule_false_pass_rate** —— 规则 + GNN 联判误判错电路为对的比例（红线 ≤ 0.5%）
* **rule_false_fail_rate** —— 规则 + GNN 联判误判对电路为错的比例（≤ 5%）
* **seal_edge_*** —— GNN 主头在 cur 实际观测到的边上的 AUC / F1 / accuracy
* **rule_runtime_ms_*** / **gnn_runtime_ms_*** —— 单次 mean / p95 延迟
* **n_disagreements** —— 规则 pass / GNN 怀疑 或 规则 fail / GNN 高自信通过

注意：GNN 永远 **不** 改 ``logic_correct``（plan §一）。本评估器把 combined_*
报告视为 "rule final + GNN advisory" 联合，等价于 rule-only 的 pass/fail，
但**额外**统计 GNN 与 rule 的分歧次数，给出 P4.1 conflict-arbitration
决策依据。

CLI 入口：``scripts/gnn_eval.py``（见同名 module）。
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from app.domain.compare.orchestrator import (
    _GNN_WRONG_EDGE_P_FLOOR,
    compare_logical_graphs,
)
from app.domain.gnn.hetero_circuit import HeteroCircuitGraph
from app.domain.gnn.perturbation import hcg_to_nx
from app.domain.gnn.port_graph import build_from_logical_reference
from app.domain.gnn.pyg_dataset import reconstruct_cur_hcg
from app.domain.logical_reference import logical_reference_to_graph

if TYPE_CHECKING:  # pragma: no cover
    from app.domain.gnn.inference import GNNAdvice, GNNAdvisor

log = logging.getLogger("gnn.evaluator")


# ---------------------------------------------------------------------------
# Default registry: ref_id → fixture payload + subtype overrides
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "references"

DEFAULT_REF_PAYLOAD_PATHS: dict[str, Path] = {
    "rc_lowpass": _FIXTURES / "test_rc_v1.json",
    "divider": _FIXTURES / "test_voltage_divider_v1.json",
    "all_signal": _FIXTURES / "test_all_signal_v1.json",
    "opamp_buffer": _FIXTURES / "test_opamp_buffer_v1.json",
    "opamp_inverting": _FIXTURES / "test_opamp_inverting_v1.json",
    "npn_switch": _FIXTURES / "test_npn_switch_v1.json",
    "lm358_dual_buffer": _FIXTURES / "test_lm358_dual_buffer_v1.json",
}

DEFAULT_SUBTYPES_BY_REF: dict[str, dict[str, str]] = {
    "opamp_buffer": {"U1": "UA741"},
    "opamp_inverting": {"U1": "UA741"},
    "lm358_dual_buffer": {"U1": "LM358"},
}


# ---------------------------------------------------------------------------
# Per-sample + aggregate results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleEvaluation:
    """One (ref, cur) pair's evaluation result.

    All fields are JSON-serialisable. SEAL raw scores are captured here
    in the **single-pass** evaluator (since v2 quality refactor) so we
    never re-invoke the advisor.
    """

    sample_id: str
    ref_id: str
    expected_positive: bool
    perturbation_chain: tuple[str, ...]
    # Rule path
    rule_logic_correct: bool
    rule_similarity: float
    rule_match_type: str
    rule_runtime_ms: float
    # GNN path (None if advisor unavailable / failed)
    gnn_inference_ms: float | None
    n_edges_scored: int
    gnn_mean_p_correct: float | None
    gnn_min_p_correct: float | None
    # Plan §七 sense: GNN says "looks ok" if no edge dropped below threshold
    gnn_predicted_positive: bool | None
    # Combined: rule wins (plan §一); kept here for stat aggregation
    combined_logic_correct: bool
    # SEAL head per-edge metric on the observed-edge subset
    n_observed_edge_labels: int
    n_correct_observed_edge_preds: int
    # Raw (score, label) pairs for observed-edge SEAL aggregate AUC/F1.
    # Same length; len == n_observed_edge_labels.
    observed_edge_scores: tuple[float, ...] = ()
    observed_edge_labels: tuple[int, ...] = ()
    # P4.1 R2 — how many edges fell below the
    # _GNN_WRONG_EDGE_P_FLOOR (0.3) threshold. Source of
    # rule↔GNN disagreement statistics.
    n_suspicious_edges: int = 0


@dataclass
class EvaluationReport:
    """Aggregate metrics over a whole split. Mirrors plan §八 columns."""

    n_samples: int
    by_ref_id: dict[str, int]
    by_perturbation: dict[str, int]
    # plan §八 main metrics
    seal_edge_n: int
    seal_edge_auc: float | None
    seal_edge_f1: float | None
    seal_edge_precision: float | None
    seal_edge_recall: float | None
    seal_edge_accuracy: float | None
    rule_false_pass_rate: float
    rule_false_fail_rate: float
    rule_accuracy: float
    combined_false_pass_rate: float
    combined_false_fail_rate: float
    combined_accuracy: float
    rule_runtime_ms_mean: float
    rule_runtime_ms_p95: float
    gnn_runtime_ms_mean: float | None
    gnn_runtime_ms_p95: float | None
    n_disagreements: int
    n_disagreements_rule_pass_gnn_fail: int
    n_disagreements_rule_fail_gnn_pass: int
    # P4.1 R2 — samples that would emit a WARN_GNN_DISAGREES_WITH_RULE
    # in the live orchestrator path (rule_pass + suspicious_edges > 0).
    n_r2_warnings: int
    advisor_unavailable: bool
    advisor_version: str | None
    # Per-perturbation false-pass breakdown (key insight for risk register)
    by_perturbation_false_pass: dict[str, float] = field(default_factory=dict)
    by_perturbation_false_fail: dict[str, float] = field(default_factory=dict)
    by_perturbation_r2_warning_rate: dict[str, float] = field(default_factory=dict)
    samples: tuple[SampleEvaluation, ...] = ()

    # -- IO helpers --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict mangles nested dataclass tuples — re-serialise samples explicitly
        d["samples"] = [asdict(s) for s in self.samples]
        return d

    def to_markdown(self) -> str:
        plan_targets = {
            "rule_false_pass_rate": 0.005,
            "rule_false_fail_rate": 0.05,
            "seal_edge_f1": 0.88,
            "seal_edge_auc": 0.92,
        }

        def _check(metric: str, value: float | None, lower_is_better: bool) -> str:
            if value is None:
                return "—"
            target = plan_targets[metric]
            ok = (value <= target) if lower_is_better else (value >= target)
            mark = "✅" if ok else "⚠️"
            return f"{value:.4f} {mark} (target {'≤' if lower_is_better else '≥'} {target})"

        lines: list[str] = []
        lines.append("# P5 · LabGuardian-Server GNN evaluation")
        lines.append("")
        lines.append(f"- **n_samples**: {self.n_samples}")
        lines.append(f"- **advisor**: `{self.advisor_version}`")
        if self.advisor_unavailable:
            lines.append("- ⚠️ GNN advisor unavailable on this run — only rule metrics reported")
        lines.append("- **by_ref_id**: " + ", ".join(
            f"`{k}`×{v}" for k, v in sorted(self.by_ref_id.items())
        ))
        lines.append("")

        lines.append("## Plan §八 hard targets (rule + GNN联判)")
        lines.append("")
        lines.append("| metric | value | gate |")
        lines.append("|---|---|---|")
        lines.append(f"| **false_pass_rate** (rule) | {self.rule_false_pass_rate:.4f} | "
                     f"{_check('rule_false_pass_rate', self.rule_false_pass_rate, True)} |")
        lines.append(f"| false_fail_rate (rule) | {self.rule_false_fail_rate:.4f} | "
                     f"{_check('rule_false_fail_rate', self.rule_false_fail_rate, True)} |")
        lines.append(f"| accuracy (rule) | {self.rule_accuracy:.4f} | — |")
        lines.append(
            f"| **false_pass_rate** (rule+GNN combined) | {self.combined_false_pass_rate:.4f} | "
            f"{_check('rule_false_pass_rate', self.combined_false_pass_rate, True)} |"
        )
        lines.append(
            f"| false_fail_rate (rule+GNN combined) | {self.combined_false_fail_rate:.4f} | "
            f"{_check('rule_false_fail_rate', self.combined_false_fail_rate, True)} |"
        )
        lines.append("")

        lines.append("## SEAL head on observed cur edges")
        lines.append("")
        if self.seal_edge_n == 0:
            lines.append("_no observed edges scored — advisor disabled or empty cur graphs._")
        else:
            lines.append("| metric | value | gate |")
            lines.append("|---|---|---|")
            lines.append(f"| n observed edges | {self.seal_edge_n} | — |")
            lines.append(f"| **AUC** | {self.seal_edge_auc:.4f} | "
                         f"{_check('seal_edge_auc', self.seal_edge_auc, False)} |")
            lines.append(f"| **F1** | {self.seal_edge_f1:.4f} | "
                         f"{_check('seal_edge_f1', self.seal_edge_f1, False)} |")
            lines.append(f"| precision | {self.seal_edge_precision:.4f} | — |")
            lines.append(f"| recall | {self.seal_edge_recall:.4f} | — |")
            lines.append(f"| accuracy | {self.seal_edge_accuracy:.4f} | — |")
        lines.append("")

        lines.append("## Runtime (CPU)")
        lines.append("")
        lines.append("| stage | mean (ms) | p95 (ms) |")
        lines.append("|---|---|---|")
        lines.append(
            f"| rule comparator | {self.rule_runtime_ms_mean:.2f} | "
            f"{self.rule_runtime_ms_p95:.2f} |"
        )
        if self.gnn_runtime_ms_mean is not None:
            lines.append(f"| GNN advise | {self.gnn_runtime_ms_mean:.2f} | "
                         f"{self.gnn_runtime_ms_p95:.2f} |")
        else:
            lines.append("| GNN advise | — | — |")
        lines.append("")

        lines.append("## Rule ↔ GNN disagreements")
        lines.append("")
        lines.append(f"- total disagreements: **{self.n_disagreements}** / {self.n_samples}")
        lines.append(
            "  - rule says pass, GNN flags wrong edge: "
            f"{self.n_disagreements_rule_pass_gnn_fail}"
        )
        lines.append(
            "  - rule says fail, GNN sees no wrong edge: "
            f"{self.n_disagreements_rule_fail_gnn_pass}"
        )
        lines.append("")

        lines.append("## P4.1 R2 — `WARN_GNN_DISAGREES_WITH_RULE` advisory warnings")
        lines.append("")
        lines.append(
            f"- would-emit warnings: **{self.n_r2_warnings}** / {self.n_samples} "
            f"({(self.n_r2_warnings / max(1, self.n_samples)):.2%})"
        )
        lines.append(
            f"- threshold: edge `p_correct < {_GNN_WRONG_EDGE_P_FLOOR}` "
            "while rule says pass"
        )
        lines.append("")

        if self.by_perturbation_false_pass:
            lines.append("## Per-perturbation rule false_pass / false_fail / R2-warn")
            lines.append("")
            lines.append("| perturbation | n samples | false_pass | false_fail | R2-warn rate |")
            lines.append("|---|---|---|---|---|")
            for p in sorted(self.by_perturbation_false_pass):
                lines.append(
                    f"| `{p}` | {self.by_perturbation.get(p, 0)} | "
                    f"{self.by_perturbation_false_pass[p]:.4f} | "
                    f"{self.by_perturbation_false_fail.get(p, 0.0):.4f} | "
                    f"{self.by_perturbation_r2_warning_rate.get(p, 0.0):.4f} |"
                )
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_ref_payload(
    ref_id: str, ref_payload_paths: dict[str, Path]
) -> dict[str, Any]:
    path = ref_payload_paths.get(ref_id)
    if path is None:
        raise KeyError(
            f"no payload path registered for ref_id={ref_id!r}; "
            "pass ref_payload_paths={...} to evaluate_split()"
        )
    payload: dict[str, Any] = json.loads(path.read_text())
    return payload


def _build_ref_artifacts(
    ref_id: str,
    ref_payload_paths: dict[str, Path],
    subtypes_by_ref: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], HeteroCircuitGraph, nx.Graph, dict[str, str]]:
    payload = _load_ref_payload(ref_id, ref_payload_paths)
    ref_hcg = build_from_logical_reference(payload)
    ref_graph = logical_reference_to_graph(payload)
    subtypes = dict(subtypes_by_ref.get(ref_id, {}))
    return payload, ref_hcg, ref_graph, subtypes


def _roc_auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _binary_f1(
    scores: list[float], labels: list[int], threshold: float = 0.5
) -> tuple[float, float, float, float]:
    """Returns (precision, recall, f1, accuracy)."""

    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max(1, len(scores))
    return prec, rec, f1, acc


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


# ---------------------------------------------------------------------------
# Sample iteration
# ---------------------------------------------------------------------------


def _iter_split_label_files(
    label_dir: Path,
    split_ids: list[str] | None,
) -> list[Path]:
    if split_ids is None:
        return sorted(label_dir.rglob("*.json"))
    found: list[Path] = []
    for entry in split_ids:
        # split.json entries look like "opamp_buffer/opamp_buffer__chained_0000"
        path = label_dir / f"{entry}.json"
        if path.is_file():
            found.append(path)
    return found


def _evaluate_sample(
    label_path: Path,
    *,
    advisor: GNNAdvisor | None,
    ref_cache: dict[str, tuple[dict[str, Any], HeteroCircuitGraph, nx.Graph, dict[str, str]]],
    ref_payload_paths: dict[str, Path],
    subtypes_by_ref: dict[str, dict[str, str]],
    seal_threshold: float,
) -> SampleEvaluation | None:
    label = json.loads(label_path.read_text())
    ref_id = label["ref_id"]
    sample_id = label["sample_id"]
    cur_meta = label.get("cur_metadata") or {}
    expected_outcome = cur_meta.get("expected_outcome")
    # PerturbedCur.expected_outcome is one of
    #   "positive"          —— cur is electrically equivalent to ref
    #   "wrong_observed"    —— cur contains a wrong (port, net) edge
    #   "missing_required"  —— cur is missing a REQUIRED ref edge
    # Both "wrong_*" and "missing_*" are negatives from the
    # logic_correct standpoint.
    if expected_outcome not in ("positive", "wrong_observed", "missing_required"):
        log.warning(
            "skip %s: unknown expected_outcome=%r",
            sample_id, expected_outcome,
        )
        return None
    expected_positive = expected_outcome == "positive"
    perturbation_chain = tuple(cur_meta.get("perturbation_chain", ()) or ())

    if ref_id not in ref_cache:
        ref_cache[ref_id] = _build_ref_artifacts(
            ref_id, ref_payload_paths, subtypes_by_ref
        )
    ref_payload, ref_hcg, ref_graph, subtypes = ref_cache[ref_id]

    try:
        cur_hcg = reconstruct_cur_hcg(
            ref_hcg, cur_meta, subtype_by_source_id=subtypes
        )
    except Exception as e:  # noqa: BLE001 — bad sample, skip not crash
        log.warning(
            "skip %s: cur reconstruction failed (%s)",
            sample_id, type(e).__name__,
        )
        return None

    cur_graph = hcg_to_nx(cur_hcg, target_side="cur")

    # ---- Rule path -----------------------------------------------------
    t0 = time.time()
    rule_result = compare_logical_graphs(
        ref_graph,
        cur_graph,
        ref_payload=None,         # disable GNN attachment here — we run advisor below
        cur_netlist_v2=None,
    )
    rule_ms = (time.time() - t0) * 1000

    # ---- GNN path ------------------------------------------------------
    gnn_ms: float | None = None
    n_edges_scored = 0
    gnn_mean: float | None = None
    gnn_min: float | None = None
    gnn_predicted_positive: bool | None = None
    advice: GNNAdvice | None = None
    if advisor is not None:
        t1 = time.time()
        advice = advisor.advise(ref_hcg, cur_hcg, timeout_ms=2000)
        gnn_ms = (time.time() - t1) * 1000
        if advice is not None:
            n_edges_scored = advice.n_edges_scored
            probs = [
                float(e["p_correct"]) for e in advice.edge_predictions
            ]
            if probs:
                gnn_mean = sum(probs) / len(probs)
                gnn_min = min(probs)
                gnn_predicted_positive = all(p >= seal_threshold for p in probs)
            else:
                # empty cur — treat as positive (no edge to flag)
                gnn_predicted_positive = True
                gnn_mean = 1.0
                gnn_min = 1.0

    # ---- SEAL head per-edge metrics (only on observed-edge labels) -----
    # GNN advise only scores cur-observed edges; cross-reference label entries
    # with subgraph.edge_present == True to compute apples-to-apples metrics.
    # Capture raw (score, label) pairs here so the outer loop doesn't need a
    # second advise() pass.
    obs_scores: list[float] = []
    obs_labels: list[int] = []
    n_suspicious_edges = 0
    if advice is not None:
        pred_by_edge = {
            (e["edge"][0], e["edge"][1]): float(e["p_correct"])
            for e in advice.edge_predictions
        }
        # R2 disagreement counter (mirrors orchestrator threshold)
        n_suspicious_edges = sum(
            1 for p in pred_by_edge.values()
            if p < _GNN_WRONG_EDGE_P_FLOOR
        )
        for s in label.get("samples", []):
            if s.get("task_type") != "wrong_edge":
                continue
            sg = s.get("subgraph") or {}
            if not sg.get("edge_present", False):
                continue
            edge = tuple(s["candidate_edge"])
            score = pred_by_edge.get((edge[0], edge[1]))
            if score is None:
                continue
            obs_scores.append(float(score))
            obs_labels.append(int(s["label"]))

    n_correct = sum(
        1 for sc, lb in zip(obs_scores, obs_labels)
        if (1 if sc >= seal_threshold else 0) == lb
    )

    rule_correct = bool(rule_result.get("logic_correct", False))
    return SampleEvaluation(
        sample_id=sample_id,
        ref_id=ref_id,
        expected_positive=expected_positive,
        perturbation_chain=perturbation_chain,
        rule_logic_correct=rule_correct,
        rule_similarity=float(rule_result.get("similarity", 0.0)),
        rule_match_type=str(
            (rule_result.get("details") or {}).get("match_type", "")
        ),
        rule_runtime_ms=rule_ms,
        gnn_inference_ms=gnn_ms,
        n_edges_scored=n_edges_scored,
        gnn_mean_p_correct=gnn_mean,
        gnn_min_p_correct=gnn_min,
        gnn_predicted_positive=gnn_predicted_positive,
        combined_logic_correct=rule_correct,  # plan §一 — GNN never overrides
        n_observed_edge_labels=len(obs_scores),
        n_correct_observed_edge_preds=n_correct,
        observed_edge_scores=tuple(obs_scores),
        observed_edge_labels=tuple(obs_labels),
        n_suspicious_edges=n_suspicious_edges,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_split(
    label_dir: Path,
    *,
    split_ids: list[str] | None = None,
    advisor: GNNAdvisor | None = None,
    ref_payload_paths: dict[str, Path] | None = None,
    subtypes_by_ref: dict[str, dict[str, str]] | None = None,
    seal_threshold: float = 0.5,
    limit: int | None = None,
) -> EvaluationReport:
    """Run rule comparator + GNN advisor over every sample listed in
    ``split_ids`` (or every label file under ``label_dir`` if None).

    Args:
        label_dir: dataset labels root, e.g. ``datasets/circuit_compare/labels``.
        split_ids: optional list of ``"<ref_id>/<sample_id>"`` keys from a
            splits json (e.g. ``splits/test.json``). When None, walks
            ``label_dir`` recursively.
        advisor: a constructed :class:`GNNAdvisor`. Pass ``None`` to skip
            the GNN path entirely (rule-only baseline).
        ref_payload_paths: ref_id → fixture path mapping. Defaults to
            :data:`DEFAULT_REF_PAYLOAD_PATHS`.
        subtypes_by_ref: ref_id → {component_source_id: subtype} mapping.
            Defaults to :data:`DEFAULT_SUBTYPES_BY_REF` (UA741 / LM358).
        seal_threshold: SEAL head decision threshold (default 0.5).
        limit: cap on samples; useful for smoke tests.
    """

    ref_payload_paths = ref_payload_paths or dict(DEFAULT_REF_PAYLOAD_PATHS)
    subtypes_by_ref = subtypes_by_ref or dict(DEFAULT_SUBTYPES_BY_REF)

    label_files = _iter_split_label_files(label_dir, split_ids)
    if limit is not None:
        label_files = label_files[:limit]
    if not label_files:
        raise ValueError(
            f"no label files found under {label_dir} for split_ids={split_ids!r}"
        )

    ref_cache: dict[str, tuple] = {}
    samples: list[SampleEvaluation] = []

    for i, label_path in enumerate(label_files):
        if (i + 1) % 100 == 0:
            log.info("evaluate_split: %d / %d", i + 1, len(label_files))
        ev = _evaluate_sample(
            label_path,
            advisor=advisor,
            ref_cache=ref_cache,
            ref_payload_paths=ref_payload_paths,
            subtypes_by_ref=subtypes_by_ref,
            seal_threshold=seal_threshold,
        )
        if ev is None:
            continue
        samples.append(ev)

    # Aggregate -----------------------------------------------------------
    n_total = len(samples)
    n_neg = sum(1 for s in samples if not s.expected_positive)
    n_pos = n_total - n_neg

    # false_pass = (rule says pass) AND (expected negative)
    rule_fp = sum(
        1 for s in samples
        if s.rule_logic_correct and not s.expected_positive
    )
    rule_ff = sum(
        1 for s in samples
        if (not s.rule_logic_correct) and s.expected_positive
    )
    rule_correct = sum(
        1 for s in samples
        if s.rule_logic_correct == s.expected_positive
    )
    combined_fp = sum(
        1 for s in samples
        if s.combined_logic_correct and not s.expected_positive
    )
    combined_ff = sum(
        1 for s in samples
        if (not s.combined_logic_correct) and s.expected_positive
    )
    combined_correct = sum(
        1 for s in samples
        if s.combined_logic_correct == s.expected_positive
    )

    by_ref: dict[str, int] = defaultdict(int)
    by_pert: dict[str, int] = defaultdict(int)
    pert_fp: dict[str, int] = defaultdict(int)
    pert_ff: dict[str, int] = defaultdict(int)
    pert_neg: dict[str, int] = defaultdict(int)
    pert_pos: dict[str, int] = defaultdict(int)
    pert_r2_warn: dict[str, int] = defaultdict(int)
    n_r2_warnings = 0
    for s in samples:
        by_ref[s.ref_id] += 1
        head = (
            s.perturbation_chain[0] if s.perturbation_chain else "identity"
        )
        op = head.split(":", 1)[0]
        by_pert[op] += 1
        if s.expected_positive:
            pert_pos[op] += 1
            if not s.rule_logic_correct:
                pert_ff[op] += 1
        else:
            pert_neg[op] += 1
            if s.rule_logic_correct:
                pert_fp[op] += 1
        # R2: would the orchestrator emit a disagreement warning here?
        if s.rule_logic_correct and s.n_suspicious_edges > 0:
            n_r2_warnings += 1
            pert_r2_warn[op] += 1

    by_perturbation_false_pass = {
        op: pert_fp[op] / pert_neg[op]
        for op in pert_neg if pert_neg[op] > 0
    }
    by_perturbation_false_fail = {
        op: pert_ff[op] / pert_pos[op]
        for op in pert_pos if pert_pos[op] > 0
    }
    by_perturbation_r2_warning_rate = {
        op: pert_r2_warn[op] / by_pert[op]
        for op in by_pert if by_pert[op] > 0
    }

    rule_rts = [s.rule_runtime_ms for s in samples]
    gnn_rts = [
        s.gnn_inference_ms for s in samples
        if s.gnn_inference_ms is not None
    ]

    # ---- SEAL aggregate (single-pass, scores captured in
    #      _evaluate_sample; no second advisor invocation needed) -------
    all_scores: list[float] = []
    all_labels: list[int] = []
    for s in samples:
        all_scores.extend(s.observed_edge_scores)
        all_labels.extend(s.observed_edge_labels)
    seal_n = len(all_scores)
    seal_auc: float | None = None
    seal_prec: float | None = None
    seal_rec: float | None = None
    seal_f1: float | None = None
    seal_acc: float | None = None
    if seal_n > 0:
        seal_auc = _roc_auc(all_scores, all_labels)
        seal_prec, seal_rec, seal_f1, seal_acc = _binary_f1(
            all_scores, all_labels, threshold=seal_threshold
        )

    # ---- Disagreements --------------------------------------------------
    n_dis = 0
    n_dis_rp_gf = 0  # rule pass, GNN says wrong edge present
    n_dis_rf_gp = 0  # rule fail, GNN sees no wrong edge
    for s in samples:
        if s.gnn_predicted_positive is None:
            continue
        if s.rule_logic_correct != s.gnn_predicted_positive:
            n_dis += 1
            if s.rule_logic_correct and not s.gnn_predicted_positive:
                n_dis_rp_gf += 1
            elif (not s.rule_logic_correct) and s.gnn_predicted_positive:
                n_dis_rf_gp += 1

    advisor_unavailable = advisor is None
    advisor_version = getattr(advisor, "model_version", None) if advisor else None

    return EvaluationReport(
        n_samples=n_total,
        by_ref_id=dict(by_ref),
        by_perturbation=dict(by_pert),
        seal_edge_n=seal_n,
        seal_edge_auc=seal_auc,
        seal_edge_f1=seal_f1,
        seal_edge_precision=seal_prec,
        seal_edge_recall=seal_rec,
        seal_edge_accuracy=seal_acc,
        rule_false_pass_rate=rule_fp / max(1, n_neg),
        rule_false_fail_rate=rule_ff / max(1, n_pos),
        rule_accuracy=rule_correct / max(1, n_total),
        combined_false_pass_rate=combined_fp / max(1, n_neg),
        combined_false_fail_rate=combined_ff / max(1, n_pos),
        combined_accuracy=combined_correct / max(1, n_total),
        rule_runtime_ms_mean=sum(rule_rts) / max(1, len(rule_rts)),
        rule_runtime_ms_p95=_percentile(rule_rts, 95),
        gnn_runtime_ms_mean=(
            sum(gnn_rts) / len(gnn_rts) if gnn_rts else None
        ),
        gnn_runtime_ms_p95=_percentile(gnn_rts, 95) if gnn_rts else None,
        n_disagreements=n_dis,
        n_disagreements_rule_pass_gnn_fail=n_dis_rp_gf,
        n_disagreements_rule_fail_gnn_pass=n_dis_rf_gp,
        n_r2_warnings=n_r2_warnings,
        advisor_unavailable=advisor_unavailable,
        advisor_version=advisor_version,
        by_perturbation_false_pass=by_perturbation_false_pass,
        by_perturbation_false_fail=by_perturbation_false_fail,
        by_perturbation_r2_warning_rate=by_perturbation_r2_warning_rate,
        samples=tuple(samples),
    )


__all__ = [
    "SampleEvaluation",
    "EvaluationReport",
    "evaluate_split",
    "DEFAULT_REF_PAYLOAD_PATHS",
    "DEFAULT_SUBTYPES_BY_REF",
]
