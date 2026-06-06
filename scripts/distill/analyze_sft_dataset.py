from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


INTENT_LABELS = {
    "diagnostic": "diagnostic",
    "lab_guidance": "lab_guidance",
    "concept_tutor": "concept_tutor",
    "mixed": "mixed",
}

SCENE_LABELS = {
    "exp_first_order_rc": "RC",
    "exp_common_emitter_amplifier": "Common-emitter",
    "exp_differential_amplifier": "Differential",
    "exp_ua741_inverting_amplifier": "UA741 inverting",
    "exp_ua741_integrator": "UA741 integrator",
    "exp_ua741_summing_amplifier": "UA741 summing",
}

ERROR_TAG_LABELS = {
    "missing_required_component": "MISSING_COMPONENT",
    "floating_connection": "FLOATING_CONNECTION",
    "wrong_node_connection": "WRONG_NODE_CONNECTION",
    "incomplete_circuit": "INCOMPLETE_CIRCUIT",
    "scope_ground_or_short_risk": "SHORT_RISK",
}


def _extract_embedded_evidence(prompt_input: str) -> dict[str, Any]:
    start = prompt_input.find("{")
    if start < 0:
        raise ValueError("frozen evidence JSON not found in input field")
    return json.loads(prompt_input[start:])


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(part * 100.0 / total, 2)


def _ordered_counter(counter: Counter[str], labels: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = sum(counter.values())
    seen: set[str] = set()

    for key, label in labels.items():
        count = counter.get(key, 0)
        rows.append(
            {
                "key": key,
                "label": label,
                "count": count,
                "pct": _pct(count, total),
            }
        )
        seen.add(key)

    for key, count in counter.most_common():
        if key in seen:
            continue
        rows.append(
            {
                "key": key,
                "label": key,
                "count": count,
                "pct": _pct(count, total),
            }
        )
    return rows


def analyze_dataset(
    *,
    dataset_path: Path,
    candidate_total: int,
    teacher_trainable: int,
    sft_kept_expected: int | None = None,
) -> dict[str, Any]:
    intents: Counter[str] = Counter()
    scenes: Counter[str] = Counter()
    risks: Counter[str] = Counter()
    packs: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    error_codes: Counter[str] = Counter()
    error_tags: Counter[str] = Counter()
    output_lengths: list[int] = []

    total = 0
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        evidence = _extract_embedded_evidence(row["input"])
        intents[evidence.get("intent", "unknown")] += 1
        scenes[evidence.get("scene_id", "unknown")] += 1

        agent_output = evidence.get("agent_output", {})
        context_pack = agent_output.get("context_pack", {})
        risks[context_pack.get("risk_level", "unknown")] += 1
        packs[context_pack.get("pack_id", "unknown")] += 1

        for tool in context_pack.get("allowed_tools", []):
            tools[tool.get("name", "unknown")] += 1
        for code in agent_output.get("evidence_error_codes", []) or []:
            error_codes[code] += 1
        for tag in agent_output.get("evidence_error_tags", []) or []:
            error_tags[tag] += 1

        output_lengths.append(len(row.get("output", "")))

    skipped_after_trainable = max(teacher_trainable - total, 0)
    filtered_before_trainable = max(candidate_total - teacher_trainable, 0)

    result = {
        "dataset_path": str(dataset_path),
        "totals": {
            "candidate_total": candidate_total,
            "teacher_trainable": teacher_trainable,
            "sft_kept": total,
            "sft_skipped_after_trainable": skipped_after_trainable,
            "filtered_before_trainable": filtered_before_trainable,
        },
        "funnel": [
            {
                "stage": "candidate_questions_with_teacher_output",
                "count": candidate_total,
                "pct_of_candidates": 100.0,
            },
            {
                "stage": "high_purity_teacher_rows",
                "count": teacher_trainable,
                "pct_of_candidates": _pct(teacher_trainable, candidate_total),
            },
            {
                "stage": "final_sft_rows",
                "count": total,
                "pct_of_candidates": _pct(total, candidate_total),
            },
        ],
        "intents": _ordered_counter(intents, INTENT_LABELS),
        "scenes": _ordered_counter(scenes, SCENE_LABELS),
        "risk_levels": _ordered_counter(risks, {"danger": "danger", "warning": "warning", "safe": "safe"}),
        "top_pack_ids": [
            {"key": key, "count": count, "pct": _pct(count, total)}
            for key, count in packs.most_common(8)
        ],
        "top_allowed_tools": [
            {"key": key, "count": count, "pct": _pct(count, total)}
            for key, count in tools.most_common(10)
        ],
        "error_codes": [
            {"key": key, "count": count, "pct": _pct(count, total)}
            for key, count in error_codes.most_common(10)
        ],
        "error_tags": _ordered_counter(error_tags, ERROR_TAG_LABELS),
        "output_length": {
            "min_chars": min(output_lengths) if output_lengths else 0,
            "mean_chars": round(mean(output_lengths), 2) if output_lengths else 0.0,
            "max_chars": max(output_lengths) if output_lengths else 0,
        },
    }

    if sft_kept_expected is not None:
        result["totals"]["sft_kept_expected"] = sft_kept_expected
        result["totals"]["matches_expected"] = total == sft_kept_expected

    return result


def _write_markdown(stats: dict[str, Any], output_path: Path) -> None:
    totals = stats["totals"]
    lines: list[str] = []
    lines.append("# SFT Dataset Distribution Summary")
    lines.append("")
    lines.append(f"- Candidate teacher rows: {totals['candidate_total']}")
    lines.append(f"- High-purity teacher rows: {totals['teacher_trainable']}")
    lines.append(f"- Final SFT rows: {totals['sft_kept']}")
    lines.append(f"- Filtered before trainable: {totals['filtered_before_trainable']}")
    lines.append(f"- Skipped after trainable: {totals['sft_skipped_after_trainable']}")
    lines.append("")

    def add_table(title: str, rows: list[dict[str, Any]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Label | Count | Pct |")
        lines.append("|---|---:|---:|")
        for row in rows:
            lines.append(f"| {row['label']} | {row['count']} | {row['pct']}% |")
        lines.append("")

    add_table("Intent distribution", stats["intents"])
    add_table("Scene distribution", stats["scenes"])
    add_table("Risk distribution", stats["risk_levels"])
    add_table("Error tag distribution", stats["error_tags"])

    lines.append("## Output length")
    lines.append("")
    lines.append(
        f"- min / mean / max chars: {stats['output_length']['min_chars']} / "
        f"{stats['output_length']['mean_chars']} / {stats['output_length']['max_chars']}"
    )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(stats: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#444444",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), dpi=180)
    fig.patch.set_facecolor("white")

    def _style_axis(ax: Any, title: str) -> None:
        ax.set_title(title, fontsize=11, pad=8, weight="bold")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")
        ax.tick_params(axis="both", labelsize=9, colors="#333333")

    def _annotate_bars(ax: Any, bars: Any, values: list[int], suffix: str = "") -> None:
        ymax = max(values) if values else 1
        offset = ymax * 0.025
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{value}{suffix}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#333333",
            )

    funnel = stats["funnel"]
    funnel_labels = ["Candidates", "High-purity", "Final SFT"]
    funnel_values = [row["count"] for row in funnel]
    bars = axes[0, 0].bar(
        funnel_labels,
        funnel_values,
        color=["#CFE8F3", "#7DB7D9", "#2F7FB9"],
        edgecolor="none",
        width=0.68,
    )
    _style_axis(axes[0, 0], "SFT Funnel")
    _annotate_bars(axes[0, 0], bars, funnel_values)
    axes[0, 0].set_ylabel("Samples", fontsize=9.5)

    intents = stats["intents"]
    intent_labels = ["Diagnostic", "Lab guidance", "Concept tutor", "Mixed"]
    intent_values = [row["count"] for row in intents]
    bars = axes[0, 1].bar(
        intent_labels,
        intent_values,
        color=["#6DBE73"] * len(intent_labels),
        edgecolor="none",
        width=0.68,
    )
    _style_axis(axes[0, 1], "Intent Distribution")
    _annotate_bars(axes[0, 1], bars, intent_values)
    axes[0, 1].set_ylabel("Samples", fontsize=9.5)

    scenes = stats["scenes"]
    scene_labels = ["RC", "CE", "Diff", "Inv.", "Integrator", "Summing"]
    scene_values = [row["count"] for row in scenes]
    bars = axes[1, 0].bar(
        scene_labels,
        scene_values,
        color=["#F29A4A"] * len(scene_labels),
        edgecolor="none",
        width=0.68,
    )
    _style_axis(axes[1, 0], "Scene Distribution")
    _annotate_bars(axes[1, 0], bars, scene_values)
    axes[1, 0].set_ylabel("Samples", fontsize=9.5)

    tags = stats["error_tags"][:5]
    tag_labels = ["Missing", "Floating", "Wrong node", "Incomplete", "Short risk"]
    tag_values = [row["count"] for row in tags]
    bars = axes[1, 1].bar(
        tag_labels,
        tag_values,
        color=["#9D97C9"] * len(tag_labels),
        edgecolor="none",
        width=0.68,
    )
    _style_axis(axes[1, 1], "Top Error Tags")
    _annotate_bars(axes[1, 1], bars, tag_values)
    axes[1, 1].set_ylabel("Samples", fontsize=9.5)

    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(0)
            label.set_ha("center")

    fig.suptitle("Distillation Dataset Overview", fontsize=13, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.945,
        "Funnel, intent balance, scene coverage and dominant error tags",
        ha="center",
        fontsize=9.5,
        color="#666666",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.10, wspace=0.22, hspace=0.32)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze the final SFT dataset and export report-ready stats.")
    parser.add_argument(
        "--dataset",
        default="datasets/distill/train_sft_alpaca.jsonl",
        type=Path,
        help="Path to the final Alpaca-format SFT dataset.",
    )
    parser.add_argument(
        "--candidate-total",
        default=4990,
        type=int,
        help="Total candidate rows with teacher generation before filtering.",
    )
    parser.add_argument(
        "--teacher-trainable",
        default=3466,
        type=int,
        help="High-purity teacher rows retained after filtering.",
    )
    parser.add_argument(
        "--sft-kept-expected",
        default=3450,
        type=int,
        help="Expected final SFT row count after build_sft_dataset.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-png", type=Path)
    args = parser.parse_args(argv)

    stats = analyze_dataset(
        dataset_path=args.dataset,
        candidate_total=args.candidate_total,
        teacher_trainable=args.teacher_trainable,
        sft_kept_expected=args.sft_kept_expected,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(stats, args.output_md)

    if args.output_png is not None:
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        _write_plot(stats, args.output_png)

    print(f"wrote {args.output_json}")
    if args.output_md is not None:
        print(f"wrote {args.output_md}")
    if args.output_png is not None:
        print(f"wrote {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
