from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


OVERALL_COLORS = ["#8FB7E8", "#9FD7C9", "#F3C982"]
INTENT_COLORS = ["#8FB7E8", "#9FD7C9", "#F3C982", "#CBB8E9"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a P0 manual scoring summary figure for the final report."
    )
    parser.add_argument("--summary-json", type=Path, required=True, help="Manual summary JSON path.")
    parser.add_argument("--output-png", type=Path, required=True, help="Output PNG figure path.")
    return parser.parse_args()


def _pct(value: float | None) -> float:
    return float(value or 0.0)


def render(summary: dict, output_png: Path) -> None:
    overall = summary["overall"]
    by_intent = summary["by_intent"]

    overall_labels = ["Avg Score %", "Pass %", "Full Score %"]
    overall_values = [
        _pct(overall.get("avg_score_pct")),
        _pct(overall.get("pass_rate_pct")),
        _pct(overall.get("full_score_rate_pct")),
    ]

    intent_order = ["concept_tutor", "mixed", "lab_guidance", "diagnostic"]
    intent_labels = ["Concept", "Mixed", "Guidance", "Diagnostic"]
    intent_values = [_pct(by_intent[key]["avg_score_pct"]) for key in intent_order]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#7F8C8D",
            "axes.labelcolor": "#37474F",
            "xtick.color": "#37474F",
            "ytick.color": "#37474F",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=180)
    fig.patch.set_facecolor("white")

    ax0, ax1 = axes

    bars0 = ax0.bar(overall_labels, overall_values, color=OVERALL_COLORS, width=0.58)
    ax0.set_ylim(0, 100)
    ax0.set_ylabel("Percent (%)")
    ax0.set_title("P0 Manual Rubric Summary", fontsize=12, pad=10)
    ax0.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax0.set_axisbelow(True)
    for bar, value in zip(bars0, overall_values):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#37474F",
        )

    bars1 = ax1.bar(intent_labels, intent_values, color=INTENT_COLORS, width=0.58)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Avg Score %")
    ax1.set_title("P0 Average Score By Intent", fontsize=12, pad=10)
    ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax1.set_axisbelow(True)
    for bar, value in zip(bars1, intent_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#37474F",
        )

    fig.suptitle("P0 Fixed Question-Set Evaluation", fontsize=13, y=1.03)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary_json = args.summary_json.resolve()
    output_png = args.output_png.resolve()
    if not summary_json.exists():
        raise SystemExit(f"Summary JSON not found: {summary_json}")

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    render(summary, output_png)
    print(f"wrote {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
