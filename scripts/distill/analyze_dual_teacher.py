from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(part * 100.0 / total, 2)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_usable(row: dict[str, Any]) -> bool:
    generation = row.get("generation", {})
    audit = generation.get("contract_audit", {})
    return (
        generation.get("ok") is True
        and audit.get("supported_citation_count", 0) > 0
        and not audit.get("downgraded_to_evidence_insufficient", False)
    )


def _has_citations(row: dict[str, Any]) -> bool:
    citations = row.get("teacher_output", {}).get("citations") or []
    return bool(citations)


def _avg_latency_ms(rows: list[dict[str, Any]]) -> float:
    values = [
        row.get("generation", {}).get("latency_ms")
        for row in rows
        if isinstance(row.get("generation", {}).get("latency_ms"), (int, float))
    ]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _teacher_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    ok = sum(1 for row in rows if row.get("generation", {}).get("ok") is True)
    cited = sum(1 for row in rows if _has_citations(row))
    usable = sum(1 for row in rows if _is_usable(row))
    downgraded = sum(
        1
        for row in rows
        if row.get("generation", {})
        .get("contract_audit", {})
        .get("downgraded_to_evidence_insufficient")
        is True
    )
    return {
        "total": total,
        "ok": ok,
        "cited": cited,
        "usable": usable,
        "downgraded": downgraded,
        "ok_rate_pct": _pct(ok, total),
        "citation_rate_pct": _pct(cited, total),
        "usable_rate_pct": _pct(usable, total),
        "downgraded_rate_pct": _pct(downgraded, total),
        "avg_latency_ms": _avg_latency_ms(rows),
    }


def analyze_dual_teacher(*, deepseek_path: Path, qwen_path: Path) -> dict[str, Any]:
    deepseek_rows = _load_jsonl(deepseek_path)
    qwen_rows = _load_jsonl(qwen_path)

    deepseek_by_qid = {row["qid"]: row for row in deepseek_rows}
    qwen_by_qid = {row["qid"]: row for row in qwen_rows}
    overlap_qids = sorted(set(deepseek_by_qid) & set(qwen_by_qid))

    overlap_total = len(overlap_qids)
    both_usable = 0
    deepseek_only = 0
    qwen_only = 0
    neither = 0

    for qid in overlap_qids:
        deepseek_usable = _is_usable(deepseek_by_qid[qid])
        qwen_usable = _is_usable(qwen_by_qid[qid])
        if deepseek_usable and qwen_usable:
            both_usable += 1
        elif deepseek_usable:
            deepseek_only += 1
        elif qwen_usable:
            qwen_only += 1
        else:
            neither += 1

    deepseek_overlap_usable = both_usable + deepseek_only
    qwen_overlap_usable = both_usable + qwen_only
    pooled_usable = both_usable + deepseek_only + qwen_only

    overlap = {
        "total": overlap_total,
        "both_usable": both_usable,
        "deepseek_only_usable": deepseek_only,
        "qwen_only_usable": qwen_only,
        "neither_usable": neither,
        "both_usable_rate_pct": _pct(both_usable, overlap_total),
        "deepseek_only_rate_pct": _pct(deepseek_only, overlap_total),
        "qwen_only_rate_pct": _pct(qwen_only, overlap_total),
        "neither_rate_pct": _pct(neither, overlap_total),
        "deepseek_usable_rate_pct": _pct(deepseek_overlap_usable, overlap_total),
        "qwen_usable_rate_pct": _pct(qwen_overlap_usable, overlap_total),
        "pooled_usable_rate_pct": _pct(pooled_usable, overlap_total),
        "pooled_gain_vs_deepseek_pp": round(
            _pct(pooled_usable, overlap_total) - _pct(deepseek_overlap_usable, overlap_total),
            2,
        ),
        "pooled_gain_vs_qwen_pp": round(
            _pct(pooled_usable, overlap_total) - _pct(qwen_overlap_usable, overlap_total),
            2,
        ),
    }

    return {
        "deepseek_path": str(deepseek_path),
        "qwen_path": str(qwen_path),
        "teachers": {
            "deepseek": _teacher_stats(deepseek_rows),
            "qwen": _teacher_stats(qwen_rows),
        },
        "overlap": overlap,
    }


def _write_markdown(stats: dict[str, Any], output_path: Path) -> None:
    deepseek = stats["teachers"]["deepseek"]
    qwen = stats["teachers"]["qwen"]
    overlap = stats["overlap"]

    lines = [
        "# Dual-teacher Candidate Pool Summary",
        "",
        "## Teacher-level ratios",
        "",
        "| Teacher | Success rate | Citation rate | Usable rate | Downgraded rate | Avg latency |",
        "|---|---:|---:|---:|---:|---:|",
        f"| DeepSeek-V3 | {deepseek['ok_rate_pct']}% | {deepseek['citation_rate_pct']}% | {deepseek['usable_rate_pct']}% | {deepseek['downgraded_rate_pct']}% | {deepseek['avg_latency_ms']} ms |",
        f"| Qwen3-32B | {qwen['ok_rate_pct']}% | {qwen['citation_rate_pct']}% | {qwen['usable_rate_pct']}% | {qwen['downgraded_rate_pct']}% | {qwen['avg_latency_ms']} ms |",
        "",
        "## Overlap-set coverage",
        "",
        "| Metric | Ratio |",
        "|---|---:|",
        f"| DeepSeek usable coverage | {overlap['deepseek_usable_rate_pct']}% |",
        f"| Qwen usable coverage | {overlap['qwen_usable_rate_pct']}% |",
        f"| Dual-teacher pooled coverage | {overlap['pooled_usable_rate_pct']}% |",
        f"| Both usable | {overlap['both_usable_rate_pct']}% |",
        f"| DeepSeek-only gain | {overlap['deepseek_only_rate_pct']}% |",
        f"| Qwen-only gain | {overlap['qwen_only_rate_pct']}% |",
        f"| Neither usable | {overlap['neither_rate_pct']}% |",
        "",
        "## Key takeaway",
        "",
        f"- Dual-teacher pooled coverage improves usable coverage by {overlap['pooled_gain_vs_deepseek_pp']} percentage points over DeepSeek alone on the overlap set.",
        f"- Qwen contributes an extra {overlap['qwen_only_rate_pct']}% long-tail coverage beyond the stable teacher branch.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(stats: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    overlap = stats["overlap"]
    deepseek = stats["teachers"]["deepseek"]
    qwen = stats["teachers"]["qwen"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#94A3B8",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), dpi=200)

    def _style_axis(ax: Any, title: str) -> None:
        ax.set_title(title, fontsize=11.2, pad=10, weight="semibold", color="#0F172A")
        ax.grid(axis="x", linestyle="-", linewidth=0.55, alpha=0.12, color="#64748B")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=9, colors="#475569")
        ax.spines["left"].set_color("#94A3B8")
        ax.spines["bottom"].set_color("#94A3B8")

    coverage_labels = ["DeepSeek", "Qwen", "Dual-teacher pool"]
    coverage_values = [
        overlap["deepseek_usable_rate_pct"],
        overlap["qwen_usable_rate_pct"],
        overlap["pooled_usable_rate_pct"],
    ]
    coverage_colors = ["#8FB7E8", "#9FD7C9", "#F3C982"]
    bars = axes[0].barh(
        coverage_labels,
        coverage_values,
        color=coverage_colors,
        edgecolor="none",
        height=0.56,
    )
    _style_axis(axes[0], "Usable Coverage On Overlap Set")
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Coverage (%)", fontsize=9.5, color="#334155")
    for bar, value in zip(bars, coverage_values):
        axes[0].text(
            value + 1.1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            color="#334155",
            weight="semibold",
        )

    category_labels = ["Both usable", "DeepSeek-only", "Qwen-only", "Neither"]
    category_values = [
        overlap["both_usable_rate_pct"],
        overlap["deepseek_only_rate_pct"],
        overlap["qwen_only_rate_pct"],
        overlap["neither_rate_pct"],
    ]
    category_colors = ["#8FB7E8", "#B8D0F0", "#A8DDD0", "#F6E3B4"]
    left = 0.0
    for label, value, color in zip(category_labels, category_values, category_colors):
        axes[1].barh(
            ["Candidate pool composition"],
            [value],
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            height=0.52,
            label=label,
        )
        if value >= 7.0:
            axes[1].text(
                left + value / 2,
                0,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8.8,
                color="#334155",
                weight="semibold",
            )
        left += value
    _style_axis(axes[1], "Dual-teacher Pool Composition")
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Share (%)", fontsize=9.5, color="#334155")
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=8.3,
        labelcolor="#475569",
    )

    fig.suptitle("Dual-teacher Candidate Pool Analysis", fontsize=13, weight="semibold", y=0.99, color="#0F172A")
    fig.text(
        0.5,
        0.93,
        (
            f"DeepSeek usable rate {deepseek['usable_rate_pct']:.1f}%   |   "
            f"Qwen usable rate {qwen['usable_rate_pct']:.1f}%   |   "
            f"pooled overlap coverage +{overlap['pooled_gain_vs_deepseek_pp']:.1f} pp vs stable branch"
        ),
        ha="center",
        fontsize=9.2,
        color="#64748B",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.83, bottom=0.22, wspace=0.28)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze dual-teacher candidate pool coverage and stability.")
    parser.add_argument(
        "--deepseek",
        default="datasets/distill/teacher_deepseek_v3.jsonl",
        type=Path,
        help="Path to the DeepSeek teacher output JSONL.",
    )
    parser.add_argument(
        "--qwen",
        default="datasets/distill/teacher_qwen3_32b.jsonl",
        type=Path,
        help="Path to the Qwen teacher output JSONL.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-png", type=Path)
    args = parser.parse_args(argv)

    stats = analyze_dual_teacher(deepseek_path=args.deepseek, qwen_path=args.qwen)
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
