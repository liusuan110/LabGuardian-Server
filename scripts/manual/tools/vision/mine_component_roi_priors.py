#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from labelme_pose_dataset_utils import iter_labelme_instances


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    q = max(0.0, min(1.0, q))
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine ROI priors from LabelMe annotations.")
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("/Users/liusuan/Desktop/LabGuardian-Server/train_demo"),
        help="Root directory containing LabelMe JSON files.",
    )
    parser.add_argument(
        "--component-class",
        type=str,
        default="resistor",
        help="Component class to analyze, e.g. resistor / diode / transistor_3pin.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/labguardian_roi_priors"),
        help="Directory to write JSON summary into.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    component_class = args.component_class.strip().lower()
    ratios_dx: list[float] = []
    ratios_dy: list[float] = []
    abs_dx: list[float] = []
    abs_dy: list[float] = []
    by_pin_count: dict[int, int] = defaultdict(int)
    sample_count = 0

    for instance in iter_labelme_instances(args.annotations_root):
        if instance.class_name.lower() != component_class:
            continue
        pin_points = [instance.points[k] for k in ("pin1", "pin2", "pin3") if k in instance.points]
        if not pin_points:
            continue
        sample_count += 1
        by_pin_count[len(pin_points)] += 1
        cx, cy = instance.center
        w, h = instance.size
        max_dx = max(abs(px - cx) for px, _ in pin_points)
        max_dy = max(abs(py - cy) for _, py in pin_points)
        abs_dx.append(max_dx)
        abs_dy.append(max_dy)
        ratios_dx.append(max_dx / max(1.0, w))
        ratios_dy.append(max_dy / max(1.0, h))

    ratios_dx.sort()
    ratios_dy.sort()
    abs_dx.sort()
    abs_dy.sort()

    summary = {
        "component_class": component_class,
        "sample_count": sample_count,
        "pin_count_histogram": dict(sorted(by_pin_count.items())),
        "ratio_summary": {
            "max_dx_over_w": _describe(ratios_dx),
            "max_dy_over_h": _describe(ratios_dy),
        },
        "absolute_summary": {
            "max_dx_px": _describe(abs_dx),
            "max_dy_px": _describe(abs_dy),
        },
        "suggested_roi_rule": {
            "major_expand_ratio": round(percentile(ratios_dx, 0.99), 4),
            "minor_expand_ratio": round(percentile(ratios_dy, 0.99), 4),
            "note": "Use p99 as a safer ROI crop prior; max is available but more sensitive to annotation outliers.",
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"{component_class}_roi_prior_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved summary to: {out_path}")
    return 0


def _describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(values[0], 4),
        "median": round(median(values), 4),
        "p90": round(percentile(values, 0.90), 4),
        "p95": round(percentile(values, 0.95), 4),
        "p99": round(percentile(values, 0.99), 4),
        "max": round(values[-1], 4),
    }


if __name__ == "__main__":
    raise SystemExit(main())
