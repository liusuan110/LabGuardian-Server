#!/usr/bin/env python3
"""
Official pipeline debug runner.

用途:
- 只调用正式 `run_pipeline()` 入口
- 将完整 `S1 -> S1.5 -> S2 -> S3 -> S4` 结果落盘
- 作为演示/联调用的标准离线入口

边界:
- 不引入实验性 full-image pose 归属逻辑
- 不绕开 orchestrator 直接拼接阶段结果
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.orchestrator import run_pipeline


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official LabGuardian vision pipeline on local images.")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="1-3 image paths, ordered as top / left / right when available.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/labguardian_pipeline_debug"),
        help="Directory used to write json outputs.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional reference circuit json path.",
    )
    parser.add_argument("--conf", type=float, default=None, help="Optional detect confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="Optional detect IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional inference image size.")
    return parser


def _stage_summary(raw: dict[str, Any]) -> dict[str, Any]:
    stages = raw.get("stages", {}) or {}
    detect = stages.get("detect", {}) or {}
    pin_detect = stages.get("pin_detect", {}) or {}
    mapping = stages.get("mapping", {}) or {}
    topology = stages.get("topology", {}) or {}
    validate = stages.get("validate", {}) or {}
    return {
        "stage_order": ["detect", "pin_detect", "mapping", "topology", "validate"],
        "detect": {
            "component_count": len(detect.get("detections", []) or []),
            "duration_ms": detect.get("duration_ms"),
            "interface_version": detect.get("interface_version"),
        },
        "pin_detect": {
            "component_count": len(pin_detect.get("components", []) or []),
            "duration_ms": pin_detect.get("duration_ms"),
            "interface_version": pin_detect.get("interface_version"),
        },
        "mapping": {
            "component_count": len(mapping.get("components", []) or []),
            "duration_ms": mapping.get("duration_ms"),
            "interface_version": mapping.get("interface_version"),
            "calibration_mode": mapping.get("calibration_mode"),
        },
        "topology": {
            "component_count": topology.get("component_count"),
            "duration_ms": topology.get("duration_ms"),
            "interface_version": topology.get("interface_version"),
        },
        "validate": {
            "risk_level": validate.get("risk_level"),
            "match_level": validate.get("match_level"),
            "duration_ms": validate.get("duration_ms"),
            "interface_version": validate.get("interface_version"),
        },
        "total_duration_ms": raw.get("total_duration_ms"),
    }


def main() -> int:
    args = _build_arg_parser().parse_args()
    image_paths = [Path(item).expanduser().resolve() for item in args.images]
    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")

    images_b64 = [_encode_image(path) for path in image_paths]
    reference_payload: dict[str, Any] | str | None = None
    if args.reference is not None:
        reference_payload = str(args.reference.expanduser().resolve())

    raw = run_pipeline(
        images_b64=images_b64,
        reference_circuit=reference_payload,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    out_dir = args.output_root / image_paths[0].stem
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "mode": "official_pipeline_debug",
        "images": [str(path) for path in image_paths],
        "reference": str(args.reference.expanduser().resolve()) if args.reference else None,
        "output_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "pipeline_result.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "stage_summary.json").write_text(json.dumps(_stage_summary(raw), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
