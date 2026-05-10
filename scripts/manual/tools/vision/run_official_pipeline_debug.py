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
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
import os

import cv2
import numpy as np

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


def _build_component_bbox_index(raw: dict[str, Any]) -> dict[str, list[float]]:
    detect = ((raw.get("stages") or {}).get("detect") or {})
    index: dict[str, list[float]] = {}
    for det in detect.get("detections") or []:
        comp_id = str(det.get("component_id") or "")
        bbox = det.get("bbox") or []
        if comp_id and len(bbox) >= 4:
            index[comp_id] = [float(v) for v in bbox[:4]]
    return index


def _infer_disconnected_component_ids(raw: dict[str, Any]) -> list[str]:
    topology = ((raw.get("stages") or {}).get("topology") or {})
    topo_graph = topology.get("topology_graph") or {}
    nodes = topo_graph.get("nodes") or []
    links = topo_graph.get("links") or topo_graph.get("edges") or []
    if not nodes:
        return []

    node_kind: dict[str, str] = {}
    comp_ids: set[str] = set()
    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        kind = str(node.get("kind") or "")
        node_kind[node_id] = kind
        if kind == "comp":
            comp_ids.add(node_id)
    if not comp_ids:
        return []

    adj: dict[str, set[str]] = defaultdict(set)
    for link in links:
        src = str(link.get("source") or "")
        dst = str(link.get("target") or "")
        if not src or not dst:
            continue
        adj[src].add(dst)
        adj[dst].add(src)

    visited: set[str] = set()
    groups: list[list[str]] = []
    for comp in sorted(comp_ids):
        if comp in visited:
            continue
        q: deque[str] = deque([comp])
        visited.add(comp)
        group: list[str] = []
        while q:
            cur = q.popleft()
            if node_kind.get(cur) == "comp":
                group.append(cur)
            for nxt in adj.get(cur, set()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                q.append(nxt)
        if group:
            groups.append(group)

    if len(groups) <= 1:
        return []

    groups.sort(key=lambda g: len(g), reverse=True)
    return sorted([comp for group in groups[1:] for comp in group])


def _semantic_wiring_items(raw: dict[str, Any]) -> list[dict[str, Any]]:
    semantic = ((raw.get("stages") or {}).get("semantic_analysis") or {})
    out: list[dict[str, Any]] = []
    for err in semantic.get("wiring_errors") or []:
        severity_raw = str(err.get("severity") or "warning").lower()
        severity = "error" if severity_raw in {"danger", "error"} else "warning"
        out.append(
            {
                "error_code": str(err.get("error_code") or "SEMANTIC_WIRING_ERROR"),
                "category": "semantic_errors",
                "severity": severity,
                "message": str(err.get("message") or ""),
                "component_id": err.get("component_id"),
                "current_hole_id": err.get("current_hole_id"),
                "current_node_id": err.get("current_net_id"),
                "evidence_refs": [],
            }
        )
    return out


def _collect_render_targets(
    item: dict[str, Any],
    *,
    raw: dict[str, Any],
    bbox_index: dict[str, list[float]],
) -> list[dict[str, Any]]:
    targets = list(item.get("highlight_targets") or [])
    fallback: list[dict[str, Any]] = list(targets)

    for ref in item.get("evidence_refs") or []:
        kind = ref.get("kind")
        if kind == "component_bbox_ref" and ref.get("bbox"):
            fallback.append(
                {
                    "kind": "component_bbox_ref",
                    "render": "box",
                    "target_type": "component",
                    "component_id": ref.get("component_id"),
                    "view_id": "top",
                    "bbox": ref.get("bbox"),
                }
            )
        elif kind == "pin_keypoint_ref" and ref.get("keypoint"):
            fallback.append(
                {
                    "kind": "pin_keypoint_ref",
                    "render": "point",
                    "target_type": "pin",
                    "component_id": ref.get("component_id"),
                    "pin_name": ref.get("pin_name"),
                    "view_id": ref.get("view_id", "top"),
                    "keypoint": ref.get("keypoint"),
                    "radius_px": 8,
                }
            )

    existing_bbox_components = {
        str(target.get("component_id") or "")
        for target in fallback
        if str(target.get("kind") or "") == "component_bbox_ref"
    }

    component_candidates = {
        str(item.get("component_id") or ""),
        str(item.get("current_component_id") or ""),
    }
    if str(item.get("error_code") or "") == "MULTIPLE_DISCONNECTED_SUBGRAPHS":
        component_candidates.update(_infer_disconnected_component_ids(raw))

    for comp_id in sorted(component_candidates):
        if not comp_id or comp_id in existing_bbox_components:
            continue
        bbox = bbox_index.get(comp_id)
        if not bbox:
            continue
        fallback.append(
            {
                "kind": "component_bbox_ref",
                "render": "box",
                "target_type": "component",
                "component_id": comp_id,
                "view_id": "top",
                "bbox": bbox,
            }
        )

    return fallback


def _draw_target(canvas: np.ndarray, target: dict[str, Any], color: tuple[int, int, int]) -> None:
    kind = str(target.get("kind") or "")
    if kind == "component_bbox_ref":
        bbox = target.get("bbox") or []
        if len(bbox) >= 4:
            x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            comp = str(target.get("component_id") or "")
            if comp:
                _draw_text_safe(canvas, comp, (x1, max(24, y1 - 8)), 0.7, color, 2)
    elif kind == "pin_keypoint_ref":
        keypoint = target.get("keypoint") or []
        if len(keypoint) >= 2:
            px, py = int(float(keypoint[0])), int(float(keypoint[1]))
            radius = int(target.get("radius_px") or 8)
            cv2.circle(canvas, (px, py), max(4, radius), color, 2)
            pin_name = str(target.get("pin_name") or "")
            if pin_name:
                _draw_text_safe(canvas, pin_name, (px + 8, py - 8), 0.55, color, 2)


def _find_cjk_font_path() -> str | None:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for font_path in candidates:
        if os.path.exists(font_path):
            return font_path
    return None


def _draw_text_safe(
    canvas: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    # OpenCV 的内置字体不支持中文；ASCII 仍走 cv2，非 ASCII 用 Pillow + CJK 字体。
    if text.isascii():
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        return

    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        fallback = text.encode("ascii", errors="replace").decode("ascii")
        cv2.putText(canvas, fallback, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        return

    font_path = _find_cjk_font_path()
    if not font_path:
        fallback = text.encode("ascii", errors="replace").decode("ascii")
        cv2.putText(canvas, fallback, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        return

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    font_size = max(14, int(30 * font_scale))
    font = ImageFont.truetype(font_path, font_size)
    draw.text(org, text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
    canvas[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _render_issue_images(
    *,
    raw: dict[str, Any],
    top_image_path: Path,
    out_dir: Path,
) -> None:
    img = cv2.imread(str(top_image_path))
    if img is None:
        print(f"skip issue render: cannot read image {top_image_path}")
        return

    validate = ((raw.get("stages") or {}).get("validate") or {})
    report = validate.get("comparison_report") or {}
    items = list(report.get("items") or [])
    items.extend(_semantic_wiring_items(raw))
    if not items:
        print("skip issue render: no diagnostic items")
        return

    issue_dir = out_dir / "issue_images"
    issue_dir.mkdir(parents=True, exist_ok=True)

    bbox_index = _build_component_bbox_index(raw)
    overview = img.copy()
    for idx, item in enumerate(items, start=1):
        code = str(item.get("error_code") or "UNKNOWN")
        message = str(item.get("message") or "")
        targets = _collect_render_targets(item, raw=raw, bbox_index=bbox_index)
        item_canvas = img.copy()
        color = (0, 0, 255) if str(item.get("severity")) == "error" else (0, 165, 255)
        for target in targets:
            view_id = str(target.get("view_id") or "top")
            if view_id != "top":
                continue
            _draw_target(item_canvas, target, color)
            _draw_target(overview, target, color)

        _draw_text_safe(item_canvas, f"{idx:02d}. {code}", (24, 38), 0.9, color, 2)
        if message:
            clipped_msg = message[:120]
            _draw_text_safe(item_canvas, clipped_msg, (24, 72), 0.62, color, 2)
        item_file = issue_dir / f"{idx:02d}_{code}.jpg"
        cv2.imwrite(str(item_file), item_canvas)

    _draw_text_safe(overview, f"Detected issues: {len(items)}", (24, 38), 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(issue_dir / "overview.jpg"), overview)


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
    _render_issue_images(raw=raw, top_image_path=image_paths[0], out_dir=out_dir)

    print(f"saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
