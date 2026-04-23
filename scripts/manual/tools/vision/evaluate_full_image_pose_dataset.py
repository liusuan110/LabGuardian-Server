#!/usr/bin/env python3
"""
Evaluate full-image YOLO-Pose predictions against dataset pixel labels.

目标:
1. 直接在整图上运行 pose 模型
2. 读取 YOLO-Pose 标签中的 pin 像素真值
3. 统计 pin1/pin2(/pin3) 的像素误差
4. 导出误差摘要和可视化对照图
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.pipeline.vision.label_mapping import (
    default_package_type,
    default_pin_count,
    is_pin_order_exchangeable,
    normalize_component_type,
)
from app.pipeline.vision.pin_model import PinRoiDetector, _parse_model_keypoints


@dataclass
class PoseInstance:
    component_type: str
    class_name: str
    bbox: list[float]
    confidence: float
    keypoints: list[tuple[float, float] | None]


def _load_names(data_yaml: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if ": " not in stripped:
            continue
        left, right = stripped.split(": ", 1)
        if left.isdigit():
            names[int(left)] = right.strip()
    return names


def _norm_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> list[float]:
    bw = w * img_w
    bh = h * img_h
    x1 = (cx * img_w) - bw / 2.0
    y1 = (cy * img_h) - bh / 2.0
    x2 = x1 + bw
    y2 = y1 + bh
    return [float(x1), float(y1), float(x2), float(y2)]


def _parse_gt_label_file(label_path: Path, img_w: int, img_h: int, names: dict[int, str]) -> list[PoseInstance]:
    items: list[PoseInstance] = []
    if not label_path.exists():
        return items
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        vals = stripped.split()
        if len(vals) < 5:
            continue
        cls_id = int(vals[0])
        raw_name = names.get(cls_id, str(cls_id))
        cx, cy, bw, bh = [float(v) for v in vals[1:5]]
        bbox = _norm_xyxy(cx, cy, bw, bh, img_w, img_h)
        kp_vals = vals[5:]
        keypoints: list[tuple[float, float] | None] = []
        for idx in range(0, len(kp_vals), 3):
            if idx + 2 >= len(kp_vals):
                break
            kx = float(kp_vals[idx])
            ky = float(kp_vals[idx + 1])
            vis = int(float(kp_vals[idx + 2]))
            if vis <= 0 or (abs(kx) < 1e-9 and abs(ky) < 1e-9):
                keypoints.append(None)
            else:
                keypoints.append((float(kx * img_w), float(ky * img_h)))

        component_type = normalize_component_type(raw_name)
        package_type = default_package_type(component_type)
        pin_count = default_pin_count(component_type, package_type)
        items.append(
            PoseInstance(
                component_type=component_type,
                class_name=raw_name,
                bbox=bbox,
                confidence=1.0,
                keypoints=keypoints[:pin_count],
            )
        )
    return items


def _load_pred_instances(image: np.ndarray, detector: PinRoiDetector) -> list[PoseInstance]:
    results = detector.model(image, verbose=False, device=detector.device)  # type: ignore[union-attr]
    if not results:
        return []
    first = results[0]
    if first.boxes is None or first.keypoints is None or first.keypoints.xy is None:
        return []
    names_map = getattr(detector.model, "names", {})  # type: ignore[union-attr]
    xyxy = first.boxes.xyxy.cpu().numpy()
    cls_ids = first.boxes.cls.cpu().numpy() if first.boxes.cls is not None else np.zeros((len(xyxy),), dtype=np.float32)
    confs = first.boxes.conf.cpu().numpy() if first.boxes.conf is not None else np.ones((len(xyxy),), dtype=np.float32)
    all_xy = first.keypoints.xy.cpu().numpy()
    kp_conf = first.keypoints.conf.cpu().numpy() if first.keypoints.conf is not None else None

    items: list[PoseInstance] = []
    for idx in range(len(xyxy)):
        raw_name = str(names_map.get(int(cls_ids[idx]), int(cls_ids[idx])))
        component_type = normalize_component_type(raw_name)
        package_type = default_package_type(component_type)
        pin_count = default_pin_count(component_type, package_type)
        parsed = _parse_model_keypoints(
            points=all_xy[idx],
            confs=kp_conf[idx] if kp_conf is not None and idx < len(kp_conf) else None,
            pin_count=pin_count,
        )
        items.append(
            PoseInstance(
                component_type=component_type,
                class_name=raw_name,
                bbox=[float(v) for v in xyxy[idx].tolist()],
                confidence=float(confs[idx]),
                keypoints=list(parsed.ordered_keypoints),
            )
        )
    return items


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_distance(a: list[float], b: list[float]) -> float:
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def _expand_bbox(bbox: list[float], pad_ratio: float = 0.22, min_pad: float = 12.0) -> list[float]:
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    pad_x = max(min_pad, w * pad_ratio)
    pad_y = max(min_pad, h * pad_ratio)
    return [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]


def _point_in_bbox(point: tuple[float, float] | None, bbox: list[float]) -> bool:
    if point is None:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _keypoints_inside_ratio(points: list[tuple[float, float] | None], bbox: list[float]) -> float:
    valid = [p for p in points if p is not None]
    if not valid:
        return 0.0
    inside = sum(1 for p in valid if _point_in_bbox(p, bbox))
    return inside / len(valid)


def _bbox_orientation(bbox: list[float]) -> str:
    x1, y1, x2, y2 = bbox
    return "horizontal" if (x2 - x1) >= (y2 - y1) else "vertical"


def _keypoint_orientation(points: list[tuple[float, float] | None]) -> str | None:
    valid = [p for p in points if p is not None]
    if len(valid) < 2:
        return None
    p1, p2 = valid[0], valid[-1]
    return "horizontal" if abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1]) else "vertical"


def _span_consistency(points: list[tuple[float, float] | None], bbox: list[float]) -> float:
    valid = [p for p in points if p is not None]
    if len(valid) < 2:
        return 0.35
    p1, p2 = valid[0], valid[-1]
    span = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    major = max(bw, bh)
    minor = max(1.0, min(bw, bh))
    if span < minor * 0.4:
        base = 0.2
    elif span > major * 2.6:
        base = 0.25
    else:
        base = 1.0 - min(abs(span - major) / max(major, 1.0), 1.0) * 0.5
    kp_orient = _keypoint_orientation(points)
    bbox_orient = _bbox_orientation(bbox)
    if kp_orient is None:
        return base
    return base if kp_orient == bbox_orient else base * 0.55


def _pair_score(gt: PoseInstance, pred: PoseInstance) -> float:
    iou = _iou_xyxy(gt.bbox, pred.bbox)
    dist = _center_distance(gt.bbox, pred.bbox)
    diag = max(1.0, math.hypot(gt.bbox[2] - gt.bbox[0], gt.bbox[3] - gt.bbox[1]))
    proximity = max(0.0, 1.0 - dist / (diag * 1.5))
    expanded_gt = _expand_bbox(gt.bbox)
    kp_fit = _keypoints_inside_ratio(pred.keypoints, expanded_gt)
    span_fit = _span_consistency(pred.keypoints, gt.bbox)
    return (
        iou * 1.8
        + proximity * 0.45
        + kp_fit * 0.9
        + span_fit * 0.65
        + pred.confidence * 0.1
    )


def _match_predictions(gt_items: list[PoseInstance], pred_items: list[PoseInstance]) -> list[tuple[PoseInstance, PoseInstance | None]]:
    scored_pairs: list[tuple[float, int, int]] = []
    for gt_idx, gt in enumerate(gt_items):
        for pred_idx, pred in enumerate(pred_items):
            if pred.class_name != gt.class_name:
                continue
            score = _pair_score(gt, pred)
            if score > 0.05:
                scored_pairs.append((score, gt_idx, pred_idx))

    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    gt_to_pred: dict[int, int] = {}
    used_gt = set()
    used_pred = set()
    for score, gt_idx, pred_idx in scored_pairs:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        gt_to_pred[gt_idx] = pred_idx
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)

    pairs: list[tuple[PoseInstance, PoseInstance | None]] = []
    for gt_idx, gt in enumerate(gt_items):
        pred_idx = gt_to_pred.get(gt_idx)
        pairs.append((gt, pred_items[pred_idx] if pred_idx is not None else None))
    return pairs


def _aligned_pred_keypoints(gt: PoseInstance, pred: PoseInstance) -> list[tuple[float, float] | None]:
    pred_points = list(pred.keypoints)
    if not is_pin_order_exchangeable(gt.component_type):
        return pred_points
    if len(gt.keypoints) < 2 or len(pred_points) < 2:
        return pred_points
    if gt.keypoints[0] is None or gt.keypoints[1] is None:
        return pred_points
    if pred_points[0] is None or pred_points[1] is None:
        return pred_points
    direct = (
        math.hypot(pred_points[0][0] - gt.keypoints[0][0], pred_points[0][1] - gt.keypoints[0][1])
        + math.hypot(pred_points[1][0] - gt.keypoints[1][0], pred_points[1][1] - gt.keypoints[1][1])
    )
    swapped = (
        math.hypot(pred_points[1][0] - gt.keypoints[0][0], pred_points[1][1] - gt.keypoints[0][1])
        + math.hypot(pred_points[0][0] - gt.keypoints[1][0], pred_points[0][1] - gt.keypoints[1][1])
    )
    if swapped + 1e-6 < direct:
        pred_points[0], pred_points[1] = pred_points[1], pred_points[0]
    return pred_points


def _draw_overlay(
    image_path: Path,
    pairs: list[tuple[PoseInstance, PoseInstance | None]],
    out_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    canvas = image.copy()
    for gt, pred in pairs:
        gx1, gy1, gx2, gy2 = [int(round(v)) for v in gt.bbox]
        cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        cv2.putText(canvas, f"GT:{gt.class_name}", (gx1, max(18, gy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)
        for idx, kp in enumerate(gt.keypoints, start=1):
            if kp is None:
                continue
            px, py = int(round(kp[0])), int(round(kp[1]))
            cv2.circle(canvas, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(canvas, f"g{idx}", (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if pred is None:
            continue
        pred_points = _aligned_pred_keypoints(gt, pred)
        px1, py1, px2, py2 = [int(round(v)) for v in pred.bbox]
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 165, 255), 2)
        cv2.putText(canvas, f"P:{pred.class_name}", (px1, min(canvas.shape[0] - 8, py2 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
        for idx, kp in enumerate(pred_points, start=1):
            if kp is None:
                continue
            px, py = int(round(kp[0])), int(round(kp[1]))
            cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(canvas, f"p{idx}", (px + 4, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            if idx - 1 < len(gt.keypoints) and gt.keypoints[idx - 1] is not None:
                gx, gy = gt.keypoints[idx - 1]
                cv2.line(canvas, (int(round(gx)), int(round(gy))), (px, py), (255, 255, 0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _safe_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
            "within_3px": 0,
            "within_5px": 0,
            "within_10px": 0,
        }
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": round(float(sum(values) / len(values)), 4),
        "median": round(float(statistics.median(values)), 4),
        "p90": round(float(np.percentile(np.array(values, dtype=np.float32), 90)), 4),
        "max": round(float(max(values)), 4),
        "within_3px": int(sum(1 for v in values if v <= 3.0)),
        "within_5px": int(sum(1 for v in values if v <= 5.0)),
        "within_10px": int(sum(1 for v in values if v <= 10.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate full-image YOLO-Pose pixel error against dataset labels.")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "train_demo" / "yolo_packed2_pose")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of images to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/labguardian_pose_eval"))
    args = parser.parse_args()

    data_yaml = args.dataset_root / "data.yaml"
    names = _load_names(data_yaml)
    image_dir = args.dataset_root / "images" / args.split
    label_dir = args.dataset_root / "labels" / args.split
    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    detector = PinRoiDetector(model_path=str(settings.PIN_MODEL_PATH), device=settings.PIN_MODEL_DEVICE)
    if detector.model is None:
        raise RuntimeError(f"Pin pose model failed to load: {settings.PIN_MODEL_PATH}")

    global_pin_errors: dict[str, list[float]] = defaultdict(list)
    class_pin_errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    image_summaries: list[dict[str, Any]] = []
    total_gt_instances = 0
    total_matched_instances = 0

    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]
        gt_items = _parse_gt_label_file(label_dir / f"{image_path.stem}.txt", img_w, img_h, names)
        pred_items = _load_pred_instances(image, detector)
        pairs = _match_predictions(gt_items, pred_items)
        total_gt_instances += len(gt_items)
        matched_here = sum(1 for _, pred in pairs if pred is not None)
        total_matched_instances += matched_here

        image_errors: list[float] = []
        for gt, pred in pairs:
            if pred is None:
                continue
            pred_points = _aligned_pred_keypoints(gt, pred)
            for pin_idx, gt_kp in enumerate(gt.keypoints, start=1):
                if gt_kp is None:
                    continue
                pred_kp = pred_points[pin_idx - 1] if pin_idx - 1 < len(pred_points) else None
                if pred_kp is None:
                    continue
                err = float(math.hypot(pred_kp[0] - gt_kp[0], pred_kp[1] - gt_kp[1]))
                pin_name = f"pin{pin_idx}"
                global_pin_errors[pin_name].append(err)
                class_pin_errors[gt.class_name][pin_name].append(err)
                image_errors.append(err)

        image_summaries.append(
            {
                "image": image_path.name,
                "gt_instances": len(gt_items),
                "pred_instances": len(pred_items),
                "matched_instances": matched_here,
                "mean_pin_error": round(float(sum(image_errors) / len(image_errors)), 4) if image_errors else None,
                "pairs": pairs,
                "image_path": str(image_path),
            }
        )
        if idx % 20 == 0:
            print(f"[pose-eval] processed {idx}/{len(image_paths)} images", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    ranked = sorted(
        image_summaries,
        key=lambda item: item["mean_pin_error"] if item["mean_pin_error"] is not None else -1.0,
        reverse=True,
    )
    selected = ranked[:5] + [item for item in image_summaries[:3] if item not in ranked[:5]]
    selected_unique: list[dict[str, Any]] = []
    seen_images = set()
    for item in selected:
        if item["image"] in seen_images:
            continue
        seen_images.add(item["image"])
        selected_unique.append(item)
    for item in selected_unique:
        _draw_overlay(Path(item["image_path"]), item["pairs"], vis_dir / f"{Path(item['image']).stem}_overlay.png")

    summary = {
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "image_count": len(image_paths),
        "gt_instance_count": total_gt_instances,
        "matched_instance_count": total_matched_instances,
        "match_rate": round(float(total_matched_instances / total_gt_instances), 4) if total_gt_instances else None,
        "global_pin_error": {pin_name: _safe_stats(vals) for pin_name, vals in sorted(global_pin_errors.items())},
        "per_class_pin_error": {
            class_name: {pin_name: _safe_stats(vals) for pin_name, vals in sorted(pin_map.items())}
            for class_name, pin_map in sorted(class_pin_errors.items())
        },
        "sample_images": [
            {
                "image": item["image"],
                "matched_instances": item["matched_instances"],
                "mean_pin_error": item["mean_pin_error"],
                "overlay": str(vis_dir / f"{Path(item['image']).stem}_overlay.png"),
            }
            for item in selected_unique
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
