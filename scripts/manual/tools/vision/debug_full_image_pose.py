#!/usr/bin/env python3
"""
Experimental full-image YOLO-Pose runner.

实验目的:
- 不使用 ROI 裁切
- 直接在整图上跑 YOLO-Pose
- 将 pose 实例按类别 + IoU 关联回 S1 组件检测结果
- 输出可视化与 hole mapping 结果, 用于比较 ROI pose 与 full-image pose

注意:
- 该脚本不是正式 pipeline 入口
- 它会绕开正式 S1.5 的 ROI pin 检测主语义
- 仅用于研究 full-image pose 方案是否值得进入主链
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.pipeline.stages.s1_detect import run_detect
from app.pipeline.stages.s2_mapping import run_mapping
from app.pipeline.vision.calibrator import BreadboardCalibrator
from app.pipeline.vision.detector import ComponentDetector
from app.pipeline.vision.label_mapping import (
    default_package_type,
    default_pin_count,
    default_pin_names,
    default_pin_schema_id,
    is_pin_order_exchangeable,
    is_supported_component_type,
    default_symmetry_group,
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
    parse_meta: dict[str, Any]
    used: bool = False


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
    span = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
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


def _load_full_image_pose_instances(
    *,
    image: np.ndarray,
    pin_detector: PinRoiDetector,
) -> list[PoseInstance]:
    results = pin_detector.model(image, verbose=False, device=pin_detector.device)  # type: ignore[union-attr]
    if not results:
        return []
    first = results[0]
    boxes = first.boxes
    keypoints = first.keypoints
    if boxes is None or keypoints is None or keypoints.xy is None:
        return []

    names_map = getattr(pin_detector.model, "names", {})  # type: ignore[union-attr]
    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros((len(xyxy),), dtype=np.float32)
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy),), dtype=np.float32)
    all_xy = keypoints.xy.cpu().numpy()
    kp_conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None

    instances: list[PoseInstance] = []
    for idx in range(len(xyxy)):
        raw_class = str(names_map.get(int(cls_ids[idx]), int(cls_ids[idx])))
        component_type = normalize_component_type(raw_class)
        package_type = default_package_type(component_type)
        pin_count = default_pin_count(component_type, package_type)
        parsed = _parse_model_keypoints(
            points=all_xy[idx],
            confs=kp_conf[idx] if kp_conf is not None and idx < len(kp_conf) else None,
            pin_count=pin_count,
        )
        instances.append(
            PoseInstance(
                component_type=component_type,
                class_name=raw_class,
                bbox=[float(v) for v in xyxy[idx].tolist()],
                confidence=float(confs[idx]),
                keypoints=list(parsed.ordered_keypoints),
                parse_meta={
                    "raw_keypoint_count": parsed.raw_keypoint_count,
                    "raw_visible_keypoint_count": parsed.raw_visible_keypoint_count,
                    "used_keypoint_count": parsed.used_keypoint_count,
                    "extra_keypoints_ignored": parsed.extra_keypoints_ignored,
                    "ignored_keypoints_reason": parsed.ignored_keypoints_reason,
                },
            )
        )
    return instances


def _match_pose_instance(
    det: dict,
    pose_instances: list[PoseInstance],
) -> PoseInstance | None:
    det_bbox = [float(v) for v in det.get("bbox") or [0, 0, 0, 0]]
    component_type = normalize_component_type(str(det.get("component_type") or det.get("class_name") or "UNKNOWN"))

    def _score(inst: PoseInstance) -> float:
        iou = _iou_xyxy(det_bbox, inst.bbox)
        det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
        det_cy = (det_bbox[1] + det_bbox[3]) / 2.0
        inst_cx = (inst.bbox[0] + inst.bbox[2]) / 2.0
        inst_cy = (inst.bbox[1] + inst.bbox[3]) / 2.0
        center_dist = ((det_cx - inst_cx) ** 2 + (det_cy - inst_cy) ** 2) ** 0.5
        diag = max(1.0, ((det_bbox[2] - det_bbox[0]) ** 2 + (det_bbox[3] - det_bbox[1]) ** 2) ** 0.5)
        proximity = max(0.0, 1.0 - center_dist / (diag * 1.5))
        kp_fit = _keypoints_inside_ratio(inst.keypoints, _expand_bbox(det_bbox))
        span_fit = _span_consistency(inst.keypoints, det_bbox)
        return iou * 1.2 + proximity * 0.45 + kp_fit * 0.9 + span_fit * 0.7 + inst.confidence * 0.1

    typed_candidates = [
        inst
        for inst in pose_instances
        if not inst.used and inst.component_type == component_type and is_supported_component_type(inst.component_type)
    ]
    fallback_candidates = [inst for inst in pose_instances if not inst.used]

    candidates = typed_candidates or fallback_candidates
    best: tuple[float, PoseInstance] | None = None
    for inst in candidates:
        if inst.used:
            continue
        score = _score(inst)
        if best is None or score > best[0]:
            best = (score, inst)

    if best is None:
        return None
    best[1].used = True
    return best[1]


def _aligned_keypoints(component_type: str, keypoints: list[tuple[float, float] | None], bbox: list[float]) -> list[tuple[float, float] | None]:
    if not is_pin_order_exchangeable(component_type):
        return list(keypoints)
    if len(keypoints) < 2 or keypoints[0] is None or keypoints[1] is None:
        return list(keypoints)
    p1, p2 = keypoints[0], keypoints[1]
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    if _bbox_orientation(bbox) == "horizontal":
        ordered = [p1, p2] if p1[0] <= p2[0] else [p2, p1]
    else:
        ordered = [p1, p2] if p1[1] <= p2[1] else [p2, p1]
    result = list(keypoints)
    result[0], result[1] = ordered[0], ordered[1]
    return result


def _build_components_from_full_pose(
    detections: list[dict],
    pose_instances: list[PoseInstance],
) -> list[dict]:
    components: list[dict] = []
    for det in detections:
        component_type = normalize_component_type(str(det.get("component_type") or det.get("class_name") or "UNKNOWN"))
        package_type = str(det.get("package_type") or default_package_type(component_type))
        pin_schema_id = default_pin_schema_id(component_type, package_type)
        pin_names = default_pin_names(component_type, default_pin_count(component_type, package_type))
        matched = _match_pose_instance(det, pose_instances)
        aligned_points = _aligned_keypoints(component_type, matched.keypoints if matched else [], list(det.get("bbox") or [0, 0, 0, 0]))

        pins = []
        for idx, pin_name in enumerate(pin_names, start=1):
            kp = aligned_points[idx - 1] if matched and idx - 1 < len(aligned_points) else None
            pins.append(
                {
                    "pin_id": idx,
                    "pin_name": pin_name,
                    "keypoints_by_view": {"top": [float(kp[0]), float(kp[1])] if kp is not None else None},
                    "visibility_by_view": {"top": 2 if kp is not None else 0},
                    "score_by_view": {"top": float(det.get("confidence", 1.0)) if kp is not None else 0.0},
                    "source_by_view": {"top": "model" if kp is not None else "unavailable"},
                    "confidence": float(det.get("confidence", 1.0)) if kp is not None else 0.0,
                    "source": "model" if kp is not None else "unavailable",
                    "metadata": {
                        "per_view": {
                            "top": {
                                "backend_type": "yolo_pose",
                                "backend_mode": "full_image_model",
                                "interface_version": "full_image_pose_debug_v1",
                                "roi_source": "full_image_pose",
                                **(matched.parse_meta if matched else {}),
                            }
                        }
                    },
                }
            )

        components.append(
            {
                "component_id": det.get("component_id"),
                "component_type": component_type,
                "class_name": component_type,
                "package_type": package_type,
                "pin_schema_id": pin_schema_id,
                "symmetry_group": default_symmetry_group(component_type),
                "bbox": list(det.get("bbox") or [0, 0, 0, 0]),
                "confidence": float(det.get("confidence", 1.0)),
                "orientation": float(det.get("orientation", 0.0)),
                "full_image_pose_match": {
                    "matched": matched is not None,
                    "pose_bbox": matched.bbox if matched else None,
                    "pose_class_name": matched.class_name if matched else None,
                    "pose_component_type": matched.component_type if matched else None,
                    "pose_confidence": matched.confidence if matched else 0.0,
                },
                "pins": pins,
            }
        )
    return components


def _draw_debug(
    *,
    image: np.ndarray,
    detections: list[dict],
    components: list[dict],
    out_path: Path,
) -> None:
    canvas = image.copy()
    for det, comp in zip(detections, components):
        x1, y1, x2, y2 = [int(v) for v in det.get("bbox") or [0, 0, 0, 0]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{comp.get('component_id')}:{comp.get('component_type')}",
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 0),
            2,
        )
        match = comp.get("full_image_pose_match") or {}
        pose_bbox = match.get("pose_bbox")
        if pose_bbox:
            px1, py1, px2, py2 = [int(round(v)) for v in pose_bbox]
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 180, 255), 2)

        for pin in comp.get("pins") or []:
            kp = (pin.get("keypoints_by_view") or {}).get("top")
            if kp is None:
                observations = pin.get("observations") or []
                top_obs = next((obs for obs in observations if obs.get("view_id") == "top"), None)
                if top_obs is not None:
                    kp = top_obs.get("keypoint")
            if not kp:
                continue
            px, py = int(round(kp[0])), int(round(kp[1]))
            cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                str(pin.get("pin_name")),
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 220),
                1,
            )
    cv2.imwrite(str(out_path), canvas)


def run_on_image(image_path: Path, out_root: Path, pin_model_path: str | None = None) -> None:
    detector = ComponentDetector(
        model_path=settings.YOLO_MODEL_PATH,
        obb_model_path=settings.YOLO_OBB_MODEL_PATH,
        device=settings.YOLO_DEVICE,
    )
    pin_detector = PinRoiDetector(
        model_path=pin_model_path or settings.PIN_MODEL_PATH,
        device=settings.PIN_MODEL_DEVICE,
    )

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    s1 = run_detect(
        [img_b64],
        detector=detector,
        conf=settings.YOLO_CONF_THRESHOLD,
        iou=settings.YOLO_IOU_THRESHOLD,
        imgsz=settings.YOLO_IMGSZ,
    )
    pose_instances = _load_full_image_pose_instances(image=image, pin_detector=pin_detector)
    components = _build_components_from_full_pose(s1["detections"], pose_instances)
    calibrator = BreadboardCalibrator(
        rows=settings.BREADBOARD_ROWS,
        cols_per_side=settings.BREADBOARD_COLS_PER_SIDE,
    )
    s2 = run_mapping(
        components,
        calibrator=calibrator,
        image_shape=s1["primary_image_shape"],
        images_b64=[img_b64],
    )

    sample_dir = out_root / image_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    _draw_debug(
        image=image,
        detections=s1["detections"],
        components=s2["components"],
        out_path=sample_dir / "annotated_full_image_pose.png",
    )

    with open(sample_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": str(image_path),
                "pin_model_path": str(pin_model_path or settings.PIN_MODEL_PATH),
                "detect_count": len(s1["detections"]),
                "pose_instance_count": len(pose_instances),
                "calibration": s2["calibration"],
                "components": s2["components"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="Image paths")
    parser.add_argument("--out", default="/tmp/labguardian_full_image_pose_debug", help="Output directory")
    parser.add_argument("--pin-model", default=None, help="Explicit full-image pose weight path")
    args = parser.parse_args()

    if args.images:
        images = [Path(p) for p in args.images]
    else:
        images = [
            Path("/Users/liusuan/Desktop/LabGuardian-Server/train_demo/Camera Roll(1)/Camera Roll/WIN_20260415_21_27_54_Pro.jpg"),
            Path("/Users/liusuan/Desktop/LabGuardian-Server/train_demo/Camera Roll(1)/Camera Roll/WIN_20260413_21_36_48_Pro.jpg"),
            Path("/Users/liusuan/Desktop/LabGuardian-Server/train_demo/Camera Roll(1)/Camera Roll/WIN_20260413_21_39_28_Pro.jpg"),
        ]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        run_on_image(image_path, out_root, pin_model_path=args.pin_model)
    print(out_root)


if __name__ == "__main__":
    main()
