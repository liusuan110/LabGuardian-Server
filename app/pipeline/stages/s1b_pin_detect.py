"""
Stage 1.5: Component ROI pin detection.

当前正式主路径:
- 不再按单组件 ROI 裁切后做 pin 识别
- 改为整图 full-image YOLO-Pose
- 再按类别 + bbox 几何把 pose 实例关联回 S1 组件检测结果

保留 legacy ROI 路径仅用于:
- 无 full-image pose 模型时的启发式 fallback
- 测试中的 mock pin detector
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, List

import numpy as np

from app.pipeline.vision.pin_model import PinRoiDetector, _parse_model_keypoints
from app.pipeline.vision.image_io import decode_images_b64, decode_summary
from app.pipeline.vision.label_mapping import (
    component_id_prefix,
    default_pin_count,
    default_pin_names,
    normalize_component_type,
)
from app.pipeline.vision.pin_schema import (
    default_package_type,
    default_pin_schema_id,
    default_symmetry_group,
)
from app.pipeline.vision.roi_cropper import crop_component_roi
from app.pipeline.vision.view_association import SideViewRoiResolver

logger = logging.getLogger(__name__)


ROI_RETRY_SCALES = {
    "top": [1.0, 1.4, 1.85, 2.35],
    "left_front": [1.0, 1.3, 1.7],
    "right_front": [1.0, 1.3, 1.7],
}
ROI_EDGE_MARGIN_PX = 12


@dataclass
class PoseInstance:
    component_type: str
    class_name: str
    bbox: list[float]
    confidence: float
    keypoints: list[tuple[float, float] | None]
    parse_meta: dict[str, Any]
    used: bool = False

def run_pin_detect(
    detections: List[dict],
    images_b64: List[str],
    pin_detector: PinRoiDetector,
    supplemental_detections: List[dict] | None = None,
) -> Dict[str, Any]:
    """为每个组件生成 ordered pin predictions.

    默认主路径:
    - top 整图 full-image pose
    - 关联回 S1 检测框
    - 产出与旧 S1.5 相同的组件 pin JSON 外壳
    """
    decoded = decode_images_b64(images_b64, logger=logger, stage_name="S1.5")
    summary = decode_summary(decoded)
    view_ids = _view_ids_from_images(images_b64)
    top_item = next((item for item in decoded if item["view_id"] == "top" and item["decoded"]), None)

    if _should_use_full_image_pose(pin_detector=pin_detector, top_item=top_item):
        return _run_pin_detect_full_image_pose(
            detections=detections,
            decoded=decoded,
            summary=summary,
            view_ids=view_ids,
            pin_detector=pin_detector,
        )

    return _run_pin_detect_legacy(
        detections=detections,
        decoded=decoded,
        summary=summary,
        view_ids=view_ids,
        pin_detector=pin_detector,
        supplemental_detections=supplemental_detections,
    )


def _should_use_full_image_pose(*, pin_detector: PinRoiDetector, top_item: dict | None) -> bool:
    if top_item is None or top_item.get("image") is None:
        return False
    if getattr(pin_detector, "backend_type", "") != "yolo_pose":
        return False
    return getattr(pin_detector, "model", None) is not None


def _run_pin_detect_full_image_pose(
    *,
    detections: List[dict],
    decoded: List[dict],
    summary: dict[str, Any],
    view_ids: List[str],
    pin_detector: PinRoiDetector,
) -> Dict[str, Any]:
    t0 = time.time()
    top_item = next((item for item in decoded if item["view_id"] == "top" and item["decoded"]), None)
    top_image = top_item["image"] if top_item else None
    if top_image is None:
        return {
            "interface_version": "component_pin_detect_v1",
            "pin_detector_backend": pin_detector.backend_type,
            "pin_detector_mode": "unavailable",
            "pin_detector_contract": dict(getattr(pin_detector, "model_contract", {}) or {}),
            "side_roi_assoc_backend": "not_applicable_full_image_pose",
            "components": [],
            **summary,
            "duration_ms": (time.time() - t0) * 1000,
        }

    pose_instances = _load_full_image_pose_instances(image=top_image, pin_detector=pin_detector)
    components = _build_components_from_full_pose(
        detections=detections,
        pose_instances=pose_instances,
        view_ids=view_ids,
        image_shape=top_image.shape[:2],
        pin_detector=pin_detector,
    )

    return {
        "interface_version": "component_pin_detect_v1",
        "pin_detector_backend": pin_detector.backend_type,
        "pin_detector_mode": "full_image_model",
        "pin_detector_contract": dict(getattr(pin_detector, "model_contract", {}) or {}),
        "side_roi_assoc_backend": "not_applicable_full_image_pose",
        "components": components,
        **summary,
        "duration_ms": (time.time() - t0) * 1000,
    }


def _run_pin_detect_legacy(
    *,
    detections: List[dict],
    decoded: List[dict],
    summary: dict[str, Any],
    view_ids: List[str],
    pin_detector: PinRoiDetector,
    supplemental_detections: List[dict] | None = None,
) -> Dict[str, Any]:
    t0 = time.time()
    roi_resolver = SideViewRoiResolver()

    counters: Dict[str, int] = {}
    components: List[dict] = []
    for det in detections:
        component_type = normalize_component_type(str(det.get("component_type") or det.get("class_name") or "UNKNOWN"))
        component_id = det.get("component_id") or _next_component_id(component_type, counters)
        package_type = str(det.get("package_type") or default_package_type(component_type))
        bbox = tuple(det.get("bbox") or (0, 0, 0, 0))
        orientation = float(det.get("orientation", 0.0))
        obb_corners = det.get("obb_corners")
        rois_by_view = _build_rois_by_view(
            decoded,
            bbox,
            component_type=component_type,
            package_type=package_type,
            orientation=orientation,
            obb_corners=obb_corners,
            supplemental_detections=supplemental_detections,
            component_detection=det,
            roi_resolver=roi_resolver,
        )

        pin_schema_id = default_pin_schema_id(component_type, package_type)
        component = {
            "component_id": component_id,
            "component_type": component_type,
            "class_name": component_type,
            "package_type": package_type,
            "pin_schema_id": pin_schema_id,
            "input_pin_detect_interface_version": "component_pin_detect_v1",
            "input_detection_interface_version": det.get("input_detection_interface_version") or "component_detect_v1",
            "part_subtype": det.get("part_subtype") or "",
            "symmetry_group": det.get("symmetry_group") or default_symmetry_group(component_type),
            "bbox": list(bbox),
            "confidence": float(det.get("confidence", 1.0)),
            "orientation": orientation,
        }

        predictions_by_view: Dict[str, List[dict]] = {}
        for view_id in view_ids:
            roi_spec = rois_by_view.get(view_id) or {}
            predictions, roi_spec = _predict_with_adaptive_roi(
                pin_detector=pin_detector,
                component_id=component_id,
                component_type=component_type,
                package_type=package_type,
                pin_schema_id=pin_schema_id,
                component_confidence=float(det.get("confidence", 1.0)),
                view_id=view_id,
                base_roi_spec=roi_spec,
                image_item=next((item for item in decoded if item["view_id"] == view_id), None),
                component_detection=det,
                bbox=bbox,
                orientation=orientation,
                obb_corners=obb_corners,
            )
            rois_by_view[view_id] = roi_spec
            roi_available = bool(roi_spec.get("image") is not None)
            predictions_by_view[view_id] = [
                {
                    "pin_id": pred.pin_id,
                    "pin_name": pred.pin_name,
                    "keypoint": [float(pred.keypoint[0]), float(pred.keypoint[1])] if pred.keypoint else None,
                    "visibility": pred.visibility if roi_available else 0,
                    "confidence": float(pred.confidence) if roi_available else 0.0,
                    "source": pred.source if roi_available else "unavailable",
                    "metadata": {
                        **dict(pred.metadata),
                        "roi_source": roi_spec.get("source", "unavailable"),
                    },
                }
                for pred in predictions
            ]

        component["pins"] = _merge_predictions_by_view(
            predictions_by_view=predictions_by_view,
            view_ids=view_ids,
        )
        top_roi = rois_by_view.get("top") or {}
        component["roi"] = {
            "offset": list(top_roi.get("offset") or [0, 0]),
            "shape": list(top_roi.get("shape") or [0, 0]),
            "source": top_roi.get("source", "unavailable"),
            "crop_source": top_roi.get("crop_source", "unavailable"),
            "crop_profile": top_roi.get("crop_profile", "none"),
            "crop_bounds": top_roi.get("crop_bounds"),
            "body_bbox": top_roi.get("body_bbox"),
            "body_size": top_roi.get("body_size"),
            "roi_size": top_roi.get("roi_size"),
            "scale_multiplier": top_roi.get("scale_multiplier", 1.0),
            "retry_attempts": top_roi.get("retry_attempts", 1),
        }
        component["roi_by_view"] = {
            view_id: {
                "offset": list((rois_by_view.get(view_id) or {}).get("offset") or [0, 0]),
                "shape": list((rois_by_view.get(view_id) or {}).get("shape") or [0, 0]),
                "source": (rois_by_view.get(view_id) or {}).get("source", "unavailable"),
                "crop_source": (rois_by_view.get(view_id) or {}).get("crop_source", "unavailable"),
                "crop_profile": (rois_by_view.get(view_id) or {}).get("crop_profile", "none"),
                "crop_bounds": (rois_by_view.get(view_id) or {}).get("crop_bounds"),
                "body_bbox": (rois_by_view.get(view_id) or {}).get("body_bbox"),
                "body_size": (rois_by_view.get(view_id) or {}).get("body_size"),
                "roi_size": (rois_by_view.get(view_id) or {}).get("roi_size"),
                "scale_multiplier": (rois_by_view.get(view_id) or {}).get("scale_multiplier", 1.0),
                "retry_attempts": (rois_by_view.get(view_id) or {}).get("retry_attempts", 1),
                "association": (rois_by_view.get(view_id) or {}).get("association") or {},
                "available": bool((rois_by_view.get(view_id) or {}).get("image") is not None),
            }
            for view_id in view_ids
        }
        component["pin_detector"] = {
            "interface_version": pin_detector.interface_version,
            "backend_type": pin_detector.backend_type,
            "backend_mode": pin_detector.backend_mode,
        }
        components.append(component)

    return {
        "interface_version": "component_pin_detect_v1",
        "pin_detector_backend": pin_detector.backend_type,
        "pin_detector_mode": pin_detector.backend_mode,
        "pin_detector_contract": dict(getattr(pin_detector, "model_contract", {}) or {}),
        "side_roi_assoc_backend": roi_resolver.interface_version,
        "components": components,
        **summary,
        "duration_ms": (time.time() - t0) * 1000,
    }


def _view_ids_from_images(images_b64: List[str]) -> List[str]:
    defaults = ["top", "left_front", "right_front"]
    if not images_b64:
        return ["top"]
    view_ids = defaults[: len(images_b64)]
    if len(images_b64) > len(defaults):
        for idx in range(len(defaults), len(images_b64)):
            view_ids.append(f"aux_view_{idx - len(defaults) + 1}")
    return view_ids


def _load_full_image_pose_instances(
    *,
    image: np.ndarray,
    pin_detector: PinRoiDetector,
) -> list[PoseInstance]:
    results = pin_detector.model(image, verbose=False, device=pin_detector.device)  # type: ignore[union-attr]
    if not results:
        return []
    first = results[0]
    boxes = getattr(first, "boxes", None)
    keypoints = getattr(first, "keypoints", None)
    if boxes is None or keypoints is None or getattr(keypoints, "xy", None) is None:
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


def _bbox_orientation(bbox: list[float]) -> str:
    x1, y1, x2, y2 = bbox
    return "horizontal" if (x2 - x1) >= (y2 - y1) else "vertical"


def _keypoint_orientation(points: list[tuple[float, float] | None]) -> str | None:
    valid = [p for p in points if p is not None]
    if len(valid) < 2:
        return None
    p1, p2 = valid[0], valid[-1]
    return "horizontal" if abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1]) else "vertical"


def _point_in_bbox(point: tuple[float, float] | None, bbox: list[float]) -> bool:
    if point is None:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _expand_bbox(bbox: list[float], pad_ratio: float = 0.22, min_pad: float = 12.0) -> list[float]:
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    pad_x = max(min_pad, w * pad_ratio)
    pad_y = max(min_pad, h * pad_ratio)
    return [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]


def _keypoints_inside_ratio(points: list[tuple[float, float] | None], bbox: list[float]) -> float:
    valid = [p for p in points if p is not None]
    if not valid:
        return 0.0
    inside = sum(1 for p in valid if _point_in_bbox(p, bbox))
    return inside / len(valid)


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


def _match_pose_instance(det: dict, pose_instances: list[PoseInstance]) -> PoseInstance | None:
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
        if not inst.used and inst.component_type == component_type
    ]
    candidates = typed_candidates or [inst for inst in pose_instances if not inst.used]
    best: tuple[float, PoseInstance] | None = None
    for inst in candidates:
        score = _score(inst)
        if best is None or score > best[0]:
            best = (score, inst)
    if best is None:
        return None
    best[1].used = True
    return best[1]


def _aligned_keypoints(component_type: str, keypoints: list[tuple[float, float] | None], bbox: list[float]) -> list[tuple[float, float] | None]:
    if len(keypoints) < 2 or keypoints[0] is None or keypoints[1] is None:
        return list(keypoints)
    if component_type not in {"Resistor", "CapacitorCeramic", "Diode", "LED", "Wire", "CapacitorElectrolytic"}:
        return list(keypoints)
    p1, p2 = keypoints[0], keypoints[1]
    if _bbox_orientation(bbox) == "horizontal":
        ordered = [p1, p2] if p1[0] <= p2[0] else [p2, p1]
    else:
        ordered = [p1, p2] if p1[1] <= p2[1] else [p2, p1]
    result = list(keypoints)
    result[0], result[1] = ordered[0], ordered[1]
    return result


def _build_components_from_full_pose(
    *,
    detections: list[dict],
    pose_instances: list[PoseInstance],
    view_ids: list[str],
    image_shape: tuple[int, int],
    pin_detector: PinRoiDetector,
) -> list[dict]:
    components: list[dict] = []
    unavailable_views = [vid for vid in view_ids if vid != "top"]
    for det in detections:
        component_type = normalize_component_type(str(det.get("component_type") or det.get("class_name") or "UNKNOWN"))
        package_type = str(det.get("package_type") or default_package_type(component_type))
        pin_schema_id = default_pin_schema_id(component_type, package_type)
        pin_count = default_pin_count(component_type, package_type)
        pin_names = default_pin_names(component_type, pin_count)
        matched = _match_pose_instance(det, pose_instances)
        aligned_points = _aligned_keypoints(
            component_type,
            matched.keypoints if matched else [],
            list(det.get("bbox") or [0, 0, 0, 0]),
        )

        pins = []
        for idx, pin_name in enumerate(pin_names, start=1):
            kp = aligned_points[idx - 1] if matched and idx - 1 < len(aligned_points) else None
            keypoints_by_view = {vid: None for vid in view_ids}
            visibility_by_view = {vid: 0 for vid in view_ids}
            score_by_view = {vid: 0.0 for vid in view_ids}
            source_by_view = {vid: "unavailable" for vid in view_ids}
            per_view = {vid: {} for vid in unavailable_views}
            if kp is not None:
                keypoints_by_view["top"] = [float(kp[0]), float(kp[1])]
                visibility_by_view["top"] = 2
                score_by_view["top"] = float(det.get("confidence", 1.0))
                source_by_view["top"] = "model"
            per_view["top"] = {
                "backend_type": pin_detector.backend_type,
                "backend_mode": "full_image_model",
                "interface_version": "full_image_pose_v1",
                "roi_source": "full_image_pose",
                **(matched.parse_meta if matched else {}),
            }
            pins.append(
                {
                    "pin_id": idx,
                    "pin_name": pin_name,
                    "keypoints_by_view": keypoints_by_view,
                    "visibility_by_view": visibility_by_view,
                    "score_by_view": score_by_view,
                    "source_by_view": source_by_view,
                    "confidence": float(det.get("confidence", 1.0)) if kp is not None else 0.0,
                    "source": "model" if kp is not None else "unavailable",
                    "metadata": {"per_view": per_view},
                }
            )

        roi_by_view = {
            "top": {
                "offset": [0, 0],
                "shape": [int(image_shape[0]), int(image_shape[1])],
                "source": "full_image_pose",
                "crop_source": "full_image_pose",
                "crop_profile": "none",
                "crop_bounds": None,
                "body_bbox": list(det.get("bbox") or [0, 0, 0, 0]),
                "body_size": None,
                "roi_size": [int(image_shape[1]), int(image_shape[0])],
                "scale_multiplier": 1.0,
                "retry_attempts": 0,
                "association": {},
                "available": True,
            }
        }
        for view_id in unavailable_views:
            roi_by_view[view_id] = {
                "offset": [0, 0],
                "shape": [0, 0],
                "source": "unavailable",
                "crop_source": "unavailable",
                "crop_profile": "none",
                "crop_bounds": None,
                "body_bbox": None,
                "body_size": None,
                "roi_size": [0, 0],
                "scale_multiplier": 1.0,
                "retry_attempts": 0,
                "association": {},
                "available": False,
            }

        components.append(
            {
                "component_id": det.get("component_id"),
                "component_type": component_type,
                "class_name": component_type,
                "package_type": package_type,
                "pin_schema_id": pin_schema_id,
                "input_pin_detect_interface_version": "component_pin_detect_v1",
                "input_detection_interface_version": det.get("input_detection_interface_version") or "component_detect_v1",
                "part_subtype": det.get("part_subtype") or "",
                "symmetry_group": det.get("symmetry_group") or default_symmetry_group(component_type),
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
                "roi": roi_by_view["top"],
                "roi_by_view": roi_by_view,
                "pin_detector": {
                    "interface_version": pin_detector.interface_version,
                    "backend_type": pin_detector.backend_type,
                    "backend_mode": "full_image_model",
                },
            }
        )
    return components


def _build_rois_by_view(
    decoded_images: List[dict],
    bbox: tuple[int, int, int, int],
    *,
    component_type: str,
    package_type: str,
    orientation: float,
    obb_corners: Any,
    supplemental_detections: List[dict] | None,
    component_detection: dict,
    roi_resolver: SideViewRoiResolver,
) -> Dict[str, Dict[str, Any]]:
    rois: Dict[str, Dict[str, Any]] = {}
    for item in decoded_images:
        view_id = item["view_id"]
        image = item["image"]
        if image is None:
            rois[view_id] = {
                "image": None,
                "offset": (0, 0),
                "shape": [0, 0],
                "source": "unavailable",
                "crop_source": "unavailable",
                "crop_profile": "none",
                "crop_bounds": None,
            }
            continue
        view_bbox = bbox
        assoc_source = "detected_bbox" if view_id == "top" else "shared_bbox_fallback"
        assoc_meta: dict[str, Any] = {}
        if view_id != "top":
            association = roi_resolver.resolve(
                component_detection=component_detection,
                view_id=view_id,
                supplemental_detections=supplemental_detections,
            )
            if association is not None:
                view_bbox = association.bbox
                assoc_source = association.source
                assoc_meta = dict(association.metadata or {})
                assoc_meta["matched"] = association.matched
                assoc_meta["candidate_id"] = association.candidate_id
                assoc_meta["association_confidence"] = association.confidence
        roi_image, roi_offset, roi_meta = crop_component_roi(
            image,
            view_bbox,
            component_type=component_type,
            package_type=package_type,
            orientation=orientation,
            obb_corners=obb_corners if view_id == "top" else None,
            view_id=view_id,
        )
        rois[view_id] = {
            "image": roi_image,
            "offset": roi_offset,
            "shape": list(roi_image.shape[:2]) if roi_image is not None else [0, 0],
            "bbox": list(view_bbox),
            "source": assoc_source,
            "crop_source": roi_meta.get("source", "package_profile_crop"),
            "crop_profile": roi_meta.get("profile_name", "generic"),
            "crop_bounds": roi_meta.get("bounds"),
            "body_bbox": roi_meta.get("body_bbox"),
            "body_size": roi_meta.get("body_size"),
            "roi_size": roi_meta.get("roi_size"),
            "association": assoc_meta,
        }
    return rois


def _predict_with_adaptive_roi(
    *,
    pin_detector: PinRoiDetector,
    component_id: str,
    component_type: str,
    package_type: str,
    pin_schema_id: str,
    component_confidence: float,
    view_id: str,
    base_roi_spec: dict[str, Any],
    image_item: dict | None,
    component_detection: dict,
    bbox: tuple[int, int, int, int],
    orientation: float,
    obb_corners: Any,
) -> tuple[list, dict[str, Any]]:
    image = (image_item or {}).get("image")
    if image is None:
        predictions = pin_detector.predict_component_pins(
            component_id=component_id,
            component_type=component_type,
            package_type=package_type,
            pin_schema_id=pin_schema_id,
            roi_image=None,
            roi_offset=(0, 0),
            view_id=view_id,
            confidence=component_confidence,
        )
        base_roi_spec = dict(base_roi_spec)
        base_roi_spec["retry_scale"] = None
        base_roi_spec["retry_attempts"] = 0
        return predictions, base_roi_spec

    retry_scales = ROI_RETRY_SCALES.get(view_id, [1.0, 1.2, 1.45])
    assoc_source = str(base_roi_spec.get("source", "detected_bbox"))
    crop_source = str(base_roi_spec.get("crop_source", "package_profile_crop"))
    best_predictions = None
    best_roi_spec = dict(base_roi_spec)

    view_bbox = bbox
    if base_roi_spec.get("bbox") and len(base_roi_spec.get("bbox")) == 4:
        view_bbox = tuple(int(v) for v in base_roi_spec.get("bbox"))
    if view_id != "top" and assoc_source != "shared_bbox_fallback":
        association = (base_roi_spec.get("association") or {})
        candidate_bbox = association.get("candidate_bbox")
        if candidate_bbox and len(candidate_bbox) == 4:
            view_bbox = tuple(int(v) for v in candidate_bbox)

    for attempt, scale in enumerate(retry_scales, start=1):
        roi_image, roi_offset, roi_meta = crop_component_roi(
            image,
            view_bbox,
            component_type=component_type,
            package_type=package_type,
            orientation=orientation,
            obb_corners=obb_corners if view_id == "top" else None,
            view_id=view_id,
            scale_multiplier=scale,
        )
        predictions = pin_detector.predict_component_pins(
            component_id=component_id,
            component_type=component_type,
            package_type=package_type,
            pin_schema_id=pin_schema_id,
            roi_image=roi_image,
            roi_offset=tuple(roi_offset or (0, 0)),
            view_id=view_id,
            confidence=component_confidence,
        )
        roi_spec = dict(base_roi_spec)
        roi_spec.update(
            {
                "image": roi_image,
                "offset": roi_offset,
                "shape": list(roi_image.shape[:2]) if roi_image is not None else [0, 0],
                "source": assoc_source,
                "crop_source": roi_meta.get("source", crop_source),
                "crop_profile": roi_meta.get("profile_name", "generic"),
                "crop_bounds": roi_meta.get("bounds"),
                "body_bbox": roi_meta.get("body_bbox"),
                "body_size": roi_meta.get("body_size"),
                "roi_size": roi_meta.get("roi_size"),
                "scale_multiplier": float(scale),
                "retry_scale": float(scale),
                "retry_attempts": attempt,
            }
        )
        best_predictions = predictions
        best_roi_spec = roi_spec
        if _predictions_are_usable(predictions) and not _predictions_need_more_context(
            predictions,
            roi_spec,
            margin_px=ROI_EDGE_MARGIN_PX,
        ):
            break

    return best_predictions or [], best_roi_spec


def _predictions_are_usable(predictions: list) -> bool:
    if not predictions:
        return False
    visible = 0
    model_visible = 0
    for pred in predictions:
        if getattr(pred, "keypoint", None) is not None:
            visible += 1
        if getattr(pred, "source", "") == "model" and getattr(pred, "keypoint", None) is not None:
            model_visible += 1
    return visible >= 2 and model_visible >= max(1, visible // 2)


def _predictions_need_more_context(
    predictions: list,
    roi_spec: dict[str, Any],
    *,
    margin_px: float,
) -> bool:
    crop_bounds = roi_spec.get("crop_bounds")
    if not crop_bounds or len(crop_bounds) != 4:
        return False

    x1, y1, x2, y2 = [float(v) for v in crop_bounds]
    for pred in predictions:
        keypoint = getattr(pred, "keypoint", None)
        if keypoint is None:
            continue
        source = str(getattr(pred, "source", "") or "")
        if source in {"unavailable", "heuristic_fallback"}:
            continue

        px, py = float(keypoint[0]), float(keypoint[1])
        margin = min(px - x1, x2 - px, py - y1, y2 - py)
        if margin < float(margin_px):
            return True
    return False


def _merge_predictions_by_view(
    *,
    predictions_by_view: Dict[str, List[dict]],
    view_ids: List[str],
) -> List[dict]:
    merged: Dict[int, dict] = {}
    for view_id in view_ids:
        for pred in predictions_by_view.get(view_id, []):
            pin_id = int(pred["pin_id"])
            pin_entry = merged.setdefault(
                pin_id,
                {
                    "pin_id": pin_id,
                    "pin_name": pred["pin_name"],
                    "keypoints_by_view": {vid: None for vid in view_ids},
                    "visibility_by_view": {vid: 0 for vid in view_ids},
                    "score_by_view": {vid: 0.0 for vid in view_ids},
                    "source_by_view": {vid: "unavailable" for vid in view_ids},
                    "confidence": 0.0,
                    "source": "unavailable",
                    "metadata": {"per_view": {}},
                },
            )
            pin_entry["keypoints_by_view"][view_id] = pred["keypoint"]
            pin_entry["visibility_by_view"][view_id] = int(pred["visibility"])
            pin_entry["score_by_view"][view_id] = float(pred["confidence"])
            pin_entry["source_by_view"][view_id] = str(pred["source"])
            pin_entry["metadata"]["per_view"][view_id] = dict(pred.get("metadata") or {})

    ordered = []
    for pin_id in sorted(merged):
        item = merged[pin_id]
        scores = [score for score in item["score_by_view"].values() if score > 0]
        item["confidence"] = max(scores) if scores else 0.0
        if any(source == "model" for source in item["source_by_view"].values()):
            item["source"] = "model"
        elif any(source == "heuristic_fallback" for source in item["source_by_view"].values()):
            item["source"] = "heuristic_fallback"
        ordered.append(item)
    return ordered


def _next_component_id(component_type: str, counters: Dict[str, int]) -> str:
    normalized = normalize_component_type(component_type)
    prefix = component_id_prefix(normalized)
    counters[normalized] = counters.get(normalized, 0) + 1
    return f"{prefix}{counters[normalized]}"
