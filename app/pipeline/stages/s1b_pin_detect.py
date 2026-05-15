"""
Stage 1.5: Full-image component pin detection.

当前正式主路径:
- 不再按单组件裁切小图后做 pin 识别
- 改为整图 full-image YOLO-Pose
- 再按类别 + bbox 几何把 pose 实例关联回 S1 组件检测结果
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, List

import cv2
import numpy as np

from app.pipeline.vision.pin_model import PinRoiDetector, _parse_model_keypoints
from app.pipeline.vision.image_io import decode_images_b64, decode_summary
from app.pipeline.vision.label_mapping import (
    default_pin_count,
    default_pin_names,
    normalize_component_type,
)
from app.pipeline.vision.pin_schema import (
    default_package_type,
    default_pin_schema_id,
    default_symmetry_group,
)
from app.pipeline.vision.transistor_polarity import infer_transistor_pin_roles

logger = logging.getLogger(__name__)


POTENTIOMETER_PIN_NAMES = ["terminal_a", "wiper", "terminal_b"]


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
    calibrator: Any | None = None,
) -> Dict[str, Any]:
    """为每个组件生成 ordered pin predictions.

    默认主路径:
    - top 整图 full-image pose
    - 关联回 S1 检测框
    - 产出与旧 S1.5 相同的组件 pin JSON 外壳

    IC 例外: 不依赖引脚模型, 直接按 bbox + e/f 行约束生成 8/14 个引脚
    (参见 _build_ic_geometry_pins). 可选 calibrator 仅用于补 board_2d_point,
    最终 hole_id 仍交给 S2 映射。
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
            calibrator=calibrator,
        )

    return _run_pin_detect_unavailable(
        detections=detections,
        decoded=decoded,
        summary=summary,
        view_ids=view_ids,
        pin_detector=pin_detector,
        calibrator=calibrator,
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
    calibrator: Any | None = None,
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
        top_image=top_image,
        image_shape=top_image.shape[:2],
        pin_detector=pin_detector,
        calibrator=calibrator,
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


def _run_pin_detect_unavailable(
    *,
    detections: List[dict],
    decoded: List[dict],
    summary: dict[str, Any],
    view_ids: List[str],
    pin_detector: PinRoiDetector,
    calibrator: Any | None = None,
) -> Dict[str, Any]:
    """Return schema-compatible pin shells without running model inference."""
    t0 = time.time()
    top_decoded = next((item for item in decoded if item["view_id"] == "top" and item.get("decoded")), None)
    top_image = top_decoded["image"] if top_decoded and top_decoded.get("image") is not None else None
    top_image_shape = top_image.shape[:2] if top_image is not None else (0, 0)
    components = _build_components_from_full_pose(
        detections=detections,
        pose_instances=[],
        view_ids=view_ids,
        top_image=top_image,
        image_shape=top_image_shape,
        pin_detector=pin_detector,
        calibrator=calibrator,
        pin_detector_mode="unavailable",
    )

    return {
        "interface_version": "component_pin_detect_v1",
        "pin_detector_backend": pin_detector.backend_type,
        "pin_detector_mode": "unavailable",
        "pin_detector_contract": dict(getattr(pin_detector, "model_contract", {}) or {}),
        "side_roi_assoc_backend": "removed",
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
    if component_type == "Potentiometer" and not typed_candidates:
        return None
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


def _potentiometer_projection_axis(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 2:
        return (1.0, 0.0)
    arr = np.asarray(points, dtype=np.float32)
    centered = arr - arr.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        ax, ay = float(axis[0]), float(axis[1])
    except Exception:
        p0, p1 = points[0], points[-1]
        ax, ay = float(p1[0] - p0[0]), float(p1[1] - p0[1])
    norm = max((ax * ax + ay * ay) ** 0.5, 1e-6)
    ax, ay = ax / norm, ay / norm
    if ax < 0 or (abs(ax) < 1e-6 and ay < 0):
        ax, ay = -ax, -ay
    return (ax, ay)


def _project_point(point: tuple[float, float], axis: tuple[float, float]) -> float:
    return float(point[0] * axis[0] + point[1] * axis[1])


_POT_LETTERS = list("abcdefghij")
# Within a half (a-e or f-j), 3 adjacent letters: indices 0-2, 1-3, 2-4 / 5-7, 6-8, 7-9.
_POT_VERTICAL_LETTER_STARTS = (0, 1, 2, 5, 6, 7)


def _semanticize_potentiometer_pins(
    *,
    pins: list[dict],
    bbox: list[float],
    view_id: str = "top",
    calibrator: Any | None = None,
    top_image: Any | None = None,
) -> list[dict]:
    """Rename/reorder POT pins as terminal_a/wiper/terminal_b and snap to 3-collinear holes.

    物理约束: POT 三个引脚必须同时落在面包板的一条线上 ——
    - 横插 (horizontal): 三脚同一字母行 (a-j 中某一个), 跨 3 个相邻数字列;
    - 竖插 (vertical):  三脚同一数字列, 跨同一半 (a-e 或 f-j) 内 3 个相邻字母行.

    当 calibrator 就绪时, 将模型 keypoint (缺失则用 bbox 几何补齐) 投到 board plane,
    枚举所有合法三孔三元组, 选总平方距离最小的一组, 把三脚的 keypoint_by_view
    重写为候选孔在 frame pixel 上的精确位置.

    当 calibrator 不可用时 -> 拒绝输出 pin 坐标 (与 IC 同一策略), 仅保留 pin schema
    + degraded metadata, 让下游跳过此元件而不是吸附到错孔.
    """
    if len(pins) != 3:
        return pins

    if not _ensure_calibrator_ready(calibrator=calibrator, top_image=top_image):
        return _pot_refused_output(pins=pins, view_id=view_id, reason="calibrator_unavailable")

    detected_pixel_by_idx = _pot_extract_view_pixels(
        pins=pins, view_id=view_id, bbox=bbox, calibrator=calibrator, top_image=top_image,
    )
    if detected_pixel_by_idx is None:
        return _pot_refused_output(pins=pins, view_id=view_id, reason="no_visible_potentiometer_keypoints")

    detected_board: list[tuple[int, tuple[float, float]]] = []
    for src_idx, pixel in detected_pixel_by_idx:
        try:
            bp = calibrator.frame_pixel_to_board_point(float(pixel[0]), float(pixel[1]))
        except Exception as exc:
            logger.debug("S1.5 POT geometry: frame_pixel_to_board_point failed: %s", exc)
            return _pot_refused_output(pins=pins, view_id=view_id, reason="calibrator_projection_failed")
        detected_board.append((src_idx, (float(bp[0]), float(bp[1]))))

    bbox_footprint = _pot_bbox_board_footprint(bbox=bbox, calibrator=calibrator)
    snap = _pot_best_snap(
        detected=detected_board,
        calibrator=calibrator,
        bbox_footprint=bbox_footprint,
    )
    if snap is None:
        return _pot_refused_output(pins=pins, view_id=view_id, reason="no_legal_triplet")

    return _pot_apply_snap(
        pins=pins,
        view_id=view_id,
        snap=snap,
        calibrator=calibrator,
        keypoint_sources={src_idx: src for src_idx, src in _pot_keypoint_sources(pins, view_id).items()},
    )


def _ensure_calibrator_ready(*, calibrator: Any | None, top_image: Any | None) -> bool:
    if calibrator is None:
        return False
    if getattr(calibrator, "is_grid_ready", False):
        return True
    if top_image is None or not hasattr(calibrator, "ensure_calibrated"):
        return False
    try:
        calibrator.ensure_calibrated(top_image)
    except Exception as exc:
        logger.debug("S1.5 POT geometry: ensure_calibrated failed: %s", exc)
        return False
    return bool(getattr(calibrator, "is_grid_ready", False))


def _pot_keypoint_sources(pins: list[dict], view_id: str) -> dict[int, str]:
    """Map original pin index → source label ('model' / 'potentiometer_bbox_fallback' / 'unavailable')."""
    sources: dict[int, str] = {}
    for idx, pin in enumerate(pins):
        if _pin_has_view_keypoint(pin, view_id):
            src = (pin.get("source_by_view") or {}).get(view_id) or pin.get("source") or "model"
            sources[idx] = str(src)
        else:
            sources[idx] = "potentiometer_bbox_fallback"
    return sources


def _pot_extract_view_pixels(
    *,
    pins: list[dict],
    view_id: str,
    bbox: list[float],
    calibrator: Any | None = None,
    top_image: Any | None = None,
) -> list[tuple[int, tuple[float, float]]] | None:
    """Return [(pin_idx, (px, py))] x3 using model keypoints where present, bbox fallback otherwise.

    Returns None when there are zero visible keypoints AND bbox is degenerate
    (caller should refuse output in that case).
    """
    valid: list[tuple[int, tuple[float, float]]] = []
    missing_idxs: list[int] = []
    for idx, pin in enumerate(pins):
        point = (pin.get("keypoints_by_view") or {}).get(view_id)
        if point and len(point) >= 2:
            valid.append((idx, (float(point[0]), float(point[1]))))
        else:
            missing_idxs.append(idx)

    if not valid and not _bbox_is_usable(bbox):
        return None

    if not missing_idxs:
        return valid

    fallback = _potentiometer_bbox_fallback_points(bbox, calibrator=calibrator, top_image=top_image)
    fallback_points = [fallback["terminal_a"], fallback["wiper"], fallback["terminal_b"]]
    used = [False, False, False]
    if valid:
        # Greedily assign each valid keypoint to its nearest fallback slot,
        # so the remaining slots are used for the *missing* original indices.
        for _, pt in valid:
            dists = [
                ((pt[0] - fp[0]) ** 2 + (pt[1] - fp[1]) ** 2, slot)
                for slot, fp in enumerate(fallback_points)
                if not used[slot]
            ]
            if not dists:
                break
            dists.sort()
            used[dists[0][1]] = True
    free_slots = [slot for slot, taken in enumerate(used) if not taken]
    for missing_idx, slot in zip(missing_idxs, free_slots):
        valid.append((missing_idx, fallback_points[slot]))

    valid.sort(key=lambda item: item[0])
    return valid


def _bbox_is_usable(bbox: list[float]) -> bool:
    try:
        x1, y1, x2, y2 = [float(v) for v in (bbox or [0.0, 0.0, 0.0, 0.0])[:4]]
    except (TypeError, ValueError):
        return False
    return (x2 - x1) > 1.0 and (y2 - y1) > 1.0


def _pot_enumerate_triplets(
    *,
    calibrator: Any,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Yield (orientation, [logic_loc x3]) for every legal 3-collinear hole triplet."""
    row_coords = getattr(calibrator, "row_coords", None)
    if row_coords is None or len(row_coords) < 3:
        return []
    n_digits = int(len(row_coords))
    out: list[tuple[str, list[tuple[str, str]]]] = []
    # Horizontal: same letter row, 3 adjacent digit columns.
    for letter in _POT_LETTERS:
        for d in range(1, n_digits - 1):
            out.append(("horizontal", [(str(d), letter), (str(d + 1), letter), (str(d + 2), letter)]))
    # Vertical: same digit column, 3 adjacent letters within same half (a-e or f-j).
    for d in range(1, n_digits + 1):
        for start in _POT_VERTICAL_LETTER_STARTS:
            out.append((
                "vertical",
                [
                    (str(d), _POT_LETTERS[start]),
                    (str(d), _POT_LETTERS[start + 1]),
                    (str(d), _POT_LETTERS[start + 2]),
                ],
            ))
    return out


def _pot_bbox_board_footprint(
    *,
    bbox: list[float],
    calibrator: Any,
) -> tuple[float, float, float, float] | None:
    """Project bbox 4 corners to board plane → axis-aligned (x_lo, y_lo, x_hi, y_hi).

    Returns ``None`` when the bbox is degenerate or the calibrator can't project.
    The 3296 trim-pot pins always sit *underneath* the body in plan view, so the
    legal 3-hole triplet must lie inside this footprint — only holes covered by
    the body bbox are physical pin candidates.
    """
    try:
        x1, y1, x2, y2 = [float(v) for v in (bbox or [0.0, 0.0, 0.0, 0.0])[:4]]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
        return None
    try:
        corners = [
            calibrator.frame_pixel_to_board_point(x1, y1),
            calibrator.frame_pixel_to_board_point(x2, y1),
            calibrator.frame_pixel_to_board_point(x1, y2),
            calibrator.frame_pixel_to_board_point(x2, y2),
        ]
    except Exception as exc:
        logger.debug("S1.5 POT footprint: frame_pixel_to_board_point failed: %s", exc)
        return None
    bxs = [float(c[0]) for c in corners]
    bys = [float(c[1]) for c in corners]
    return (min(bxs), min(bys), max(bxs), max(bys))


def _pot_best_snap(
    *,
    detected: list[tuple[int, tuple[float, float]]],
    calibrator: Any,
    bbox_footprint: tuple[float, float, float, float] | None = None,
) -> dict[str, Any] | None:
    """Find the legal 3-collinear hole triplet minimizing sum-sq distance to `detected` board points.

    When ``bbox_footprint`` is provided, candidates whose 3 holes all lie inside
    (with a half-pitch tolerance for bbox slop) are evaluated first; the
    unconstrained set is only used as a fallback if no triplet fits inside.
    This enforces the physical reality that 3296 trim-pot pins are hidden
    underneath the component body — visible adjacent holes are never the
    physical pin positions.

    Returns dict with:
      - orientation: "horizontal" | "vertical"
      - logic_locs: list[(digit, letter)] sorted along the candidate axis
      - board_points: list[(bx, by)] for those 3 holes, same order
      - sorted_detected_idx: [original pin idx] in the same projection order
      - cost_sq: total squared distance in board-plane units
      - body_constrained: True when the selected triplet sits inside bbox_footprint
    """
    if len(detected) != 3:
        return None
    triplets = _pot_enumerate_triplets(calibrator=calibrator)
    if not triplets:
        return None

    tol = 0.0
    if bbox_footprint is not None:
        pitch = float(getattr(calibrator, "representative_pitch_px", lambda: 10.0)())
        # Half-pitch slop absorbs slight detection-bbox under-tightness.
        tol = max(pitch * 0.5, 1.0)
        fx_lo, fy_lo, fx_hi, fy_hi = bbox_footprint

    inside_best: dict[str, Any] | None = None
    inside_cost = float("inf")
    any_best: dict[str, Any] | None = None
    any_cost = float("inf")

    for orientation, locs in triplets:
        cps: list[tuple[float, float]] = []
        ok = True
        for loc in locs:
            pt = calibrator.logic_to_board_point(loc)
            if pt is None:
                ok = False
                break
            cps.append((float(pt[0]), float(pt[1])))
        if not ok or len(cps) != 3:
            continue
        dx = cps[2][0] - cps[0][0]
        dy = cps[2][1] - cps[0][1]
        norm = (dx * dx + dy * dy) ** 0.5
        if norm < 1e-6:
            continue
        ax = dx / norm
        ay = dy / norm
        sorted_det = sorted(detected, key=lambda item: item[1][0] * ax + item[1][1] * ay)
        cost = 0.0
        for (_idx, bp), cp in zip(sorted_det, cps):
            ex = bp[0] - cp[0]
            ey = bp[1] - cp[1]
            cost += ex * ex + ey * ey
        record = {
            "orientation": orientation,
            "logic_locs": list(locs),
            "board_points": list(cps),
            "sorted_detected_idx": [item[0] for item in sorted_det],
            "cost_sq": cost,
        }
        if cost < any_cost:
            any_cost = cost
            any_best = record
        if bbox_footprint is not None:
            inside = all(
                fx_lo - tol <= cp[0] <= fx_hi + tol and fy_lo - tol <= cp[1] <= fy_hi + tol
                for cp in cps
            )
            if inside and cost < inside_cost:
                inside_cost = cost
                inside_best = record

    if inside_best is not None:
        inside_best = dict(inside_best)
        inside_best["body_constrained"] = True
        return inside_best
    if any_best is not None:
        any_best = dict(any_best)
        any_best["body_constrained"] = False
    return any_best


def _pot_apply_snap(
    *,
    pins: list[dict],
    view_id: str,
    snap: dict[str, Any],
    calibrator: Any,
    keypoint_sources: dict[int, str],
) -> list[dict]:
    """Rewrite the 3 pin entries to reflect the chosen 3-collinear hole triplet."""
    entries = [dict(pin) for pin in pins]
    sorted_idx: list[int] = snap["sorted_detected_idx"]
    logic_locs: list[tuple[str, str]] = snap["logic_locs"]
    board_points: list[tuple[float, float]] = snap["board_points"]
    orientation: str = snap["orientation"]
    cost_sq: float = float(snap.get("cost_sq", 0.0))

    # Frame-pixel positions of the 3 candidate holes.
    frame_pixels: list[tuple[float, float]] = []
    if hasattr(calibrator, "board_point_to_frame_pixel"):
        for bp in board_points:
            try:
                fp = calibrator.board_point_to_frame_pixel(float(bp[0]), float(bp[1]))
                frame_pixels.append((float(fp[0]), float(fp[1])))
            except Exception:
                frame_pixels.append((float(bp[0]), float(bp[1])))
    else:
        frame_pixels = [tuple(bp) for bp in board_points]

    out: list[dict] = []
    # ordered_role_idx[0] = wire to terminal_a, [1] = wiper, [2] = terminal_b.
    for slot, pin_name in enumerate(POTENTIOMETER_PIN_NAMES):
        src_idx = sorted_idx[slot]
        pin = dict(entries[src_idx])
        kx, ky = frame_pixels[slot]
        keypoints_by_view = dict(pin.get("keypoints_by_view") or {})
        visibility_by_view = dict(pin.get("visibility_by_view") or {})
        score_by_view = dict(pin.get("score_by_view") or {})
        source_by_view = dict(pin.get("source_by_view") or {})
        keypoints_by_view[view_id] = [kx, ky]
        visibility_by_view[view_id] = 2
        score_by_view[view_id] = max(float(score_by_view.get(view_id, 0.0) or 0.0), 0.75)
        source_by_view[view_id] = "potentiometer_board_logic"

        metadata = dict(pin.get("metadata") or {})
        metadata["potentiometer_role_source"] = "board_plane_3collinear_snap"
        metadata["potentiometer_source_pin_id"] = entries[src_idx].get("pin_id")
        metadata["potentiometer_input_source"] = keypoint_sources.get(src_idx, "unknown")
        metadata["pot_orientation"] = orientation
        metadata["pot_logic_slots"] = [list(loc) for loc in logic_locs]
        metadata["pot_snap_cost_sq"] = cost_sq
        metadata["pot_body_constrained"] = bool(snap.get("body_constrained", False))
        metadata["board_2d_point"] = [float(board_points[slot][0]), float(board_points[slot][1])]
        if orientation == "horizontal":
            metadata["row_lock"] = logic_locs[slot][1]
        else:
            metadata["column_lock"] = logic_locs[slot][0]

        pin["pin_id"] = slot + 1
        pin["pin_name"] = pin_name
        pin["pin_display_name"] = pin_name
        pin["keypoints_by_view"] = keypoints_by_view
        pin["visibility_by_view"] = visibility_by_view
        pin["score_by_view"] = score_by_view
        pin["source_by_view"] = source_by_view
        pin["confidence"] = max(float(pin.get("confidence", 0.0) or 0.0), 0.75)
        pin["source"] = "potentiometer_board_logic"
        pin["metadata"] = metadata
        out.append(pin)
    return out


def _pot_refused_output(*, pins: list[dict], view_id: str, reason: str) -> list[dict]:
    """Return 3 pin entries with no usable keypoint and a degraded reason.

    Used when the calibrator is unavailable so downstream stages know to
    skip the component instead of snapping to the wrong hole.
    """
    logger.warning("S1.5 POT geometry: refusing pin output (%s)", reason)
    out: list[dict] = []
    for slot, pin_name in enumerate(POTENTIOMETER_PIN_NAMES):
        src = dict(pins[slot]) if slot < len(pins) else {}
        metadata = dict(src.get("metadata") or {})
        metadata["potentiometer_role_source"] = "refused"
        metadata["potentiometer_role_degraded_reason"] = reason
        metadata.pop("board_2d_point", None)
        metadata.pop("pot_orientation", None)
        metadata.pop("pot_logic_slots", None)
        pin = {
            "pin_id": slot + 1,
            "pin_name": pin_name,
            "pin_display_name": pin_name,
            "polarity_role": src.get("polarity_role", "UNKNOWN"),
            "polarity_candidate_role": src.get("polarity_candidate_role", "UNKNOWN"),
            "keypoints_by_view": {view_id: None},
            "visibility_by_view": {view_id: 0},
            "score_by_view": {view_id: 0.0},
            "source_by_view": {view_id: "unavailable"},
            "confidence": 0.0,
            "source": "unavailable",
            "metadata": metadata,
        }
        out.append(pin)
    return out


def _pin_has_view_keypoint(pin: dict, view_id: str) -> bool:
    point = (pin.get("keypoints_by_view") or {}).get(view_id)
    return bool(point and len(point) >= 2)


def _potentiometer_bbox_fallback_points(
    bbox: list[float],
    *,
    calibrator: Any | None = None,
    top_image: Any | None = None,
) -> dict[str, tuple[float, float]]:
    """Fallback 3-pin positions when model keypoints are missing.

    With a calibrator: project bbox 4 corners to board plane, pick the dominant
    axis (digit vs letter), snap the bbox center to the nearest legal hole, and
    emit 3 adjacent legal hole positions (in frame pixels). These are already
    "legal 3-collinear holes" so the board-plane snap in
    ``_semanticize_potentiometer_pins`` finds them with cost ≈ 0.

    Without a calibrator: degrade to the old image-frame heuristic so callers
    that don't pass a calibrator (refused path, unit tests) keep working.
    """
    try:
        x1, y1, x2, y2 = [float(v) for v in (bbox or [0.0, 0.0, 0.0, 0.0])[:4]]
    except (TypeError, ValueError):
        x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    top_box_fallback = _potentiometer_top_box_fallback_points(
        bbox=(x1, y1, x2, y2),
        top_image=top_image,
        calibrator=calibrator,
    )
    if top_box_fallback is not None:
        return top_box_fallback

    board_fallback = _potentiometer_bbox_fallback_points_board(
        bbox=(x1, y1, x2, y2),
        calibrator=calibrator,
    )
    if board_fallback is not None:
        return board_fallback

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    if width >= height:
        y = y2 - height * 0.18
        return {
            "terminal_a": (x1 + width * 0.22, y),
            "wiper": (x1 + width * 0.50, y),
            "terminal_b": (x1 + width * 0.78, y),
        }
    x = x1 + width * 0.50
    return {
        "terminal_a": (x, y1 + height * 0.22),
        "wiper": (x, y1 + height * 0.50),
        "terminal_b": (x, y1 + height * 0.78),
    }


def _potentiometer_top_box_fallback_points(
    *,
    bbox: tuple[float, float, float, float],
    top_image: Any | None,
    calibrator: Any | None = None,
) -> dict[str, tuple[float, float]] | None:
    """Estimate POT pins from the visible top rectangle.

    The top-down view often hides the three metal legs. The blue top cap,
    however, remains visible. We rectify that cap to a rotated rectangle and
    place the three electrical slots along its long axis.
    """
    if top_image is None or not isinstance(top_image, np.ndarray) or top_image.size == 0:
        return None
    if top_image.ndim < 2:
        return None

    x1, y1, x2, y2 = bbox
    h, w = top_image.shape[:2]
    ix1 = max(0, int(np.floor(min(x1, x2))))
    iy1 = max(0, int(np.floor(min(y1, y2))))
    ix2 = min(w, int(np.ceil(max(x1, x2))))
    iy2 = min(h, int(np.ceil(max(y1, y2))))
    if ix2 - ix1 < 6 or iy2 - iy1 < 6:
        return None

    crop = top_image[iy1:iy2, ix1:ix2]
    rect = _detect_potentiometer_top_rect(crop)
    if rect is None:
        return None

    (cx, cy), (rw, rh), angle = rect
    if rw < 3.0 or rh < 3.0:
        return None

    theta = np.deg2rad(float(angle))
    if rw >= rh:
        ax, ay = float(np.cos(theta)), float(np.sin(theta))
        span = float(rw)
    else:
        ax, ay = float(-np.sin(theta)), float(np.cos(theta))
        span = float(rh)

    norm = max((ax * ax + ay * ay) ** 0.5, 1e-6)
    ax, ay = ax / norm, ay / norm
    gcx = float(ix1 + cx)
    gcy = float(iy1 + cy)

    spacing = _potentiometer_fallback_spacing_px(
        calibrator=calibrator,
        bbox=bbox,
        top_box_span=span,
    )
    points = [
        (gcx - ax * spacing, gcy - ay * spacing),
        (gcx, gcy),
        (gcx + ax * spacing, gcy + ay * spacing),
    ]
    points = _order_pot_points_by_image_axis(points)
    return {
        "terminal_a": points[0],
        "wiper": points[1],
        "terminal_b": points[2],
    }


def _potentiometer_fallback_spacing_px(
    *,
    calibrator: Any | None,
    bbox: tuple[float, float, float, float],
    top_box_span: float,
) -> float:
    """Return approximate adjacent-hole spacing for POT fallback points."""
    if calibrator is not None and hasattr(calibrator, "representative_pitch_px"):
        try:
            pitch = float(calibrator.representative_pitch_px())
            if np.isfinite(pitch) and pitch > 1.0:
                return pitch
        except Exception:
            pass
    x1, y1, x2, y2 = bbox
    bbox_major = max(abs(float(x2) - float(x1)), abs(float(y2) - float(y1)))
    return float(max(8.0, min(bbox_major * 0.32, top_box_span * 0.55)))


def _detect_potentiometer_top_rect(crop: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([85, 35, 25], dtype=np.uint8), np.array([150, 255, 255], dtype=np.uint8))

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 45, 130)
    mask = cv2.bitwise_or(blue_mask, cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    crop_area = float(crop.shape[0] * crop.shape[1])
    best: tuple[float, tuple[tuple[float, float], tuple[float, float], float]] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(12.0, crop_area * 0.03):
            continue
        rect = cv2.minAreaRect(contour)
        (_cx, _cy), (rw, rh), _angle = rect
        rect_area = max(1.0, float(rw) * float(rh))
        fill = area / rect_area
        if fill < 0.25:
            continue
        aspect = max(float(rw), float(rh)) / max(1.0, min(float(rw), float(rh)))
        if aspect > 5.0:
            continue
        score = area * min(fill, 1.0)
        if best is None or score > best[0]:
            best = (score, rect)
    return best[1] if best is not None else None


def _order_pot_points_by_image_axis(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) != 3:
        return points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if (max(xs) - min(xs)) >= (max(ys) - min(ys)):
        return sorted(points, key=lambda p: p[0])
    return sorted(points, key=lambda p: p[1])


def _potentiometer_bbox_fallback_points_board(
    *,
    bbox: tuple[float, float, float, float],
    calibrator: Any | None,
) -> dict[str, tuple[float, float]] | None:
    """Compute 3 fallback hole positions in board plane → frame pixels.

    Returns ``None`` when the calibrator is unusable; caller then degrades to
    image-frame heuristics.
    """
    if calibrator is None or not getattr(calibrator, "is_grid_ready", False):
        return None
    if not hasattr(calibrator, "frame_pixel_to_board_point"):
        return None
    if not hasattr(calibrator, "logic_to_board_point"):
        return None

    x1, y1, x2, y2 = bbox
    try:
        bp_corners = [
            calibrator.frame_pixel_to_board_point(x1, y1),
            calibrator.frame_pixel_to_board_point(x2, y1),
            calibrator.frame_pixel_to_board_point(x1, y2),
            calibrator.frame_pixel_to_board_point(x2, y2),
        ]
    except Exception as exc:
        logger.debug("S1.5 POT fallback: bbox board projection failed: %s", exc)
        return None

    bxs = [pt[0] for pt in bp_corners]
    bys = [pt[1] for pt in bp_corners]
    spread_x = max(bxs) - min(bxs)
    spread_y = max(bys) - min(bys)
    center_x = 0.5 * (max(bxs) + min(bxs))
    center_y = 0.5 * (max(bys) + min(bys))

    landscape = bool(getattr(calibrator, "landscape", False) or getattr(calibrator, "_landscape", False))
    # Map board axes back to "digit axis" / "letter axis":
    #   landscape=True  -> digits along board-X, letters along board-Y
    #   landscape=False -> digits along board-Y, letters along board-X
    if landscape:
        digit_spread, letter_spread = spread_x, spread_y
    else:
        digit_spread, letter_spread = spread_y, spread_x

    horizontal_insert = digit_spread >= letter_spread

    anchor = None
    if hasattr(calibrator, "board_point_to_logic"):
        try:
            anchor = calibrator.board_point_to_logic(center_x, center_y)
        except Exception as exc:
            logger.debug("S1.5 POT fallback: board_point_to_logic failed: %s", exc)
            anchor = None
    if anchor is None or len(anchor) < 2:
        return None

    digit_str, letter_str = str(anchor[0]).strip(), str(anchor[1]).strip().lower()
    try:
        digit_idx = int(digit_str) - 1
    except ValueError:
        return None
    if letter_str not in _POT_LETTERS:
        return None
    letter_idx = _POT_LETTERS.index(letter_str)

    row_coords = getattr(calibrator, "row_coords", None)
    n_digits = int(len(row_coords)) if row_coords is not None else 0
    if n_digits < 3:
        return None

    if horizontal_insert:
        # 3 adjacent digits, same letter. Clamp window so all 3 are in [1, n_digits].
        start = max(1, min(digit_idx, n_digits - 2))
        digits = [str(start), str(start + 1), str(start + 2)]
        triplet = [(d, letter_str) for d in digits]
    else:
        # 3 adjacent letters within same half (a-e or f-j), same digit.
        if letter_idx <= 4:
            start = max(0, min(letter_idx - 1, 2))
        else:
            start = max(5, min(letter_idx - 1, 7))
        triplet = [(digit_str, _POT_LETTERS[start + i]) for i in range(3)]

    frame_pixels: list[tuple[float, float]] = []
    for loc in triplet:
        try:
            bp = calibrator.logic_to_board_point(loc)
            if bp is None:
                return None
            if hasattr(calibrator, "board_point_to_frame_pixel"):
                fp = calibrator.board_point_to_frame_pixel(float(bp[0]), float(bp[1]))
                frame_pixels.append((float(fp[0]), float(fp[1])))
            else:
                frame_pixels.append((float(bp[0]), float(bp[1])))
        except Exception as exc:
            logger.debug("S1.5 POT fallback: hole projection failed for %s: %s", loc, exc)
            return None

    return {
        "terminal_a": frame_pixels[0],
        "wiper": frame_pixels[1],
        "terminal_b": frame_pixels[2],
    }


def _with_potentiometer_fallback_keypoint(
    *,
    pin: dict,
    pin_name: str,
    keypoint: tuple[float, float],
    view_id: str,
) -> dict:
    updated = dict(pin)
    kx, ky = float(keypoint[0]), float(keypoint[1])
    keypoints_by_view = dict(updated.get("keypoints_by_view") or {})
    visibility_by_view = dict(updated.get("visibility_by_view") or {})
    score_by_view = dict(updated.get("score_by_view") or {})
    source_by_view = dict(updated.get("source_by_view") or {})
    keypoints_by_view[view_id] = [kx, ky]
    visibility_by_view[view_id] = 1
    score_by_view[view_id] = max(float(score_by_view.get(view_id, 0.0) or 0.0), 0.35)
    source_by_view[view_id] = "potentiometer_bbox_fallback"
    updated["keypoints_by_view"] = keypoints_by_view
    updated["visibility_by_view"] = visibility_by_view
    updated["score_by_view"] = score_by_view
    updated["source_by_view"] = source_by_view
    updated["confidence"] = max(float(updated.get("confidence", 0.0) or 0.0), 0.35)
    updated["source"] = "potentiometer_bbox_fallback"
    metadata = dict(updated.get("metadata") or {})
    metadata["potentiometer_fallback_pin_name"] = pin_name
    metadata["potentiometer_fallback_keypoint"] = [kx, ky]
    updated["metadata"] = metadata
    return updated


def _build_components_from_full_pose(
    *,
    detections: list[dict],
    pose_instances: list[PoseInstance],
    view_ids: list[str],
    top_image: np.ndarray | None,
    image_shape: tuple[int, int],
    pin_detector: PinRoiDetector,
    calibrator: Any | None = None,
    pin_detector_mode: str = "full_image_model",
) -> list[dict]:
    components: list[dict] = []
    unavailable_views = [vid for vid in view_ids if vid != "top"]
    for det in detections:
        component_type = normalize_component_type(str(det.get("component_type") or det.get("class_name") or "UNKNOWN"))
        package_type = str(det.get("package_type") or default_package_type(component_type))
        if component_type == "IC":
            components.append(
                _build_ic_component_full_pose_shell(
                    det=det,
                    component_id=det.get("component_id") or "",
                    package_type=package_type,
                    view_ids=view_ids,
                    image_shape=image_shape,
                    pin_detector=pin_detector,
                    calibrator=calibrator,
                    backend_mode=pin_detector_mode,
                    top_image=top_image,
                )
            )
            continue
        pin_schema_id = default_pin_schema_id(component_type, package_type)
        pin_count = default_pin_count(component_type, package_type)
        pin_names = default_pin_names(component_type, pin_count)
        matched = _match_pose_instance(det, pose_instances)
        aligned_points = _aligned_keypoints(
            component_type,
            matched.keypoints if matched else [],
            list(det.get("bbox") or [0, 0, 0, 0]),
        )

        transistor_roles_by_pin_id: dict[int, dict[str, str]] = {}
        transistor_polarity_meta: dict[str, Any] = {}
        if component_type == "Transistor":
            polarity_input = []
            for idx, kp in enumerate(aligned_points, start=1):
                if kp is None:
                    continue
                polarity_input.append({"pin_id": idx, "pin_name": pin_names[idx - 1], "xy": [float(kp[0]), float(kp[1])]})
            polarity_info = infer_transistor_pin_roles(
                image=top_image,
                bbox_xyxy=list(det.get("bbox") or [0, 0, 0, 0]),
                pins=polarity_input,
                pinout_left_to_right=["E", "B", "C"],
            )
            transistor_polarity_meta = {
                "visible_face": polarity_info.get("visible_face"),
                "flat_arc_decision": polarity_info.get("flat_arc_decision"),
                "flat_arc_decision_confidence": polarity_info.get("flat_arc_decision_confidence"),
                "ebc_assignment_enabled": polarity_info.get("ebc_assignment_enabled"),
                "pinout_used_for_current_view_left_to_right": polarity_info.get("pinout_used_for_current_view_left_to_right"),
            }
            for item in polarity_info.get("pin_roles") or []:
                pin_id = int(item.get("pin_id") or 0)
                if pin_id <= 0:
                    continue
                transistor_roles_by_pin_id[pin_id] = {
                    "predicted_role": str(item.get("predicted_role") or "UNKNOWN"),
                    "candidate_role": str(item.get("candidate_role") or "UNKNOWN"),
                }

        pins = []
        for idx, pin_name in enumerate(pin_names, start=1):
            kp = aligned_points[idx - 1] if matched and idx - 1 < len(aligned_points) else None
            role_meta = transistor_roles_by_pin_id.get(idx, {})
            predicted_role = str(role_meta.get("predicted_role") or "UNKNOWN")
            candidate_role = str(role_meta.get("candidate_role") or "UNKNOWN")
            display_name = pin_name
            if component_type == "Transistor":
                display_name = predicted_role if predicted_role != "UNKNOWN" else (
                    candidate_role if candidate_role != "UNKNOWN" else pin_name
                )
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
                "backend_mode": pin_detector_mode,
                "interface_version": "full_image_pose_v1",
                "roi_source": "full_image_pose" if pin_detector_mode == "full_image_model" else "unavailable",
                **(matched.parse_meta if matched else {}),
            }
            pins.append(
                {
                    "pin_id": idx,
                    "pin_name": pin_name,
                    "pin_display_name": display_name,
                    "polarity_role": predicted_role if component_type == "Transistor" else "UNKNOWN",
                    "polarity_candidate_role": candidate_role if component_type == "Transistor" else "UNKNOWN",
                    "keypoints_by_view": keypoints_by_view,
                    "visibility_by_view": visibility_by_view,
                    "score_by_view": score_by_view,
                    "source_by_view": source_by_view,
                    "confidence": float(det.get("confidence", 1.0)) if kp is not None else 0.0,
                    "source": "model" if kp is not None else "unavailable",
                    "metadata": {"per_view": per_view},
                }
            )
        if component_type == "Potentiometer":
            pins = _semanticize_potentiometer_pins(
                pins=pins,
                bbox=list(det.get("bbox") or [0, 0, 0, 0]),
                calibrator=calibrator,
                top_image=top_image,
            )

        top_source = "full_image_pose" if pin_detector_mode == "full_image_model" else "unavailable"
        top_available = pin_detector_mode == "full_image_model"
        roi_by_view = {
            "top": {
                "offset": [0, 0],
                "shape": [int(image_shape[0]), int(image_shape[1])],
                "source": top_source,
                "crop_source": top_source,
                "crop_profile": "none",
                "crop_bounds": None,
                "body_bbox": list(det.get("bbox") or [0, 0, 0, 0]),
                "body_size": None,
                "roi_size": [int(image_shape[1]), int(image_shape[0])],
                "scale_multiplier": 1.0,
                "retry_attempts": 0,
                "association": {},
                "available": top_available,
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
                    "backend_mode": pin_detector_mode,
                },
                "transistor_polarity": transistor_polarity_meta if component_type == "Transistor" else {},
            }
        )
    return components


# ---------------------------------------------------------------------------
# IC e/f-bridge geometry path
#
# 引脚检测模型对 DIP 封装识别不稳, 且 IC 在标准面包板上一定跨接 e/f 两行。
# 这里直接按 bbox + e/f 行约束铺出 8 (DIP8) 或 14 (DIP14) 个引脚槽位,
# 不再依赖任何引脚模型或旧的 anchor_pair 逻辑。hole_id 仍交给 S2 映射。
# ---------------------------------------------------------------------------

IC_GEOMETRY_SOURCE = "ic_ef_bridge_geometry"
IC_DEFAULT_PACKAGE = "dip8"
IC_PIN_SCHEMA_ID = "ic_dip_ef_bridge"
_IC_PACKAGE_PIN_COUNT = {"dip8": 8, "dip14": 14}


def _normalize_ic_package_type(package_type: str | None) -> str:
    pkg = (package_type or "").lower()
    return pkg if pkg in _IC_PACKAGE_PIN_COUNT else IC_DEFAULT_PACKAGE


def _ic_pin_count(package_type: str) -> int:
    return _IC_PACKAGE_PIN_COUNT[_normalize_ic_package_type(package_type)]


def _build_ic_geometry_pins(
    *,
    det: dict,
    package_type: str,
    view_ids: List[str],
    calibrator: Any | None = None,
    top_image: np.ndarray | None = None,
    _allow_image_frame_fallback: bool = False,
) -> List[dict]:
    """根据 IC bbox + 面包板 e/f 行约束推断 DIP8/DIP14 引脚槽位。

    物理模型:
    - 两排引脚永远锁在面包板 e 行 / f 行 (字母行轴).
    - 引脚沿"数字列"方向展开: DIP8 占 4 个连续数字列, DIP14 占 7 个.
    - **绝对不**用 image-frame 的 bbox horizontal/vertical 来决定 pin 排布方向 ——
      否则 IC 在面包板上会被"旋转 90 度". image-frame 的轴向只是相机视角, 与
      面包板物理轴向无关.
    - notch_direction 只决定 pin_id 编号绕向 (notch 端起逆时针), 不影响 e/f
      行的几何槽位.

    优先级:
    1) calibrator 就绪 / 能 ensure_calibrated 成功 -> 用 logic_to_board_point 选
       4 / 7 个连续数字列, board_2d_point 直接来自面包板逻辑坐标, 反变换回 frame
       pixel 得到 keypoint.
    2) calibrator 拿不到 -> 拒绝输出引脚 (返回 []), 同时把 degraded 信号写到 det
       的 metadata. 下游 (S2) 应直接跳过此 IC 而不是吸附到错孔. 旧的 image-frame
       兜底仅保留给 ``_allow_image_frame_fallback=True`` 的离线测试入口.
    """
    pkg = _normalize_ic_package_type(package_type)
    pin_count = _ic_pin_count(pkg)
    half = pin_count // 2

    bbox = _parse_ic_bbox(det.get("bbox"))
    notch_direction = str(
        det.get("notch_direction")
        or (det.get("metadata") or {}).get("notch_direction")
        or "left"
    ).lower()

    layout = _try_board_logic_layout(
        bbox=bbox,
        half=half,
        calibrator=calibrator,
        top_image=top_image,
    )
    if layout is None:
        if _allow_image_frame_fallback:
            layout = _image_frame_fallback_layout(bbox=bbox, half=half)
        else:
            logger.warning(
                "S1.5 IC geometry: calibrator unavailable, refusing pin output "
                "(package=%s, bbox=%s).",
                pkg, bbox,
            )
            return []

    notch_at_low_index = notch_direction in {"left", "up"}
    pins_by_id: Dict[int, dict] = {}
    for slot in range(half):
        if notch_at_low_index:
            e_pin_id = slot + 1
            f_pin_id = 2 * half - slot
        else:
            e_pin_id = half - slot
            f_pin_id = half + 1 + slot
        pins_by_id[e_pin_id] = _make_ic_pin_entry(
            pin_id=e_pin_id,
            keypoint=layout["e_frame_pixels"][slot],
            board_point=layout["e_board_points"][slot] if layout.get("e_board_points") else None,
            row_lock="e",
            estimated_column=slot,
            digit_column_label=layout["digit_column_labels"][slot] if layout.get("digit_column_labels") else None,
            package_type=pkg,
            notch_direction=notch_direction,
            view_ids=view_ids,
            column_source=layout["column_source"],
        )
        pins_by_id[f_pin_id] = _make_ic_pin_entry(
            pin_id=f_pin_id,
            keypoint=layout["f_frame_pixels"][slot],
            board_point=layout["f_board_points"][slot] if layout.get("f_board_points") else None,
            row_lock="f",
            estimated_column=slot,
            digit_column_label=layout["digit_column_labels"][slot] if layout.get("digit_column_labels") else None,
            package_type=pkg,
            notch_direction=notch_direction,
            view_ids=view_ids,
            column_source=layout["column_source"],
        )
    return [pins_by_id[pid] for pid in range(1, pin_count + 1)]


def _parse_ic_bbox(raw_bbox: Any) -> tuple[float, float, float, float]:
    try:
        x1, y1, x2, y2 = (float(v) for v in (raw_bbox or [0.0, 0.0, 0.0, 0.0])[:4])
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0, 0.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _try_board_logic_layout(
    *,
    bbox: tuple[float, float, float, float],
    half: int,
    calibrator: Any | None,
    top_image: np.ndarray | None,
) -> Dict[str, Any] | None:
    """尝试用 calibrator 在 board plane 上选 half 个连续数字列。失败返回 None。"""
    if calibrator is None:
        return None
    if not getattr(calibrator, "is_grid_ready", False):
        if top_image is None or not hasattr(calibrator, "ensure_calibrated"):
            return None
        try:
            calibrator.ensure_calibrated(top_image)
        except Exception as exc:
            logger.debug("S1.5 IC geometry: ensure_calibrated failed: %s", exc)
            return None
        if not getattr(calibrator, "is_grid_ready", False):
            return None

    if not hasattr(calibrator, "frame_pixel_to_board_point"):
        return None
    if not hasattr(calibrator, "logic_to_board_point"):
        return None
    row_coords = getattr(calibrator, "row_coords", None)
    if row_coords is None or len(row_coords) < half:
        return None

    x1, y1, x2, y2 = bbox
    try:
        bp_tl = calibrator.frame_pixel_to_board_point(x1, y1)
        bp_tr = calibrator.frame_pixel_to_board_point(x2, y1)
        bp_bl = calibrator.frame_pixel_to_board_point(x1, y2)
        bp_br = calibrator.frame_pixel_to_board_point(x2, y2)
    except Exception as exc:
        logger.debug("S1.5 IC geometry: bbox board projection failed: %s", exc)
        return None

    landscape = bool(getattr(calibrator, "landscape", False) or getattr(calibrator, "_landscape", False))
    if landscape:
        # 数字列轴在 board X 方向; 字母行 a-j 在 board Y 方向.
        bxs = [bp_tl[0], bp_tr[0], bp_bl[0], bp_br[0]]
    else:
        bxs = [bp_tl[1], bp_tr[1], bp_bl[1], bp_br[1]]
    bbox_lo, bbox_hi = min(bxs), max(bxs)
    bbox_center = 0.5 * (bbox_lo + bbox_hi)

    coords = np.asarray(row_coords, dtype=np.float32)
    if coords.size < half:
        return None
    # 选 half 个相邻数字列 (列号连续), 让该列窗口与 bbox 数字列投影 [lo,hi] 的 1D IoU 最大.
    # 与旧的"bbox 中点选窗口"相比, IoU 对 bbox 单侧含 label / 留白偏移更鲁棒.
    best_start = 0
    best_key: tuple[float, float] = (-1.0, -float("inf"))
    for start in range(coords.size - half + 1):
        win_lo = float(coords[start])
        win_hi = float(coords[start + half - 1])
        inter = max(0.0, min(win_hi, bbox_hi) - max(win_lo, bbox_lo))
        union = max(win_hi, bbox_hi) - min(win_lo, bbox_lo)
        iou = inter / union if union > 1e-6 else 0.0
        win_center = 0.5 * (win_lo + win_hi)
        # Tiebreaker: prefer the window whose center is closer to the bbox center
        # (covers degenerate cases where every window has IoU 0 — bbox far outside).
        key = (iou, -abs(win_center - bbox_center))
        if key > best_key:
            best_key = key
            best_start = start

    digit_column_labels = [str(best_start + i + 1) for i in range(half)]
    e_board_points: list[tuple[float, float]] = []
    f_board_points: list[tuple[float, float]] = []
    try:
        for col_label in digit_column_labels:
            pe = calibrator.logic_to_board_point((col_label, "e"))
            pf = calibrator.logic_to_board_point((col_label, "f"))
            if pe is None or pf is None:
                return None
            e_board_points.append((float(pe[0]), float(pe[1])))
            f_board_points.append((float(pf[0]), float(pf[1])))
    except Exception as exc:
        logger.debug("S1.5 IC geometry: logic_to_board_point failed: %s", exc)
        return None

    if hasattr(calibrator, "board_point_to_frame_pixel"):
        try:
            e_frame_pixels = [tuple(calibrator.board_point_to_frame_pixel(*pt)) for pt in e_board_points]
            f_frame_pixels = [tuple(calibrator.board_point_to_frame_pixel(*pt)) for pt in f_board_points]
        except Exception as exc:
            logger.debug("S1.5 IC geometry: board_point_to_frame_pixel failed: %s", exc)
            e_frame_pixels = [tuple(pt) for pt in e_board_points]
            f_frame_pixels = [tuple(pt) for pt in f_board_points]
    else:
        # 兼容老版 calibrator: 直接把 board point 当 frame pixel 用 (仅 synthetic 模式正确).
        e_frame_pixels = [tuple(pt) for pt in e_board_points]
        f_frame_pixels = [tuple(pt) for pt in f_board_points]

    return {
        "digit_column_labels": digit_column_labels,
        "e_board_points": e_board_points,
        "f_board_points": f_board_points,
        "e_frame_pixels": e_frame_pixels,
        "f_frame_pixels": f_frame_pixels,
        "column_source": "board_logic",
    }


def _image_frame_fallback_layout(
    *,
    bbox: tuple[float, float, float, float],
    half: int,
) -> Dict[str, Any]:
    """没有 calibrator 时的兜底: 假设标准俯拍 (数字列 ≈ image-X, 字母行 ≈ image-Y)。

    与原实现相比, 这里**强制** e 行在 image-Y 较小一侧, f 行在较大一侧,
    pin 沿 image-X 均匀展开 —— 不再根据 bbox 长短轴决定排布方向, 避免在
    竖向 bbox 时把 pin 摆成沿 Y 的一列, 造成 IC 旋转 90 度的错觉.
    """
    logger.warning(
        "S1.5 IC geometry: calibrator not ready, falling back to image-frame layout. "
        "DIP pin slots may not align with breadboard e/f rows."
    )
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    pad_ratio = 0.1
    if half > 1:
        xs = [
            x1 + bw * (pad_ratio + (1.0 - 2.0 * pad_ratio) * i / (half - 1))
            for i in range(half)
        ]
    else:
        xs = [cx]
    e_y = y1 + bh * 0.3
    f_y = y2 - bh * 0.3
    e_frame_pixels = [(xs[i], e_y) for i in range(half)]
    f_frame_pixels = [(xs[i], f_y) for i in range(half)]
    return {
        "digit_column_labels": None,
        "e_board_points": None,
        "f_board_points": None,
        "e_frame_pixels": e_frame_pixels,
        "f_frame_pixels": f_frame_pixels,
        "column_source": "image_frame_fallback",
    }


def _make_ic_pin_entry(
    *,
    pin_id: int,
    keypoint: tuple[float, float],
    board_point: tuple[float, float] | None,
    row_lock: str,
    estimated_column: int,
    digit_column_label: str | None,
    package_type: str,
    notch_direction: str,
    view_ids: List[str],
    column_source: str,
) -> dict:
    kx, ky = float(keypoint[0]), float(keypoint[1])
    board_point_list: list[float] | None = None
    if board_point is not None and len(board_point) >= 2:
        board_point_list = [float(board_point[0]), float(board_point[1])]

    keypoints_by_view = {vid: None for vid in view_ids}
    visibility_by_view = {vid: 0 for vid in view_ids}
    score_by_view = {vid: 0.0 for vid in view_ids}
    source_by_view = {vid: "unavailable" for vid in view_ids}
    per_view: Dict[str, Dict[str, Any]] = {vid: {} for vid in view_ids}

    keypoints_by_view["top"] = [kx, ky]
    visibility_by_view["top"] = 2
    score_by_view["top"] = 1.0
    source_by_view["top"] = IC_GEOMETRY_SOURCE

    top_per_view: Dict[str, Any] = {
        "roi_source": IC_GEOMETRY_SOURCE,
        "row_lock": row_lock,
        "estimated_column": estimated_column,
        "digit_column_label": digit_column_label,
        "column_source": column_source,
        "package_type": package_type,
        "notch_direction": notch_direction,
        "numbering_rule": "counterclockwise",
    }
    if board_point_list is not None:
        top_per_view["board_2d_point"] = board_point_list
    per_view["top"] = top_per_view

    pin_name = f"pin{pin_id}"
    metadata: Dict[str, Any] = {
        "per_view": per_view,
        "package_type": package_type,
        "row_lock": row_lock,
        "estimated_column": estimated_column,
        "digit_column_label": digit_column_label,
        "column_source": column_source,
        "notch_direction": notch_direction,
        "numbering_rule": "counterclockwise",
    }
    if board_point_list is not None:
        metadata["board_2d_point"] = board_point_list

    return {
        "pin_id": pin_id,
        "pin_name": pin_name,
        "pin_display_name": pin_name,
        "polarity_role": "UNKNOWN",
        "polarity_candidate_role": "UNKNOWN",
        "keypoints_by_view": keypoints_by_view,
        "visibility_by_view": visibility_by_view,
        "score_by_view": score_by_view,
        "source_by_view": source_by_view,
        "confidence": 1.0,
        "source": IC_GEOMETRY_SOURCE,
        "metadata": metadata,
    }


def _build_ic_component_full_pose_shell(
    *,
    det: dict,
    component_id: str,
    package_type: str,
    view_ids: List[str],
    image_shape: tuple[int, int],
    pin_detector: PinRoiDetector,
    calibrator: Any | None,
    backend_mode: str,
    top_image: np.ndarray | None = None,
) -> dict:
    """统一封装 IC 组件外壳, 让两条 S1.5 主路径走同一份 IC 输出契约。"""
    pkg = _normalize_ic_package_type(package_type)
    pins = _build_ic_geometry_pins(
        det=det,
        package_type=pkg,
        view_ids=view_ids,
        calibrator=calibrator,
        top_image=top_image,
    )
    bbox = list(det.get("bbox") or [0, 0, 0, 0])
    h = int(image_shape[0]) if image_shape and len(image_shape) >= 1 else 0
    w = int(image_shape[1]) if image_shape and len(image_shape) >= 2 else 0
    unavailable_views = [vid for vid in view_ids if vid != "top"]
    roi_by_view: Dict[str, Dict[str, Any]] = {
        "top": {
            "offset": [0, 0],
            "shape": [h, w],
            "source": IC_GEOMETRY_SOURCE,
            "crop_source": IC_GEOMETRY_SOURCE,
            "crop_profile": "ic_ef_bridge",
            "crop_bounds": None,
            "body_bbox": list(bbox),
            "body_size": None,
            "roi_size": [w, h],
            "scale_multiplier": 1.0,
            "retry_attempts": 0,
            "association": {},
            "available": True,
        }
    }
    for vid in unavailable_views:
        roi_by_view[vid] = {
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

    return {
        "component_id": component_id or det.get("component_id") or "",
        "component_type": "IC",
        "class_name": "IC",
        "package_type": pkg,
        "pin_schema_id": IC_PIN_SCHEMA_ID,
        "input_pin_detect_interface_version": "component_pin_detect_v1",
        "input_detection_interface_version": det.get("input_detection_interface_version") or "component_detect_v1",
        "part_subtype": det.get("part_subtype") or "",
        "symmetry_group": det.get("symmetry_group") or default_symmetry_group("IC"),
        "bbox": list(bbox),
        "confidence": float(det.get("confidence", 1.0)),
        "orientation": float(det.get("orientation", 0.0)),
        "pins": pins,
        "roi": roi_by_view["top"],
        "roi_by_view": roi_by_view,
        "pin_detector": {
            "interface_version": getattr(pin_detector, "interface_version", "pin_detector_v1"),
            "backend_type": getattr(pin_detector, "backend_type", "ic_ef_bridge_geometry"),
            "backend_mode": backend_mode or IC_GEOMETRY_SOURCE,
        },
        "ic_geometry": {
            "package_type": pkg,
            "pin_count": _ic_pin_count(pkg),
            "notch_direction": pins[0]["metadata"]["notch_direction"] if pins else "left",
            "numbering_rule": "counterclockwise",
            "calibrator_used": bool(calibrator is not None and getattr(calibrator, "is_grid_ready", False)),
        },
    }
