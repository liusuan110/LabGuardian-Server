from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.orchestrator import run_pipeline


def _order_points_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points[order]


def _line_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _rotate_image(image: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return rotated, matrix


def _mask_foreground(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _extract_body_mask(fg_mask: np.ndarray) -> np.ndarray:
    h = fg_mask.shape[0]
    k = max(5, int(h * 0.02) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    body = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return body


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = mask.shape[:2]

    def _touches_border(cnt: np.ndarray) -> bool:
        x, y, cw, ch = cv2.boundingRect(cnt)
        return x <= 1 or y <= 1 or (x + cw) >= (w - 1) or (y + ch) >= (h - 1)

    candidates = []
    total_area = float(h * w)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < total_area * 0.0002:
            continue
        if area > total_area * 0.65:
            continue
        if _touches_border(cnt):
            continue
        candidates.append(cnt)

    if not candidates:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours[0]
    candidates = sorted(candidates, key=cv2.contourArea, reverse=True)
    return candidates[0]


def _estimate_flat_side(body_contour: np.ndarray) -> Dict[str, Any]:
    rect = cv2.minAreaRect(body_contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = _order_points_clockwise(box)

    edges: List[Tuple[int, int, float]] = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        edges.append((i, (i + 1) % 4, _line_length(p1, p2)))
    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    longest = edges[0]
    second = edges[1]
    ratio = float(longest[2] / (second[2] + 1e-6))

    p1 = box[longest[0]]
    p2 = box[longest[1]]
    mid = (p1 + p2) / 2.0
    vec = p2 - p1
    angle_deg = float(np.degrees(np.arctan2(vec[1], vec[0])))

    return {
        "rect_center": [float(rect[0][0]), float(rect[0][1])],
        "rect_size": [float(rect[1][0]), float(rect[1][1])],
        "rect_angle": float(rect[2]),
        "box_points": box.tolist(),
        "flat_side_edge": [int(longest[0]), int(longest[1])],
        "flat_side_midpoint": [float(mid[0]), float(mid[1])],
        "flat_side_angle_deg": angle_deg,
        "flat_confidence": min(1.0, max(0.0, (ratio - 1.0) * 1.8)),
    }


def _detect_lead_columns(rotated_fg: np.ndarray, body_bbox: Tuple[int, int, int, int]) -> List[int]:
    x, y, w, h = body_bbox
    lead_region_top = min(rotated_fg.shape[0] - 1, y + h + 2)
    if lead_region_top >= rotated_fg.shape[0] - 3:
        return []

    lead_region = rotated_fg[lead_region_top:, :]
    col_sum = lead_region.sum(axis=0).astype(np.float32)
    if np.max(col_sum) <= 0:
        return []

    col_sum = cv2.GaussianBlur(col_sum.reshape(1, -1), (1, 15), 0).reshape(-1)
    threshold = float(np.max(col_sum) * 0.35)
    candidate_cols = np.where(col_sum >= threshold)[0]
    if candidate_cols.size == 0:
        return []

    groups: List[List[int]] = []
    current = [int(candidate_cols[0])]
    for col in candidate_cols[1:]:
        col = int(col)
        if col - current[-1] <= 5:
            current.append(col)
        else:
            groups.append(current)
            current = [col]
    groups.append(current)

    peaks = [int(round(float(np.mean(g)))) for g in groups if len(g) >= 2]
    peaks = sorted(peaks, key=lambda c: col_sum[c], reverse=True)[:3]
    peaks.sort()
    return peaks


def _detect_lead_columns_hough(rotated_fg: np.ndarray, body_bbox: Tuple[int, int, int, int]) -> List[int]:
    x, y, w, h = body_bbox
    lead_top = min(rotated_fg.shape[0] - 1, y + h + 2)
    if lead_top >= rotated_fg.shape[0] - 4:
        return []

    roi = rotated_fg[lead_top:, :]
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180.0,
        threshold=30,
        minLineLength=max(20, int(roi.shape[0] * 0.15)),
        maxLineGap=8,
    )
    if lines is None:
        return []

    x_hits: List[int] = []
    yb = roi.shape[0] - 1
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in l]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < 8:
            continue
        if abs(dx) > max(10, int(abs(dy) * 0.45)):
            continue
        # 线延长到底部，取 x 截距
        t = (yb - y1) / (dy + 1e-6)
        xb = int(round(x1 + t * dx))
        if 0 <= xb < roi.shape[1]:
            x_hits.append(xb)

    if not x_hits:
        return []

    x_hits = sorted(x_hits)
    clusters: List[List[int]] = [[x_hits[0]]]
    for xv in x_hits[1:]:
        if xv - clusters[-1][-1] <= 10:
            clusters[-1].append(xv)
        else:
            clusters.append([xv])

    clusters = sorted(clusters, key=lambda g: len(g), reverse=True)[:3]
    peaks = [int(round(float(np.mean(g)))) for g in clusters]
    peaks.sort()
    return peaks


def _invert_affine(matrix: np.ndarray, points: List[Tuple[float, float]]) -> List[List[float]]:
    inv = cv2.invertAffineTransform(matrix)
    out: List[List[float]] = []
    for x, y in points:
        px = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]
        py = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]
        out.append([float(px), float(py)])
    return out


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _expand_bbox(bbox_xyxy: list[float], ratio: float) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    cx, cy = _bbox_center(bbox_xyxy)
    w = max(1.0, x2 - x1) * ratio
    h = max(1.0, y2 - y1) * ratio
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _point_in_bbox(point_xy: list[float], bbox_xyxy: list[float]) -> bool:
    x, y = float(point_xy[0]), float(point_xy[1])
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    return x1 <= x <= x2 and y1 <= y <= y2


def _get_transistor_components(raw: dict[str, Any], component_id: str | None) -> list[dict[str, Any]]:
    detect = ((raw.get("stages") or {}).get("detect") or {})
    detections = detect.get("detections") or []
    candidates = [d for d in detections if str(d.get("component_type") or "") == "Transistor"]
    if component_id:
        candidates = [d for d in candidates if str(d.get("component_id") or "") == component_id]
    return candidates


def _find_pin_component(raw: dict[str, Any], component_id: str) -> dict[str, Any] | None:
    pin_detect = ((raw.get("stages") or {}).get("pin_detect") or {})
    for comp in pin_detect.get("components") or []:
        if str(comp.get("component_id") or "") == component_id:
            return comp
    return None


def _extract_top_pin_points(pin_component: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pin in pin_component.get("pins") or []:
        kp = ((pin.get("keypoints_by_view") or {}).get("top")) or pin.get("keypoint")
        if not kp or len(kp) < 2:
            continue
        out.append(
            {
                "pin_id": int(pin.get("pin_id") or 0),
                "pin_name": str(pin.get("pin_name") or ""),
                "xy": [float(kp[0]), float(kp[1])],
            }
        )
    return out


def _collect_all_transistor_pin_points(raw: dict[str, Any]) -> list[dict[str, Any]]:
    pin_detect = ((raw.get("stages") or {}).get("pin_detect") or {})
    out: list[dict[str, Any]] = []
    for comp in pin_detect.get("components") or []:
        if str(comp.get("component_type") or "") != "Transistor":
            continue
        comp_id = str(comp.get("component_id") or "")
        for pin in _extract_top_pin_points(comp):
            out.append({**pin, "source_component_id": comp_id})
    return out


def _associate_pins_for_transistor(
    *,
    raw: dict[str, Any],
    component_id: str,
    bbox_xyxy: list[float],
    all_pin_pool: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tight = _expand_bbox(bbox_xyxy, 1.45)
    loose = _expand_bbox(bbox_xyxy, 2.1)
    cx, cy = _bbox_center(bbox_xyxy)

    own = _extract_top_pin_points(_find_pin_component(raw, component_id) or {})
    own_in_tight = [p for p in own if _point_in_bbox(p["xy"], tight)]

    selected = list(own_in_tight)
    source_mode = "own_component_pins"
    if len(selected) < 3:
        source_mode = "global_pool_fallback"
        candidates = [p for p in all_pin_pool if _point_in_bbox(p["xy"], loose)]
        candidates = sorted(
            candidates,
            key=lambda p: (float(p["xy"][0] - cx) ** 2 + float(p["xy"][1] - cy) ** 2),
        )
        dedup: list[dict[str, Any]] = []
        used_xy: set[tuple[int, int]] = set()
        for p in candidates:
            key = (int(round(float(p["xy"][0]))), int(round(float(p["xy"][1]))))
            if key in used_xy:
                continue
            used_xy.add(key)
            dedup.append(p)
            if len(dedup) >= 3:
                break
        selected = dedup

    meta = {
        "association_mode": source_mode,
        "own_pin_count": len(own),
        "own_pin_in_tight_count": len(own_in_tight),
        "selected_pin_count": len(selected),
        "tight_bbox_xyxy": [round(v, 2) for v in tight],
        "loose_bbox_xyxy": [round(v, 2) for v in loose],
    }
    return selected, meta


def _edge_straightness_score(contour: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    pts = contour.reshape(-1, 2).astype(np.float32)
    v = p2 - p1
    norm = float(np.linalg.norm(v)) + 1e-6
    dir_u = v / norm
    rel = pts - p1[None, :]
    proj = rel @ dir_u
    in_seg = (proj >= -3.0) & (proj <= norm + 3.0)
    if not np.any(in_seg):
        return 0.0
    rel_seg = rel[in_seg]
    dist = np.abs(rel_seg[:, 0] * dir_u[1] - rel_seg[:, 1] * dir_u[0])
    if dist.size == 0:
        return 0.0
    return float(np.mean(dist <= 2.6))


def _fit_line_score(points: np.ndarray) -> tuple[float, float]:
    if points.shape[0] < 8:
        return 0.0, 999.0
    line = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    line = np.asarray(line, dtype=np.float32).reshape(-1)
    if line.size < 4:
        return 0.0, 999.0
    vx, vy, x0, y0 = [float(v) for v in line[:4]]
    v = np.array([vx, vy], dtype=np.float32).reshape(2)
    v_norm = float(np.linalg.norm(v)) + 1e-6
    u = v / v_norm
    rel = points.astype(np.float32) - np.array([x0, y0], dtype=np.float32)
    dist = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    mean_dist = float(np.mean(dist))
    score = float(np.exp(-mean_dist / 2.6))
    return score, mean_dist


def _fit_ellipse_score(points: np.ndarray) -> tuple[float, float]:
    if points.shape[0] < 20:
        return 0.0, 999.0
    try:
        ellipse = cv2.fitEllipse(points.astype(np.float32))
    except cv2.error:
        return 0.0, 999.0
    cx, cy = ellipse[0]
    ax, by = ellipse[1][0] / 2.0, ellipse[1][1] / 2.0
    if ax < 1e-3 or by < 1e-3:
        return 0.0, 999.0
    theta = np.deg2rad(float(ellipse[2]))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    p = points.astype(np.float32)
    x = p[:, 0] - float(cx)
    y = p[:, 1] - float(cy)
    xr = c * x + s * y
    yr = -s * x + c * y
    val = np.sqrt((xr / float(ax)) ** 2 + (yr / float(by)) ** 2)
    resid = np.abs(val - 1.0)
    mean_resid = float(np.mean(resid))
    score = float(np.exp(-mean_resid / 0.22))
    return score, mean_resid


def _head_mask_by_pins(crop: np.ndarray, pin_points_local: list[list[float]]) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    fg = _mask_foreground(gray)
    if len(pin_points_local) < 2:
        return _extract_body_mask(fg)

    pts = np.array(pin_points_local, dtype=np.float32)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    vals, vecs = np.linalg.eig(cov)
    axis = vecs[:, int(np.argmax(vals))].astype(np.float32)
    axis = axis / (float(np.linalg.norm(axis)) + 1e-6)
    n = np.array([-axis[1], axis[0]], dtype=np.float32)  # 法向

    # 判定“头部在法向哪一侧”：取前景点在两侧计数，较多的一侧作为头部候选
    ys, xs = np.where(fg > 0)
    if xs.size == 0:
        return _extract_body_mask(fg)
    samples = np.stack([xs, ys], axis=1).astype(np.float32)
    signed = (samples - mean[None, :]) @ n
    pos = int(np.sum(signed >= 0))
    neg = int(np.sum(signed < 0))
    sign_keep = 1.0 if pos >= neg else -1.0

    mask_side = np.zeros_like(fg, dtype=np.uint8)
    keep = signed * sign_keep >= 0
    keep_pts = samples[keep].astype(np.int32)
    mask_side[keep_pts[:, 1], keep_pts[:, 0]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    head = cv2.morphologyEx(mask_side, cv2.MORPH_OPEN, kernel, iterations=1)
    head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, kernel, iterations=1)
    return head


def _flat_arc_decision(
    body_contour: np.ndarray,
    edge_scores: list[float],
) -> dict[str, Any]:
    pts = body_contour.reshape(-1, 2).astype(np.float32)
    flat_idx = int(np.argmax(np.array(edge_scores)))
    box = cv2.boxPoints(cv2.minAreaRect(body_contour)).astype(np.float32)
    box = _order_points_clockwise(box)
    p1 = box[flat_idx]
    p2 = box[(flat_idx + 1) % 4]
    v = p2 - p1
    vn = float(np.linalg.norm(v)) + 1e-6
    u = v / vn
    rel = pts - p1[None, :]
    proj = rel @ u
    near_seg = (proj >= -4.0) & (proj <= vn + 4.0)
    line_points = pts[near_seg]
    line_fit_score, line_mean_dist = _fit_line_score(line_points)

    ellipse_score, ellipse_resid = _fit_ellipse_score(pts)
    flat_raw = 0.6 * float(edge_scores[flat_idx]) + 0.4 * line_fit_score
    arc_raw = ellipse_score
    margin = flat_raw - arc_raw
    if margin > 0.14:
        decision = "FLAT_SIDE_CONFIDENT"
    elif margin < -0.14:
        decision = "ARC_SIDE_CONFIDENT"
    else:
        decision = "UNKNOWN"
    confidence = float(min(1.0, max(0.0, abs(margin) * 3.0)))
    return {
        "flat_score": round(float(flat_raw), 4),
        "arc_score": round(float(arc_raw), 4),
        "line_fit_score": round(float(line_fit_score), 4),
        "ellipse_fit_score": round(float(ellipse_score), 4),
        "line_mean_dist_px": round(float(line_mean_dist), 4),
        "ellipse_mean_residual": round(float(ellipse_resid), 4),
        "decision": decision,
        "decision_confidence": round(confidence, 4),
    }


def _estimate_flat_side_in_bbox(
    image: np.ndarray,
    bbox_xyxy: list[float],
    pin_points_global: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy[:4]]
    h, w = image.shape[:2]
    pad = 18
    sx1 = max(0, x1 - pad)
    sy1 = max(0, y1 - pad)
    sx2 = min(w, x2 + pad)
    sy2 = min(h, y2 + pad)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    crop = image[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return None

    pin_local: list[list[float]] = []
    for pin in pin_points_global or []:
        px, py = [float(v) for v in pin.get("xy", [0, 0])]
        if sx1 <= px <= sx2 and sy1 <= py <= sy2:
            pin_local.append([px - sx1, py - sy1])

    head_mask = _head_mask_by_pins(crop, pin_local)
    contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    ccx, ccy = (sx2 - sx1) / 2.0, (sy2 - sy1) / 2.0
    # 选择“靠近框中心”的轮廓，降低把引脚/背景当本体的概率
    contours = sorted(
        contours,
        key=lambda cnt: (
            ((cv2.moments(cnt)["m10"] / (cv2.moments(cnt)["m00"] + 1e-6) - ccx) ** 2
             + (cv2.moments(cnt)["m01"] / (cv2.moments(cnt)["m00"] + 1e-6) - ccy) ** 2),
            -cv2.contourArea(cnt),
        ),
    )
    body_contour = contours[0]
    if body_contour is None or cv2.contourArea(body_contour) < 30:
        return None
    flat = _estimate_flat_side(body_contour)

    box = np.array(flat["box_points"], dtype=np.float32)
    edge_scores: list[float] = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        edge_scores.append(_edge_straightness_score(body_contour, p1, p2))
    best_idx = int(np.argmax(np.array(edge_scores)))
    second_score = sorted(edge_scores, reverse=True)[1] if len(edge_scores) > 1 else 0.0
    confidence = float(max(0.0, min(1.0, (edge_scores[best_idx] - second_score) * 2.2)))
    flat["flat_side_edge"] = [best_idx, (best_idx + 1) % 4]
    p1 = box[best_idx]
    p2 = box[(best_idx + 1) % 4]
    mid = (p1 + p2) / 2.0
    vec = p2 - p1
    flat["flat_side_midpoint"] = [float(mid[0]), float(mid[1])]
    flat["flat_side_angle_deg"] = float(np.degrees(np.arctan2(vec[1], vec[0])))
    flat["flat_confidence"] = confidence
    flat["edge_straightness_scores"] = [round(float(s), 4) for s in edge_scores]
    flat["flat_arc_metrics"] = _flat_arc_decision(body_contour, edge_scores)

    # local -> global
    for p in flat["box_points"]:
        p[0] += sx1
        p[1] += sy1
    flat["flat_side_midpoint"][0] += sx1
    flat["flat_side_midpoint"][1] += sy1
    return flat


def _predict_roles_by_flat_axis(
    points: list[dict[str, Any]],
    flat_side_angle_deg: float,
    pinout_left_to_right: list[str],
    *,
    assign_roles: bool,
) -> list[dict[str, Any]]:
    # 将 flat side 旋转到水平后，按 x 从小到大当作“左->右”
    theta = np.deg2rad(-flat_side_angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    enriched: list[dict[str, Any]] = []
    for p in points:
        x, y = p["xy"]
        xr = c * x - s * y
        enriched.append({**p, "_proj_x": float(xr)})
    enriched.sort(key=lambda it: it["_proj_x"])

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(enriched):
        candidate = pinout_left_to_right[idx] if idx < len(pinout_left_to_right) else f"PIN{idx + 1}"
        role = candidate if assign_roles else "UNKNOWN"
        out.append(
            {
                "index_left_to_right": idx + 1,
                "predicted_role": role,
                "candidate_role": candidate,
                "pin_id": item["pin_id"],
                "pin_name": item["pin_name"],
                "point_xy": [round(item["xy"][0], 2), round(item["xy"][1], 2)],
            }
        )
    return out


def analyze_to92(
    image_path: Path,
    output_dir: Path,
    pinout_left_to_right: List[str],
    component_id: str | None = None,
) -> Dict[str, Any]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    raw = run_pipeline(images_b64=[_encode_image(image_path)])
    transistors = _get_transistor_components(raw, component_id=component_id)
    if not transistors:
        raise ValueError("No Transistor detected in S1 output")
    all_pin_pool = _collect_all_transistor_pin_points(raw)

    vis = img.copy()
    outputs: list[dict[str, Any]] = []
    for det in transistors:
        comp_id = str(det.get("component_id") or "")
        bbox = [float(v) for v in (det.get("bbox") or [0, 0, 0, 0])[:4]]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 120, 30), 2)
        cv2.putText(vis, comp_id, (x1 + 2, max(24, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 120, 30), 2)

        pin_points, assoc_meta = _associate_pins_for_transistor(
            raw=raw,
            component_id=comp_id,
            bbox_xyxy=bbox,
            all_pin_pool=all_pin_pool,
        )
        flat_info = _estimate_flat_side_in_bbox(img, bbox, pin_points_global=pin_points)
        if flat_info is not None:
            flat_angle = float(flat_info.get("flat_side_angle_deg", 0.0))
            for p in np.array(flat_info["box_points"], dtype=np.int32):
                cv2.circle(vis, tuple(p.tolist()), 2, (255, 100, 0), -1)
        else:
            # fallback: 使用检测框朝向信息（弱）
            flat_angle = float(det.get("orientation") or 0.0)

        flat_conf = float((flat_info or {}).get("flat_confidence", 0.0))
        arc_metrics = (flat_info or {}).get("flat_arc_metrics") or {}
        decision = str(arc_metrics.get("decision") or "UNKNOWN")
        decision_conf = float(arc_metrics.get("decision_confidence") or 0.0)
        visible_face = "UNKNOWN"
        if decision == "FLAT_SIDE_CONFIDENT":
            visible_face = "FLAT"
        elif decision == "ARC_SIDE_CONFIDENT":
            visible_face = "ARC"

        if visible_face == "FLAT":
            role_order_for_current_view = list(pinout_left_to_right)
        elif visible_face == "ARC":
            # Viewing from the curved side mirrors left-right pin order.
            role_order_for_current_view = list(reversed(pinout_left_to_right))
        else:
            role_order_for_current_view = list(pinout_left_to_right)

        can_assign = (
            len(pin_points) == 3
            and decision_conf >= 0.35
            and visible_face in {"FLAT", "ARC"}
        )
        pin_roles = _predict_roles_by_flat_axis(
            pin_points,
            flat_angle,
            role_order_for_current_view,
            assign_roles=can_assign,
        )
        pin_map_by_id: dict[str, str] = {}
        pin_map_candidate_by_id: dict[str, str] = {}
        for pin in pin_roles:
            px, py = int(pin["point_xy"][0]), int(pin["point_xy"][1])
            role = str(pin["predicted_role"])
            candidate = str(pin.get("candidate_role") or role)
            pin_id = int(pin.get("pin_id") or 0)
            cv2.circle(vis, (px, py), 6, (30, 220, 30), -1)
            # Visual label only shows pin polarity role, e.g. E/B/C.
            label = candidate if role == "UNKNOWN" else role
            cv2.putText(vis, label, (px + 6, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 220, 30), 2)
            pin_map_by_id[str(pin_id)] = role
            pin_map_candidate_by_id[str(pin_id)] = candidate
        cv2.putText(
            vis,
            f"flat_conf={flat_conf:.2f} {decision}",
            (x1 + 2, min(y2 + 18, img.shape[0] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 255),
            1,
        )

        outputs.append(
            {
                "component_id": comp_id,
                "bbox_xyxy": bbox,
                "pin_association": assoc_meta,
                "flat_side": flat_info,
                "flat_arc_decision": decision,
                "visible_face": visible_face,
                "flat_angle_deg_used": flat_angle,
                "flat_confidence_threshold": 0.22,
                "flat_arc_decision_confidence": decision_conf,
                "ebc_assignment_enabled": can_assign,
                "pinout_assumption_left_to_right": pinout_left_to_right,
                "pinout_used_for_current_view_left_to_right": role_order_for_current_view,
                "pin_id_to_role": pin_map_by_id,
                "pin_id_to_candidate_role": pin_map_candidate_by_id,
                "pins": pin_roles,
                "ok": len(pin_roles) == 3 and can_assign,
            }
        )

    result: Dict[str, Any] = {
        "image": str(image_path),
        "pipeline_stage_summary": {
            "detect_count": len((((raw.get("stages") or {}).get("detect") or {}).get("detections") or [])),
            "pin_component_count": len((((raw.get("stages") or {}).get("pin_detect") or {}).get("components") or [])),
        },
        "transistors": outputs,
        "ok": all(item.get("ok") for item in outputs) if outputs else False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "to92_probe_result.json"
    vis_path = output_dir / "to92_probe_vis.jpg"
    raw_path = output_dir / "to92_probe_pipeline_raw.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    cv2.imwrite(str(vis_path), vis)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype TO-92 E/B/C by official model outputs (S1 + S1.5).")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--output-dir",
        default="D:/LabGuardian/debug_out/to92_probe",
        help="Directory to write JSON and visualization.",
    )
    parser.add_argument(
        "--pinout",
        default="E,B,C",
        help="Assumed left-to-right roles when flat side faces camera, e.g. E,B,C.",
    )
    parser.add_argument(
        "--component-id",
        default=None,
        help="Optional target transistor component_id, e.g. Q1.",
    )
    args = parser.parse_args()

    pinout = [item.strip().upper() for item in str(args.pinout).split(",") if item.strip()]
    if not pinout:
        pinout = ["E", "B", "C"]

    result = analyze_to92(
        image_path=Path(args.image),
        output_dir=Path(args.output_dir),
        pinout_left_to_right=pinout,
        component_id=args.component_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
