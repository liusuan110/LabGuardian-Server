from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _order_points_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points[order]


def _mask_foreground(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


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


def _head_mask_by_pins(crop: np.ndarray, pin_points_local: list[list[float]]) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    fg = _mask_foreground(gray)
    if len(pin_points_local) < 2:
        return fg

    pts = np.array(pin_points_local, dtype=np.float32)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    vals, vecs = np.linalg.eig(cov)
    axis = vecs[:, int(np.argmax(vals))].astype(np.float32)
    axis = axis / (float(np.linalg.norm(axis)) + 1e-6)
    n = np.array([-axis[1], axis[0]], dtype=np.float32)

    ys, xs = np.where(fg > 0)
    if xs.size == 0:
        return fg
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


def _flat_arc_decision(body_contour: np.ndarray, edge_scores: list[float]) -> dict[str, Any]:
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
    line_fit_score, _ = _fit_line_score(line_points)

    ellipse_score, _ = _fit_ellipse_score(pts)
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
        "decision": decision,
        "decision_confidence": round(confidence, 4),
        "margin": round(float(margin), 4),
        "flat_score": round(float(flat_raw), 4),
        "arc_score": round(float(arc_raw), 4),
    }


def _estimate_flat_info(
    image: np.ndarray,
    bbox_xyxy: list[float],
    pin_points_global: list[dict[str, Any]],
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
    for pin in pin_points_global:
        px, py = [float(v) for v in pin.get("xy", [0, 0])]
        if sx1 <= px <= sx2 and sy1 <= py <= sy2:
            pin_local.append([px - sx1, py - sy1])

    head_mask = _head_mask_by_pins(crop, pin_local)
    contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    body_contour = max(contours, key=cv2.contourArea)
    if body_contour is None or cv2.contourArea(body_contour) < 30:
        return None

    rect = cv2.minAreaRect(body_contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = _order_points_clockwise(box)

    edge_scores: list[float] = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        edge_scores.append(_edge_straightness_score(body_contour, p1, p2))
    best_idx = int(np.argmax(np.array(edge_scores)))
    second = sorted(edge_scores, reverse=True)[1] if len(edge_scores) > 1 else 0.0
    confidence = float(max(0.0, min(1.0, (edge_scores[best_idx] - second) * 2.2)))
    p1 = box[best_idx]
    p2 = box[(best_idx + 1) % 4]
    vec = p2 - p1
    angle_deg = float(np.degrees(np.arctan2(vec[1], vec[0])))
    arc = _flat_arc_decision(body_contour, edge_scores)
    return {
        "flat_side_angle_deg": angle_deg,
        "flat_confidence": confidence,
        "flat_arc_metrics": arc,
    }


def _predict_roles_by_flat_axis(
    points: list[dict[str, Any]],
    flat_side_angle_deg: float,
    pinout_left_to_right: list[str],
    assign_roles: bool,
) -> list[dict[str, Any]]:
    # Prefer pin-geometry axis for ordering: this is more stable than body contour
    # when TO-92 body extraction is noisy, and avoids B/C accidental swap.
    axis = None
    if len(points) >= 2:
        pts = np.array([[float(p["xy"][0]), float(p["xy"][1])] for p in points], dtype=np.float32)
        mean = np.mean(pts, axis=0)
        centered = pts - mean[None, :]
        cov = centered.T @ centered
        vals, vecs = np.linalg.eig(cov)
        axis = vecs[:, int(np.argmax(vals))].astype(np.float32)
    if axis is None or float(np.linalg.norm(axis)) < 1e-6:
        theta = np.deg2rad(float(flat_side_angle_deg))
        axis = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    axis = axis / (float(np.linalg.norm(axis)) + 1e-6)
    if axis[0] < 0 or (abs(float(axis[0])) < 1e-6 and axis[1] < 0):
        axis = -axis
    enriched: list[dict[str, Any]] = []
    for p in points:
        x, y = p["xy"]
        proj = float(x * axis[0] + y * axis[1])
        enriched.append({**p, "_proj_x": proj})
    enriched.sort(key=lambda it: it["_proj_x"])

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(enriched):
        candidate = pinout_left_to_right[idx] if idx < len(pinout_left_to_right) else f"PIN{idx + 1}"
        role = candidate if assign_roles else "UNKNOWN"
        out.append(
            {
                "pin_id": int(item["pin_id"]),
                "predicted_role": role,
                "candidate_role": candidate,
            }
        )
    return out


def infer_transistor_pin_roles(
    *,
    image: np.ndarray,
    bbox_xyxy: list[float],
    pins: list[dict[str, Any]],
    pinout_left_to_right: list[str] | None = None,
) -> dict[str, Any]:
    pinout = [item.strip().upper() for item in (pinout_left_to_right or ["E", "B", "C"]) if item]
    if not pinout:
        pinout = ["E", "B", "C"]

    flat_info = _estimate_flat_info(image=image, bbox_xyxy=bbox_xyxy, pin_points_global=pins)
    flat_angle = float((flat_info or {}).get("flat_side_angle_deg", 0.0))
    arc_metrics = (flat_info or {}).get("flat_arc_metrics") or {}
    decision = str(arc_metrics.get("decision") or "UNKNOWN")
    decision_conf = float(arc_metrics.get("decision_confidence") or 0.0)
    margin = float(arc_metrics.get("margin") or 0.0)
    flat_score = float(arc_metrics.get("flat_score") or 0.0)
    arc_score = float(arc_metrics.get("arc_score") or 0.0)

    # Conservative policy:
    # - FLAT assignment is the default;
    # - Only switch to ARC when evidence is strongly confident.
    visible_face = "FLAT"
    if decision == "FLAT_SIDE_CONFIDENT":
        visible_face = "FLAT"
    elif (
        decision == "ARC_SIDE_CONFIDENT"
        and decision_conf >= 0.65
        and (arc_score - flat_score) >= 0.12
    ):
        visible_face = "ARC"

    if visible_face == "FLAT":
        role_order = list(pinout)
    elif visible_face == "ARC":
        role_order = list(reversed(pinout))
    else:
        role_order = list(pinout)

    can_assign = (
        len(pins) == 3
        and visible_face in {"FLAT", "ARC"}
        and (decision_conf >= 0.35 or abs(margin) >= 0.03 or visible_face == "FLAT")
    )
    roles = _predict_roles_by_flat_axis(
        points=pins,
        flat_side_angle_deg=flat_angle,
        pinout_left_to_right=role_order,
        assign_roles=can_assign,
    )
    return {
        "visible_face": visible_face,
        "flat_arc_decision": decision,
        "flat_arc_decision_confidence": decision_conf,
        "ebc_assignment_enabled": can_assign,
        "pinout_used_for_current_view_left_to_right": role_order,
        "pin_roles": roles,
    }
