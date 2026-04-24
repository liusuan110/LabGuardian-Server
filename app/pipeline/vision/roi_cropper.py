"""
ROI crop helpers for component-centered pin detection.

当前 ROI 裁剪策略明确按封装工作, 而不是对所有元件统一 margin:
- 轴向 2-pin 器件沿主轴给更多 lead 空间
- DIP 封装沿短轴给更多 pin 排空间
- top 视图优先使用 bbox 几何方向; 若历史兼容输入提供 OBB, 则可辅助裁剪
- side 视图在缺真实 bbox 时仍走封装驱动的保守 crop
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.pipeline.vision.pin_schema import roi_crop_profile


def crop_component_roi(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    component_type: str = "UNKNOWN",
    package_type: str = "generic",
    orientation: float = 0.0,
    obb_corners: list[list[float]] | np.ndarray | None = None,
    view_id: str = "top",
    scale_multiplier: float = 1.0,
) -> tuple[np.ndarray | None, tuple[int, int], dict[str, Any]]:
    """按封装、朝向和 bbox 几何裁剪单元件 ROI.

    当前 detect 主路径主要依赖标准 bbox。
    这里将 detect 产出的 bbox 视为“元件本体范围”，再按封装把 ROI
    外扩到更接近“主体 + 可见引脚”的区域。
    `obb_corners` 仅作为历史兼容输入参与细化裁剪。
    """
    if _bbox_mostly_outside_image(bbox, image.shape[:2]):
        # 某些测试/离线联调会直接把“单元件 ROI 图”作为输入传进来,
        # 这时 bbox 仍可能是原始整图坐标。为了让第二阶段接口稳定,
        # 这里显式退回整图 ROI, 但把来源标记为 fallback, 避免误认为严格 crop。
        return image, (0, 0), {
            "source": "full_image_fallback",
            "profile_name": "full_image_fallback",
            "package_type": package_type,
            "component_type": component_type,
            "view_id": view_id,
            "bounds": [0, 0, int(image.shape[1]), int(image.shape[0])],
        }

    profile = roi_crop_profile(component_type, package_type, view_id=view_id)
    profile = _scaled_profile(profile, scale_multiplier=scale_multiplier)
    bounds = _compute_crop_bounds(
        bbox=bbox,
        image_shape=image.shape[:2],
        orientation=orientation,
        obb_corners=obb_corners,
        profile=profile,
    )
    if bounds is None:
        return None, (0, 0), {
            "source": "invalid_bbox",
            "profile_name": str(profile.get("profile_name", "generic")),
            "package_type": package_type,
            "component_type": component_type,
            "view_id": view_id,
        }

    x1, y1, x2, y2 = bounds
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1), {
            "source": "empty_crop",
            "profile_name": str(profile.get("profile_name", "generic")),
            "package_type": package_type,
            "component_type": component_type,
            "view_id": view_id,
            "bounds": [x1, y1, x2, y2],
        }

    roi = image[y1:y2, x1:x2]
    return roi, (x1, y1), {
        "source": "package_profile_crop",
        "profile_name": str(profile.get("profile_name", "generic")),
        "expand_mode": str(profile.get("expand_mode", "body_bbox_expand")),
        "package_type": package_type,
        "component_type": component_type,
        "view_id": view_id,
        "scale_multiplier": float(scale_multiplier),
        "body_bbox": [int(v) for v in bbox],
        "body_size": [int(max(0, bbox[2] - bbox[0])), int(max(0, bbox[3] - bbox[1]))],
        "bounds": [x1, y1, x2, y2],
        "roi_size": [int(x2 - x1), int(y2 - y1)],
    }


def _scaled_profile(profile: dict[str, Any], *, scale_multiplier: float) -> dict[str, Any]:
    if abs(scale_multiplier - 1.0) < 1e-6:
        return dict(profile)
    scaled = dict(profile)
    for key in ("major_pad_ratio", "minor_pad_ratio"):
        if key in scaled:
            scaled[key] = float(scaled[key]) * float(scale_multiplier)
    for key in (
        "major_pad_before_ratio",
        "major_pad_after_ratio",
        "minor_pad_before_ratio",
        "minor_pad_after_ratio",
    ):
        if key in scaled:
            scaled[key] = float(scaled[key]) * float(scale_multiplier)
    for key in (
        "min_major_pad_px",
        "min_minor_pad_px",
        "min_major_span_px",
        "min_minor_span_px",
        "min_roi_w",
        "min_roi_h",
    ):
        if key in scaled:
            scaled[key] = max(1, int(round(float(scaled[key]) * float(scale_multiplier))))
    scaled["profile_name"] = f"{profile.get('profile_name', 'generic')}@x{scale_multiplier:.2f}"
    scaled["scale_multiplier"] = float(scale_multiplier)
    return scaled


def _bbox_mostly_outside_image(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> bool:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    img_h, img_w = image_shape
    if x2 <= x1 or y2 <= y1:
        return True
    inter_x1 = max(0, x1)
    inter_y1 = max(0, y1)
    inter_x2 = min(img_w, x2)
    inter_y2 = min(img_h, y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    return inter_area / float(bbox_area) < 0.15


def _compute_crop_bounds(
    *,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    orientation: float,
    obb_corners: list[list[float]] | np.ndarray | None,
    profile: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    img_h, img_w = image_shape
    if x2 <= x1 or y2 <= y1:
        return None

    if obb_corners is not None:
        corners = np.asarray(obb_corners, dtype=np.float32).reshape(-1, 2)
        if len(corners) == 4:
            bounds = _expanded_bounds_from_obb(corners, profile, image_shape=(img_h, img_w))
            if bounds is not None:
                return bounds

    return _expanded_bounds_from_bbox(
        bbox=(x1, y1, x2, y2),
        image_shape=(img_h, img_w),
        orientation=orientation,
        profile=profile,
    )


def _expanded_bounds_from_obb(
    corners: np.ndarray,
    profile: dict[str, Any],
    *,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    center = corners.mean(axis=0)
    edge_specs = []
    for idx in range(4):
        p0 = corners[idx]
        p1 = corners[(idx + 1) % 4]
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length > 1e-6:
            edge_specs.append((length, vec / length))
    if not edge_specs:
        return None

    edge_specs.sort(key=lambda item: item[0], reverse=True)
    major_len, major_dir = edge_specs[0]
    minor_len = edge_specs[-1][0]
    minor_dir = np.array([-major_dir[1], major_dir[0]], dtype=np.float32)

    major_before_pad = _pad_from_profile(
        profile,
        axis="major",
        side="before",
        extent=major_len,
    )
    major_after_pad = _pad_from_profile(
        profile,
        axis="major",
        side="after",
        extent=major_len,
    )
    minor_before_pad = _pad_from_profile(
        profile,
        axis="minor",
        side="before",
        extent=minor_len,
    )
    minor_after_pad = _pad_from_profile(
        profile,
        axis="minor",
        side="after",
        extent=minor_len,
    )

    min_major_span = float(profile.get("min_major_span_px", 0))
    min_minor_span = float(profile.get("min_minor_span_px", 0))
    current_major_span = major_len + major_before_pad + major_after_pad
    current_minor_span = minor_len + minor_before_pad + minor_after_pad
    if min_major_span > 0 and current_major_span < min_major_span:
        extra = (min_major_span - current_major_span) / 2.0
        major_before_pad += extra
        major_after_pad += extra
    if min_minor_span > 0 and current_minor_span < min_minor_span:
        extra = (min_minor_span - current_minor_span) / 2.0
        minor_before_pad += extra
        minor_after_pad += extra
    expanded = np.array(
        [
            center - major_dir * (major_len / 2.0 + major_before_pad) - minor_dir * (minor_len / 2.0 + minor_before_pad),
            center + major_dir * (major_len / 2.0 + major_after_pad) - minor_dir * (minor_len / 2.0 + minor_before_pad),
            center + major_dir * (major_len / 2.0 + major_after_pad) + minor_dir * (minor_len / 2.0 + minor_after_pad),
            center - major_dir * (major_len / 2.0 + major_before_pad) + minor_dir * (minor_len / 2.0 + minor_after_pad),
        ],
        dtype=np.float32,
    )

    x1 = int(np.floor(expanded[:, 0].min()))
    y1 = int(np.floor(expanded[:, 1].min()))
    x2 = int(np.ceil(expanded[:, 0].max()))
    y2 = int(np.ceil(expanded[:, 1].max()))
    return _enforce_min_roi_size((x1, y1, x2, y2), image_shape=image_shape, profile=profile)


def _expanded_bounds_from_bbox(
    *,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    orientation: float,
    profile: dict[str, Any],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    img_h, img_w = image_shape
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    angle = abs(orientation % 180.0)
    horizontal = angle < 45.0 or angle > 135.0
    major_extent = width if horizontal else height
    minor_extent = height if horizontal else width
    major_before_pad = _pad_from_profile(
        profile,
        axis="major",
        side="before",
        extent=major_extent,
    )
    major_after_pad = _pad_from_profile(
        profile,
        axis="major",
        side="after",
        extent=major_extent,
    )
    minor_before_pad = _pad_from_profile(
        profile,
        axis="minor",
        side="before",
        extent=minor_extent,
    )
    minor_after_pad = _pad_from_profile(
        profile,
        axis="minor",
        side="after",
        extent=minor_extent,
    )
    min_major_span = float(profile.get("min_major_span_px", 0))
    min_minor_span = float(profile.get("min_minor_span_px", 0))

    if horizontal:
        body_major_start = x1
        body_major_end = x2
        body_minor_start = y1
        body_minor_end = y2
    else:
        body_major_start = y1
        body_major_end = y2
        body_minor_start = x1
        body_minor_end = x2

    major_start = int(np.floor(body_major_start - major_before_pad))
    major_end = int(np.ceil(body_major_end + major_after_pad))
    minor_start = int(np.floor(body_minor_start - minor_before_pad))
    minor_end = int(np.ceil(body_minor_end + minor_after_pad))

    major_span = major_end - major_start
    minor_span = minor_end - minor_start
    if min_major_span > 0 and major_span < min_major_span:
        major_center = (body_major_start + body_major_end) / 2.0
        half_major = min_major_span / 2.0
        major_start = int(np.floor(major_center - half_major))
        major_end = int(np.ceil(major_center + half_major))
    if min_minor_span > 0 and minor_span < min_minor_span:
        minor_center = (body_minor_start + body_minor_end) / 2.0
        half_minor = min_minor_span / 2.0
        minor_start = int(np.floor(minor_center - half_minor))
        minor_end = int(np.ceil(minor_center + half_minor))

    if horizontal:
        ex1, ex2 = major_start, major_end
        ey1, ey2 = minor_start, minor_end
    else:
        ex1, ex2 = minor_start, minor_end
        ey1, ey2 = major_start, major_end

    return _enforce_min_roi_size((ex1, ey1, ex2, ey2), image_shape=(img_h, img_w), profile=profile)


def _pad_from_profile(
    profile: dict[str, Any],
    *,
    axis: str,
    side: str,
    extent: float,
) -> float:
    if axis not in {"major", "minor"}:
        raise ValueError(f"unsupported axis: {axis}")
    if side not in {"before", "after"}:
        raise ValueError(f"unsupported side: {side}")

    ratio_key = f"{axis}_pad_{side}_ratio"
    fallback_ratio_key = f"{axis}_pad_ratio"
    min_key = f"min_{axis}_pad_px"

    ratio = float(profile.get(ratio_key, profile.get(fallback_ratio_key, 0.18)))
    min_pad = float(profile.get(min_key, 6))
    return max(min_pad, extent * ratio)


def _enforce_min_roi_size(
    bounds: tuple[int, int, int, int],
    *,
    image_shape: tuple[int, int],
    profile: dict[str, Any],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bounds
    img_h, img_w = image_shape
    min_w = int(profile.get("min_roi_w", 32))
    min_h = int(profile.get("min_roi_h", 32))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    width = x2 - x1
    height = y2 - y1
    if width < min_w:
        cx = (x1 + x2) / 2.0
        half = min_w / 2.0
        x1 = max(0, int(np.floor(cx - half)))
        x2 = min(img_w, int(np.ceil(cx + half)))
    if height < min_h:
        cy = (y1 + y2) / 2.0
        half = min_h / 2.0
        y1 = max(0, int(np.floor(cy - half)))
        y2 = min(img_h, int(np.ceil(cy + half)))

    return x1, y1, x2, y2
