"""
IC 封装识别 (DIP8 / DIP14).

只回答一个问题: 给定一个 IC 检测框, 它是 DIP8 还是 DIP14?
- 不生成引脚, 不处理 notch_direction, 也不碰 S1.5 / S2.
- 上层在 S1 调用, 把结果直接写进 detection 的 package_type 字段.

识别优先级:
1. 模型类别名已经能区分 (ic_dip8 / IC_DIP8 / dip8 / ic_dip14 / ...) -> 直接用.
2. 模型只输出 IC 时, 用 bbox 在面包板数字列轴上的覆盖数推断:
     ~4 列 -> dip8, ~7 列 -> dip14.
3. 否则 unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


PACKAGE_DIP8 = "dip8"
PACKAGE_DIP14 = "dip14"
PACKAGE_UNKNOWN = "unknown"

SOURCE_MODEL_CLASS = "model_class"
SOURCE_BBOX_COLUMN = "bbox_column_inference"
SOURCE_UNKNOWN = "unknown"

# 包名匹配: 大小写不敏感, 既兼容 ic_dip8 这种带前缀的, 也兼容裸 dip8.
_DIP_TAG_RE = re.compile(r"dip[\s_\-]*(\d+)", re.IGNORECASE)

# bbox 列覆盖判别区间, 略宽于 4 / 7 给一点容差.
_DIP8_COLUMN_SOFT = {3, 5}
_DIP14_COLUMN_SOFT = {6, 8}


@dataclass(frozen=True)
class PackageInference:
    package_type: str
    package_confidence: float
    package_source: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "package_type": self.package_type,
            "package_confidence": float(self.package_confidence),
            "package_source": self.package_source,
            "package_inference_metadata": dict(self.metadata),
        }


def infer_ic_package(
    *,
    class_name: str,
    bbox: tuple[float, float, float, float] | list[float] | None,
    calibrator: Any | None = None,
    top_image: np.ndarray | None = None,
) -> PackageInference:
    """对单个 IC 检测框推断封装类型. 不修改入参."""
    by_class = _from_class_name(class_name)
    if by_class is not None:
        return by_class

    by_bbox = _from_bbox_column_coverage(bbox=bbox, calibrator=calibrator, top_image=top_image)
    if by_bbox is not None:
        return by_bbox

    return PackageInference(
        package_type=PACKAGE_UNKNOWN,
        package_confidence=0.0,
        package_source=SOURCE_UNKNOWN,
        metadata={"reason": "no_class_hint_and_no_grid"},
    )


def _from_class_name(class_name: str | None) -> PackageInference | None:
    raw = str(class_name or "").strip()
    if not raw:
        return None
    match = _DIP_TAG_RE.search(raw)
    if not match:
        return None
    try:
        pin_count = int(match.group(1))
    except (TypeError, ValueError):
        return None
    if pin_count == 8:
        return PackageInference(
            package_type=PACKAGE_DIP8,
            package_confidence=1.0,
            package_source=SOURCE_MODEL_CLASS,
            metadata={"raw_class_name": raw},
        )
    if pin_count == 14:
        return PackageInference(
            package_type=PACKAGE_DIP14,
            package_confidence=1.0,
            package_source=SOURCE_MODEL_CLASS,
            metadata={"raw_class_name": raw},
        )
    return None


def _from_bbox_column_coverage(
    *,
    bbox: tuple[float, float, float, float] | list[float] | None,
    calibrator: Any | None,
    top_image: np.ndarray | None,
) -> PackageInference | None:
    if bbox is None or len(bbox) < 4:
        return None
    if calibrator is None:
        return None

    if not getattr(calibrator, "is_grid_ready", False):
        if top_image is None or not hasattr(calibrator, "ensure_calibrated"):
            return None
        try:
            calibrator.ensure_calibrated(top_image)
        except Exception as exc:
            logger.debug("IC package inference: ensure_calibrated failed: %s", exc)
            return None
        if not getattr(calibrator, "is_grid_ready", False):
            return None

    row_coords = getattr(calibrator, "row_coords", None)
    if row_coords is None or len(row_coords) == 0:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    try:
        bp1 = calibrator.frame_pixel_to_board_point(x1, y1)
        bp2 = calibrator.frame_pixel_to_board_point(x2, y2)
    except Exception as exc:
        logger.debug("IC package inference: frame_pixel_to_board_point failed: %s", exc)
        return None

    landscape = bool(getattr(calibrator, "landscape", False) or getattr(calibrator, "_landscape", False))
    if landscape:
        lo = min(bp1[0], bp2[0])
        hi = max(bp1[0], bp2[0])
    else:
        lo = min(bp1[1], bp2[1])
        hi = max(bp1[1], bp2[1])

    coords = np.asarray(row_coords, dtype=np.float32)
    diffs = np.diff(np.sort(coords))
    diffs = diffs[diffs > 1e-3]
    pitch = float(np.median(diffs)) if diffs.size > 0 else 0.0
    margin = pitch * 0.25 if pitch > 0 else 0.0
    inside_mask = (coords >= (lo - margin)) & (coords <= (hi + margin))
    column_count = int(inside_mask.sum())

    package_type, confidence = _column_count_to_package(column_count)
    metadata = {
        "column_count": column_count,
        "bbox_board_range": [float(lo), float(hi)],
        "pitch_px": round(pitch, 4),
    }
    if package_type == PACKAGE_UNKNOWN:
        return PackageInference(
            package_type=PACKAGE_UNKNOWN,
            package_confidence=0.0,
            package_source=SOURCE_UNKNOWN,
            metadata={**metadata, "reason": "column_count_out_of_range"},
        )
    return PackageInference(
        package_type=package_type,
        package_confidence=confidence,
        package_source=SOURCE_BBOX_COLUMN,
        metadata=metadata,
    )


def _column_count_to_package(count: int) -> tuple[str, float]:
    if count == 4:
        return PACKAGE_DIP8, 0.9
    if count in _DIP8_COLUMN_SOFT:
        return PACKAGE_DIP8, 0.6
    if count == 7:
        return PACKAGE_DIP14, 0.9
    if count in _DIP14_COLUMN_SOFT:
        return PACKAGE_DIP14, 0.6
    return PACKAGE_UNKNOWN, 0.0
