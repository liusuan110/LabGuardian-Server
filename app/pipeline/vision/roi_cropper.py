"""
ROI crop helpers for component-centered pin detection.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    margin_ratio: float = 0.15,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]
    pad_x = max(4, int((x2 - x1) * margin_ratio))
    pad_y = max(4, int((y2 - y1) * margin_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )


def crop_component_roi(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin_ratio: float = 0.15,
) -> tuple[np.ndarray, tuple[int, int]]:
    expanded = expand_bbox(bbox, image.shape[:2], margin_ratio=margin_ratio)
    x1, y1, x2, y2 = expanded
    return image[y1:y2, x1:x2], (x1, y1)
