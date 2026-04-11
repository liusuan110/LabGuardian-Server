"""
Shared image decode helpers for pipeline stages.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List

import cv2
import numpy as np


def view_id_for_index(index: int) -> str:
    defaults = ["top", "left_front", "right_front"]
    if index < len(defaults):
        return defaults[index]
    return f"aux_view_{index - len(defaults) + 1}"


def decode_images_b64(
    images_b64: List[str],
    *,
    logger: logging.Logger,
    stage_name: str,
) -> List[Dict[str, Any]]:
    """Decode base64 images and preserve per-view status.

    Returns a list with stable order and explicit decode metadata:
    - `view_id`
    - `image`
    - `decoded`
    - `error`
    """
    decoded: List[Dict[str, Any]] = []
    for idx, image_b64 in enumerate(images_b64):
        view_id = view_id_for_index(idx)
        try:
            data = base64.b64decode(image_b64)
            arr = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("cv2.imdecode returned None")
            decoded.append(
                {
                    "view_id": view_id,
                    "image": image,
                    "decoded": True,
                    "error": None,
                }
            )
        except Exception as exc:
            logger.warning("%s image decode failed for view %s: %s", stage_name, view_id, exc)
            decoded.append(
                {
                    "view_id": view_id,
                    "image": None,
                    "decoded": False,
                    "error": str(exc),
                }
            )
    return decoded


def decode_summary(decoded_images: List[Dict[str, Any]]) -> Dict[str, Any]:
    available_view_ids = [item["view_id"] for item in decoded_images if item.get("decoded")]
    dropped_view_ids = [item["view_id"] for item in decoded_images if not item.get("decoded")]
    errors = {
        item["view_id"]: item.get("error")
        for item in decoded_images
        if not item.get("decoded") and item.get("error")
    }
    return {
        "decoded_view_count": len(available_view_ids),
        "available_view_ids": available_view_ids,
        "dropped_view_ids": dropped_view_ids,
        "decode_errors": errors,
    }
