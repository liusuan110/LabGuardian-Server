"""Projection helpers for mapping visual pin evidence onto the 2D board plane."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


TRUE_3D_METHODS = {
    "project_points_3d_to_top_2d",
    "orthographic_3d_to_board_2d",
}

BOARD_DIRECT_METHODS = TRUE_3D_METHODS | {
    "provided_board_2d_point",
    "provided_projected_top_keypoint",
}


@dataclass(frozen=True)
class BoardProjection:
    """Resolved mapping point for one pin observation."""

    source_view_id: str
    method: str
    input_keypoint: tuple[float, float] | None = None
    board_point: tuple[float, float] | None = None
    projected_frame_point: tuple[float, float] | None = None
    target_view_id: str = "top"
    used_3d: bool = False
    reason: str = ""

    @property
    def should_use_board_point_for_mapping(self) -> bool:
        return self.board_point is not None and self.method in BOARD_DIRECT_METHODS

    def to_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_view_id": self.source_view_id,
            "target_view_id": self.target_view_id,
            "method": self.method,
            "used_3d": self.used_3d,
        }
        if self.input_keypoint is not None:
            payload["input_keypoint"] = _point2_to_list(self.input_keypoint)
        if self.projected_frame_point is not None:
            payload["projected_frame_point"] = _point2_to_list(self.projected_frame_point)
        if self.board_point is not None:
            payload["board_2d_point"] = _point2_to_list(self.board_point)
        if self.reason:
            payload["reason"] = self.reason
        return payload


def resolve_pin_board_projection(
    *,
    view_id: str,
    keypoint: Any,
    per_view_metadata: Mapping[str, Any] | None,
    pin_metadata: Mapping[str, Any] | None,
    calibrator: Any,
) -> BoardProjection:
    """Resolve a pin keypoint to the shared 2D board plane.

    The preferred path consumes an upstream 3D pin point plus camera projection data.
    When that data is absent, top-view keypoints and legacy side-view pixels keep the
    existing behavior while explicitly marking the fallback method.
    """
    input_point = _coerce_point2(keypoint)
    if input_point is None:
        return BoardProjection(
            source_view_id=view_id,
            method="unavailable",
            reason="missing_keypoint",
        )

    per_view = dict(per_view_metadata or {})
    pin_meta = dict(pin_metadata or {})
    target_view_id = str(
        per_view.get("target_view_id")
        or (per_view.get("projection") or {}).get("target_view_id")
        or (pin_meta.get("projection") or {}).get("target_view_id")
        or "top"
    )

    provided_board = _first_point2(
        per_view,
        ("board_2d_point", "projected_board_point", "mapping_board_point"),
    )
    if provided_board is None:
        provided_board = _point_by_view(pin_meta, "board_2d_point_by_view", view_id)
    if provided_board is not None:
        return BoardProjection(
            source_view_id=view_id,
            target_view_id=target_view_id,
            input_keypoint=input_point,
            board_point=provided_board,
            method="provided_board_2d_point",
        )

    point_3d = _first_point3(per_view, ("point_3d", "world_point_3d", "pin_point_3d"))
    if point_3d is None:
        point_3d = _point_by_view(pin_meta, "point_3d_by_view", view_id, dims=3)
    if point_3d is not None:
        projection_cfg = _projection_config(
            pin_meta=pin_meta,
            per_view=per_view,
            target_view_id=target_view_id,
        )
        projected_frame = project_point_3d_to_2d(point_3d, projection_cfg)
        if projected_frame is not None:
            board_point = _frame_to_board_point(calibrator, projected_frame)
            return BoardProjection(
                source_view_id=view_id,
                target_view_id=target_view_id,
                input_keypoint=input_point,
                projected_frame_point=projected_frame,
                board_point=board_point,
                method="project_points_3d_to_top_2d",
                used_3d=True,
            )

        board_point = (float(point_3d[0]), float(point_3d[1]))
        return BoardProjection(
            source_view_id=view_id,
            target_view_id=target_view_id,
            input_keypoint=input_point,
            board_point=board_point,
            method="orthographic_3d_to_board_2d",
            used_3d=True,
            reason="missing_camera_matrix",
        )

    projected_top = _first_point2(
        per_view,
        (
            "projected_top_keypoint",
            "projected_frame_point",
            "top_view_keypoint",
            "projected_keypoint",
        ),
    )
    if projected_top is None:
        projected_top = _point_by_view(pin_meta, "projected_keypoint_by_view", view_id)
    if projected_top is not None:
        return BoardProjection(
            source_view_id=view_id,
            target_view_id=target_view_id,
            input_keypoint=input_point,
            projected_frame_point=projected_top,
            board_point=_frame_to_board_point(calibrator, projected_top),
            method="provided_projected_top_keypoint",
        )

    board_point = _frame_to_board_point(calibrator, input_point)
    return BoardProjection(
        source_view_id=view_id,
        target_view_id=target_view_id,
        input_keypoint=input_point,
        board_point=board_point,
        method=(
            "top_frame_to_board_2d"
            if view_id == "top"
            else "legacy_frame_pixel_fallback"
        ),
        reason="" if view_id == "top" else "no_3d_projection_metadata",
    )


def project_point_3d_to_2d(
    point_3d: Sequence[float],
    projection_cfg: Mapping[str, Any],
) -> tuple[float, float] | None:
    """Project one 3D point into a 2D image point with OpenCV camera parameters."""
    camera_matrix = projection_cfg.get("camera_matrix")
    if camera_matrix is None:
        return None

    try:
        obj = np.asarray(point_3d, dtype=np.float32).reshape(1, 1, 3)
        matrix = np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3)
        rvec = np.asarray(
            projection_cfg.get("rvec", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        ).reshape(3, 1)
        tvec = np.asarray(
            projection_cfg.get("tvec", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        ).reshape(3, 1)
        dist_coeffs = np.asarray(
            projection_cfg.get("dist_coeffs", []),
            dtype=np.float32,
        ).reshape(-1, 1)
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, dist_coeffs)
    except Exception as exc:
        logger.warning("3D pin projection failed: %s", exc)
        return None

    point = projected.reshape(-1, 2)[0]
    return (float(point[0]), float(point[1]))


def _projection_config(
    *,
    pin_meta: Mapping[str, Any],
    per_view: Mapping[str, Any],
    target_view_id: str,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for source in (pin_meta.get("projection"), per_view.get("projection"), per_view):
        if isinstance(source, Mapping):
            cfg.update({k: v for k, v in source.items() if k not in {"cameras", "camera_by_view"}})
            cameras = source.get("cameras") or source.get("camera_by_view")
            if isinstance(cameras, Mapping):
                target_cfg = cameras.get(target_view_id)
                if isinstance(target_cfg, Mapping):
                    cfg.update(target_cfg)
    return cfg


def _frame_to_board_point(calibrator: Any, point: tuple[float, float]) -> tuple[float, float]:
    if hasattr(calibrator, "frame_pixel_to_board_point"):
        return tuple(float(v) for v in calibrator.frame_pixel_to_board_point(point[0], point[1]))
    return (float(point[0]), float(point[1]))


def _first_point2(payload: Mapping[str, Any], keys: Iterable[str]) -> tuple[float, float] | None:
    for key in keys:
        point = _coerce_point2(payload.get(key))
        if point is not None:
            return point
    return None


def _first_point3(payload: Mapping[str, Any], keys: Iterable[str]) -> tuple[float, float, float] | None:
    for key in keys:
        point = _coerce_point3(payload.get(key))
        if point is not None:
            return point
    return None


def _point_by_view(
    payload: Mapping[str, Any],
    key: str,
    view_id: str,
    *,
    dims: int = 2,
) -> Any:
    by_view = payload.get(key)
    if not isinstance(by_view, Mapping):
        return None
    value = by_view.get(view_id)
    return _coerce_point3(value) if dims == 3 else _coerce_point2(value)


def _coerce_point2(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        if isinstance(value, Mapping):
            value = (value.get("x"), value.get("y"))
        if len(value) < 2:  # type: ignore[arg-type]
            return None
        return (float(value[0]), float(value[1]))  # type: ignore[index]
    except (TypeError, ValueError):
        return None


def _coerce_point3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        if isinstance(value, Mapping):
            value = (value.get("x"), value.get("y"), value.get("z"))
        if len(value) < 3:  # type: ignore[arg-type]
            return None
        return (float(value[0]), float(value[1]), float(value[2]))  # type: ignore[index]
    except (TypeError, ValueError):
        return None


def _point2_to_list(point: tuple[float, float]) -> list[float]:
    return [float(point[0]), float(point[1])]
