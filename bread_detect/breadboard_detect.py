from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class GridModel:
    main_x0: float
    main_pitch_x: float
    upper_y0: float
    upper_pitch_y: float
    lower_y0: float
    lower_pitch_y: float
    rail_x0: float
    rail_pitch_x: float
    top_rail_y0: float
    top_rail_pitch_y: float
    bottom_rail_y0: float
    bottom_rail_pitch_y: float
    main_score: float
    rail_score: float


@dataclass(frozen=True)
class Region:
    name: str
    label: str
    color: tuple[int, int, int]
    corners_warp: list[list[float]]
    corners_image: list[list[float]]


@dataclass(frozen=True)
class Connection:
    name: str
    connection_type: str
    color: tuple[int, int, int]
    points_warp: list[list[float]]
    points_image: list[list[float]]
    axis: str = ""
    member_ids: list[str] = field(default_factory=list)
    member_count: int = 0
    adjacent_pairs: list[list[str]] = field(default_factory=list)


def odd_size(value: float, minimum: int = 3, maximum: int | None = None) -> int:
    size = int(round(value))
    if maximum is not None:
        size = min(size, maximum)
    size = max(size, minimum)
    return size if size % 2 == 1 else size + 1


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def contour_quad(contour: np.ndarray) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        return order_points(approx.reshape(4, 2))

    rect = cv2.minAreaRect(contour)
    return order_points(cv2.boxPoints(rect))


def detect_board_quad(image: np.ndarray) -> np.ndarray:
    """Return board corners in TL, TR, BR, BL order."""
    h, w = image.shape[:2]
    image_area = float(h * w)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    pale_mask = ((value > 145) & (saturation < 105)).astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (odd_size(w * 0.015, 15, 41), odd_size(h * 0.025, 15, 41))
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (odd_size(w * 0.008, 7, 25), odd_size(h * 0.012, 7, 25))
    )
    pale_mask = cv2.morphologyEx(pale_mask, cv2.MORPH_CLOSE, close_kernel)
    pale_mask = cv2.morphologyEx(pale_mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(pale_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 0.04 * image_area <= area <= 0.85 * image_area:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / max(ch, 1)
            if aspect > 1.8:
                candidates.append((area, contour))

    if candidates:
        contour = max(candidates, key=lambda item: item[0])[1]
        return contour_quad(contour)

    # Fallback for pale tables: connect the dark holes, printed marks, wires, and rails.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature_mask = ((gray < 170) | (saturation > 45)).astype(np.uint8) * 255
    feature_mask = cv2.morphologyEx(
        feature_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (odd_size(w * 0.028, 15, 61), odd_size(h * 0.012, 9, 35))),
    )
    feature_mask = cv2.morphologyEx(
        feature_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (odd_size(w * 0.010, 9, 31), odd_size(h * 0.035, 15, 61))),
    )
    feature_mask = cv2.morphologyEx(
        feature_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    contours, _ = cv2.findContours(feature_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fallback: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 0.03 * image_area <= area <= 0.90 * image_area:
            x, y, cw, ch = cv2.boundingRect(contour)
            if cw / max(ch, 1) > 1.8:
                fallback.append((area, contour))

    if not fallback:
        raise RuntimeError("Could not find a breadboard-shaped region.")

    contour = max(fallback, key=lambda item: item[0])[1]
    x, y, cw, ch = cv2.boundingRect(contour)
    return order_points(np.array([[x, y], [x + cw, y], [x + cw, y + ch], [x, y + ch]], dtype=np.float32))


def warp_board(image: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quad = order_points(quad)
    width_top = np.linalg.norm(quad[1] - quad[0])
    width_bottom = np.linalg.norm(quad[2] - quad[3])
    height_left = np.linalg.norm(quad[3] - quad[0])
    height_right = np.linalg.norm(quad[2] - quad[1])
    width = max(1, int(round(max(width_top, width_bottom))))
    height = max(1, int(round(max(height_left, height_right))))

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, dst)
    inverse = cv2.getPerspectiveTransform(dst, quad)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix, inverse


def score_image(warped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    kernel_size = odd_size(min(h, w) * 0.04, 11, 25)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)))
    return cv2.GaussianBlur(blackhat, (3, 3), 0).astype(np.float32)


def mean_score_at_points(integral: np.ndarray, width: int, height: int, xs: Iterable[float], ys: Iterable[float], radius: int = 4) -> float:
    x_values = np.asarray(list(xs), dtype=np.float32)
    y_values = np.asarray(list(ys), dtype=np.float32)
    if x_values.size == 0 or y_values.size == 0:
        return -1.0

    grid_x, grid_y = np.meshgrid(x_values, y_values)
    xi = np.rint(grid_x).astype(np.int32).reshape(-1)
    yi = np.rint(grid_y).astype(np.int32).reshape(-1)
    inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    if not np.any(inside):
        return -1.0

    xi = xi[inside]
    yi = yi[inside]
    x1 = np.clip(xi - radius, 0, width)
    x2 = np.clip(xi + radius + 1, 0, width)
    y1 = np.clip(yi - radius, 0, height)
    y2 = np.clip(yi + radius + 1, 0, height)

    sums = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    areas = (x2 - x1) * (y2 - y1)
    return float(np.mean(sums / np.maximum(areas, 1)))


def two_stage_search(
    scorer: Callable[[float, float], float],
    first_range: tuple[float, float],
    second_range: tuple[float, float],
    coarse_steps: int = 25,
    refine_steps: int = 17,
) -> tuple[float, float, float]:
    best_score = -1.0
    best_first = first_range[0]
    best_second = second_range[0]

    def scan(r1: tuple[float, float], r2: tuple[float, float], steps: int) -> tuple[float, float, float]:
        local_score = -1.0
        local_first = r1[0]
        local_second = r2[0]
        for first in np.linspace(r1[0], r1[1], steps):
            for second in np.linspace(r2[0], r2[1], steps):
                score = scorer(float(first), float(second))
                if score > local_score:
                    local_score = score
                    local_first = float(first)
                    local_second = float(second)
        return local_score, local_first, local_second

    best_score, best_first, best_second = scan(first_range, second_range, coarse_steps)
    first_span = (first_range[1] - first_range[0]) / max(coarse_steps - 1, 1)
    second_span = (second_range[1] - second_range[0]) / max(coarse_steps - 1, 1)
    refined = scan(
        (best_first - first_span, best_first + first_span),
        (best_second - second_span, best_second + second_span),
        refine_steps,
    )
    if refined[0] > best_score:
        best_score, best_first, best_second = refined
    return best_score, best_first, best_second


def fit_grid(warped: np.ndarray, main_columns: int) -> GridModel:
    scores = score_image(warped)
    height, width = scores.shape[:2]
    integral = cv2.integral(scores)

    main_row_guess = np.array([0.249, 0.296, 0.343, 0.390, 0.437, 0.576, 0.623, 0.670, 0.716, 0.763]) * height

    def main_x_scorer(x0: float, pitch: float) -> float:
        xs = [x0 + i * pitch for i in range(main_columns)]
        if xs[0] < 0 or xs[-1] > width - 1:
            return -1.0
        return mean_score_at_points(integral, width, height, xs, main_row_guess)

    main_score, main_x0, main_pitch_x = two_stage_search(
        main_x_scorer,
        (0.000 * width, 0.045 * width),
        (0.0140 * width, 0.0165 * width),
        coarse_steps=61,
        refine_steps=21,
    )
    main_phase_target = 0.023 * width
    phase_candidates: list[tuple[float, float, float]] = []
    for shift in range(-3, 4):
        shifted_x0 = main_x0 + shift * main_pitch_x
        last_x = shifted_x0 + (main_columns - 1) * main_pitch_x
        if 0 <= shifted_x0 and last_x <= width - 1:
            phase_candidates.append((abs(shifted_x0 - main_phase_target), shifted_x0, main_x_scorer(shifted_x0, main_pitch_x)))
    if phase_candidates:
        _, main_x0, main_score = min(phase_candidates, key=lambda item: item[0])

    main_xs = [main_x0 + i * main_pitch_x for i in range(main_columns)]

    def upper_y_scorer(y0: float, pitch: float) -> float:
        return mean_score_at_points(integral, width, height, main_xs, [y0 + i * pitch for i in range(5)])

    _, upper_y0, upper_pitch_y = two_stage_search(
        upper_y_scorer,
        (0.225 * height, 0.270 * height),
        (0.040 * height, 0.055 * height),
    )

    def lower_y_scorer(y0: float, pitch: float) -> float:
        return mean_score_at_points(integral, width, height, main_xs, [y0 + i * pitch for i in range(5)])

    _, lower_y0, lower_pitch_y = two_stage_search(
        lower_y_scorer,
        (0.545 * height, 0.600 * height),
        (0.040 * height, 0.055 * height),
    )

    rail_offsets = [group * 6 + hole for group in range(10) for hole in range(5)]
    rail_row_guess = np.array([0.060, 0.106, 0.897, 0.944]) * height

    def rail_x_scorer(x0: float, pitch: float) -> float:
        xs = [x0 + offset * pitch for offset in rail_offsets]
        return mean_score_at_points(integral, width, height, xs, rail_row_guess)

    rail_score, rail_x0, rail_pitch_x = two_stage_search(
        rail_x_scorer,
        (main_x0 + 1.20 * main_pitch_x, main_x0 + 2.35 * main_pitch_x),
        (0.950 * main_pitch_x, 1.050 * main_pitch_x),
        coarse_steps=41,
        refine_steps=21,
    )
    rail_phase_target = 0.052 * width
    rail_phase_candidates: list[tuple[float, float, float]] = []
    for shift in range(-3, 4):
        shifted_x0 = rail_x0 + shift * rail_pitch_x
        last_x = shifted_x0 + (9 * 6 + 4) * rail_pitch_x
        if 0 <= shifted_x0 and last_x <= width - 1:
            rail_phase_candidates.append((abs(shifted_x0 - rail_phase_target), shifted_x0, rail_x_scorer(shifted_x0, rail_pitch_x)))
    if rail_phase_candidates:
        _, rail_x0, rail_score = min(rail_phase_candidates, key=lambda item: item[0])

    rail_xs = [rail_x0 + offset * rail_pitch_x for offset in rail_offsets]

    def top_rail_y_scorer(y0: float, pitch: float) -> float:
        return mean_score_at_points(integral, width, height, rail_xs, [y0, y0 + pitch])

    _, top_rail_y0, top_rail_pitch_y = two_stage_search(
        top_rail_y_scorer,
        (0.045 * height, 0.075 * height),
        (0.035 * height, 0.060 * height),
    )

    def bottom_rail_y_scorer(y0: float, pitch: float) -> float:
        return mean_score_at_points(integral, width, height, rail_xs, [y0, y0 + pitch])

    _, bottom_rail_y0, bottom_rail_pitch_y = two_stage_search(
        bottom_rail_y_scorer,
        (0.870 * height, 0.920 * height),
        (0.035 * height, 0.060 * height),
    )

    return GridModel(
        main_x0=main_x0,
        main_pitch_x=main_pitch_x,
        upper_y0=upper_y0,
        upper_pitch_y=upper_pitch_y,
        lower_y0=lower_y0,
        lower_pitch_y=lower_pitch_y,
        rail_x0=rail_x0,
        rail_pitch_x=rail_pitch_x,
        top_rail_y0=top_rail_y0,
        top_rail_pitch_y=top_rail_pitch_y,
        bottom_rail_y0=bottom_rail_y0,
        bottom_rail_pitch_y=bottom_rail_pitch_y,
        main_score=main_score,
        rail_score=rail_score,
    )


def perspective_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, matrix)
    return transformed.reshape(-1, 2)


def build_holes(model: GridModel, inverse_matrix: np.ndarray, score_map: np.ndarray, main_columns: int) -> list[dict[str, object]]:
    height, width = score_map.shape[:2]
    integral = cv2.integral(score_map.astype(np.float32))
    holes: list[dict[str, object]] = []

    def add_hole(hole_id: str, section: str, row: str, column: int, x_warp: float, y_warp: float, group: int | None = None, group_index: int | None = None) -> None:
        original = perspective_points(np.array([[x_warp, y_warp]], dtype=np.float32), inverse_matrix)[0]
        local_score = mean_score_at_points(integral, width, height, [x_warp], [y_warp])
        holes.append(
            {
                "id": hole_id,
                "section": section,
                "row": row,
                "column": column,
                "group": group,
                "group_index": group_index,
                "x_warp": round(float(x_warp), 3),
                "y_warp": round(float(y_warp), 3),
                "x_image": round(float(original[0]), 3),
                "y_image": round(float(original[1]), 3),
                "visible_score": round(float(local_score), 3),
            }
        )

    main_rows = [
        ("A", model.upper_y0 + 0 * model.upper_pitch_y),
        ("B", model.upper_y0 + 1 * model.upper_pitch_y),
        ("C", model.upper_y0 + 2 * model.upper_pitch_y),
        ("D", model.upper_y0 + 3 * model.upper_pitch_y),
        ("E", model.upper_y0 + 4 * model.upper_pitch_y),
        ("F", model.lower_y0 + 0 * model.lower_pitch_y),
        ("G", model.lower_y0 + 1 * model.lower_pitch_y),
        ("H", model.lower_y0 + 2 * model.lower_pitch_y),
        ("I", model.lower_y0 + 3 * model.lower_pitch_y),
        ("J", model.lower_y0 + 4 * model.lower_pitch_y),
    ]
    for row_label, y_warp in main_rows:
        section = "terminal_upper" if row_label in "ABCDE" else "terminal_lower"
        for column in range(main_columns):
            x_warp = model.main_x0 + column * model.main_pitch_x
            add_hole(f"{row_label}{column:02d}", section, row_label, column, x_warp, y_warp)

    rail_rows = [
        ("top_pos", model.top_rail_y0),
        ("top_neg", model.top_rail_y0 + model.top_rail_pitch_y),
        ("bottom_pos", model.bottom_rail_y0),
        ("bottom_neg", model.bottom_rail_y0 + model.bottom_rail_pitch_y),
    ]
    for row_label, y_warp in rail_rows:
        for group in range(10):
            for group_index in range(5):
                column = group * 5 + group_index
                offset = group * 6 + group_index
                x_warp = model.rail_x0 + offset * model.rail_pitch_x
                add_hole(
                    f"{row_label}_{column:02d}",
                    "power_rail",
                    row_label,
                    column,
                    x_warp,
                    y_warp,
                    group=group,
                    group_index=group_index,
                )

    return holes


def clamp_rect(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    clamped = corners.copy().astype(np.float32)
    clamped[:, 0] = np.clip(clamped[:, 0], 0, width - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, height - 1)
    return clamped


def rect_corners(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def rounded_points(points: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in points]


def build_regions(model: GridModel, inverse_matrix: np.ndarray, warped_size: tuple[int, int], main_columns: int) -> list[Region]:
    width, height = warped_size
    main_x1 = model.main_x0 - 0.55 * model.main_pitch_x
    main_x2 = model.main_x0 + (main_columns - 1) * model.main_pitch_x + 0.55 * model.main_pitch_x
    rail_x1 = model.rail_x0 - 0.65 * model.rail_pitch_x
    rail_x2 = model.rail_x0 + (9 * 6 + 4) * model.rail_pitch_x + 0.65 * model.rail_pitch_x

    specs = [
        (
            "top_power_rail",
            "Top power rail",
            (255, 120, 0),
            rect_corners(
                rail_x1,
                model.top_rail_y0 - 0.75 * model.top_rail_pitch_y,
                rail_x2,
                model.top_rail_y0 + 1.75 * model.top_rail_pitch_y,
            ),
        ),
        (
            "upper_terminal_strip",
            "A-E terminal strip",
            (0, 170, 255),
            rect_corners(
                main_x1,
                model.upper_y0 - 0.70 * model.upper_pitch_y,
                main_x2,
                model.upper_y0 + 4.70 * model.upper_pitch_y,
            ),
        ),
        (
            "lower_terminal_strip",
            "F-J terminal strip",
            (0, 220, 120),
            rect_corners(
                main_x1,
                model.lower_y0 - 0.70 * model.lower_pitch_y,
                main_x2,
                model.lower_y0 + 4.70 * model.lower_pitch_y,
            ),
        ),
        (
            "bottom_power_rail",
            "Bottom power rail",
            (255, 120, 0),
            rect_corners(
                rail_x1,
                model.bottom_rail_y0 - 0.75 * model.bottom_rail_pitch_y,
                rail_x2,
                model.bottom_rail_y0 + 1.75 * model.bottom_rail_pitch_y,
            ),
        ),
    ]

    regions: list[Region] = []
    for name, label, color, corners in specs:
        corners = clamp_rect(corners, width, height)
        image_corners = perspective_points(corners, inverse_matrix)
        regions.append(Region(name, label, color, rounded_points(corners), rounded_points(image_corners)))
    return regions


def build_connections(model: GridModel, inverse_matrix: np.ndarray, main_columns: int) -> list[Connection]:
    connections: list[Connection] = []

    def add_connection(name: str, connection_type: str, color: tuple[int, int, int], points: np.ndarray) -> None:
        image_points = perspective_points(points, inverse_matrix)
        connections.append(Connection(name, connection_type, color, rounded_points(points), rounded_points(image_points)))

    terminal_color = (0, 255, 255)
    for column in range(main_columns):
        x_warp = model.main_x0 + column * model.main_pitch_x
        add_connection(
            f"terminal_upper_col_{column:02d}",
            "vertical_terminal_strip",
            terminal_color,
            np.array(
                [
                    [x_warp, model.upper_y0],
                    [x_warp, model.upper_y0 + 4 * model.upper_pitch_y],
                ],
                dtype=np.float32,
            ),
        )
        add_connection(
            f"terminal_lower_col_{column:02d}",
            "vertical_terminal_strip",
            terminal_color,
            np.array(
                [
                    [x_warp, model.lower_y0],
                    [x_warp, model.lower_y0 + 4 * model.lower_pitch_y],
                ],
                dtype=np.float32,
            ),
        )

    rail_color = (255, 0, 0)
    rail_x1 = model.rail_x0
    rail_x2 = model.rail_x0 + (9 * 6 + 4) * model.rail_pitch_x
    rail_rows = [
        ("top_pos", model.top_rail_y0),
        ("top_neg", model.top_rail_y0 + model.top_rail_pitch_y),
        ("bottom_pos", model.bottom_rail_y0),
        ("bottom_neg", model.bottom_rail_y0 + model.bottom_rail_pitch_y),
    ]
    for rail_name, y_warp in rail_rows:
        add_connection(
            f"power_rail_{rail_name}",
            "horizontal_power_rail",
            rail_color,
            np.array([[rail_x1, y_warp], [rail_x2, y_warp]], dtype=np.float32),
        )

    return connections


def annotate_connectivity(
    holes: list[dict[str, object]],
    connections: list[Connection],
) -> tuple[list[dict[str, object]], list[Connection]]:
    row_order = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "J": 9,
        "top_pos": 10,
        "top_neg": 11,
        "bottom_pos": 12,
        "bottom_neg": 13,
    }
    connection_meta: dict[str, dict[str, object]] = {}

    upper_groups: dict[int, list[dict[str, object]]] = {}
    lower_groups: dict[int, list[dict[str, object]]] = {}
    rail_groups: dict[str, list[dict[str, object]]] = {}

    for hole in holes:
        section = str(hole["section"])
        if section == "terminal_upper":
            upper_groups.setdefault(int(hole["column"]), []).append(hole)
        elif section == "terminal_lower":
            lower_groups.setdefault(int(hole["column"]), []).append(hole)
        elif section == "power_rail":
            rail_groups.setdefault(str(hole["row"]), []).append(hole)

    def register_group(
        name: str,
        connection_type: str,
        axis: str,
        members: list[dict[str, object]],
    ) -> None:
        member_ids = [str(member["id"]) for member in members]
        connection_meta[name] = {
            "name": name,
            "connection_type": connection_type,
            "axis": axis,
            "member_ids": member_ids,
            "member_count": len(member_ids),
            "adjacent_pairs": [[member_ids[index], member_ids[index + 1]] for index in range(len(member_ids) - 1)],
        }

    for column, members in sorted(upper_groups.items()):
        ordered = sorted(members, key=lambda item: row_order[str(item["row"])])
        register_group(f"terminal_upper_col_{column:02d}", "vertical_terminal_strip", "vertical", ordered)

    for column, members in sorted(lower_groups.items()):
        ordered = sorted(members, key=lambda item: row_order[str(item["row"])])
        register_group(f"terminal_lower_col_{column:02d}", "vertical_terminal_strip", "vertical", ordered)

    for row_name, members in sorted(rail_groups.items(), key=lambda item: row_order.get(item[0], 999)):
        ordered = sorted(members, key=lambda item: int(item["column"]))
        register_group(f"power_rail_{row_name}", "horizontal_power_rail", "horizontal", ordered)

    hole_lookup = {str(hole["id"]): hole for hole in holes}
    for meta in connection_meta.values():
        member_ids = list(meta["member_ids"])
        member_count = int(meta["member_count"])
        for index, hole_id in enumerate(member_ids):
            adjacent_ids: list[str] = []
            if index > 0:
                adjacent_ids.append(member_ids[index - 1])
            if index + 1 < member_count:
                adjacent_ids.append(member_ids[index + 1])
            hole = hole_lookup[hole_id]
            hole["connection_name"] = str(meta["name"])
            hole["connection_type"] = str(meta["connection_type"])
            hole["connection_axis"] = str(meta["axis"])
            hole["connection_index"] = index
            hole["connection_member_count"] = member_count
            hole["adjacent_connected_ids"] = adjacent_ids

    enriched_connections: list[Connection] = []
    for connection in connections:
        meta = connection_meta.get(connection.name)
        if meta is None:
            enriched_connections.append(connection)
            continue
        enriched_connections.append(
            Connection(
                name=connection.name,
                connection_type=connection.connection_type,
                color=connection.color,
                points_warp=connection.points_warp,
                points_image=connection.points_image,
                axis=str(meta["axis"]),
                member_ids=list(meta["member_ids"]),
                member_count=int(meta["member_count"]),
                adjacent_pairs=[list(pair) for pair in meta["adjacent_pairs"]],
            )
        )

    return holes, enriched_connections


def draw_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    annotated = image.copy()
    pts = order_points(quad).astype(np.int32)
    cv2.polylines(annotated, [pts], True, (0, 0, 255), 4, cv2.LINE_AA)
    for index, (x, y) in enumerate(pts):
        cv2.circle(annotated, (int(x), int(y)), 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(annotated, str(index + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return annotated


def draw_holes(image: np.ndarray, holes: list[dict[str, object]], coordinate_key: str, draw_labels: bool = False) -> np.ndarray:
    annotated = image.copy()
    for hole in holes:
        x = int(round(float(hole[f"x_{coordinate_key}"])))
        y = int(round(float(hole[f"y_{coordinate_key}"])))
        is_rail = hole["section"] == "power_rail"
        color = (255, 0, 0) if is_rail else (0, 0, 255)
        radius = 2 if coordinate_key == "warp" else 3
        cv2.circle(annotated, (x, y), radius, color, -1, cv2.LINE_AA)
        if draw_labels:
            cv2.putText(annotated, str(hole["id"]), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1, cv2.LINE_AA)
    return annotated


def draw_region_and_connection_overlay(
    image: np.ndarray,
    holes: list[dict[str, object]],
    regions: list[Region],
    connections: list[Connection],
    coordinate_key: str,
    draw_labels: bool = False,
) -> np.ndarray:
    annotated = image.copy()
    overlay = annotated.copy()

    for region in regions:
        corners = np.asarray(region.corners_warp if coordinate_key == "warp" else region.corners_image, dtype=np.float32).astype(np.int32)
        cv2.fillPoly(overlay, [corners], region.color)
        cv2.polylines(annotated, [corners], True, region.color, 3, cv2.LINE_AA)
        label_point = tuple(corners[0])
        cv2.putText(
            annotated,
            region.label,
            (int(label_point[0]) + 8, int(label_point[1]) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 if coordinate_key == "warp" else 0.6,
            region.color,
            2,
            cv2.LINE_AA,
        )

    annotated = cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0)

    connection_layer = annotated.copy()
    for connection in connections:
        points = np.asarray(connection.points_warp if coordinate_key == "warp" else connection.points_image, dtype=np.float32).astype(np.int32)
        thickness = 2 if connection.connection_type == "vertical_terminal_strip" else 4
        cv2.polylines(connection_layer, [points], False, connection.color, thickness, cv2.LINE_AA)
    annotated = cv2.addWeighted(connection_layer, 0.55, annotated, 0.45, 0)

    return draw_holes(annotated, holes, coordinate_key, draw_labels)


def write_csv(path: Path, holes: list[dict[str, object]]) -> None:
    fieldnames = [
        "id",
        "section",
        "row",
        "column",
        "group",
        "group_index",
        "x_warp",
        "y_warp",
        "x_image",
        "y_image",
        "visible_score",
        "connection_name",
        "connection_type",
        "connection_axis",
        "connection_index",
        "connection_member_count",
        "adjacent_connected_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for hole in holes:
            row = {key: hole.get(key) for key in fieldnames}
            row["adjacent_connected_ids"] = ";".join(str(value) for value in hole.get("adjacent_connected_ids", []))
            writer.writerow(row)


def write_connections_csv(path: Path, connections: list[Connection]) -> None:
    fieldnames = [
        "name",
        "connection_type",
        "axis",
        "member_count",
        "member_ids",
        "adjacent_pairs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for connection in connections:
            writer.writerow(
                {
                    "name": connection.name,
                    "connection_type": connection.connection_type,
                    "axis": connection.axis,
                    "member_count": connection.member_count,
                    "member_ids": ";".join(connection.member_ids),
                    "adjacent_pairs": ";".join(f"{start}<->{end}" for start, end in connection.adjacent_pairs),
                }
            )


def process_image(image_path: Path, out_dir: Path, prefix: str, draw_labels: bool = False, main_columns: int = 63) -> dict[str, object]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    quad = detect_board_quad(image)
    warped, _, inverse_matrix = warp_board(image, quad)
    model = fit_grid(warped, main_columns)
    score_map = score_image(warped)
    holes = build_holes(model, inverse_matrix, score_map, main_columns)
    regions = build_regions(model, inverse_matrix, (warped.shape[1], warped.shape[0]), main_columns)
    connections = build_connections(model, inverse_matrix, main_columns)
    holes, connections = annotate_connectivity(holes, connections)

    paths = {
        "corners": out_dir / f"{prefix}_corners.png",
        "warped": out_dir / f"{prefix}_warped.png",
        "annotated_warped": out_dir / f"{prefix}_annotated_warped.png",
        "annotated_original": out_dir / f"{prefix}_annotated_original.png",
        "connectivity_warped": out_dir / f"{prefix}_connectivity_warped.png",
        "connectivity_original": out_dir / f"{prefix}_connectivity_original.png",
        "csv": out_dir / f"{prefix}_holes.csv",
        "connections_csv": out_dir / f"{prefix}_connections.csv",
        "json": out_dir / f"{prefix}_holes.json",
    }

    cv2.imwrite(str(paths["corners"]), draw_quad(image, quad))
    cv2.imwrite(str(paths["warped"]), warped)
    cv2.imwrite(str(paths["annotated_warped"]), draw_holes(warped, holes, "warp", draw_labels))
    cv2.imwrite(str(paths["annotated_original"]), draw_holes(image, holes, "image", draw_labels))
    cv2.imwrite(str(paths["connectivity_warped"]), draw_region_and_connection_overlay(warped, holes, regions, connections, "warp", draw_labels))
    cv2.imwrite(str(paths["connectivity_original"]), draw_region_and_connection_overlay(image, holes, regions, connections, "image", draw_labels))
    write_csv(paths["csv"], holes)
    write_connections_csv(paths["connections_csv"], connections)

    metadata = {
        "image": str(image_path),
        "hole_count": len(holes),
        "main_columns": main_columns,
        "quad_tl_tr_br_bl": [[round(float(x), 3), round(float(y), 3)] for x, y in order_points(quad)],
        "warped_size": {"width": warped.shape[1], "height": warped.shape[0]},
        "grid": model.__dict__,
        "regions": [asdict(region) for region in regions],
        "connections": [asdict(connection) for connection in connections],
        "holes": holes,
    }
    with paths["json"].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return {"paths": {key: str(value) for key, value in paths.items()}, **metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and annotate breadboard hole coordinates.")
    parser.add_argument("--image", default="bread.png", type=Path, help="Input breadboard image.")
    parser.add_argument("--out-dir", default=Path("outputs"), type=Path, help="Directory for generated files.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to input stem.")
    parser.add_argument("--main-columns", default=63, type=int, help="Terminal-strip columns. Standard 830-point boards use 63.")
    parser.add_argument("--draw-labels", action="store_true", help="Draw every hole id on annotated images. This is very cluttered.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.prefix or args.image.stem
    result = process_image(args.image, args.out_dir, prefix, args.draw_labels, args.main_columns)
    print(f"Detected {result['hole_count']} holes.")
    print(f"Board corners TL/TR/BR/BL: {result['quad_tl_tr_br_bl']}")
    for name, path in result["paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
