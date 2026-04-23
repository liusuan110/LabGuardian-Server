from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


POINT_LABELS = {"pin1", "pin2", "pin3"}


@dataclass
class LabeledInstance:
    image_path: Path
    image_width: int
    image_height: int
    class_name: str
    group_id: int | str
    bbox: tuple[float, float, float, float]
    points: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (max(1.0, x2 - x1), max(1.0, y2 - y1))


def iter_labelme_instances(root: Path) -> Iterable[LabeledInstance]:
    for json_path in sorted(root.rglob("*.json")):
        yield from parse_labelme_json(json_path)


def parse_labelme_json(json_path: Path) -> list[LabeledInstance]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_path = resolve_image_path(json_path, data.get("imagePath"))
    image_width = int(data.get("imageWidth") or 0)
    image_height = int(data.get("imageHeight") or 0)
    grouped: dict[int | str, dict[str, object]] = {}

    for shape in data.get("shapes") or []:
        label = str(shape.get("label") or "").strip()
        shape_type = str(shape.get("shape_type") or "")
        group_id = shape.get("group_id")
        if group_id is None:
            continue
        slot = grouped.setdefault(group_id, {"class_name": None, "bbox": None, "points": {}})
        if label in POINT_LABELS and shape_type == "point":
            point = _normalize_point(shape.get("points") or [])
            if point is not None:
                slot_points = slot["points"]
                assert isinstance(slot_points, dict)
                slot_points[label] = point
            continue

        if shape_type == "rectangle":
            bbox = rectangle_points_to_bbox(shape.get("points") or [])
            if bbox is None:
                continue
            slot["class_name"] = label
            slot["bbox"] = bbox

    instances: list[LabeledInstance] = []
    for group_id, slot in grouped.items():
        class_name = slot.get("class_name")
        bbox = slot.get("bbox")
        if not class_name or bbox is None:
            continue
        instances.append(
            LabeledInstance(
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                class_name=str(class_name),
                group_id=group_id,
                bbox=bbox,
                points=dict(slot.get("points") or {}),
            )
        )
    return instances


def resolve_image_path(json_path: Path, image_path_value: str | None) -> Path:
    if image_path_value:
        candidate = Path(image_path_value)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        sibling = json_path.parent / image_path_value
        if sibling.exists():
            return sibling
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        sibling = json_path.with_suffix(ext)
        if sibling.exists():
            return sibling
    raise FileNotFoundError(f"Could not resolve image for {json_path}")


def rectangle_points_to_bbox(points: list) -> tuple[float, float, float, float] | None:
    normalized = [tuple(map(float, pt[:2])) for pt in points if isinstance(pt, list) and len(pt) >= 2]
    if len(normalized) < 2:
        return None
    xs = [pt[0] for pt in normalized]
    ys = [pt[1] for pt in normalized]
    return (min(xs), min(ys), max(xs), max(ys))


def _normalize_point(points: list) -> tuple[float, float] | None:
    if not points:
        return None
    first = points[0]
    if not isinstance(first, list) or len(first) < 2:
        return None
    return (float(first[0]), float(first[1]))
