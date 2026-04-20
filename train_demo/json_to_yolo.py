import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _read_png_size(path: Path) -> Optional[ImageSize]:
    try:
        with path.open("rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return None
            length = int.from_bytes(f.read(4), "big")
            chunk_type = f.read(4)
            if chunk_type != b"IHDR":
                return None
            data = f.read(length)
            if len(data) < 8:
                return None
            width = int.from_bytes(data[0:4], "big")
            height = int.from_bytes(data[4:8], "big")
            if width <= 0 or height <= 0:
                return None
            return ImageSize(width=width, height=height)
    except OSError:
        return None


def _read_jpeg_size(path: Path) -> Optional[ImageSize]:
    try:
        with path.open("rb") as f:
            if f.read(2) != b"\xFF\xD8":
                return None
            while True:
                marker_prefix = f.read(1)
                if not marker_prefix:
                    return None
                if marker_prefix != b"\xFF":
                    continue
                marker = f.read(1)
                if not marker:
                    return None
                while marker == b"\xFF":
                    marker = f.read(1)
                    if not marker:
                        return None

                if marker in (b"\xC0", b"\xC1", b"\xC2", b"\xC3", b"\xC5", b"\xC6", b"\xC7", b"\xC9", b"\xCA", b"\xCB", b"\xCD", b"\xCE", b"\xCF"):
                    segment_len = int.from_bytes(f.read(2), "big")
                    segment = f.read(segment_len - 2)
                    if len(segment) < 7:
                        return None
                    height = int.from_bytes(segment[1:3], "big")
                    width = int.from_bytes(segment[3:5], "big")
                    if width <= 0 or height <= 0:
                        return None
                    return ImageSize(width=width, height=height)

                if marker in (b"\xD8", b"\xD9"):
                    continue

                seg_len_bytes = f.read(2)
                if len(seg_len_bytes) != 2:
                    return None
                seg_len = int.from_bytes(seg_len_bytes, "big")
                if seg_len < 2:
                    return None
                f.seek(seg_len - 2, os.SEEK_CUR)
    except OSError:
        return None


def _read_image_size(path: Path) -> Optional[ImageSize]:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return _read_png_size(path)
    if suffix in (".jpg", ".jpeg"):
        return _read_jpeg_size(path)
    return None


def _iter_json_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from input_dir.rglob("*.json")
        return
    yield from input_dir.glob("*.json")


def _parse_classes_arg(classes: Optional[str]) -> Optional[List[str]]:
    if classes is None:
        return None
    parts = [p.strip() for p in classes.split(",")]
    parts = [p for p in parts if p]
    return parts if parts else None


def _load_classes_file(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    classes = [ln.strip() for ln in lines if ln.strip()]
    return classes if classes else None


def _shape_to_bbox(points: Sequence[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    if not points:
        return None
    xs: List[float] = []
    ys: List[float] = []
    for p in points:
        if len(p) < 2:
            continue
        x, y = float(p[0]), float(p[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if not (math.isfinite(x_min) and math.isfinite(x_max) and math.isfinite(y_min) and math.isfinite(y_max)):
        return None
    return (x_min, y_min, x_max, y_max)


def _bbox_to_yolo(
    bbox_xyxy: Tuple[float, float, float, float],
    image_size: ImageSize,
) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = bbox_xyxy
    w = float(image_size.width)
    h = float(image_size.height)
    if w <= 0 or h <= 0:
        return None
    x1 = _clamp(x1, 0.0, w)
    x2 = _clamp(x2, 0.0, w)
    y1 = _clamp(y1, 0.0, h)
    y2 = _clamp(y2, 0.0, h)
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    cx = _clamp(cx, 0.0, 1.0)
    cy = _clamp(cy, 0.0, 1.0)
    bw = _clamp(bw, 0.0, 1.0)
    bh = _clamp(bh, 0.0, 1.0)
    return (cx, cy, bw, bh)


def _guess_image_path(json_path: Path, image_path_field: Optional[str]) -> Optional[Path]:
    if image_path_field:
        candidate = (json_path.parent / image_path_field).resolve()
        if candidate.exists():
            return candidate
    stem = json_path.stem
    for ext in IMAGE_EXTS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _detect_format(obj: object) -> str:
    if isinstance(obj, dict):
        if "shapes" in obj and ("imageWidth" in obj or "imageHeight" in obj or "imagePath" in obj):
            return "labelme"
        if "images" in obj and "annotations" in obj and "categories" in obj:
            return "coco"
    return "unknown"


def _collect_labelme_labels(json_files: Sequence[Path], include_points: bool) -> List[str]:
    labels: List[str] = []
    seen = set()
    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _detect_format(data) != "labelme":
            continue
        for shape in data.get("shapes", []) or []:
            if not isinstance(shape, dict):
                continue
            label = shape.get("label")
            if not label:
                continue
            shape_type = (shape.get("shape_type") or "").lower()
            if shape_type in ("line", "linestrip"):
                continue
            if shape_type == "point" and not include_points:
                continue
            if label not in seen:
                seen.add(label)
                labels.append(label)
    labels.sort()
    return labels


def _point_to_bbox(
    points: Sequence[Sequence[float]],
    point_box_px: int,
    image_size: ImageSize,
) -> Optional[Tuple[float, float, float, float]]:
    if point_box_px <= 0 or not points or len(points[0]) < 2:
        return None
    x = float(points[0][0])
    y = float(points[0][1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    half = float(point_box_px) / 2.0
    return (x - half, y - half, x + half, y + half)


def _convert_labelme_file(
    json_path: Path,
    out_txt_path: Path,
    class_to_id: Dict[str, int],
    include_labels: Optional[set],
    include_points: bool,
    point_box_px: int,
) -> Tuple[int, int, int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if _detect_format(data) != "labelme":
        return (0, 0, 0)

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")
    image_path = _guess_image_path(json_path, data.get("imagePath"))

    image_size: Optional[ImageSize] = None
    if isinstance(img_w, int) and isinstance(img_h, int) and img_w > 0 and img_h > 0:
        image_size = ImageSize(width=img_w, height=img_h)
    elif image_path is not None:
        image_size = _read_image_size(image_path)

    if image_size is None:
        raise RuntimeError(f"无法获取图片尺寸: {json_path}")

    out_lines: List[str] = []
    shapes = data.get("shapes", []) or []
    kept = 0
    skipped = 0
    unknown = 0

    for shape in shapes:
        if not isinstance(shape, dict):
            skipped += 1
            continue
        label = shape.get("label")
        if not label:
            skipped += 1
            continue
        if include_labels is not None and label not in include_labels:
            skipped += 1
            continue
        shape_type = (shape.get("shape_type") or "").lower()
        if shape_type in ("line", "linestrip"):
            skipped += 1
            continue
        if label not in class_to_id:
            unknown += 1
            continue

        points = shape.get("points") or []
        if shape_type == "point":
            if not include_points:
                skipped += 1
                continue
            bbox = _point_to_bbox(points, point_box_px=point_box_px, image_size=image_size)
        else:
            bbox = _shape_to_bbox(points)
        if bbox is None:
            skipped += 1
            continue
        yolo_box = _bbox_to_yolo(bbox, image_size)
        if yolo_box is None:
            skipped += 1
            continue

        class_id = class_to_id[label]
        cx, cy, bw, bh = yolo_box
        out_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        kept += 1

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return (kept, skipped, unknown)


def _convert_coco_file(
    coco_json_path: Path,
    out_dir: Path,
    class_to_id: Dict[str, int],
    include_labels: Optional[set],
) -> Tuple[int, int, int]:
    data = json.loads(coco_json_path.read_text(encoding="utf-8"))
    if _detect_format(data) != "coco":
        return (0, 0, 0)

    images = data.get("images") or []
    annotations = data.get("annotations") or []
    categories = data.get("categories") or []

    cat_id_to_name: Dict[int, str] = {}
    for c in categories:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        name = c.get("name")
        if isinstance(cid, int) and isinstance(name, str) and name:
            cat_id_to_name[cid] = name

    img_id_to_info: Dict[int, dict] = {}
    for im in images:
        if not isinstance(im, dict):
            continue
        iid = im.get("id")
        if isinstance(iid, int):
            img_id_to_info[iid] = im

    img_to_lines: Dict[int, List[str]] = {}
    kept = 0
    skipped = 0
    unknown = 0

    for ann in annotations:
        if not isinstance(ann, dict):
            skipped += 1
            continue
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        bbox = ann.get("bbox")
        if not (isinstance(img_id, int) and isinstance(cat_id, int) and isinstance(bbox, list) and len(bbox) >= 4):
            skipped += 1
            continue
        label = cat_id_to_name.get(cat_id)
        if not label:
            skipped += 1
            continue
        if include_labels is not None and label not in include_labels:
            skipped += 1
            continue
        if label not in class_to_id:
            unknown += 1
            continue

        im = img_id_to_info.get(img_id)
        if not im:
            skipped += 1
            continue
        w = im.get("width")
        h = im.get("height")
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            skipped += 1
            continue

        x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        xyxy = (x, y, x + bw, y + bh)
        yolo_box = _bbox_to_yolo(xyxy, ImageSize(width=w, height=h))
        if yolo_box is None:
            skipped += 1
            continue

        class_id = class_to_id[label]
        cx, cy, bw_n, bh_n = yolo_box
        img_to_lines.setdefault(img_id, []).append(f"{class_id} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")
        kept += 1

    for img_id, lines in img_to_lines.items():
        im = img_id_to_info.get(img_id)
        if not im:
            continue
        file_name = im.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            continue
        out_path = (out_dir / Path(file_name).with_suffix(".txt")).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return (kept, skipped, unknown)


def _pack_flat_labelme(
    json_paths: Sequence[Path],
    input_dir: Path,
    label_output_dir: Path,
    pack_dir: Path,
    start_index: int,
    move_files: bool,
) -> Tuple[int, int]:
    pack_dir.mkdir(parents=True, exist_ok=True)
    packed = 0
    skipped = 0
    index = start_index

    for jp in json_paths:
        rel = jp.relative_to(input_dir)
        label_path = (label_output_dir / rel).with_suffix(".txt")
        if not label_path.exists():
            skipped += 1
            continue

        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        image_path = _guess_image_path(jp, data.get("imagePath") if isinstance(data, dict) else None)
        if image_path is None or not image_path.exists():
            skipped += 1
            continue

        img_suffix = image_path.suffix.lower()
        if not img_suffix:
            img_suffix = ".jpg"
        dst_img = pack_dir / f"{index}{img_suffix}"
        dst_lbl = pack_dir / f"{index}.txt"

        if move_files:
            shutil.move(str(image_path), str(dst_img))
            shutil.move(str(label_path), str(dst_lbl))
        else:
            shutil.copy2(str(image_path), str(dst_img))
            shutil.copy2(str(label_path), str(dst_lbl))

        packed += 1
        index += 1

    return (packed, skipped)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="将标注 JSON 转换为 YOLO 训练所需的 .txt 标注文件")
    parser.add_argument("--input_dir", required=True, help="包含 json 的目录（可包含子目录）")
    parser.add_argument("--output_dir", required=True, help="输出 .txt 的目录（会保留相对目录结构）")
    parser.add_argument("--recursive", action="store_true", help="递归扫描 input_dir 下所有 json")
    parser.add_argument("--classes", help="逗号分隔的类别列表，如: resistor,capacitor")
    parser.add_argument("--classes_file", help="类别列表文件（每行一个类别名）")
    parser.add_argument("--only_labels", help="只转换这些 label（逗号分隔），其他全部忽略")
    parser.add_argument("--format", choices=["auto", "labelme", "coco"], default="auto", help="json 格式（默认自动识别）")
    parser.add_argument("--pack_dir", help="将图片和标注按 1/2/3... 重命名并平铺到该目录（默认复制，不影响原文件）")
    parser.add_argument("--pack_start", type=int, default=1, help="pack 起始编号（默认 1）")
    parser.add_argument("--pack_move", action="store_true", help="pack 时移动文件（而不是复制）")
    parser.add_argument("--include_points", action="store_true", help="将 shape_type=point 的标注也转换到 txt（会转成小框）")
    parser.add_argument("--point_box_px", type=int, default=6, help="point 转成检测框时的边长像素（默认 6）")

    args = parser.parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    json_files = sorted([p for p in _iter_json_files(input_dir, recursive=bool(args.recursive)) if p.is_file()])
    if not json_files:
        raise SystemExit(f"未找到 json 文件: {input_dir}")

    include_labels_list = _parse_classes_arg(args.only_labels)
    include_labels = set(include_labels_list) if include_labels_list else None

    classes = _parse_classes_arg(args.classes) or _load_classes_file(Path(args.classes_file).resolve() if args.classes_file else None)
    if classes is None:
        classes = _collect_labelme_labels(json_files, include_points=bool(args.include_points))
        if not classes:
            raise SystemExit("未能自动收集到任何类别名。请使用 --classes 或 --classes_file 指定。")

    class_to_id = {name: i for i, name in enumerate(classes)}

    kept_total = 0
    skipped_total = 0
    unknown_total = 0
    converted_files = 0
    labelme_jsons: List[Path] = []

    for jp in json_files:
        try:
            obj = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue

        fmt = _detect_format(obj) if args.format == "auto" else args.format
        if fmt == "labelme":
            rel = jp.relative_to(input_dir)
            out_txt = (output_dir / rel).with_suffix(".txt")
            kept, skipped, unknown = _convert_labelme_file(
                jp,
                out_txt,
                class_to_id,
                include_labels,
                include_points=bool(args.include_points),
                point_box_px=int(args.point_box_px),
            )
            kept_total += kept
            skipped_total += skipped
            unknown_total += unknown
            converted_files += 1
            labelme_jsons.append(jp)
        elif fmt == "coco":
            kept, skipped, unknown = _convert_coco_file(jp, output_dir, class_to_id, include_labels)
            kept_total += kept
            skipped_total += skipped
            unknown_total += unknown
            converted_files += 1

    print("类别映射（name -> id）:")
    for name, idx in class_to_id.items():
        print(f"  {name} -> {idx}")
    print(f"处理 json 文件数: {converted_files}")
    print(f"输出标注框数: {kept_total}")
    print(f"跳过 shape/ann 数: {skipped_total}")
    print(f"未知 label 数(不在 classes 中): {unknown_total}")

    if args.pack_dir:
        pack_dir = Path(args.pack_dir).resolve()
        if args.pack_start < 1:
            raise SystemExit("--pack_start 必须 >= 1")
        packed, pack_skipped = _pack_flat_labelme(
            json_paths=labelme_jsons,
            input_dir=input_dir,
            label_output_dir=output_dir,
            pack_dir=pack_dir,
            start_index=args.pack_start,
            move_files=bool(args.pack_move),
        )
        print(f"pack 输出目录: {pack_dir}")
        print(f"pack 成功数量: {packed}")
        print(f"pack 跳过数量: {pack_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
