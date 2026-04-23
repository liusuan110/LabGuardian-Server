#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import yaml

from labelme_pose_dataset_utils import iter_labelme_instances


CLASS_TO_INDEX = {
    "capacitor_ceramic": 0,
    "capacitor_electrolytic": 1,
    "diode": 2,
    "jumper_wire": 3,
    "led": 4,
    "resistor": 5,
    "transistor_3pin": 6,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a ROI-context YOLO-Pose dataset from LabelMe annotations.")
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("/Users/liusuan/Desktop/LabGuardian-Server/train_demo"),
        help="Root directory containing LabelMe JSON files.",
    )
    parser.add_argument(
        "--reference-dataset-root",
        type=Path,
        default=Path("/Users/liusuan/Desktop/LabGuardian-Server/train_demo/yolo_packed2_pose"),
        help="Existing YOLO pose dataset root used only to preserve train/val image split.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Users/liusuan/Desktop/LabGuardian-Server/train_demo/yolo_packed2_pose_roi_context_v1"),
        help="Output root for generated ROI-based pose dataset.",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.10,
        help="Safety margin ratio applied after taking the union of body bbox and labeled pins.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output directory before rebuilding.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory exists: {args.output_root}. Use --overwrite to rebuild.")
        shutil.rmtree(args.output_root)

    split_map = build_split_map(args.reference_dataset_root)
    ensure_layout(args.output_root)

    counts = {"train": 0, "val": 0}
    seen_signatures: set[str] = set()
    duplicate_skips = 0

    for instance in iter_labelme_instances(args.annotations_root):
        class_name = instance.class_name.lower()
        if class_name not in CLASS_TO_INDEX:
            continue
        pin_names = [name for name in ("pin1", "pin2", "pin3") if name in instance.points]
        if "pin1" not in instance.points or "pin2" not in instance.points:
            continue
        signature = instance_signature(instance)
        if signature in seen_signatures:
            duplicate_skips += 1
            continue
        seen_signatures.add(signature)

        split = split_map.get(instance.image_path.name) or infer_split_from_image_name(instance.image_path.name)
        crop_bounds = build_union_crop(instance, margin_ratio=args.margin_ratio)
        image = cv2.imread(str(instance.image_path))
        if image is None:
            continue
        x1, y1, x2, y2 = crop_bounds
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        bx1, by1, bx2, by2 = instance.bbox
        local_bbox = (bx1 - x1, by1 - y1, bx2 - x1, by2 - y1)
        bbox_cx = ((local_bbox[0] + local_bbox[2]) / 2.0) / crop_w
        bbox_cy = ((local_bbox[1] + local_bbox[3]) / 2.0) / crop_h
        bbox_w = (local_bbox[2] - local_bbox[0]) / crop_w
        bbox_h = (local_bbox[3] - local_bbox[1]) / crop_h

        keypoints = []
        for name in ("pin1", "pin2", "pin3"):
            point = instance.points.get(name)
            if point is None:
                keypoints.extend([0.0, 0.0, 0])
                continue
            px, py = point
            keypoints.extend([(px - x1) / crop_w, (py - y1) / crop_h, 2])

        source_token = short_path_token(instance.image_path)
        stem = f"{source_token}__{instance.image_path.stem}__g{instance.group_id}__{class_name}"
        image_out = args.output_root / "images" / split / f"{stem}.jpg"
        label_out = args.output_root / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(image_out), crop)
        line = " ".join(
            [
                str(CLASS_TO_INDEX[class_name]),
                f"{bbox_cx:.6f}",
                f"{bbox_cy:.6f}",
                f"{bbox_w:.6f}",
                f"{bbox_h:.6f}",
                *[f"{value:.6f}" if isinstance(value, float) else str(value) for value in keypoints],
            ]
        )
        label_out.write_text(line + "\n", encoding="utf-8")
        counts[split] += 1

    write_data_yaml(args.output_root)
    summary = {
        "output_root": str(args.output_root),
        "counts": counts,
        "margin_ratio": args.margin_ratio,
        "class_to_index": CLASS_TO_INDEX,
        "split_strategy": "reference_dataset_name_match_or_stable_hash_by_original_image_name",
        "duplicate_instances_skipped": duplicate_skips,
    }
    (args.output_root / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_split_map(reference_root: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for split in ("train", "val"):
        image_dir = reference_root / "images" / split
        if not image_dir.exists():
            continue
        for image_path in image_dir.iterdir():
            if image_path.is_file():
                split_map[image_path.name] = split
    return split_map


def infer_split_from_image_name(image_name: str) -> str:
    digest = hashlib.sha1(image_name.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    return "val" if bucket < 2 else "train"


def short_path_token(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return digest[:8]


def instance_signature(instance) -> str:
    parts = [
        instance.image_path.name,
        str(instance.group_id),
        instance.class_name.lower(),
        ",".join(f"{v:.3f}" for v in instance.bbox),
    ]
    for name in ("pin1", "pin2", "pin3"):
        point = instance.points.get(name)
        if point is None:
            parts.append(f"{name}:none")
        else:
            parts.append(f"{name}:{point[0]:.3f},{point[1]:.3f}")
    return "|".join(parts)


def ensure_layout(output_root: Path) -> None:
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def build_union_crop(instance, *, margin_ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = instance.bbox
    xs = [x1, x2]
    ys = [y1, y2]
    for px, py in instance.points.values():
        xs.append(px)
        ys.append(py)
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    pad_x = max(4.0, width * margin_ratio)
    pad_y = max(4.0, height * margin_ratio)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    half_w = width / 2.0 + pad_x
    half_h = height / 2.0 + pad_y
    crop_x1 = max(0, int(round(cx - half_w)))
    crop_y1 = max(0, int(round(cy - half_h)))
    crop_x2 = min(instance.image_width, int(round(cx + half_w)))
    crop_y2 = min(instance.image_height, int(round(cy + half_h)))
    return crop_x1, crop_y1, crop_x2, crop_y2


def write_data_yaml(output_root: Path) -> None:
    data = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for name, index in CLASS_TO_INDEX.items()},
        "kpt_shape": [3, 3],
    }
    with (output_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    raise SystemExit(main())
