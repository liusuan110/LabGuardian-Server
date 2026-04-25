#!/usr/bin/env python3
"""
Minimal OpenVINO YOLO-Pose smoke runner.

用途:
- 独立验证导出的 OpenVINO pose 模型目录是否可加载
- 在单张图片上跑一次最小推理
- 输出基础可视化与 keypoint 摘要, 方便部署前排查
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def _draw_debug(image, detections: list[dict]) -> object:
    canvas = image.copy()
    for det in detections:
        bbox = det.get("bbox") or []
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 220), 2)
            cv2.putText(
                canvas,
                f"{det.get('class_name')} {det.get('confidence', 0.0):.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 180, 255),
                2,
            )
        for point in det.get("keypoints") or []:
            px = int(round(point["x"]))
            py = int(round(point["y"]))
            cv2.circle(canvas, (px, py), 4, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                f"k{point['index']}",
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 220),
                1,
            )
    return canvas


def run_smoke(*, model_path: Path, image_path: Path, out_dir: Path, device: str) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    model = YOLO(str(model_path), task="pose")
    results = model(image, verbose=False, device=device)
    if not results:
        raise RuntimeError("Model returned no results")

    result = results[0]
    names = getattr(model, "names", {}) or {}
    boxes = result.boxes
    keypoints = result.keypoints

    detections = []
    if boxes is not None and boxes.xyxy is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        kp_xy = keypoints.xy.cpu().numpy() if keypoints is not None and keypoints.xy is not None else None
        kp_conf = keypoints.conf.cpu().numpy() if keypoints is not None and keypoints.conf is not None else None

        for idx in range(len(xyxy)):
            points = []
            if kp_xy is not None and idx < len(kp_xy):
                for kp_idx, point in enumerate(kp_xy[idx].tolist()):
                    conf = None
                    if kp_conf is not None and idx < len(kp_conf) and kp_idx < len(kp_conf[idx]):
                        conf = float(kp_conf[idx][kp_idx])
                    points.append(
                        {
                            "index": kp_idx,
                            "x": float(point[0]),
                            "y": float(point[1]),
                            "confidence": conf,
                        }
                    )
            detections.append(
                {
                    "index": idx,
                    "class_id": int(cls_ids[idx]) if len(cls_ids) > idx else None,
                    "class_name": str(names.get(int(cls_ids[idx]), int(cls_ids[idx]))) if len(cls_ids) > idx else "unknown",
                    "confidence": float(confs[idx]) if len(confs) > idx else None,
                    "bbox": [float(v) for v in xyxy[idx].tolist()],
                    "keypoints": points,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = _draw_debug(image, detections)
    cv2.imwrite(str(out_dir / "annotated.png"), annotated)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": str(model_path),
                "image_path": str(image_path),
                "device": device,
                "count": len(detections),
                "detections": detections,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Image path")
    parser.add_argument("--model", required=True, help="OpenVINO model directory path")
    parser.add_argument("--out", default="/tmp/labguardian_openvino_pose_smoke", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Runtime device passed to Ultralytics")
    args = parser.parse_args()

    out_dir = run_smoke(
        model_path=Path(args.model),
        image_path=Path(args.image),
        out_dir=Path(args.out),
        device=args.device,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
