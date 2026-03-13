"""
测试新训练模型 unit_v1_yolov8s_960 的检测效果
- 检测元件 (resistor, capacitor, led, diode, IC, potentiometer)
- 检测引脚插入孔洞 (pinned)
- 可视化检测结果，保存标注图片
"""

import sys
import os
import time
import json
import cv2
import numpy as np

# ─── 配置 ───
MODEL_PATH = r"D:\desktop\LabGuardian-Server-main\LabGuardian-Server-main\models\unit_v1_yolov8s_960\weights\best.pt"
TEST_IMG_DIR = r"D:\desktop\inter\LabGuardian\dataset\images\demo_self\valid\images"
OUTPUT_DIR = r"D:\desktop\LabGuardian-Server-main\LabGuardian-Server-main\test_output"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
IMGSZ = 960  # 模型训练时的 imgsz
DEVICE = "cpu"  # 无 CUDA 时用 CPU

# 每个类别的颜色 (BGR)
CLASS_COLORS = {
    "Breadboard":    (200, 200, 200),
    "IC":            (255, 0, 0),
    "Line_area":     (0, 165, 255),
    "capacitor":     (0, 255, 0),
    "diode":         (0, 255, 255),
    "led":           (0, 0, 255),
    "pinned":        (255, 0, 255),
    "potentiometer": (255, 255, 0),
    "resistor":      (128, 0, 128),
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─── 1. 加载模型 ───
    print("=" * 60)
    print("模型加载")
    print("=" * 60)
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  任务类型: {model.task}")
    print(f"  类别映射: {model.names}")
    print(f"  imgsz:    {IMGSZ}")
    print(f"  conf:     {CONF_THRESHOLD}")
    print(f"  iou:      {IOU_THRESHOLD}")
    print()

    # ─── 2. 收集测试图片 ───
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    test_images = []
    if os.path.isdir(TEST_IMG_DIR):
        for f in os.listdir(TEST_IMG_DIR):
            if os.path.splitext(f)[1].lower() in img_extensions:
                test_images.append(os.path.join(TEST_IMG_DIR, f))
    else:
        print(f"[WARN] 测试图片目录不存在: {TEST_IMG_DIR}")

    if not test_images:
        print("未找到测试图片，请手动指定路径。")
        return

    print(f"找到 {len(test_images)} 张测试图片")
    print()

    # ─── 3. 逐张检测 ───
    all_stats = []
    for idx, img_path in enumerate(test_images):
        print("=" * 60)
        print(f"[{idx+1}/{len(test_images)}] {os.path.basename(img_path)}")
        print("=" * 60)

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [ERROR] 无法读取图片")
            continue

        h, w = img.shape[:2]
        print(f"  原始尺寸: {w}x{h}")

        # 推理
        t0 = time.time()
        results = model(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMGSZ, device=DEVICE, verbose=False)
        inference_ms = (time.time() - t0) * 1000
        print(f"  推理耗时: {inference_ms:.0f}ms")

        result = results[0]
        boxes = result.boxes

        # 统计
        class_counts = {}
        pinned_count = 0
        component_count = 0
        detections = []

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = model.names[cls_id]
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            det = {
                "class": cls_name,
                "conf": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                "size": [int(x2 - x1), int(y2 - y1)],
            }
            detections.append(det)

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            if cls_name == "pinned":
                pinned_count += 1
            elif cls_name not in ("Breadboard", "Line_area"):
                component_count += 1

        print(f"\n  检测结果汇总:")
        print(f"    总检测数: {len(detections)}")
        print(f"    元件数:   {component_count}")
        print(f"    引脚孔洞: {pinned_count}")
        print(f"    各类别:")
        for cls_name, count in sorted(class_counts.items()):
            print(f"      {cls_name:15s}: {count}")

        # ─── 4. 可视化 ───
        annotated = img.copy()

        # 4a. 画所有检测框
        for det in detections:
            cls_name = det["class"]
            conf = det["conf"]
            x1, y1, x2, y2 = det["bbox"]
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))

            if cls_name == "pinned":
                # 引脚孔洞用小圆点标注
                cx, cy = det["center"]
                radius = max(3, min(det["size"]) // 3)
                cv2.circle(annotated, (cx, cy), radius, color, 2)
                # 添加标签
                label = f"pin {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.putText(annotated, label, (cx - label_size[0] // 2, cy - radius - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            elif cls_name in ("Breadboard", "Line_area"):
                # 面包板和线区域用虚线框
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # 元件用矩形框
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 4b. 单独保存引脚孔洞检测图
        pinned_img = img.copy()
        pinned_dets = [d for d in detections if d["class"] == "pinned"]
        for det in pinned_dets:
            cx, cy = det["center"]
            conf = det["conf"]
            radius = max(4, min(det["size"]) // 3)
            cv2.circle(pinned_img, (cx, cy), radius, (255, 0, 255), 2)
            cv2.circle(pinned_img, (cx, cy), 2, (0, 0, 255), -1)  # 中心红点

        # 4c. 单独保存元件检测图 (不含面包板)
        component_img = img.copy()
        comp_dets = [d for d in detections if d["class"] not in ("Breadboard", "Line_area", "pinned")]
        for det in comp_dets:
            cls_name = det["class"]
            conf = det["conf"]
            x1, y1, x2, y2 = det["bbox"]
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            cv2.rectangle(component_img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(component_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ─── 5. 元件-引脚关联分析 ───
        print(f"\n  元件与引脚孔洞关联分析:")
        comp_pin_associations = []
        for comp_det in comp_dets:
            cx1, cy1, cx2, cy2 = comp_det["bbox"]
            comp_center = comp_det["center"]
            cls_name = comp_det["class"]

            # 找到元件 bbox 附近的引脚孔洞
            nearby_pins = []
            expand = 30  # 扩展搜索范围 (像素)
            for pin_det in pinned_dets:
                px, py = pin_det["center"]
                # 检查引脚是否在元件框附近
                if (cx1 - expand <= px <= cx2 + expand and
                    cy1 - expand <= py <= cy2 + expand):
                    dist = np.sqrt((px - comp_center[0])**2 + (py - comp_center[1])**2)
                    nearby_pins.append({"pin_center": [px, py], "distance": float(dist), "conf": pin_det["conf"]})

            nearby_pins.sort(key=lambda x: x["distance"])
            assoc = {
                "component": cls_name,
                "comp_bbox": comp_det["bbox"],
                "nearby_pins": len(nearby_pins),
                "pins": nearby_pins[:6],  # 最多显示 6 个
            }
            comp_pin_associations.append(assoc)
            print(f"    {cls_name} @ {comp_det['bbox']}: {len(nearby_pins)} 个引脚孔洞")
            for p in nearby_pins[:4]:
                print(f"      pin @ {p['pin_center']} dist={p['distance']:.0f}px conf={p['conf']:.2f}")

        # ─── 6. 保存结果 ───
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 保存标注图
        out_all = os.path.join(OUTPUT_DIR, f"{base_name}_all_detections.jpg")
        out_pins = os.path.join(OUTPUT_DIR, f"{base_name}_pinned_only.jpg")
        out_comps = os.path.join(OUTPUT_DIR, f"{base_name}_components_only.jpg")
        cv2.imwrite(out_all, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(out_pins, pinned_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(out_comps, component_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\n  已保存:")
        print(f"    全部检测: {out_all}")
        print(f"    引脚孔洞: {out_pins}")
        print(f"    元件:     {out_comps}")

        # 保存 JSON 详情
        out_json = os.path.join(OUTPUT_DIR, f"{base_name}_results.json")
        report = {
            "image": img_path,
            "image_size": [w, h],
            "model": MODEL_PATH,
            "inference_ms": inference_ms,
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "imgsz": IMGSZ,
            "total_detections": len(detections),
            "component_count": component_count,
            "pinned_count": pinned_count,
            "class_counts": class_counts,
            "detections": detections,
            "component_pin_associations": comp_pin_associations,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"    JSON报告: {out_json}")

        all_stats.append(report)

    # ─── 7. 总结 ───
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    for stat in all_stats:
        print(f"  {os.path.basename(stat['image'])}:")
        print(f"    推理耗时: {stat['inference_ms']:.0f}ms")
        print(f"    元件数: {stat['component_count']}  引脚孔洞: {stat['pinned_count']}")
        for cls, cnt in sorted(stat["class_counts"].items()):
            print(f"      {cls}: {cnt}")
    print(f"\n  输出目录: {OUTPUT_DIR}")
    print("  请查看生成的标注图片确认检测效果。")


if __name__ == "__main__":
    main()
