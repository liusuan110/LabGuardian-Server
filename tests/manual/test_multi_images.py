"""
综合测试脚本: 用新模型在多张图片上测试完整 Pipeline (类名修复 + pinned 辅助信息)

选取多个数据集分区的测试图片:
  1. demo_self (自拍面包板)
  2. v1i test (5 张场景图)
  3. Brainboard_4cls test (随机选 5 张)
"""

import sys
import os
import base64
import json
import time
import glob
import random
import cv2
import numpy as np

# ─── 环境配置 ───
os.environ['YOLO_MODEL_PATH'] = r'D:/desktop/LabGuardian-Server-main/LabGuardian-Server-main/models/unit_v1_yolov8s_960/weights/best.pt'
os.environ['YOLO_DEVICE'] = 'cpu'
os.environ['YOLO_IMGSZ'] = '960'
os.environ['YOLO_CONF_THRESHOLD'] = '0.25'
os.environ['YOLO_IOU_THRESHOLD'] = '0.5'
os.environ['BREADBOARD_ROWS'] = '63'
os.environ['BREADBOARD_COLS_PER_SIDE'] = '5'
# 不加载参考电路 — 仅测试检测 + 映射 + 拓扑
os.environ['REFERENCE_CIRCUIT_PATH'] = ''

sys.path.insert(0, '.')

# 清除旧 context
import app.pipeline.orchestrator as orch
orch._shared_ctx = None

from app.pipeline.orchestrator import run_pipeline

OUTPUT_DIR = r'D:\desktop\LabGuardian-Server-main\LabGuardian-Server-main\test_output_multi'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 每个类别的颜色 (BGR)
CLASS_COLORS = {
    "Resistor":      (128, 0, 128),
    "Capacitor":     (0, 255, 0),
    "Wire":          (0, 165, 255),
    "LED":           (0, 0, 255),
    "Diode":         (0, 255, 255),
    "IC":            (255, 0, 0),
    "Potentiometer": (255, 255, 0),
}

# ─── 收集测试图片 ───
def collect_test_images():
    images = []

    # 1. demo_self
    demo_dir = r'D:\desktop\inter\LabGuardian\dataset\images\demo_self\valid\images'
    if os.path.isdir(demo_dir):
        for f in os.listdir(demo_dir):
            if f.endswith('.jpg'):
                images.append(("demo_self", os.path.join(demo_dir, f)))

    # 2. v1i test (全部 5 张)
    v1i_dir = r'D:\desktop\inter\LabGuardian\dataset\images\---------.v1i.yolov8\test\images'
    if os.path.isdir(v1i_dir):
        for f in os.listdir(v1i_dir):
            if f.endswith('.jpg'):
                images.append(("v1i_test", os.path.join(v1i_dir, f)))

    # 3. Brainboard_4cls test (随机选 5 张)
    bb4_dir = r'D:\desktop\inter\LabGuardian\dataset\images\Brainboard_4cls\test\images'
    if os.path.isdir(bb4_dir):
        all_bb4 = [os.path.join(bb4_dir, f) for f in os.listdir(bb4_dir) if f.endswith('.jpg')]
        random.seed(42)
        selected = random.sample(all_bb4, min(5, len(all_bb4)))
        for fp in selected:
            images.append(("Brainboard_4cls", fp))

    return images


def draw_detections(img, s1_data, pinned_hints):
    """在图片上绘制检测结果"""
    annotated = img.copy()

    # 画元件检测
    for det in s1_data['detections']:
        cls = det['class_name']
        conf = det['confidence']
        x1, y1, x2, y2 = det['bbox']
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{cls} {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(annotated, (int(x1), int(y1) - label_size[1] - 8),
                     (int(x1) + label_size[0], int(y1)), color, -1)
        cv2.putText(annotated, label, (int(x1), int(y1) - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 画 pinned 孔洞 (洋红色小圆)
    for ph in pinned_hints:
        cx, cy = int(ph['center'][0]), int(ph['center'][1])
        conf = ph['confidence']
        cv2.circle(annotated, (cx, cy), 5, (255, 0, 255), 2)
        cv2.circle(annotated, (cx, cy), 1, (0, 0, 255), -1)
        cv2.putText(annotated, f"pin {conf:.2f}", (cx + 6, cy - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

    return annotated


def main():
    images = collect_test_images()
    print(f"收集到 {len(images)} 张测试图片\n")

    all_results = []

    for idx, (source, img_path) in enumerate(images):
        basename = os.path.basename(img_path)
        short_name = basename[:50] + "..." if len(basename) > 50 else basename
        print(f"{'=' * 70}")
        print(f"[{idx+1}/{len(images)}] [{source}] {short_name}")
        print(f"{'=' * 70}")

        # 加载图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [ERROR] 无法读取图片")
            continue
        h, w = img.shape[:2]
        print(f"  尺寸: {w}x{h}")

        # 编码为 base64
        _, buf = cv2.imencode('.jpg', img)
        b64 = base64.b64encode(buf).decode()

        # 运行 Pipeline
        t0 = time.time()
        try:
            result = run_pipeline(
                images_b64=[b64],
                rail_assignments={
                    "top_plus": "VCC", "top_minus": "GND",
                    "bot_plus": "VCC", "bot_minus": "GND",
                },
            )
        except Exception as e:
            print(f"  [ERROR] Pipeline 异常: {e}")
            import traceback
            traceback.print_exc()
            continue
        total_ms = (time.time() - t0) * 1000

        stages = result['stages']
        s1 = stages['detect']
        s2 = stages['mapping']
        s3 = stages['topology']
        s4 = stages['validate']
        pinned_hints = s1.get('pinned_hints', [])

        # ── S1 结果 ──
        class_counts = {}
        for d in s1['detections']:
            cn = d['class_name']
            class_counts[cn] = class_counts.get(cn, 0) + 1
        print(f"\n  S1 检测 ({s1['duration_ms']:.0f}ms):")
        print(f"    元件: {len(s1['detections'])} 个  |  pinned 孔洞: {len(pinned_hints)} 个")
        print(f"    类别: {class_counts}")
        for d in s1['detections']:
            print(f"      {d['class_name']:15s} conf={d['confidence']:.3f} bbox={d['bbox']}")

        # ── S2 结果 ──
        print(f"\n  S2 映射 ({s2['duration_ms']:.0f}ms):")
        for c in s2['components']:
            print(f"    {c['class_name']:15s} pin1={c.get('pin1_logic')} pin2={c.get('pin2_logic')}")

        # ── S3 结果 ──
        print(f"\n  S3 拓扑 ({s3['duration_ms']:.0f}ms): {s3['component_count']} 个元件")
        # 只打印电路描述的前几行
        desc_lines = s3['circuit_description'].split('\n')
        for line in desc_lines[:8]:
            print(f"    {line}")
        if len(desc_lines) > 8:
            print(f"    ... ({len(desc_lines) - 8} more lines)")

        # ── S4 结果 ──
        print(f"\n  S4 验证 ({s4['duration_ms']:.0f}ms): risk={s4['risk_level']}")
        for diag in s4.get('diagnostics', [])[:5]:
            print(f"    - {diag}")

        print(f"\n  总耗时: {total_ms:.0f}ms")

        # ── 保存可视化 ──
        safe_name = basename.replace('.', '_')[:60]
        annotated = draw_detections(img, s1, pinned_hints)
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_annotated.jpg")
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  保存标注图: {out_path}")

        # 收集统计
        all_results.append({
            "source": source,
            "image": basename,
            "size": [w, h],
            "total_ms": total_ms,
            "s1_ms": s1['duration_ms'],
            "num_detections": len(s1['detections']),
            "num_pinned": len(pinned_hints),
            "class_counts": class_counts,
            "s2_components": len(s2['components']),
            "s3_component_count": s3['component_count'],
            "s4_risk": s4['risk_level'],
            "s4_diagnostics_count": len(s4.get('diagnostics', [])),
        })

    # ── 汇总报告 ──
    print(f"\n{'=' * 70}")
    print("汇总报告")
    print(f"{'=' * 70}")
    print(f"{'源':15s} {'图片':35s} {'元件':>4s} {'pinned':>6s} {'S1ms':>6s} {'总ms':>6s} {'风险':>6s}")
    print("-" * 80)
    for r in all_results:
        name = r['image'][:33] + ".." if len(r['image']) > 35 else r['image']
        print(f"{r['source']:15s} {name:35s} {r['num_detections']:4d} {r['num_pinned']:6d} "
              f"{r['s1_ms']:6.0f} {r['total_ms']:6.0f} {r['s4_risk']:>6s}")

    # 统计各类别总数
    total_class = {}
    total_pinned = 0
    for r in all_results:
        for cls, cnt in r['class_counts'].items():
            total_class[cls] = total_class.get(cls, 0) + cnt
        total_pinned += r['num_pinned']

    print(f"\n总计 {len(all_results)} 张图片:")
    print(f"  各类别检测总数: {total_class}")
    print(f"  pinned 孔洞总数: {total_pinned}")
    print(f"  输出目录: {OUTPUT_DIR}")

    # 保存 JSON 报告
    report_path = os.path.join(OUTPUT_DIR, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  JSON报告: {report_path}")


if __name__ == "__main__":
    main()
