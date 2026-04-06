"""
UA741 运放电路 — 全流程测试 (S1→S2→S3→S4)

测试目标:
  1. YOLO 检测面包板上的元件 (IC/Resistor/Wire/pinned)
  2. 像素→逻辑坐标映射, 含 pinned 缺失纠正
  3. 构建电路拓扑网表 (UA741 理解为 8-pin DIP)
  4. 输出可视化检测结果 + 电路描述
"""

import sys, os, base64, json, time, logging

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONIOENCODING"] = "utf-8"

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("ua741_test")

# ── 配置 ──
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "unit_v1_yolov8s_960", "weights", "best.pt")
CONF = 0.25
IMGSZ = 960
DEVICE = "cpu"

# 颜色定义
COLORS = {
    "IC": (255, 0, 0), "Resistor": (128, 0, 128), "Capacitor": (0, 255, 0),
    "Wire": (0, 165, 255), "LED": (0, 0, 255), "Potentiometer": (255, 255, 0),
    "Diode": (0, 255, 255), "pinned": (255, 0, 255),
}

# UA741 引脚定义 (DIP-8)
UA741_PINOUT = {
    1: "Offset Null",
    2: "Inv Input (-)",
    3: "Non-Inv Input (+)",
    4: "V- (Neg Supply)",
    5: "Offset Null",
    6: "Output",
    7: "V+ (Pos Supply)",
    8: "NC",
}


def run_full_pipeline(image_path: str, output_dir: str):
    """运行完整 S1→S4 流水线"""
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return
    h, w = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"  UA741 Circuit Pipeline Test")
    print(f"  Image: {os.path.basename(image_path)} ({w}x{h})")
    print(f"{'='*60}")

    # 编码为 base64
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    # ===== S1: YOLO 检测 =====
    print("\n[S1] YOLO Detection...")
    from app.pipeline.vision.detector import ComponentDetector
    detector = ComponentDetector(model_path=MODEL_PATH, device=DEVICE)

    from app.pipeline.stages.s1_detect import run_detect
    s1 = run_detect(
        images_b64=[img_b64],
        detector=detector,
        conf=CONF,
        iou=0.5,
        imgsz=IMGSZ,
    )

    detections = s1["detections"]
    pinned_hints = s1.get("pinned_hints", [])
    print(f"  Components: {len(detections)}")
    for d in detections:
        print(f"    {d['class_name']:15s} conf={d['confidence']:.2f}  bbox={d['bbox']}")
    print(f"  Pinned holes: {len(pinned_hints)}")

    # ===== 可视化 S1 检测结果 =====
    vis_s1 = img.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cn = d["class_name"]
        cf = d["confidence"]
        color = COLORS.get(cn, (255, 255, 255))
        cv2.rectangle(vis_s1, (x1, y1), (x2, y2), color, 2)
        label = f"{cn} {cf:.2f}"
        ls = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.rectangle(vis_s1, (x1, y1-ls[1]-8), (x1+ls[0], y1), color, -1)
        cv2.putText(vis_s1, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    for p in pinned_hints:
        cx, cy = p["center"]
        cv2.circle(vis_s1, (int(cx), int(cy)), 8, (255, 0, 255), 2)
        cv2.circle(vis_s1, (int(cx), int(cy)), 2, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "s1_detect.jpg"), vis_s1, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: s1_detect.jpg")

    # ===== IC pinned 缺失纠正分析 =====
    print("\n[Pin Correction Analysis]")
    ic_dets = [d for d in detections if d["class_name"] == "IC"]
    resistor_dets = [d for d in detections if d["class_name"] == "Resistor"]

    # 分析哪些元件有 pinned hints 在附近，哪些没有
    for det in detections:
        cn = det["class_name"]
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        expand = 40
        nearby_pins = []
        for p in pinned_hints:
            px, py = p["center"]
            if (x1 - expand <= px <= x2 + expand and
                y1 - expand <= py <= y2 + expand):
                nearby_pins.append(p)

        has_p1 = det.get("pin1_pixel") is not None
        has_p2 = det.get("pin2_pixel") is not None
        pin_status = f"pin1={'Y' if has_p1 else 'N'} pin2={'Y' if has_p2 else 'N'}"
        print(f"  {cn:15s} {pin_status}  nearby_pinned={len(nearby_pins)}", end="")

        if len(nearby_pins) < 2 and cn == "Resistor":
            print("  << MISSING PINNED - will use bbox inference >>", end="")
        print()

    # ===== S2: 坐标映射 =====
    print("\n[S2] Pixel -> Logic Mapping...")
    from app.pipeline.vision.calibrator import BreadboardCalibrator
    from app.pipeline.stages.s2_mapping import run_mapping

    calibrator = BreadboardCalibrator(rows=63, cols_per_side=5)
    # 预建合成网格，避免视觉校准耗时或失败
    calibrator._build_synthetic_grid((h, w))
    print(f"  Calibrator: synthetic grid {calibrator.rows}x{calibrator.total_cols}")
    s2 = run_mapping(
        detections,
        calibrator=calibrator,
        image_shape=s1["primary_image_shape"],
        images_b64=[img_b64],
        pinned_hints=pinned_hints,
    )
    components = s2["components"]
    print(f"  Mapped {len(components)} components:")
    for c in components:
        cn = c["class_name"]
        p1 = c.get("pin1_logic", "?")
        p2 = c.get("pin2_logic", "?")
        print(f"    {cn:15s} pin1={p1}  pin2={p2}")

    # ===== IC 多引脚解析 (UA741 8-pin DIP) =====
    print("\n[IC Pin Analysis - UA741 DIP-8]")
    for ic in [c for c in components if c["class_name"] == "IC"]:
        bbox = ic["bbox"]
        x1, y1, x2, y2 = bbox
        ic_cx, ic_cy = (x1+x2)/2, (y1+y2)/2
        ic_h = y2 - y1
        ic_w = x2 - x1

        print(f"  IC bbox: ({x1},{y1})-({x2},{y2})")
        print(f"  IC center: ({ic_cx:.0f}, {ic_cy:.0f})")
        is_vertical = ic_h > ic_w
        print(f"  IC spans {'vertically' if is_vertical else 'horizontally'}")

        # 找 IC 附近的所有 pinned holes，按位置排列出各引脚
        ic_pins = []
        for p in pinned_hints:
            px, py = p["center"]
            expand = 60
            if (x1 - expand <= px <= x2 + expand and
                y1 - expand <= py <= y2 + expand):
                ic_pins.append((px, py, p["confidence"]))

        print(f"  Found {len(ic_pins)} pinned holes near IC")

        if ic_pins:
            # DIP-8 方向判断:
            # 垂直: 引脚 1-4 在左侧 (x < cx), 5-8 在右侧 (x > cx)
            #        Pin 1-4: 从上到下排列, Pin 5-8: 从下到上排列
            # 水平: 引脚 1-4 在上侧 (y < cy), 5-8 在下侧 (y > cy)
            if is_vertical:
                left_pins = sorted([p for p in ic_pins if p[0] < ic_cx], key=lambda p: p[1])
                right_pins = sorted([p for p in ic_pins if p[0] >= ic_cx], key=lambda p: -p[1])
            else:
                left_pins = sorted([p for p in ic_pins if p[1] < ic_cy], key=lambda p: p[0])
                right_pins = sorted([p for p in ic_pins if p[1] >= ic_cy], key=lambda p: -p[0])

            print(f"  {'Left' if is_vertical else 'Top'} side pins ({len(left_pins)}): Pin 1→4")
            for i, (px, py, cf) in enumerate(left_pins):
                pin_num = i + 1
                logic = calibrator.frame_pixel_to_logic(px, py)
                func = UA741_PINOUT.get(pin_num, "?")
                print(f"    Pin {pin_num} ({func:20s}): pixel=({px:.0f},{py:.0f}) -> logic={logic}")

            print(f"  {'Right' if is_vertical else 'Bottom'} side pins ({len(right_pins)}): Pin 5→8")
            for i, (px, py, cf) in enumerate(right_pins):
                pin_num = i + 5
                logic = calibrator.frame_pixel_to_logic(px, py)
                func = UA741_PINOUT.get(pin_num, "?")
                print(f"    Pin {pin_num} ({func:20s}): pixel=({px:.0f},{py:.0f}) -> logic={logic}")

    # ===== S3: 拓扑构建 =====
    print("\n[S3] Topology Construction...")
    from app.pipeline.stages.s3_topology import run_topology

    rail_assignments = {
        "top_plus": "VCC",
        "top_minus": "GND",
        "bot_plus": "VCC",
        "bot_minus": "GND",
    }
    s3 = run_topology(components, rail_assignments=rail_assignments)
    print(f"  Component count: {s3['component_count']}")
    print(f"\n  Circuit Description:")
    for line in s3["circuit_description"].split("\n"):
        print(f"    {line}")

    # ===== S4: 验证 =====
    print("\n[S4] Validation...")
    from app.pipeline.stages.s4_validate import run_validate
    s4 = run_validate(
        s3["topology_graph"],
        reference_path=None,
        components=components,
    )
    print(f"  Risk level: {s4['risk_level']}")
    print(f"  Diagnostics:")
    for diag in s4.get("diagnostics", []):
        print(f"    {diag}")

    # ===== 综合可视化 =====
    vis_final = img.copy()

    # 画 IC 和 pinned
    for ic in [c for c in components if c["class_name"] == "IC"]:
        x1, y1, x2, y2 = ic["bbox"]
        cv2.rectangle(vis_final, (x1,y1), (x2,y2), (255,0,0), 3)
        cv2.putText(vis_final, "UA741", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    # 画其他元件及其逻辑坐标
    for c in components:
        if c["class_name"] == "IC":
            continue
        x1, y1, x2, y2 = c["bbox"]
        cn = c["class_name"]
        color = COLORS.get(cn, (255,255,255))
        cv2.rectangle(vis_final, (x1,y1), (x2,y2), color, 2)
        p1 = c.get("pin1_logic", ["?","?"])
        p2 = c.get("pin2_logic", ["?","?"])
        label = f"{cn} ({p1[0]}{p1[1]}-{p2[0]}{p2[1]})"
        cv2.putText(vis_final, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 画 pinned holes
    for p in pinned_hints:
        cx, cy = int(p["center"][0]), int(p["center"][1])
        cv2.circle(vis_final, (cx, cy), 6, (255, 0, 255), 2)

    cv2.imwrite(os.path.join(output_dir, "s2_mapped.jpg"), vis_final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n  Saved: s2_mapped.jpg")

    # ===== 输出完整 JSON 结果 =====
    result_json = {
        "image": os.path.basename(image_path),
        "s1_detections": len(detections),
        "s1_pinned": len(pinned_hints),
        "s2_mapped_components": [{
            "class": c["class_name"],
            "pin1_logic": c.get("pin1_logic"),
            "pin2_logic": c.get("pin2_logic"),
            "confidence": round(c.get("confidence", 0), 3),
        } for c in components],
        "s3_circuit_description": s3["circuit_description"],
        "s3_netlist": s3["netlist"],
        "s4_risk_level": s4["risk_level"],
        "s4_diagnostics": s4.get("diagnostics", []),
    }
    json_path = os.path.join(output_dir, "pipeline_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved: pipeline_result.json")

    print(f"\n{'='*60}")
    print(f"  Pipeline Complete!")
    duration = s1["duration_ms"] + s2["duration_ms"] + s3["duration_ms"] + s4["duration_ms"]
    print(f"  S1: {s1['duration_ms']:.0f}ms | S2: {s2['duration_ms']:.0f}ms | S3: {s3['duration_ms']:.0f}ms | S4: {s4['duration_ms']:.0f}ms")
    print(f"  Total: {duration:.0f}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # 使用已复制到 detect_results 的场景5 (IC + resistors)
        img_path = os.path.join(ROOT, "detect_results", "scene5.jpg")
    out_dir = os.path.join(ROOT, "detect_results", "ua741_pipeline")
    run_full_pipeline(img_path, out_dir)
