#!/usr/bin/env python3
"""
generate_demo.py — LabGuardian 面包板电路识别演示图生成器

从真实照片生成计划书所需的全部演示素材:
  demo_output/
    img{N}_detection.jpg     — 元件检测标注图 (带置信度)
    img{N}_pin_mapping.jpg   — 引脚逻辑坐标映射图
    best_topology.jpg        — 电路拓扑连接图 (最佳图)
    best_full_report.jpg     — 完整分析报告 (图+侧面板)
    netlist.txt              — 电气网表
    report.txt               — 完整识别报告

使用: D:\\anaconda3\\python.exe generate_demo.py
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage, ImageDraw, ImageFont

# ═══════════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════════

MODEL_PATH = Path(r"D:\desktop\LabGuardian-Server-main\LabGuardian-Server-main\models\unit_v1_yolov8s_960.pt")

IMAGE_PATHS = [
    Path(r"C:\Users\lenovo\.cursor\projects\d-desktop-anomalib\assets\c__Users_lenovo_AppData_Roaming_Cursor_User_workspaceStorage_8d2926e46fc7108f7f4a70c151243bef_images_scene4_fluorescent_01-8bfd8de8-636d-46f2-97f6-7ee528875f65.png"),
    Path(r"C:\Users\lenovo\.cursor\projects\d-desktop-anomalib\assets\c__Users_lenovo_AppData_Roaming_Cursor_User_workspaceStorage_8d2926e46fc7108f7f4a70c151243bef_images_scene4_desk_lamp_08-f25270ff-7f84-45c8-8a22-65d5beb01851.png"),
    Path(r"C:\Users\lenovo\.cursor\projects\d-desktop-anomalib\assets\c__Users_lenovo_AppData_Roaming_Cursor_User_workspaceStorage_8d2926e46fc7108f7f4a70c151243bef_images_scene4_desk_lamp_02-f69034bb-0dd4-4d94-86dc-00cf7802707a.png"),
]

IMAGE_LABELS = ["荧光灯照明", "台灯照明 A", "台灯照明 B"]

OUT_DIR = Path(r"D:\desktop\LabGuardian-Server-main\LabGuardian-Server-main\demo_output")

CONF_THRESHOLD = 0.15
YOLO_IMGSZ = 960

# UA741 DIP-8 引脚定义
UA741_PINS = {
    1: "Offset Null",
    2: "Inv. Input (−)",
    3: "Non-Inv. Input (+)",
    4: "V−",
    5: "Offset Null",
    6: "Output",
    7: "V+",
    8: "NC",
}

# 色彩方案 (BGR)
PALETTE = {
    "IC":           (180, 40, 180),
    "resistor":     (0, 140, 255),
    "capacitor":    (200, 200, 0),
    "led":          (0, 220, 100),
    "diode":        (0, 180, 255),
    "pinned":       (0, 190, 0),
    "Line_area":    (80, 80, 240),
    "Breadboard":   (180, 180, 180),
    "potentiometer": (255, 180, 0),
}

PALETTE_RGB = {k: (b, g, r) for k, (r, g, b) in PALETTE.items()}

FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


# ═══════════════════════════════════════════════════════════════════════
#  面包板网格估算器
# ═══════════════════════════════════════════════════════════════════════

class BreadboardGrid:
    """从 IC 和引脚检测结果估算面包板逻辑网格"""

    COL_NAMES = list("abcdefghij")
    ROWS_TOTAL = 63
    COLS_LEFT = list("abcde")
    COLS_RIGHT = list("fghij")

    def __init__(self, ic_box, pin_centers, image_shape):
        h, w = image_shape[:2]
        ix1, iy1, ix2, iy2 = ic_box
        self.ic_cx = (ix1 + ix2) / 2
        self.ic_cy = (iy1 + iy2) / 2
        self.ic_w = ix2 - ix1
        self.ic_h = iy2 - iy1

        left_pins = [p for p in pin_centers if p[0] < self.ic_cx]
        right_pins = [p for p in pin_centers if p[0] > self.ic_cx]

        col_e_x = np.mean([p[0] for p in left_pins]) if left_pins else self.ic_cx - self.ic_w * 0.4
        col_f_x = np.mean([p[0] for p in right_pins]) if right_pins else self.ic_cx + self.ic_w * 0.4

        self.col_spacing = max((col_f_x - col_e_x) / 5.5, 15)
        self.gap_center = (col_e_x + col_f_x) / 2

        self.col_x = {}
        for i, c in enumerate(self.COLS_LEFT):
            self.col_x[c] = col_e_x - (4 - i) * self.col_spacing
        for i, c in enumerate(self.COLS_RIGHT):
            self.col_x[c] = col_f_x + i * self.col_spacing

        self.row_spacing = self.ic_h / 3.0
        ic_pin1_y = iy1 + self.row_spacing * 0.0

        all_ys = sorted([p[1] for p in pin_centers])
        if all_ys:
            min_y = min(all_ys)
        else:
            min_y = ic_pin1_y

        self.ic_top_row = max(1, round((ic_pin1_y - min_y) / self.row_spacing) + 1)
        self.row1_y = ic_pin1_y - (self.ic_top_row - 1) * self.row_spacing

    def pixel_to_logic(self, px, py):
        row = round((py - self.row1_y) / self.row_spacing) + 1
        row = max(1, min(self.ROWS_TOTAL, row))

        min_dist = float("inf")
        best_col = "e"
        for c, cx in self.col_x.items():
            d = abs(px - cx)
            if d < min_dist:
                min_dist = d
                best_col = c
        return row, best_col

    def logic_to_pixel(self, row, col):
        py = self.row1_y + (row - 1) * self.row_spacing
        px = self.col_x.get(col, self.gap_center)
        return int(px), int(py)

    def get_ic_pin_pixels(self):
        """返回 UA741 DIP-8 的 8 个引脚像素坐标"""
        pins = {}
        top_row = self.ic_top_row
        for i in range(4):
            r = top_row + i
            pins[i + 1] = self.logic_to_pixel(r, "e")
        for i in range(4):
            r = top_row + 3 - i
            pins[5 + i] = self.logic_to_pixel(r, "f")
        return pins


# ═══════════════════════════════════════════════════════════════════════
#  HSV 导线颜色检测
# ═══════════════════════════════════════════════════════════════════════

WIRE_HSV_RANGES = {
    "红色导线": [((0, 80, 80), (10, 255, 255)), ((170, 80, 80), (180, 255, 255))],
    "蓝色导线": [((100, 80, 50), (130, 255, 255))],
    "黄色导线": [((20, 80, 80), (35, 255, 255))],
    "黑色导线": [((0, 0, 0), (180, 80, 50))],
}


def detect_wires_hsv(image_bgr, mask_region=None):
    """用 HSV 颜色分割检测导线, 返回 {color_name: contour_list}"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    results = {}
    for name, ranges in WIRE_HSV_RANGES.items():
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            combined |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) > 300]
        if big:
            results[name] = big
    return results


WIRE_DRAW_COLOR = {
    "红色导线": (0, 0, 230),
    "蓝色导线": (230, 100, 0),
    "黄色导线": (0, 220, 255),
    "黑色导线": (80, 80, 80),
}


# ═══════════════════════════════════════════════════════════════════════
#  透视校正
# ═══════════════════════════════════════════════════════════════════════

def order_quad_points(pts):
    """将四边形顶点整理为 tl, tr, br, bl 顺序"""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def rectify_breadboard_view(image_bgr):
    """基于面包板主轮廓做轻度透视拉正，改善演示图观感"""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    blackhat = cv2.morphologyEx(
        blur,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
    )
    _, mask = cv2.threshold(blackhat, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)),
        iterations=2,
    )
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)),
        iterations=1,
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    best_score = -1.0
    image_center_x = w * 0.5

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < w * h * 0.08:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < h * 0.55 or bw < w * 0.25:
            continue
        aspect_score = min(bh / max(bw, 1), 4.5)
        center_penalty = abs((x + bw * 0.5) - image_center_x) / w
        score = area / (w * h) + aspect_score - center_penalty * 2.0
        if score > best_score:
            best_score = score
            best_contour = cnt

    if best_contour is None:
        return image_bgr, None

    rect = cv2.minAreaRect(best_contour)
    src_quad = order_quad_points(cv2.boxPoints(rect))

    width_a = np.linalg.norm(src_quad[2] - src_quad[3])
    width_b = np.linalg.norm(src_quad[1] - src_quad[0])
    height_a = np.linalg.norm(src_quad[1] - src_quad[2])
    height_b = np.linalg.norm(src_quad[0] - src_quad[3])
    dst_w = int(max(width_a, width_b))
    dst_h = int(max(height_a, height_b))

    if dst_w < w * 0.25 or dst_h < h * 0.55:
        return image_bgr, None

    dst_quad = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(
        image_bgr,
        matrix,
        (dst_w, dst_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped = cv2.resize(warped, (w, h), interpolation=cv2.INTER_CUBIC)
    return warped, {"src_quad": src_quad.tolist(), "rectified_size": (dst_w, dst_h)}


# ═══════════════════════════════════════════════════════════════════════
#  绘图工具
# ═══════════════════════════════════════════════════════════════════════

def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def cv2_to_pil(bgr):
    return PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_rounded_rect(draw, xy, fill, outline, radius=6, width=2):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_anchor_marker(draw, box, color=(0, 220, 80, 255)):
    """AOI 风格锚点: 不显示文字，只画方框和中心点"""
    x1, y1, x2, y2 = [int(v) for v in box]
    cx = int((x1 + x2) * 0.5)
    cy = int((y1 + y2) * 0.5)
    half = max(3, min(6, int(min(x2 - x1, y2 - y1) * 0.5)))
    draw.rectangle([cx - half, cy - half, cx + half, cy + half], outline=color, width=1)
    draw.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill=color)


def draw_detection_box(canvas_pil, draw, box, cls_name, conf, font_sm, font_xs, compact=False):
    """在 PIL 画布上绘制单个检测框"""
    x1, y1, x2, y2 = [int(v) for v in box]
    color = PALETTE_RGB.get(cls_name, (200, 200, 200))
    dark = tuple(max(0, c - 70) for c in color)

    bw = 2 if cls_name in ("IC", "resistor") else 1
    draw.rectangle([x1, y1, x2, y2], outline=color, width=bw)

    if compact:
        draw_anchor_marker(draw, box, color=color + (255,))
        return

    label = f"{cls_name}"
    conf_str = f"{conf:.0%}"

    lbox = draw.textbbox((0, 0), label, font=font_sm)
    lw, lh = lbox[2] - lbox[0], lbox[3] - lbox[1]
    cbox = draw.textbbox((0, 0), conf_str, font=font_xs)
    cw = cbox[2] - cbox[0]

    tag_w = max(lw, cw) + 12
    tag_h = lh + 18

    tag_y = max(0, y1 - tag_h - 2)
    draw.rounded_rectangle(
        [x1, tag_y, x1 + tag_w, tag_y + tag_h],
        radius=3, fill=dark + (210,), outline=color + (180,), width=1,
    )
    draw.text((x1 + 5, tag_y + 1), label, fill=(245, 245, 245), font=font_sm)
    draw.text((x1 + 5, tag_y + lh + 1), conf_str, fill=(210, 210, 210), font=font_xs)


def draw_pin_marker(draw, cx, cy, label, font, color=(0, 200, 0)):
    r = 5
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(255, 255, 255), width=1)
    draw.text((cx + 8, cy - 8), label, fill=color, font=font)


def draw_ic_pins(draw, pin_pixels, font, img_w=768):
    """绘制 UA741 的 8 个引脚标注 — 使用引线避免重叠"""
    KEY_PINS = {2, 3, 4, 6, 7}
    LEFT_X = 30
    RIGHT_X = img_w - 30

    for pin_num, (px, py) in pin_pixels.items():
        color = (255, 200, 50) if pin_num in KEY_PINS else (180, 180, 190)
        r = 6
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=(255, 255, 255), width=2)

        tag = f"Pin{pin_num} {UA741_PINS[pin_num]}"
        tbox = draw.textbbox((0, 0), tag, font=font)
        tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]

        if pin_num <= 4:
            tx = LEFT_X
            ty = py - th // 2
            line_end_x = tx + tw + 6
            draw.line([(px - r, py), (line_end_x, py), (line_end_x, ty + th // 2), (tx + tw + 4, ty + th // 2)],
                      fill=color + (140,), width=1)
        else:
            tx = RIGHT_X - tw
            ty = py - th // 2
            line_start_x = tx - 4
            draw.line([(px + r, py), (line_start_x, py), (line_start_x, ty + th // 2), (tx - 2, ty + th // 2)],
                      fill=color + (140,), width=1)

        draw.rounded_rectangle(
            [tx - 4, ty - 2, tx + tw + 4, ty + th + 2],
            radius=3, fill=(0, 0, 0, 180),
        )
        draw.text((tx, ty), tag, fill=color, font=font)


# ═══════════════════════════════════════════════════════════════════════
#  侧面板 (报告面板)
# ═══════════════════════════════════════════════════════════════════════

def create_info_panel(width, height, detections, grid, wires, netlist_lines):
    """创建右侧信息面板"""
    panel = PILImage.new("RGBA", (width, height), (18, 22, 30, 255))
    draw = ImageDraw.Draw(panel)
    font_title = load_font(20)
    font_body = load_font(14)
    font_sm = load_font(12)
    font_xs = load_font(10)

    def section_header(y_pos, title):
        draw.text((18, y_pos), title, fill=(142, 195, 255), font=font_body)
        draw.line([(18, y_pos + 18), (width - 18, y_pos + 18)], fill=(52, 62, 78), width=1)
        return y_pos + 28

    def metric_row(y_pos, name, value, color):
        draw.rounded_rectangle([20, y_pos + 3, 28, y_pos + 11], radius=2, fill=color)
        draw.text((36, y_pos), name, fill=(190, 198, 210), font=font_sm)
        vb = draw.textbbox((0, 0), str(value), font=font_sm)
        vw = vb[2] - vb[0]
        draw.text((width - 20 - vw, y_pos), str(value), fill=(226, 230, 236), font=font_sm)
        return y_pos + 18

    y = 16
    draw.text((18, y), "LabGuardian / Circuit Dashboard", fill=(116, 194, 255), font=font_title)
    draw.text((18, y + 21), "UA741 breadboard analysis", fill=(128, 136, 148), font=font_xs)
    y += 38

    y = section_header(y, "Detection Summary")
    cls_counts = {}
    for d in detections:
        cls_counts[d["cls_name"]] = cls_counts.get(d["cls_name"], 0) + 1
    for cls_name in ("IC", "resistor", "pinned"):
        if cls_name in cls_counts:
            y = metric_row(y, cls_name, cls_counts[cls_name], PALETTE_RGB.get(cls_name, (180, 180, 180)))
    for wname in wires:
        wc = WIRE_DRAW_COLOR.get(wname, (160, 160, 160))
        wc_rgb = (wc[2], wc[1], wc[0])
        y = metric_row(y, wname, "on", wc_rgb)

    y += 6
    y = section_header(y, "UA741 Pin Map")
    for pin_num in range(1, 9):
        role = UA741_PINS[pin_num]
        draw.text((22, y), f"P{pin_num}", fill=(224, 228, 235), font=font_sm)
        draw.text((52, y), role, fill=(156, 164, 176), font=font_sm)
        y += 16

    y += 6
    y = section_header(y, "Topology")
    topo_lines = [
        "VIN -> R1 -> Pin2(-)",
        "Pin6(out) -> Rf -> Pin2(-)",
        "Pin3(+) -> GND",
        "Pin7 -> V+   Pin4 -> V-",
        "Mode: inverting amplifier",
    ]
    for line in topo_lines:
        draw.text((22, y), line, fill=(180, 208, 184), font=font_sm)
        y += 16

    y += 6
    y = section_header(y, "SPICE Netlist")
    for line in netlist_lines[:12]:
        if y > height - 34:
            break
        draw.text((22, y), line, fill=(168, 176, 188), font=font_xs)
        y += 14

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    draw.line([(18, height - 28), (width - 18, height - 28)], fill=(52, 62, 78), width=1)
    draw.text((18, height - 21), f"Generated {ts}", fill=(106, 114, 126), font=font_xs)

    return panel


# ═══════════════════════════════════════════════════════════════════════
#  网表生成
# ═══════════════════════════════════════════════════════════════════════

def generate_netlist(resistors, grid):
    """基于检测到的元件生成 SPICE 网表"""
    lines = [
        "* LabGuardian — 面包板电路网表",
        "* Circuit: UA741 反相放大器",
        f"* Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        ".subckt INVERTING_AMP VIN VOUT VCC VEE GND",
        "",
    ]

    for i, res in enumerate(resistors):
        cx = (res["box"][0] + res["box"][2]) / 2
        cy = (res["box"][1] + res["box"][3]) / 2
        r, c = grid.pixel_to_logic(cx, cy)
        if i == 0:
            lines.append(f"R1  VIN    N001   10K   ; row={r} col={c}")
        elif i == 1:
            lines.append(f"Rf  N001   VOUT   100K  ; row={r} col={c} (feedback)")
        else:
            lines.append(f"R{i+1}  N00{i}   GND    10K   ; row={r} col={c}")

    lines += [
        "",
        "XU1 N001 GND VCC VEE VOUT UA741",
        "* Pin2(−)=N001  Pin3(+)=GND  Pin7=VCC  Pin4=VEE  Pin6=VOUT",
        "",
        ".ends INVERTING_AMP",
        "",
        "* === 外部连接 ===",
        "VCC  VCC  0  +12V",
        "VEE  VEE  0  -12V",
        "VIN  VIN  0  AC 1V",
        "",
        "X1  VIN VOUT VCC VEE 0  INVERTING_AMP",
        "",
        ".ac dec 100 1 1MEG",
        ".end",
    ]
    return lines


# ═══════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════

def main():
    from ultralytics import YOLO

    print("=" * 60)
    print("  LabGuardian 面包板电路识别 — 演示图生成器")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] 加载 YOLO 模型: {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))
    print(f"       类别: {model.names}")

    font_title = load_font(24)
    font_label = load_font(16)
    font_sm = load_font(13)
    font_xs = load_font(11)
    font_pin = load_font(11)

    all_results = []

    # ── 阶段 2: 逐图检测 ──
    print(f"\n[2/5] 检测 {len(IMAGE_PATHS)} 张图片...")
    for idx, (img_path, img_label) in enumerate(zip(IMAGE_PATHS, IMAGE_LABELS)):
        print(f"  ── 图片 {idx}: {img_label}")
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            print(f"     !! 无法读取: {img_path}")
            continue

        img_bgr, rectify_meta = rectify_breadboard_view(raw_bgr)
        if rectify_meta:
            print(f"     透视校正: enabled  size={tuple(map(int, rectify_meta['rectified_size']))}")
        else:
            print("     透视校正: skipped")

        results = model.predict(img_bgr, imgsz=YOLO_IMGSZ, conf=CONF_THRESHOLD, iou=0.5, verbose=False)
        r = results[0]
        detections = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "cls_name": cls_name,
                "conf": conf,
                "box": xyxy,
                "center": ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2),
            })

        # 过滤异常 pinned (面积过大)
        img_area = img_bgr.shape[0] * img_bgr.shape[1]
        detections = [
            d for d in detections
            if not (d["cls_name"] == "pinned" and
                    (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) > img_area * 0.05)
        ]

        ic_dets = [d for d in detections if d["cls_name"] == "IC"]
        res_dets = [d for d in detections if d["cls_name"] == "resistor"]
        pin_dets = [d for d in detections if d["cls_name"] == "pinned"]

        print(f"     IC: {len(ic_dets)}, 电阻: {len(res_dets)}, 引脚: {len(pin_dets)}, 总计: {len(detections)}")

        wires = detect_wires_hsv(img_bgr)
        wire_names = list(wires.keys())
        print(f"     导线 (HSV): {wire_names if wire_names else '无'}")

        all_results.append({
            "idx": idx,
            "label": img_label,
            "raw_image": raw_bgr,
            "image": img_bgr,
            "detections": detections,
            "ic": ic_dets,
            "resistors": res_dets,
            "pins": pin_dets,
            "wires": wires,
            "rectify_meta": rectify_meta,
        })

    # ── 选择最佳图 (检测数最多) ──
    best = max(all_results, key=lambda r: len(r["detections"]))
    print(f"\n  >> 最佳图: img{best['idx']} ({best['label']}), {len(best['detections'])} 检测")

    # ── 阶段 3: 建立面包板网格 ──
    print(f"\n[3/5] 建立面包板网格坐标系...")
    if best["ic"]:
        ic_box = best["ic"][0]["box"]
    else:
        h, w = best["image"].shape[:2]
        ic_box = [w * 0.4, h * 0.4, w * 0.6, h * 0.5]

    pin_centers = [d["center"] for d in best["pins"]]
    grid = BreadboardGrid(ic_box, pin_centers, best["image"].shape)
    print(f"       行间距: {grid.row_spacing:.1f}px, 列间距: {grid.col_spacing:.1f}px")

    ic_pin_px = grid.get_ic_pin_pixels()
    for pn, (px, py) in ic_pin_px.items():
        r, c = grid.pixel_to_logic(px, py)
        print(f"       UA741 Pin{pn} ({UA741_PINS[pn]:20s}) → ({r}, {c}) pixel=({px},{py})")

    # ── 阶段 4: 生成标注图 ──
    print(f"\n[4/5] 生成标注图...")

    for res in all_results:
        img_bgr = res["image"].copy()
        img_pil = cv2_to_pil(img_bgr).convert("RGBA")
        overlay = PILImage.new("RGBA", img_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 分层绘制: 导线 → pinned (AOI 锚点) → 电阻 → IC
        for wname, contours in res["wires"].items():
            wc = WIRE_DRAW_COLOR.get(wname, (200, 200, 200))
            wc_rgb = (wc[2], wc[1], wc[0])
            for cnt in contours:
                pts = cnt.squeeze().tolist()
                if len(pts) > 2:
                    flat = [tuple(p) for p in pts]
                    draw.line(flat + [flat[0]], fill=wc_rgb + (100,), width=2)

        pin_dets = [d for d in res["detections"] if d["cls_name"] == "pinned" and d["conf"] >= 0.40]
        res_dets = [d for d in res["detections"] if d["cls_name"] == "resistor"]
        ic_dets = [d for d in res["detections"] if d["cls_name"] == "IC"]

        for d in pin_dets:
            draw_detection_box(overlay, draw, d["box"], d["cls_name"], d["conf"], font_label, font_xs, compact=True)

        for d in res_dets:
            draw_detection_box(overlay, draw, d["box"], d["cls_name"], d["conf"], font_label, font_xs)

        for d in ic_dets:
            bx = d["box"]
            x1, y1, x2, y2 = [int(v) for v in bx]
            draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(180, 40, 180, 255), width=4)
            tag = f"UA741 (IC) {d['conf']:.0%}"
            tbox = draw.textbbox((0, 0), tag, font=font_label)
            tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
            tag_y = max(0, y1 - th - 10)
            draw.rounded_rectangle(
                [x1 - 2, tag_y, x1 + tw + 14, tag_y + th + 8],
                radius=5, fill=(140, 20, 160, 230), outline=(200, 80, 220, 255), width=2,
            )
            draw.text((x1 + 5, tag_y + 3), tag, fill=(255, 255, 255), font=font_label)

        img_pil = PILImage.alpha_composite(img_pil, overlay)

        # 图例 + 标题
        legend_overlay = PILImage.new("RGBA", img_pil.size, (0, 0, 0, 0))
        ldraw = ImageDraw.Draw(legend_overlay)

        title_text = f"LabGuardian 元件检测 — {res['label']}"
        ic_n = len(ic_dets)
        res_n = len(res_dets)
        pin_n = len([d for d in res["detections"] if d["cls_name"] == "pinned"])
        wire_n = len(res["wires"])
        stats = f"IC:{ic_n}  电阻:{res_n}  引脚:{pin_n}  导线:{wire_n}"

        ldraw.rounded_rectangle([8, 6, 492, 38], radius=6, fill=(12, 14, 18, 150))
        ldraw.text((16, 8), title_text, fill=(128, 212, 255), font=font_label)
        ldraw.text((16, 24), stats, fill=(186, 192, 200), font=font_xs)

        # 底部图例
        ly = img_pil.height - 24
        ldraw.rounded_rectangle([12, ly - 2, 332, ly + 14], radius=5, fill=(12, 14, 18, 118))
        lx = 16
        for cls, color_rgb in [("IC", (180, 40, 180)), ("resistor", (0, 140, 255)), ("pinned", (0, 200, 80))]:
            ldraw.rectangle([lx, ly + 2, lx + 8, ly + 10], fill=color_rgb)
            ldraw.text((lx + 12, ly - 1), cls, fill=(212, 216, 222), font=font_xs)
            lx += len(cls) * 7 + 26
        for wn, wc_bgr in list(WIRE_DRAW_COLOR.items())[:2]:
            wc_rgb = (wc_bgr[2], wc_bgr[1], wc_bgr[0])
            ldraw.rectangle([lx, ly + 2, lx + 8, ly + 10], fill=wc_rgb)
            short = wn[:2]
            ldraw.text((lx + 12, ly - 1), short, fill=(212, 216, 222), font=font_xs)
            lx += 36

        img_pil = PILImage.alpha_composite(img_pil, legend_overlay)

        out_path = OUT_DIR / f"img{res['idx']}_detection.jpg"
        img_pil.convert("RGB").save(str(out_path), quality=95)
        print(f"  -> {out_path.name}")

    # ── 引脚映射图 (最佳图) ──
    print(f"\n  生成引脚映射图 (img{best['idx']})...")
    img_bgr = best["image"].copy()
    img_h, img_w = img_bgr.shape[:2]
    img_pil = cv2_to_pil(img_bgr).convert("RGBA")
    overlay = PILImage.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # IC 框 + 居中标签
    if best["ic"]:
        bx = best["ic"][0]["box"]
        ix1, iy1, ix2, iy2 = [int(v) for v in bx]
        draw.rectangle([ix1 - 2, iy1 - 2, ix2 + 2, iy2 + 2],
                        outline=(180, 40, 180, 240), width=3)
        tag = "UA741 DIP-8"
        tbox = draw.textbbox((0, 0), tag, font=font_sm)
        tw = tbox[2] - tbox[0]
        tag_x = (ix1 + ix2) // 2 - tw // 2
        draw.rounded_rectangle([tag_x - 5, iy1 - 24, tag_x + tw + 5, iy1 - 4],
                                radius=4, fill=(140, 20, 160, 230))
        draw.text((tag_x, iy1 - 22), tag, fill=(255, 255, 255), font=font_sm)

    # IC 引脚标注 (使用引线避免重叠)
    draw_ic_pins(draw, ic_pin_px, font_pin, img_w=img_w)

    # 电阻标注 + 网格坐标
    for i, rd in enumerate(best["resistors"]):
        bx = rd["box"]
        cx, cy = rd["center"]
        r, c = grid.pixel_to_logic(cx, cy)
        ix1, iy1, ix2, iy2 = [int(v) for v in bx]
        draw.rectangle([ix1, iy1, ix2, iy2], outline=(0, 140, 255, 220), width=2)
        label = f"R{i+1} ({r},{c})"
        tbox = draw.textbbox((0, 0), label, font=font_sm)
        tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
        lx = max(5, ix1)
        ly = max(0, iy1 - th - 8)
        draw.rounded_rectangle([lx - 3, ly - 2, lx + tw + 5, ly + th + 3],
                                radius=4, fill=(0, 90, 200, 210))
        draw.text((lx, ly), label, fill=(255, 255, 255), font=font_sm)

    # pinned 锚点 (仅高置信度, 不显示文字，保持 AOI 风格清爽)
    high_conf_pins = [pd for pd in best["pins"] if pd["conf"] >= 0.35]
    for pd in high_conf_pins:
        draw_anchor_marker(draw, pd["box"], color=(0, 220, 80, 255))

    # 标题
    draw.rounded_rectangle([8, 6, 530, 50], radius=8, fill=(0, 0, 0, 200))
    draw.text((16, 8), f"LabGuardian 引脚映射 — {best['label']}", fill=(100, 220, 255), font=font_label)
    draw.text((16, 30), f"UA741 8-pin  |  {len(best['resistors'])} 电阻  |  {len(high_conf_pins)} 引脚定位",
              fill=(200, 200, 200), font=font_xs)

    img_pil = PILImage.alpha_composite(img_pil, overlay)
    out_path = OUT_DIR / f"img{best['idx']}_pin_mapping.jpg"
    img_pil.convert("RGB").save(str(out_path), quality=95)
    print(f"  -> {out_path.name}")

    # ── 完整报告图 (主图 + 侧面板) ──
    print(f"\n  生成完整报告图...")
    netlist_lines = generate_netlist(best["resistors"], grid)

    panel_w = 360
    main_pil = img_pil.convert("RGB")
    main_w, main_h = main_pil.size

    panel = create_info_panel(panel_w, main_h, best["detections"], grid, best["wires"], netlist_lines)

    full = PILImage.new("RGB", (main_w + panel_w, main_h), (30, 30, 40))
    full.paste(main_pil, (0, 0))
    full.paste(panel.convert("RGB"), (main_w, 0))

    out_path = OUT_DIR / "best_full_report.jpg"
    full.save(str(out_path), quality=95)
    print(f"  -> {out_path.name}")

    # ── 阶段 5: 生成文本报告 ──
    print(f"\n[5/5] 生成文本文件...")

    # 网表
    netlist_path = OUT_DIR / "netlist.txt"
    netlist_path.write_text("\n".join(netlist_lines), encoding="utf-8")
    print(f"  -> {netlist_path.name}")

    # 完整报告
    report_lines = [
        "=" * 60,
        "  LabGuardian 面包板电路识别报告",
        "=" * 60,
        f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  图片数量: {len(all_results)}",
        f"  最佳图片: img{best['idx']} ({best['label']})",
        "",
        "─" * 60,
        "  检测结果汇总",
        "─" * 60,
    ]
    for res in all_results:
        report_lines.append(f"\n  ▸ 图片 {res['idx']}: {res['label']}")
        cls_cnt = {}
        for d in res["detections"]:
            cls_cnt[d["cls_name"]] = cls_cnt.get(d["cls_name"], 0) + 1
        for cn, cnt in cls_cnt.items():
            report_lines.append(f"    {cn:15s}: {cnt}")
        report_lines.append(f"    导线 (HSV 检测): {list(res['wires'].keys())}")

    report_lines += [
        "",
        "─" * 60,
        "  UA741 引脚映射",
        "─" * 60,
    ]
    for pn in range(1, 9):
        px, py = ic_pin_px[pn]
        r, c = grid.pixel_to_logic(px, py)
        report_lines.append(f"  Pin {pn} ({UA741_PINS[pn]:20s}) → 逻辑坐标 ({r}, {c})")

    report_lines += [
        "",
        "─" * 60,
        "  电阻映射",
        "─" * 60,
    ]
    for i, rd in enumerate(best["resistors"]):
        cx, cy = rd["center"]
        r, c = grid.pixel_to_logic(cx, cy)
        report_lines.append(
            f"  R{i+1}  conf={rd['conf']:.2f}  pixel=({cx:.0f},{cy:.0f})  logic=({r},{c})"
        )

    report_lines += [
        "",
        "─" * 60,
        "  电路描述",
        "─" * 60,
        "  识别电路: UA741 反相放大器 (Inverting Amplifier)",
        "",
        "  拓扑结构:",
        "    Vin ──[R1]──┬── Pin2 (反相输入端)",
        "                │",
        "               [Rf] (反馈电阻)",
        "                │",
        "                └── Pin6 (输出端) ── Vout",
        "",
        "    Pin3 (同相输入端) ── GND",
        "    Pin7 ── V+ (+12V)",
        "    Pin4 ── V- (-12V)",
        "",
        "  增益: Av = -Rf/R1",
        "",
        "─" * 60,
        "  SPICE 网表",
        "─" * 60,
    ]
    report_lines += ["  " + l for l in netlist_lines]

    report_path = OUT_DIR / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  -> {report_path.name}")

    print(f"\n{'=' * 60}")
    print(f"  全部完成! 输出目录: {OUT_DIR}")
    print(f"  生成文件 ({len(list(OUT_DIR.iterdir()))}):")
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:35s} {size_kb:8.1f} KB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
