"""
为项目计划书生成高质量 PCB 缺陷伪彩色热力图

使用 WinCLIP 零样本检测 + 模板差分融合:
  - 自动选取缺陷数量最多的 PCB 图片
  - 输出: 红(严重)/黄(中等)/蓝(正常) 伪彩色热力图
  - 格式: 300DPI PNG, 适合 A4 排版
"""

import os
import sys
from pathlib import Path

# 设置环境
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── 项目路径 ──
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

DEEPPCB_ROOT = PROJECT_ROOT.parent / "DeepPCB" / "PCBData"
OUTPUT_DIR = PROJECT_ROOT / "proposal_heatmaps"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 自定义配色: 蓝(正常) → 黄(中度) → 红(严重) ──
PROPOSAL_CMAP = LinearSegmentedColormap.from_list(
    "pcb_defect",
    [
        (0.0, "#0d47a1"),   # 深蓝 - 正常
        (0.2, "#1976d2"),   # 蓝
        (0.4, "#42a5f5"),   # 浅蓝
        (0.5, "#ffee58"),   # 黄
        (0.65, "#ffa726"),  # 橙
        (0.8, "#ef5350"),   # 红
        (1.0, "#b71c1c"),   # 深红 - 严重缺陷
    ],
)

# DeepPCB 缺陷类型
DEFECT_NAMES = {1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pin-hole"}


def find_best_defect_images(top_n: int = 6) -> list[dict]:
    """扫描所有标注文件, 按缺陷数量降序, 返回信息最丰富的图片."""
    candidates = []
    for group_dir in sorted(DEEPPCB_ROOT.iterdir()):
        if not group_dir.is_dir() or not group_dir.name.startswith("group"):
            continue
        group_id = group_dir.name.replace("group", "")
        annot_dir = group_dir / f"{group_id}_not"
        image_dir = group_dir / group_id
        if not annot_dir.is_dir() or not image_dir.is_dir():
            continue

        for annot_file in sorted(annot_dir.glob("*.txt")):
            lines = annot_file.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                continue
            n_defects = len(lines)
            # 解析缺陷类型
            defect_types = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    defect_types.add(int(parts[4]))

            stem = annot_file.stem
            test_img = image_dir / f"{stem}_test.jpg"
            temp_img = image_dir / f"{stem}_temp.jpg"
            if test_img.exists() and temp_img.exists():
                candidates.append({
                    "test_path": test_img,
                    "template_path": temp_img,
                    "annot_path": annot_file,
                    "n_defects": n_defects,
                    "defect_types": defect_types,
                    "group": group_id,
                    "stem": stem,
                })

    # 按缺陷数量 + 类型多样性排序
    candidates.sort(key=lambda x: (x["n_defects"], len(x["defect_types"])), reverse=True)

    # 从不同 group 中选取, 保证多样
    selected = []
    seen_groups = set()
    for c in candidates:
        if c["group"] not in seen_groups:
            selected.append(c)
            seen_groups.add(c["group"])
        if len(selected) >= top_n:
            break
    # 不够就补
    for c in candidates:
        if c not in selected:
            selected.append(c)
        if len(selected) >= top_n:
            break

    return selected[:top_n]


def draw_gt_boxes(img_bgr: np.ndarray, annot_path: Path) -> np.ndarray:
    """在图上绘制真值缺陷矩形框."""
    vis = img_bgr.copy()
    colors = {
        1: (0, 0, 255),   2: (0, 165, 255), 3: (0, 255, 255),
        4: (0, 255, 0),   5: (255, 0, 0),   6: (255, 0, 255),
    }
    lines = annot_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        x1, y1, x2, y2, cls_id = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        color = colors.get(cls_id, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = DEFECT_NAMES.get(cls_id, str(cls_id))
        cv2.putText(vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return vis


def make_pseudo_color_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    """将 [0,1] 异常图转为自定义伪彩色 RGB 图 (H,W,3), uint8."""
    colored = PROPOSAL_CMAP(anomaly_map)[:, :, :3]  # (H, W, 3) float [0,1]
    return (colored * 255).astype(np.uint8)


def generate_panel(info: dict, anomaly_map: np.ndarray, idx: int):
    """生成 4 格面板: 模板 | 原图(标注) | 纯热力图 | 热力图叠加."""
    test_bgr = cv2.imread(str(info["test_path"]))
    temp_bgr = cv2.imread(str(info["template_path"]))
    h, w = test_bgr.shape[:2]

    # 模板
    template_rgb = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2RGB)
    # 原图 + GT 框
    annotated = draw_gt_boxes(test_bgr, info["annot_path"])
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    # 纯热力图
    heatmap_resized = cv2.resize(anomaly_map, (w, h))
    heatmap_rgb = make_pseudo_color_heatmap(heatmap_resized)
    # 热力图叠加
    test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
    overlay = (test_rgb.astype(np.float32) * 0.55 + heatmap_rgb.astype(np.float32) * 0.45)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    defect_strs = [DEFECT_NAMES.get(d, str(d)) for d in sorted(info["defect_types"])]
    title = f"PCB #{idx+1}  |  {info['n_defects']} defects: {', '.join(defect_strs)}"

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)

    labels = ["Golden Template", "Test Image (GT)", "Anomaly Heatmap", "Overlay"]
    images = [template_rgb, annotated_rgb, heatmap_rgb, overlay]
    for ax, img, lab in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(lab, fontsize=10)
        ax.axis("off")

    # 只在热力图上加 colorbar
    sm = plt.cm.ScalarMappable(cmap=PROPOSAL_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Anomaly Score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    panel_path = OUTPUT_DIR / f"panel_{idx+1:02d}.png"
    fig.savefig(str(panel_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Panel saved: {panel_path.name}")

    # 同时保存单独的纯热力图 (用于计划书单独引用)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    ax2.imshow(heatmap_rgb)
    ax2.axis("off")
    sm2 = plt.cm.ScalarMappable(cmap=PROPOSAL_CMAP, norm=plt.Normalize(0, 1))
    sm2.set_array([])
    cbar2 = fig2.colorbar(sm2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Anomaly Score", fontsize=9)
    plt.tight_layout()
    solo_path = OUTPUT_DIR / f"heatmap_{idx+1:02d}.png"
    fig2.savefig(str(solo_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  [OK] Heatmap saved: {solo_path.name}")

    # 保存叠加图
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    ax3.imshow(overlay)
    ax3.axis("off")
    plt.tight_layout()
    overlay_path = OUTPUT_DIR / f"overlay_{idx+1:02d}.png"
    fig3.savefig(str(overlay_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    print(f"  [OK] Overlay saved: {overlay_path.name}")


def compute_template_diff_heatmap(test_path: Path, template_path: Path) -> np.ndarray:
    """仅用模板差分生成 [0,1] 异常图，无需 WinCLIP。"""
    test_bgr = cv2.imread(str(test_path))
    temp_bgr = cv2.imread(str(template_path))
    h, w = test_bgr.shape[:2]
    temp_resized = cv2.resize(temp_bgr, (w, h))
    diff = cv2.absdiff(
        cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(temp_resized, cv2.COLOR_BGR2GRAY),
    )
    diff_blur = cv2.GaussianBlur(diff.astype(np.float32), (5, 5), 0)
    d_min, d_max = diff_blur.min(), diff_blur.max()
    if d_max > d_min:
        diff_norm = (diff_blur - d_min) / (d_max - d_min)
    else:
        diff_norm = np.zeros_like(diff_blur)
    return diff_norm.astype(np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PCB Defect Heatmap Generator")
    parser.add_argument("--diff-only", action="store_true", help="仅用模板差分，不加载 WinCLIP")
    args = parser.parse_args()

    print("=" * 60)
    print("  PCB Defect Heatmap Generator for Proposal")
    print("=" * 60)

    # Step 1: 选图
    print("\n[1/3] Scanning DeepPCB for images with most defects...")
    images = find_best_defect_images(top_n=5)
    for i, info in enumerate(images):
        types = [DEFECT_NAMES.get(d, str(d)) for d in sorted(info["defect_types"])]
        print(f"  #{i+1}: {info['stem']} (group {info['group']}) "
              f"- {info['n_defects']} defects [{', '.join(types)}]")

    use_winclip = not args.diff_only
    detector = None
    if use_winclip:
        print("\n[2/3] Initializing WinCLIP detector...")
        try:
            from app.pipeline.aoi.winclip_detector import WinCLIPDetector
            detector = WinCLIPDetector(
                class_name="printed circuit board",
                k_shot=0,
                score_threshold=0.35,
            )
            detector.initialize()
            print("  WinCLIP ready!")
        except Exception as e:
            print(f"  WinCLIP failed: {e}")
            print("  Falling back to template-diff only mode.")
            use_winclip = False
    else:
        print("\n[2/3] Using template-diff only (no WinCLIP)")

    # Step 3: 逐图推理并生成热力图
    print(f"\n[3/3] Generating heatmaps -> {OUTPUT_DIR}/")
    for i, info in enumerate(images):
        print(f"\n--- Image #{i+1}: {info['stem']} ---")
        if use_winclip and detector:
            result = detector.predict_with_fusion(
                test_image_path=str(info["test_path"]),
                template_image_path=str(info["template_path"]),
                diff_weight=0.35,
                winclip_weight=0.65,
            )
            anomaly_map = result["raw_anomaly_map"]
            test_bgr = cv2.imread(str(info["test_path"]))
            temp_bgr = cv2.imread(str(info["template_path"]))
            h, w = test_bgr.shape[:2]
            diff_norm = compute_template_diff_heatmap(info["test_path"], info["template_path"])
            winclip_resized = cv2.resize(anomaly_map, (w, h))
            fusion_map = 0.35 * diff_norm + 0.65 * winclip_resized
            fusion_map = np.clip(fusion_map, 0, 1)
            print(f"  Score: {result['anomaly_score']:.4f}  (WinCLIP+Diff fusion)")
        else:
            fusion_map = compute_template_diff_heatmap(info["test_path"], info["template_path"])
            print(f"  Template-diff heatmap (no WinCLIP)")

        generate_panel(info, fusion_map, i)

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(images) * 3} images saved to: {OUTPUT_DIR}/")
    print(f"  - panel_XX.png  : 4-panel comparison (template|GT|heatmap|overlay)")
    print(f"  - heatmap_XX.png: Standalone pseudo-color heatmap")
    print(f"  - overlay_XX.png: Heatmap overlaid on original")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
