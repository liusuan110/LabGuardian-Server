"""
WinCLIP 零样本 PCB 缺陷检测 — 端到端测试脚本

在 DeepPCB 数据集上运行 WinCLIP 零样本推理:
  1. 正常模板 (template) → 应得到 低异常分数
  2. 缺陷图片 (test)     → 应得到 高异常分数 + 缺陷热力图
  3. WinCLIP + 模板差分融合
  4. 批量测试 + 统计输出
"""

import os
import sys
import time
from pathlib import Path

# 确保 HuggingFace 镜像 (国内网络)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 项目根目录
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


def main():
    from app.pipeline.aoi.winclip_detector import WinCLIPDetector

    DEEPPCB_ROOT = Path(r"D:\desktop\LabGuardian-Server-main\DeepPCB\PCBData")
    OUTPUT_DIR = ROOT / "winclip_output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Step 1: 初始化 WinCLIP ──
    print("=" * 60)
    print("Step 1: 初始化 WinCLIP 零样本检测器")
    print("=" * 60)

    detector = WinCLIPDetector(
        class_name="printed circuit board",
        k_shot=0,
        score_threshold=0.5,
    )
    detector.initialize()
    print(f"[OK] WinCLIP 就绪 (zero-shot, 无需训练数据)")
    print()

    # ── Step 2: 收集测试图片 ──
    print("=" * 60)
    print("Step 2: 收集 DeepPCB 测试图片")
    print("=" * 60)

    test_pairs = []  # (template_path, defect_path)
    for group_dir in sorted(DEEPPCB_ROOT.iterdir()):
        if not group_dir.is_dir() or not group_dir.name.startswith("group"):
            continue
        inner = group_dir / group_dir.name.replace("group", "")
        if not inner.is_dir():
            continue
        for f in sorted(inner.iterdir()):
            if f.name.endswith("_test.jpg"):
                base = f.stem.replace("_test", "")
                temp = inner / f"{base}_temp.jpg"
                if temp.exists():
                    test_pairs.append((temp, f))
        if len(test_pairs) >= 30:
            break

    print(f"收集到 {len(test_pairs)} 对测试图片 (template + defect)")
    print()

    # ── Step 3: 零样本推理 — 正常模板 vs 缺陷图片 ──
    print("=" * 60)
    print("Step 3: 零样本推理 (正常 vs 缺陷)")
    print("=" * 60)

    normal_scores = []
    defect_scores = []
    results_log = []

    for i, (temp_path, test_path) in enumerate(test_pairs[:20]):
        # 检测正常模板
        r_normal = detector.predict(image_path=temp_path)
        normal_scores.append(r_normal["anomaly_score"])

        # 检测缺陷图片
        r_defect = detector.predict(image_path=test_path)
        defect_scores.append(r_defect["anomaly_score"])

        results_log.append({
            "pair": i,
            "normal_score": r_normal["anomaly_score"],
            "defect_score": r_defect["anomaly_score"],
            "normal_defective": r_normal["is_defective"],
            "defect_defective": r_defect["is_defective"],
        })

        print(
            f"  [{i+1:2d}] 正常={r_normal['anomaly_score']:.4f}"
            f"  缺陷={r_defect['anomaly_score']:.4f}"
            f"  {'OK' if r_defect['anomaly_score'] > r_normal['anomaly_score'] else 'FAIL'}"
        )

    print()
    print(f"  正常模板 - 平均分: {np.mean(normal_scores):.4f} (std={np.std(normal_scores):.4f})")
    print(f"  缺陷图片 - 平均分: {np.mean(defect_scores):.4f} (std={np.std(defect_scores):.4f})")
    sep = np.mean(defect_scores) - np.mean(normal_scores)
    print(f"  分数差异 (越大越好): {sep:.4f}")
    print()

    # ── Step 4: 生成热力图可视化 ──
    print("=" * 60)
    print("Step 4: 生成缺陷热力图可视化")
    print("=" * 60)

    viz_dir = OUTPUT_DIR / "heatmaps"
    viz_dir.mkdir(exist_ok=True)

    for i, (temp_path, test_path) in enumerate(test_pairs[:10]):
        # 读取原图
        test_bgr = cv2.imread(str(test_path))
        temp_bgr = cv2.imread(str(temp_path))

        # WinCLIP 推理
        r = detector.predict(image_bgr=test_bgr)
        anomaly_map = r["raw_anomaly_map"]

        # 构建 4 面板可视化:
        # [模板 | 缺陷原图 | WinCLIP 热力图 | 热力图叠加]
        h, w = test_bgr.shape[:2]
        sz = 320
        temp_viz = cv2.resize(temp_bgr, (sz, sz))
        test_viz = cv2.resize(test_bgr, (sz, sz))

        # 热力图 (纯色)
        hmap = cv2.resize(anomaly_map, (sz, sz))
        hmap_uint8 = (hmap * 255).astype(np.uint8)
        hmap_color = cv2.applyColorMap(hmap_uint8, cv2.COLORMAP_JET)

        # 叠加
        overlay = cv2.addWeighted(test_viz, 0.6, hmap_color, 0.4, 0)

        # 拼接
        panel = np.hstack([temp_viz, test_viz, hmap_color, overlay])

        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = ["Template (Normal)", "Test (Defect)", "WinCLIP Heatmap", "Overlay"]
        for j, label in enumerate(labels):
            x = j * sz + 5
            cv2.putText(panel, label, (x, 20), font, 0.5, (255, 255, 255), 1)

        score_text = f"Score: {r['anomaly_score']:.4f} | {'DEFECT' if r['is_defective'] else 'NORMAL'}"
        cv2.putText(panel, score_text, (5, sz - 10), font, 0.6, (0, 255, 255), 2)

        out_path = viz_dir / f"winclip_{i:03d}.jpg"
        cv2.imwrite(str(out_path), panel)
        print(f"  保存: {out_path.name}  score={r['anomaly_score']:.4f}")

    print()

    # ── Step 5: WinCLIP + 模板差分融合 ──
    print("=" * 60)
    print("Step 5: WinCLIP + 模板差分融合")
    print("=" * 60)

    fusion_dir = OUTPUT_DIR / "fusion"
    fusion_dir.mkdir(exist_ok=True)

    for i, (temp_path, test_path) in enumerate(test_pairs[:10]):
        r = detector.predict_with_fusion(
            test_image_path=test_path,
            template_image_path=temp_path,
            diff_weight=0.3,
            winclip_weight=0.7,
        )

        # 构建 5 面板: 模板 | 缺陷 | 模板差分 | WinCLIP | 融合
        test_bgr = cv2.imread(str(test_path))
        temp_bgr = cv2.imread(str(temp_path))
        h, w = test_bgr.shape[:2]
        sz = 256

        temp_viz = cv2.resize(temp_bgr, (sz, sz))
        test_viz = cv2.resize(test_bgr, (sz, sz))

        # 模板差分
        diff = cv2.absdiff(
            cv2.cvtColor(cv2.resize(test_bgr, (sz, sz)), cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(cv2.resize(temp_bgr, (sz, sz)), cv2.COLOR_BGR2GRAY),
        )
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)

        # WinCLIP 热力图
        winclip_map = r["raw_anomaly_map"]
        wmap = cv2.resize(winclip_map, (sz, sz))
        wmap_color = cv2.applyColorMap((wmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # 融合叠加 (从 fusion_map_b64 解码)
        import base64
        fus_bytes = base64.b64decode(r["fusion_map_b64"])
        fus_arr = np.frombuffer(fus_bytes, dtype=np.uint8)
        fus_img = cv2.imdecode(fus_arr, cv2.IMREAD_COLOR)
        fus_viz = cv2.resize(fus_img, (sz, sz))

        panel = np.hstack([temp_viz, test_viz, diff_color, wmap_color, fus_viz])

        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = ["Template", "Defect", "Diff", "WinCLIP", "Fusion"]
        for j, label in enumerate(labels):
            cv2.putText(panel, label, (j * sz + 5, 20), font, 0.5, (255, 255, 255), 1)

        cv2.putText(
            panel,
            f"Score={r['anomaly_score']:.4f} T={r['duration_ms']:.0f}ms",
            (5, sz - 10), font, 0.5, (0, 255, 255), 1,
        )

        out_path = fusion_dir / f"fusion_{i:03d}.jpg"
        cv2.imwrite(str(out_path), panel)
        print(f"  保存: {out_path.name}  score={r['anomaly_score']:.4f}  time={r['duration_ms']:.0f}ms")

    print()

    # ── 总结 ──
    print("=" * 60)
    print("WinCLIP 零样本 PCB 检测 — 测试完成")
    print("=" * 60)
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  热力图:   {viz_dir} ({len(list(viz_dir.glob('*.jpg')))} 张)")
    print(f"  融合图:   {fusion_dir} ({len(list(fusion_dir.glob('*.jpg')))} 张)")
    print(f"  正常平均分: {np.mean(normal_scores):.4f}")
    print(f"  缺陷平均分: {np.mean(defect_scores):.4f}")

    correct = sum(1 for s in defect_scores if s > 0.5) + sum(1 for s in normal_scores if s <= 0.5)
    total = len(defect_scores) + len(normal_scores)
    print(f"  准确率: {correct}/{total} = {correct/total*100:.1f}%")
    print()
    print("核心优势: 整个流程无需任何训练数据！")


if __name__ == "__main__":
    main()
