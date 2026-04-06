"""
test_aoi_module.py — AOI (PCB 缺陷检测) 模块端到端测试

流程:
  1. 从 Brainboard 数据集复制若干正常图片作为 golden samples
  2. 训练 PatchCore 模型
  3. 对测试图片进行推理, 输出异常分数+热力图
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# 数据源 — Brainboard_4cls 数据集
BRAINBOARD_ROOT = Path(r"D:\desktop\inter\LabGuardian\dataset\images\Brainboard_4cls")
TRAIN_IMAGES = BRAINBOARD_ROOT / "train" / "images"
TEST_IMAGES = BRAINBOARD_ROOT / "test" / "images"

# AOI 工作目录
AOI_DATASET_ROOT = PROJECT_ROOT / "datasets" / "pcb_aoi"
AOI_MODEL_DIR = PROJECT_ROOT / "models" / "aoi"
GOLDEN_DIR = AOI_DATASET_ROOT / "good"
OUTPUT_DIR = PROJECT_ROOT / "aoi_test_output"


def step1_prepare_golden_samples(n_samples: int = 10):
    """从训练集复制 n 张作为黄金样本"""
    print(f"\n{'='*60}")
    print(f"Step 1: 准备黄金样本 ({n_samples} 张)")
    print(f"{'='*60}")

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 取前 n 张
    src_images = sorted(TRAIN_IMAGES.glob("*.jpg"))[:n_samples]
    if len(src_images) == 0:
        print(f"ERROR: No images found in {TRAIN_IMAGES}")
        return False

    for img_path in src_images:
        dst = GOLDEN_DIR / img_path.name
        if not dst.exists():
            shutil.copy2(img_path, dst)
    print(f"  已复制 {len(src_images)} 张到 {GOLDEN_DIR}")
    return True


def step2_train():
    """训练 PatchCore 模型"""
    print(f"\n{'='*60}")
    print("Step 2: 训练 PatchCore 模型")
    print(f"{'='*60}")

    from app.pipeline.aoi.detector import PCBDefectDetector

    detector = PCBDefectDetector(
        model_dir=str(AOI_MODEL_DIR),
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        image_size=(256, 256),
        score_threshold=0.5,
    )

    print(f"  Golden dir: {GOLDEN_DIR}")
    print(f"  Model dir:  {AOI_MODEL_DIR}")
    print(f"  Backbone:   wide_resnet50_2")
    print(f"  开始训练...")

    result = detector.train(golden_dir=str(GOLDEN_DIR))

    print(f"  状态: {result['status']}")
    print(f"  Checkpoint: {result['checkpoint']}")
    print(f"  耗时: {result['duration_ms']:.0f}ms")
    return detector


def step3_inference(detector):
    """对测试图片运行推理"""
    print(f"\n{'='*60}")
    print("Step 3: 推理测试图片")
    print(f"{'='*60}")

    import cv2

    # 取 5 张测试图片
    test_imgs = sorted(TEST_IMAGES.glob("*.jpg"))[:5]
    if not test_imgs:
        print(f"  WARNING: No test images found in {TEST_IMAGES}")
        # 回退使用训练集的其他图片
        all_train = sorted(TRAIN_IMAGES.glob("*.jpg"))
        test_imgs = all_train[10:15]  # 跳过用于训练的前10张

    print(f"  测试图片: {len(test_imgs)} 张\n")

    results = []
    for i, img_path in enumerate(test_imgs, 1):
        print(f"  [{i}/{len(test_imgs)}] {img_path.name}")
        result = detector.predict(image_path=str(img_path))
        results.append(result)

        print(f"    异常分数:  {result['anomaly_score']:.4f}")
        print(f"    是否缺陷:  {'是 ❌' if result['is_defective'] else '否 ✓'}")
        print(f"    推理耗时:  {result['duration_ms']:.0f}ms")

        # 保存热力图
        import base64
        heatmap_bytes = base64.b64decode(result["anomaly_map_b64"])
        import numpy as np
        heatmap_arr = np.frombuffer(heatmap_bytes, dtype=np.uint8)
        heatmap_img = cv2.imdecode(heatmap_arr, cv2.IMREAD_COLOR)
        out_path = OUTPUT_DIR / f"heatmap_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), heatmap_img)
        print(f"    热力图:    {out_path.name}")

    # 汇总
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    scores = [r["anomaly_score"] for r in results]
    defective = sum(1 for r in results if r["is_defective"])
    print(f"  图片数:     {len(results)}")
    print(f"  分数范围:   {min(scores):.4f} ~ {max(scores):.4f}")
    print(f"  平均分数:   {sum(scores)/len(scores):.4f}")
    print(f"  缺陷数:     {defective}/{len(results)}")
    print(f"  输出目录:   {OUTPUT_DIR}")


def main():
    print("=" * 60)
    print("  LabGuardian AOI — PatchCore PCB 缺陷检测测试")
    print("=" * 60)

    t0 = time.time()

    # Step 1: 准备数据
    if not step1_prepare_golden_samples(n_samples=10):
        return

    # Step 2: 训练
    detector = step2_train()

    # Step 3: 推理
    step3_inference(detector)

    total = time.time() - t0
    print(f"\n总耗时: {total:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
