"""YOLO-Pose → OpenVINO FP16 + INT8 量化 (for NPU/iGPU deployment).

用法:
    python scripts/openvino_export/yolo_pose_npu_export.py

输入:
    train_demo/pose_components/weights/best.pt                  (Ultralytics YOLOv8s-pose)
    bread_detect/bread*.png + train_demo/pose_components/*.jpg  (校准图)

输出:
    train_demo/pose_components/weights/best_openvino_fp16/{best.xml, best.bin}
    train_demo/pose_components/weights/best_openvino_int8/{best.xml, best.bin}

后续:
    rsync best_openvino_{fp16,int8}/ bupt@10.133.22.42:~/labguardian/models/yolo_pose/
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PT_WEIGHT = ROOT / "train_demo" / "pose_components" / "weights" / "best.pt"
OUT_DIR = ROOT / "train_demo" / "pose_components" / "weights"
FP16_DIR = OUT_DIR / "best_openvino_fp16"
INT8_DIR = OUT_DIR / "best_openvino_int8"

IMGSZ = 640

# 校准图来源
CALIB_SOURCES = [
    ROOT / "bread_detect" / "bread.png",
    ROOT / "bread_detect" / "bread_1.png",
    ROOT / "bread_detect" / "bread_2.png",
    ROOT / "bread_detect" / "bread_3.png",
    ROOT / "bread_detect" / "bread_4.png",
    ROOT / "bread_detect" / "bread_black_point.jpg",
]
# 训练 batch composite (4-up 网格 大图 同样能做校准)
CALIB_SOURCES += sorted((ROOT / "train_demo" / "pose_components").glob("train_batch*.jpg"))
CALIB_SOURCES += sorted((ROOT / "train_demo" / "pose_components").glob("val_batch*.jpg"))

# 每张原图通过随机 crop+flip 生成 N 个样本，提升校准多样性
PATCHES_PER_IMAGE = 8


def letterbox(img: np.ndarray, new_shape: int = IMGSZ) -> np.ndarray:
    """YOLO 标准 letterbox: keep aspect ratio, pad to new_shape × new_shape."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    return cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR -> letterbox -> RGB -> /255 -> NCHW float32."""
    img = letterbox(img_bgr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return img[None, :, :, :]  # NCHW


def gen_calibration_samples(rng: np.random.Generator) -> Iterable[np.ndarray]:
    """Yields preprocessed (1,3,640,640) float32 arrays."""
    for src in CALIB_SOURCES:
        if not src.exists():
            print(f"  [skip] {src.relative_to(ROOT)} (missing)")
            continue
        img = cv2.imread(str(src))
        if img is None:
            print(f"  [skip] {src.relative_to(ROOT)} (decode fail)")
            continue
        h, w = img.shape[:2]
        # 1. 原图 letterbox
        yield preprocess(img)
        # 2. 随机 crop patch（裁剪到 ~640-1280，再 letterbox）
        for _ in range(PATCHES_PER_IMAGE - 1):
            crop = min(h, w, rng.integers(640, max(641, min(h, w) + 1)))
            y0 = rng.integers(0, max(1, h - crop + 1))
            x0 = rng.integers(0, max(1, w - crop + 1))
            patch = img[y0:y0 + crop, x0:x0 + crop]
            # 50% 概率水平翻转
            if rng.random() < 0.5:
                patch = patch[:, ::-1, :].copy()
            yield preprocess(patch)


def main() -> int:
    if not PT_WEIGHT.exists():
        print(f"ERROR: {PT_WEIGHT} 不存在")
        return 1

    print("=" * 70)
    print("Step 1: Ultralytics .pt → OpenVINO FP16 IR")
    print("=" * 70)
    from ultralytics import YOLO

    # 清空旧输出
    if FP16_DIR.exists():
        shutil.rmtree(FP16_DIR)
    if INT8_DIR.exists():
        shutil.rmtree(INT8_DIR)

    model = YOLO(str(PT_WEIGHT))
    # half=True → FP16; dynamic=False → 固定 shape (NPU 必需); nms=False → 后处理走 Python
    exported = model.export(
        format="openvino",
        imgsz=IMGSZ,
        half=True,
        dynamic=False,
        nms=False,
        simplify=True,
    )
    exported_dir = Path(exported)
    print(f"  导出路径: {exported_dir}")

    # ultralytics 把 OV IR 输出到 best_openvino_model/ 默认目录
    # 这次我们要带后缀，所以挪一下
    if exported_dir.exists() and exported_dir != FP16_DIR:
        if FP16_DIR.exists():
            shutil.rmtree(FP16_DIR)
        shutil.move(str(exported_dir), str(FP16_DIR))
    print(f"  → {FP16_DIR}")

    fp16_xml = FP16_DIR / "best.xml"
    if not fp16_xml.exists():
        # 部分版本输出文件名变化
        candidates = list(FP16_DIR.glob("*.xml"))
        if candidates:
            fp16_xml = candidates[0]
        else:
            print("ERROR: FP16 .xml 未找到")
            return 1
    print(f"  FP16 IR: {fp16_xml.name} ({(FP16_DIR / fp16_xml.with_suffix('.bin').name).stat().st_size / 1024 / 1024:.1f} MB)")

    print()
    print("=" * 70)
    print("Step 2: 准备校准数据集")
    print("=" * 70)
    rng = np.random.default_rng(42)
    samples = list(gen_calibration_samples(rng))
    print(f"  校准样本数: {len(samples)}")
    if len(samples) < 8:
        print("WARN: 校准样本 < 8 张，INT8 量化精度可能下降")

    print()
    print("=" * 70)
    print("Step 3: NNCF INT8 PTQ 量化")
    print("=" * 70)
    import openvino as ov
    import nncf

    ov_model = ov.Core().read_model(str(fp16_xml))
    print(f"  load FP16 IR: inputs={[i.shape for i in ov_model.inputs]} outputs={[o.shape for o in ov_model.outputs]}")

    def transform_fn(sample: np.ndarray) -> dict:
        return {0: sample.astype(np.float32)}

    calibration_dataset = nncf.Dataset(samples, transform_fn)
    quantized = nncf.quantize(
        ov_model,
        calibration_dataset,
        preset=nncf.QuantizationPreset.MIXED,  # YOLO 推荐 mixed (activation=sym, weight=asym)
        subset_size=min(300, len(samples)),
        # 不排除任何算子 —— YOLOv8-pose PTQ 经验上不需要 ignored_scope
    )

    INT8_DIR.mkdir(parents=True, exist_ok=True)
    int8_xml = INT8_DIR / "best.xml"
    ov.save_model(quantized, int8_xml, compress_to_fp16=False)
    int8_bin = INT8_DIR / "best.bin"
    print(f"  INT8 IR saved: {int8_xml.name} ({int8_bin.stat().st_size / 1024 / 1024:.1f} MB)")

    # 复制 metadata.yaml 到 INT8 目录（pipeline 需要）
    meta_src = FP16_DIR / "metadata.yaml"
    if meta_src.exists():
        shutil.copy2(meta_src, INT8_DIR / "metadata.yaml")
        print(f"  metadata.yaml copied to INT8 dir")

    print()
    print("=" * 70)
    print("✅ 完成。两个 IR 输出:")
    print(f"  FP16: {FP16_DIR}")
    print(f"  INT8: {INT8_DIR}")
    print(f"\nNext: rsync 上板，跑 bench_yolo_pose.py 测三设备 × 两精度")
    return 0


if __name__ == "__main__":
    sys.exit(main())
