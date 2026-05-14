"""
YOLO 集成离线冒烟测试

验证路径:
  1. config.py 正确解析到 models/component/best.pt
  2. model_inspector 合同检查通过 (names 含 ic-8 / ic-14)
  3. ComponentDetector 能加载权重不报错
  4. 对纯黑图调用 detect() 不崩溃
  5. label_mapping 映射正确 (IC-8→IC8, IC-14→IC14, potentiometer→Potentiometer)
  6. default_pin_names 对 IC8 / IC14 返回 anchor_pin 命名

运行方式 (无需服务器):
  python tests/manual/smoke/test_yolo_integration_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))
        status = PASS if condition else FAIL
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {name}{suffix}")

    # ── 1. config 路径解析 ─────────────────────────────────────────────────
    print("\n=== 1. config 路径解析 ===")
    model_path: str | None = None
    try:
        from app.core.config import settings

        model_path = settings.YOLO_MODEL_PATH
        check("YOLO_MODEL_PATH 非空", bool(model_path), model_path or "<empty>")
        check(
            "YOLO_MODEL_PATH 指向 models/component/best.pt",
            model_path is not None and Path(model_path).exists(),
            model_path or "<empty>",
        )
    except Exception as e:
        check("config 导入", False, str(e))

    # ── 2. model_inspector ────────────────────────────────────────────────
    print("\n=== 2. model_inspector ===")
    try:
        from app.pipeline.vision.model_inspector import inspect_yolo_weight

        if model_path and Path(model_path).exists():
            contract = inspect_yolo_weight(model_path)
            check("exists=True", contract["exists"] is True)
            check("task 已识别", contract["task"] != "unknown", contract["task"])
            check("task 不是 pose", contract["task"] != "pose", contract["task"])
            names = contract.get("names", [])
            check("names 非空", len(names) > 0, str(names))
            ic8_found = any("ic" in n.lower() and ("8" in n or "14" in n) for n in names)
            check("names 含 ic-8 或 ic-14", ic8_found, str(names))
        else:
            check("model_inspector(跳过:文件不存在)", False, str(model_path))
    except Exception as e:
        check("model_inspector 导入", False, str(e))

    # ── 3. ComponentDetector 加载 ─────────────────────────────────────────
    print("\n=== 3. ComponentDetector 加载 ===")
    detector = None
    try:
        from app.pipeline.vision.detector import ComponentDetector

        if model_path and Path(model_path).exists():
            detector = ComponentDetector(model_path=model_path, device="cpu")
            check("detector.load() 成功", detector.model is not None)
            check("model_contract.loaded=True", detector.model_contract.get("loaded") is True)
            check(
                "backend_type 是 detect",
                detector.backend_type == "yolo_detect_component",
                detector.backend_type,
            )
        else:
            check("ComponentDetector(跳过:文件不存在)", False, str(model_path))
    except Exception as e:
        check("ComponentDetector 导入/加载", False, str(e))

    # ── 4. detect() 纯黑图不崩溃 ──────────────────────────────────────────
    print("\n=== 4. detect() 纯黑图不崩溃 ===")
    try:
        if detector and detector.model is not None:
            black_img = np.zeros((640, 640, 3), dtype=np.uint8)
            dets = detector.detect(black_img, conf=0.25, iou=0.5, imgsz=640)
            check("detect() 返回 list", isinstance(dets, list), f"len={len(dets)}")
        else:
            check("detect()(跳过:模型未加载)", False)
    except Exception as e:
        check("detect() 调用", False, str(e))

    # ── 5. label_mapping ──────────────────────────────────────────────────
    print("\n=== 5. label_mapping 映射正确性 ===")
    try:
        from app.pipeline.vision.label_mapping import (
            default_package_type,
            default_pin_names,
            default_pin_schema_id,
            is_supported_component_type,
            normalize_component_type,
        )

        cases = [
            ("IC-8", "IC8"),
            ("IC-14", "IC14"),
            ("potentiometer", "Potentiometer"),
            ("resistor", "Resistor"),
        ]
        for raw, expected in cases:
            result = normalize_component_type(raw)
            check(f"normalize({raw!r}) → {expected!r}", result == expected, f"got {result!r}")

        for ct in ("IC8", "IC14"):
            check(f"is_supported({ct})", is_supported_component_type(ct))
            pkg = default_package_type(ct)
            expected_pkg = "dip8" if ct == "IC8" else "dip14"
            check(f"default_package_type({ct}) → {expected_pkg!r}", pkg == expected_pkg, f"got {pkg!r}")
            schema = default_pin_schema_id(ct, pkg)
            expected_schema = "dip8_anchor_pair" if ct == "IC8" else "dip14_anchor_pair"
            check(
                f"default_pin_schema_id({ct}) → {expected_schema!r}",
                schema == expected_schema,
                f"got {schema!r}",
            )
            pins = default_pin_names(ct, 2)
            check(f"default_pin_names({ct},2) → anchor_pin*", all("anchor_pin" in p for p in pins), str(pins))

    except Exception as e:
        check("label_mapping 导入", False, str(e))

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("\n=== 汇总 ===")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  {passed}/{total} 通过\n")

    if passed < total:
        print("失败项:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
