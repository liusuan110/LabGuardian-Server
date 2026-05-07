"""DK-2500 NPU smoke for OpenVINO GenAI VLM.

Phase 6 standalone smoke (NOT in CI). Run on the DK-2500 board only after
the NPU driver and Qwen-VL-Int4 weights are validated. Usage:

    python scripts/manual/tools/vlm/smoke_npu_vlm.py \\
        --model-dir /path/to/qwen-vl-int4-ov \\
        --device NPU

Falls back to GPU if NPU initialization fails. Returns a single
`vlm_explanation_v1` JSON to stdout so it's diff-comparable with the
template-provider baseline (see tests/test_vlm_provider_contract.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))


def _build_pack() -> dict:
    return {
        "scene": {"scene_id": "exp_first_order_rc", "scene_name": "first-order RC"},
        "fault_cases": [
            {"title": "导线未剥皮导致悬空", "reference_text": "请检查跳线 W1 的剥皮长度"},
        ],
        "fix_steps": ["剥皮 5mm", "重新插入孔位"],
        "error_tags": ["unstripped_wire"],
        "structured_context": {"error_codes": ["FLOATING_PIN"]},
        "references": {"images": [], "waveforms": [], "schematics": []},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="OpenVINO IR model directory")
    parser.add_argument("--device", default="NPU", help="Preferred OpenVINO device (NPU/GPU/CPU)")
    parser.add_argument("--cache-dir", default=None, help="Optional OpenVINO compile cache dir")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--current-image", default=None)
    parser.add_argument("--reference-image", default=None)
    args = parser.parse_args()

    if not Path(args.model_dir).exists():
        print(json.dumps({"status": "error", "reason": "model_dir_missing", "model_dir": args.model_dir}))
        return 2

    from app.services.vlm import MicroDefectType, analyze_micro_defect
    from app.services.vlm_service import VlmService

    service = VlmService(
        provider="openvino_genai",
        openvino_model_dir=args.model_dir,
        openvino_device=args.device,
        openvino_cache_dir=args.cache_dir,
        max_new_tokens=args.max_new_tokens,
    )

    result = analyze_micro_defect(
        vlm_service=service,
        defect_type=MicroDefectType.UNSTRIPPED_WIRE,
        mrag_pack=_build_pack(),
        user_query="请检查跳线 W1 是否未剥皮",
        current_image=args.current_image,
        reference_image=args.reference_image,
    )

    # Strip raw_response (likely large) and print compact summary.
    pretty = {
        "frame_version": result.get("result_version"),
        "provider": result.get("provider"),
        "model": result.get("model"),
        "status": result.get("status"),
        "device": args.device,
        "answer": result.get("answer"),
        "defect_type": result.get("defect_type"),
    }
    print(json.dumps(pretty, ensure_ascii=False, indent=2))
    return 0 if pretty.get("status") in {"completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
