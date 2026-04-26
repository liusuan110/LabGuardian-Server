from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.services.mrag_service import MragService  # noqa: E402
from app.services.teaching_kb_service import TeachingKbService  # noqa: E402
from app.services.vlm_service import VlmService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test local OpenVINO VLM integration.")
    parser.add_argument(
        "--model-dir",
        default=settings.VLM_OPENVINO_MODEL_DIR or "",
        help="OpenVINO VLM model directory. Defaults to VLM_OPENVINO_MODEL_DIR.",
    )
    parser.add_argument("--device", default="GPU", help="OpenVINO device, e.g. CPU, GPU, AUTO.")
    parser.add_argument("--image", default="", help="Current breadboard/scope image path.")
    parser.add_argument("--reference-image", default="", help="Reference image or waveform path.")
    parser.add_argument(
        "--query",
        default="示波器 X10 档为什么读数要乘以 10",
        help="Student question for the RC experiment.",
    )
    parser.add_argument(
        "--error-tag",
        action="append",
        default=["probe_mode_error"],
        help="RC teaching error tag. Can be repeated.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if not args.model_dir:
        raise SystemExit("Missing --model-dir and VLM_OPENVINO_MODEL_DIR is not set.")

    mrag = MragService(teaching_kb_service=TeachingKbService())
    pack = mrag.build_pack(
        query=args.query,
        error_tags=args.error_tag,
        structured_context={"manual_smoke": True},
        top_k=args.top_k,
    )
    service = VlmService(
        provider="openvino_genai",
        openvino_model_dir=args.model_dir,
        openvino_device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    started = time.perf_counter()
    result = service.explain_rc_pack(
        mrag_pack=pack,
        user_query=args.query,
        current_image=_normalize_optional_path(args.image),
        reference_image=_normalize_optional_path(args.reference_image),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    payload = {
        "smoke_version": "openvino_vlm_smoke_v2",
        "provider": "openvino_genai",
        "device": args.device,
        "model_dir": str(Path(args.model_dir).expanduser()),
        "available_devices": _available_openvino_devices(),
        "latency_ms": round(elapsed_ms, 2),
        "query": args.query,
        "error_tags": args.error_tag,
        "result": result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "completed" else 2


def _normalize_optional_path(path: str) -> str | None:
    if not path:
        return None
    return str(Path(path).expanduser())


def _available_openvino_devices() -> list[str]:
    try:
        import openvino as ov

        return list(ov.Core().available_devices)
    except Exception as exc:
        return [f"unavailable:{exc}"]


if __name__ == "__main__":
    raise SystemExit(main())
