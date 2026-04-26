from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.schemas.angnt import AngntAskRequest  # noqa: E402
from app.services.agent_service import AgentService  # noqa: E402
from app.services.classroom_state import ClassroomState  # noqa: E402
from app.services.error_tag_service import ErrorTagService  # noqa: E402
from app.services.mrag_service import MragService  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402
from app.services.teaching_kb_service import TeachingKbService  # noqa: E402
from app.services.vlm_service import VlmService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test LangGraph diagnostic_agent + OpenVINO VLM explanation."
    )
    parser.add_argument("--station-id", default="S01")
    parser.add_argument("--scenario", choices=sorted(_SCENARIOS), default="short_circuit")
    parser.add_argument("--query", default="为什么这个工位现在危险，应该怎么改？")
    parser.add_argument(
        "--model-dir",
        default=settings.VLM_OPENVINO_MODEL_DIR or "",
        help="OpenVINO VLM model directory. Defaults to VLM_OPENVINO_MODEL_DIR.",
    )
    parser.add_argument("--device", default="GPU", help="OpenVINO device, e.g. GPU, CPU, AUTO.")
    parser.add_argument("--image", default="", help="Current breadboard/scope image path.")
    parser.add_argument("--reference-image", default="", help="Reference image or waveform path.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if not args.model_dir:
        raise SystemExit("Missing --model-dir and VLM_OPENVINO_MODEL_DIR is not set.")

    teaching_kb = TeachingKbService()
    error_tag_service = ErrorTagService()
    mrag_service = MragService(teaching_kb_service=teaching_kb)
    rag_service = RagService(
        teaching_kb_service=teaching_kb,
        error_tag_service=error_tag_service,
        mrag_service=mrag_service,
    )
    agent_service = AgentService(rag_service=rag_service)
    vlm_service = VlmService(
        provider="openvino_genai",
        openvino_model_dir=args.model_dir,
        openvino_device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    classroom = ClassroomState()
    classroom.update_station(_build_station(args.scenario, args.station_id))

    lang_started = time.perf_counter()
    accepted = agent_service.submit(
        AngntAskRequest(
            station_id=args.station_id,
            query=args.query,
            mode="diagnostic_agent",
            top_k=args.top_k,
        ),
        classroom,
    )
    lang_elapsed_ms = (time.perf_counter() - lang_started) * 1000.0
    status = agent_service.get_status(accepted.job_id)
    if status.result is None:
        payload = {
            "smoke_version": "lang_vlm_integration_smoke_v1",
            "status": status.status,
            "error": status.error,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    station = classroom.get_all_stations().get(args.station_id, {})
    comparison_report = station.get("comparison_report", {}) or {}
    error_tags = error_tag_service.extract_tags(comparison_report)
    error_tag_values = [item["error_tag"] for item in error_tags]
    mrag_pack = mrag_service.build_pack(
        query=args.query,
        scene_id="exp_first_order_rc",
        error_tags=error_tag_values,
        structured_context={
            "error_codes": _extract_error_codes(comparison_report)[: args.top_k],
            "diagnostics": station.get("diagnostics", [])[: args.top_k],
            "risk_level": station.get("risk_level", "safe"),
            "circuit_snapshot": station.get("circuit_snapshot", ""),
            "lang_answer": status.result.answer,
        },
        top_k=args.top_k,
    )

    vlm_started = time.perf_counter()
    vlm_result = vlm_service.explain_rc_pack(
        mrag_pack=mrag_pack,
        user_query=args.query,
        current_image=_normalize_optional_path(args.image),
        reference_image=_normalize_optional_path(args.reference_image),
    )
    vlm_elapsed_ms = (time.perf_counter() - vlm_started) * 1000.0

    payload = {
        "smoke_version": "lang_vlm_integration_smoke_v1",
        "scenario": args.scenario,
        "station_id": args.station_id,
        "device": args.device,
        "model_dir": str(Path(args.model_dir).expanduser()),
        "available_devices": _available_openvino_devices(),
        "lang": {
            "status": status.status,
            "latency_ms": round(lang_elapsed_ms, 2),
            "job_id": accepted.job_id,
            "answer": status.result.answer,
            "citations": len(status.result.citations),
            "evidence": len(status.result.evidence),
            "actions": [action.label for action in status.result.actions],
        },
        "mrag": {
            "scene_id": mrag_pack.get("scene", {}).get("scene_id", ""),
            "error_tags": mrag_pack.get("error_tags", []),
            "fault_case_count": len(mrag_pack.get("fault_cases", [])),
            "fix_steps": mrag_pack.get("fix_steps", [])[:4],
        },
        "vlm": {
            "latency_ms": round(vlm_elapsed_ms, 2),
            "result": vlm_result,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if vlm_result.get("status") == "completed" else 2


def _build_station(scenario: str, station_id: str) -> dict[str, Any]:
    station = dict(_SCENARIOS[scenario])
    station["station_id"] = station_id
    return station


def _extract_error_codes(comparison_report: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    buckets: list[Any] = list(comparison_report.get("items", []))
    for key in (
        "topology_errors",
        "node_errors",
        "hole_errors",
        "polarity_errors",
        "component_errors",
    ):
        value = comparison_report.get(key, [])
        if isinstance(value, list):
            buckets.extend(value)
    for item in buckets:
        if not isinstance(item, dict):
            continue
        code = item.get("error_code")
        if isinstance(code, str) and code and code not in codes:
            codes.append(code)
    return codes


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


_SCENARIOS: dict[str, dict[str, Any]] = {
    "short_circuit": {
        "student_name": "demo_student",
        "risk_level": "danger",
        "progress": 0.42,
        "diagnostics": ["R1 两端短路"],
        "risk_reasons": ["R1 两端落在同一节点"],
        "circuit_snapshot": "R1.pin1 和 R1.pin2 同时落在 N1，存在短接风险。",
        "comparison_report": {
            "items": [
                {
                    "error_code": "COMPONENT_SHORTED_SAME_NET",
                    "severity": "danger",
                    "component_id": "R1",
                    "pin_name": "pin1",
                    "expected": "跨越不同节点",
                    "actual": "N1,N1",
                    "suggested_action": "断电后重新跨行插接 R1",
                    "evidence_refs": [
                        {
                            "ref_id": "R1.same_net",
                            "component_id": "R1",
                            "pin_name": "pin1",
                            "electrical_node_id": "N1",
                            "summary": "R1 两端被检测到位于同一电节点",
                        }
                    ],
                }
            ]
        },
        "netlist_v2": {
            "components": [{"component_id": "R1", "pins": []}],
            "nets": [{"net_id": "N1", "members": ["R1.pin1", "R1.pin2"]}],
        },
    },
    "floating": {
        "student_name": "demo_student",
        "risk_level": "warning",
        "progress": 0.55,
        "diagnostics": ["C1 一端悬空"],
        "risk_reasons": ["电容没有闭合到参考回路"],
        "circuit_snapshot": "C1.pin2 未连接到预期节点，当前网络不闭合。",
        "comparison_report": {
            "items": [
                {
                    "error_code": "FLOATING_PIN",
                    "severity": "warning",
                    "component_id": "C1",
                    "pin_name": "pin2",
                    "expected": "连接到 RC 输出节点",
                    "actual": "未连接",
                    "suggested_action": "将 C1.pin2 插回 RC 输出节点",
                }
            ]
        },
        "netlist_v2": {
            "components": [{"component_id": "C1", "pins": []}],
            "nets": [{"net_id": "Nout", "members": ["R1.pin2"]}],
        },
    },
    "wrong_hole": {
        "student_name": "demo_student",
        "risk_level": "warning",
        "progress": 0.68,
        "diagnostics": ["示波器输出节点接错孔位"],
        "risk_reasons": ["测量点落在错误面包板孔位"],
        "circuit_snapshot": "探头连接孔位与参考图不一致。",
        "comparison_report": {
            "items": [
                {
                    "error_code": "HOLE_MISMATCH",
                    "severity": "warning",
                    "component_id": "Probe1",
                    "pin_name": "tip",
                    "expected": "E22",
                    "actual": "E24",
                    "suggested_action": "把探头尖端移到参考节点对应孔位",
                }
            ]
        },
        "netlist_v2": {
            "components": [{"component_id": "Probe1", "pins": []}],
            "nets": [{"net_id": "Nscope", "members": ["Probe1.tip"]}],
        },
    },
}


if __name__ == "__main__":
    raise SystemExit(main())
