"""Synthetic diagnostic e2e smoke (board, Level-2 axis-c).

Injects a synthetic station snapshot (no camera needed) carrying a
``comparison_report`` + ``topology_label``, then runs the real
``diagnostic_agent`` through the full DI graph and checks that the
distilled student model (``openvino_genai_text``) produces a
GROUNDED, scene-anchored, safety-led diagnosis.

This exercises the same ``AgentService._llm_rewrite_diagnostic_answer``
path that a real pipeline submission would, proving the deployed student
runs the train≡deploy retrieval contract end-to-end on the edge.

Run on the board:
    cd /home/bupt/LabGuardian-Server
    /home/bupt/miniconda3/envs/labguardian/bin/python -m scripts.board.diagnostic_e2e_smoke
"""

from __future__ import annotations

import time

from app.agent.llm.factory import clear_llm_provider_cache
from app.core import deps
from app.schemas.angnt import AngntAskRequest


CASES = [
    {
        "name": "共射放大器 · 集电极电阻短路 (danger)",
        "station_id": "S90",
        "query": "这个电路现在危险吗？我该怎么处理？",
        "station": {
            "station_id": "S90",
            "risk_level": "danger",
            "topology_label": "common_emitter",
            "diagnostics": ["Q1 集电极电阻 R2 两端短路"],
            "risk_reasons": ["R2 两端落在同一电气节点"],
            "comparison_report": {
                "topology_label": "common_emitter",
                "items": [
                    {
                        "error_code": "COMPONENT_SHORTED_SAME_NET",
                        "severity": "danger",
                        "component_id": "R2",
                        "suggested_action": "断电后将 R2 两端跨接到不同行",
                    }
                ],
            },
            "netlist_v2": {
                "components": [{"component_id": "R2", "pins": []}],
                "nets": [{"net_id": "N1", "members": ["R2.pin1", "R2.pin2"]}],
            },
        },
    },
    {
        "name": "UA741 反相放大器 · 电解电容极性反接 (warning)",
        "station_id": "S91",
        "query": "我搭的反相放大器输出不对，怎么排查？",
        "station": {
            "station_id": "S91",
            "risk_level": "warning",
            "topology_label": "inverting_amp_ua741",
            "diagnostics": ["C1 电解电容极性反接"],
            "risk_reasons": ["C1 正负极接反"],
            "comparison_report": {
                "topology_label": "inverting_amp_ua741",
                "items": [
                    {
                        "error_code": "POLARITY_REVERSED",
                        "severity": "warning",
                        "component_id": "C1",
                        "suggested_action": "断电后按极性标记重新插接 C1",
                    }
                ],
            },
            "netlist_v2": {
                "components": [{"component_id": "C1", "pins": []}],
                "nets": [{"net_id": "N2", "members": ["C1.pin1"]}],
            },
        },
    },
]


def _scan_scene_and_packs(result) -> tuple[str, list[str]]:
    scene = ""
    ev_types: list[str] = []
    for ev in result.evidence:
        ev_types.append(ev.evidence_type)
        payload = ev.payload or {}
        for key in ("current_scene_id", "scene_id"):
            val = payload.get(key)
            if val:
                scene = str(val)
    return scene, ev_types


def main() -> None:
    clear_llm_provider_cache()
    classroom = deps.get_classroom()
    agent = deps.get_agent_service()

    for case in CASES:
        classroom.update_station(case["station"])
        t = time.time()
        accepted = agent.submit(
            AngntAskRequest(
                station_id=case["station_id"],
                query=case["query"],
                mode="diagnostic_agent",
            ),
            classroom,
        )
        result = agent.get_status(accepted.job_id).result
        elapsed = time.time() - t

        scene, ev_types = _scan_scene_and_packs(result)
        verification = next(
            (e for e in result.evidence if e.evidence_type == "verification_report"),
            None,
        )
        passed = bool(verification.payload.get("passed")) if verification else None

        print("=" * 78)
        print(f"CASE: {case['name']}")
        print(f"  query           : {case['query']}")
        print(f"  elapsed         : {elapsed:.1f}s")
        print(f"  llm_provider    : {result.actual_llm_provider}")
        print(f"  llm_model       : {result.actual_llm_model}")
        print(f"  resolved_scene  : {scene or '(none)'}")
        print(f"  verifier_passed : {passed}")
        print(f"  fault_case_pack : {'yes' if 'fault_case_pack' in ev_types else 'no'}")
        print(f"  evidence_types  : {sorted(set(ev_types))}")
        print("  ---- ANSWER ----")
        print(result.answer)
        print()


if __name__ == "__main__":
    main()
