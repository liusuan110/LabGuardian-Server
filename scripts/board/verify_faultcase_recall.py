"""Verify fault_case recall in the diagnostic path by inspecting tool_results.

The earlier smoke checked for an evidence_type ``fault_case_pack`` that the
diagnostic ReAct path never emits — fault cases come back through the
``fault_case_lookup_tool`` result, which lands in the ``tool_results``
evidence. This script inspects that correctly.

Expectation (per the KB ``related_error_codes`` cross-reference):
  - differential_pair + NODE_MISMATCH  → HIT  (tail_path_broken)
  - ua741_inverting + POLARITY_REVERSED → HIT  (input_pins_swapped)
  - common_emitter + COMPONENT_SHORTED_SAME_NET → MISS (no CE fault_case tagged)

Run on the board:
    cd /home/bupt/LabGuardian-Server
    /home/bupt/miniconda3/envs/labguardian/bin/python -m scripts.board.verify_faultcase_recall
"""

from __future__ import annotations

from typing import Any

from app.core import deps
from app.schemas.angnt import AngntAskRequest


CASES = [
    ("differential_pair", "NODE_MISMATCH", "Q2", "warning",
     "差分放大器两路输出不对称，可能是哪里接错了？", "应命中 tail_path_broken"),
    ("inverting_amp_ua741", "POLARITY_REVERSED", "C1", "warning",
     "我搭的UA741反相放大器输出不对，怎么排查？", "应命中 input_pins_swapped"),
    ("common_emitter", "COMPONENT_SHORTED_SAME_NET", "R2", "danger",
     "这个共射放大电路现在危险吗？", "对照：CE 无此 code 的 fault_case，应落空"),
]


def _find_tool_results(obj: Any, found: list[dict]) -> None:
    """Recursively collect any dict that looks like a fault_case_lookup_tool result."""
    if isinstance(obj, dict):
        if obj.get("tool_name") == "fault_case_lookup_tool" and "payload" in obj:
            found.append(obj)
        for v in obj.values():
            _find_tool_results(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _find_tool_results(v, found)


def main() -> None:
    classroom = deps.get_classroom()
    agent = deps.get_agent_service()

    for i, (topo, code, comp, sev, query, note) in enumerate(CASES):
        sid = f"FC{i:02d}"
        classroom.update_station({
            "station_id": sid,
            "risk_level": sev if sev == "danger" else "warning",
            "topology_label": topo,
            "diagnostics": [f"{comp} {code}"],
            "risk_reasons": [code],
            "comparison_report": {
                "topology_label": topo,
                "items": [{
                    "error_code": code, "severity": sev,
                    "component_id": comp, "suggested_action": f"修复 {comp}",
                }],
            },
            "netlist_v2": {"components": [{"component_id": comp, "pins": []}],
                           "nets": [{"net_id": "N1", "members": [f"{comp}.pin1"]}]},
        })
        accepted = agent.submit(
            AngntAskRequest(station_id=sid, query=query, mode="diagnostic_agent"),
            classroom,
        )
        result = agent.get_status(accepted.job_id).result

        # find fault_case_lookup_tool result in the tool_results evidence
        found: list[dict] = []
        for ev in result.evidence:
            if ev.evidence_type == "tool_results":
                _find_tool_results(ev.payload, found)

        print("=" * 76)
        print(f"{topo} + {code}@{comp}   ({note})")
        if not found:
            print("  fault_case_lookup_tool: 未在 tool_results 中找到(工具可能未被调度)")
        for fr in found:
            payload = fr.get("payload") or {}
            cases = payload.get("fault_cases") or []
            print(f"  status={fr.get('status')}  summary={fr.get('summary')}")
            print(f"  命中数={len(cases)}")
            for c in cases:
                print(f"    - {c.get('knowledge_id')} | {c.get('title')} "
                      f"| related_error_codes={c.get('related_error_codes')}")
        print()


if __name__ == "__main__":
    main()
