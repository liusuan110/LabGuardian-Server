"""Board e2e: diagnostic → teaching-case recall under REAL diff_report codes.

Mirrors the demo's "faulty board → diagnose → surface teaching case" segment.
Injects a comparison_report whose items carry the REAL diff_report vocabulary
(WRONG_CONNECTION / OPEN_CIRCUIT / SHORT_CIRCUIT / *_NODE_MISMATCH /
COMPONENT_MISSING) — exactly what the production validator emits — then runs
the full diagnostic agent on the board's student model and checks that the
intended teaching fault_case is recalled (via the retagged related_error_codes)
and that the answer is grounded + verifier-passed.

Run on the board:
    cd /home/bupt/LabGuardian-Server
    /home/bupt/miniconda3/envs/labguardian/bin/python -m scripts.board.verify_faultcase_recall_real
"""

from __future__ import annotations

from typing import Any

from app.core import deps
from app.schemas.angnt import AngntAskRequest

# (topology_label, real_diff_codes, component, severity, query, intended_case)
CASES = [
    ("common_emitter", ["WRONG_CONNECTION", "OPEN_CIRCUIT", "SHORT_CIRCUIT"], "VT", "danger",
     "我这个共射放大电路三极管好像装反了，输出不正常", "ce_bjt_pin_reversed"),
    ("inverting_amp_ua741", ["WRONG_CONNECTION", "OPEN_CIRCUIT"], "U1", "warning",
     "UA741 反相放大器输出不对，怀疑输入脚接反了", "inv_input_pins_swapped"),
    ("summing_amp_ua741", ["SHORT_CIRCUIT", "EXTRA_CONNECTION"], "R1", "danger",
     "加法器几路输入好像短接在一起了", "sum_input_resistors_shorted_at_node"),
]


def _find_tool(obj: Any, found: list[dict]) -> None:
    if isinstance(obj, dict):
        if obj.get("tool_name") == "fault_case_lookup_tool" and "payload" in obj:
            found.append(obj)
        for v in obj.values():
            _find_tool(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _find_tool(v, found)


def main() -> None:
    classroom = deps.get_classroom()
    agent = deps.get_agent_service()

    ok = 0
    for i, (topo, codes, comp, sev, query, intended) in enumerate(CASES):
        sid = f"FCR{i:02d}"
        classroom.update_station({
            "station_id": sid,
            "risk_level": sev if sev == "danger" else "warning",
            "topology_label": topo,
            "diagnostics": [f"{comp} {codes[0]}"],
            "risk_reasons": list(codes),
            "comparison_report": {
                "topology_label": topo,
                "items": [{"error_code": c, "severity": sev,
                           "component_id": comp, "suggested_action": f"修复 {comp}"} for c in codes],
            },
            "netlist_v2": {"components": [{"component_id": comp, "pins": []}],
                           "nets": [{"net_id": "N1", "members": [f"{comp}.pin1"]}]},
        })
        accepted = agent.submit(
            AngntAskRequest(station_id=sid, query=query, mode="diagnostic_agent"),
            classroom,
        )
        result = agent.get_status(accepted.job_id).result

        found: list[dict] = []
        for ev in result.evidence:
            if ev.evidence_type == "tool_results":
                _find_tool(ev.payload, found)
        recalled_ids = []
        for fr in found:
            for c in (fr.get("payload") or {}).get("fault_cases") or []:
                recalled_ids.append(c.get("knowledge_id"))

        verifier = None
        for ev in result.evidence:
            if ev.evidence_type == "verification_report":
                verifier = (ev.payload or {}).get("passed")

        ans = (result.answer or "").replace("\n", " ")
        hit = intended in recalled_ids
        if hit and verifier:
            ok += 1
        print("=" * 92)
        print(f"[{'✅' if hit else '❌'} recall | verifier={verifier}] {topo}  diff码={codes}")
        print(f"  想要案例 [{intended}] 命中: {hit}   召回: {recalled_ids}")
        print(f"  答案: {ans[:220]}")
    print("=" * 92)
    print(f"端到端通过(召回对+verifier过): {ok}/{len(CASES)}")


if __name__ == "__main__":
    main()
