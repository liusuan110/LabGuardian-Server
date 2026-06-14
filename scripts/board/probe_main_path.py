"""Board probe: the MAIN scenario path — students asking (via text prompt)
about pin tables / chip parameters / how-to-wire / circuit design.

These route through datasheet_v2 (chip pins/params) + concept (CONCEPT_LIBRARY)
+ teaching_scene, NOT the diagnostic error_code bridge. This probe runs real
questions through the full agent (agent_auto, the only mode the frontend uses)
on the board's distilled student model, and reports for each:
  - classified intent
  - which retrieval channels fired (datasheet_chunk / concept_pack / teaching_scene)
  - which datasheet doc + how many chunks were recalled (recall quality)
  - the answer (truncated) + whether the verifier passed

Run on the board:
    cd /home/bupt/LabGuardian-Server
    /home/bupt/miniconda3/envs/labguardian/bin/python -m scripts.board.probe_main_path
"""

from __future__ import annotations

import time
from typing import Any

from app.core import deps
from app.schemas.angnt import AngntAskRequest

# (question, topology_label_for_scene_or_empty, what_we_hope_to_see)
CASES = [
    ("UA741 运放的 8 个引脚分别是什么功能？", "inverting_amp_ua741", "datasheet:ua741 引脚"),
    ("NE555 的工作电源电压范围是多少？", "", "datasheet:ne555 参数"),
    ("LM324 是几通道运放，单电源能用吗？", "", "datasheet:lm324 参数"),
    ("S8050 三极管的三个引脚顺序怎么区分？", "common_emitter", "datasheet:bjt_8050 引脚"),
    ("RC 电路的时间常数是什么，怎么算？", "first_order_rc", "concept:rc_time_constant"),
    ("UA741 反相放大器应该怎么接线？", "inverting_amp_ua741", "datasheet+scene 怎么连"),
    ("我想做一个一阶 RC 低通滤波器，怎么设计选参数？", "first_order_rc", "scene+concept 设计"),
]


def _evidence_summary(result: Any) -> dict[str, Any]:
    types: list[str] = []
    datasheet_docs: list[str] = []
    datasheet_chunks = 0
    concept_ids: list[str] = []
    scene_ids: list[str] = []
    for ev in result.evidence:
        et = ev.evidence_type
        types.append(et)
        payload = ev.payload if isinstance(ev.payload, dict) else {}
        if et == "datasheet_chunk":
            datasheet_chunks += 1
            doc = payload.get("document_id") or payload.get("doc_id") or payload.get("source_id") or ""
            if doc:
                datasheet_docs.append(str(doc))
        elif et == "concept_pack":
            cid = payload.get("concept_id") or ""
            if cid:
                concept_ids.append(str(cid))
        elif et == "teaching_scene":
            sid = payload.get("scene_id") or payload.get("source_id") or ""
            if sid:
                scene_ids.append(str(sid))
        elif et == "tool_results":
            # datasheet may arrive as a tool result
            def _walk(o: Any) -> None:
                nonlocal datasheet_chunks
                if isinstance(o, dict):
                    if o.get("tool_name") == "datasheet_lookup_tool":
                        pl = o.get("payload") or {}
                        for ch in (pl.get("chunks") or pl.get("results") or []):
                            datasheet_chunks += 1
                            doc = (ch.get("document_id") if isinstance(ch, dict) else "") or ""
                            if doc:
                                datasheet_docs.append(str(doc))
                    for v in o.values():
                        _walk(v)
                elif isinstance(o, list):
                    for v in o:
                        _walk(v)
            _walk(payload)
    verifier = None
    for ev in result.evidence:
        if ev.evidence_type == "verification_report":
            verifier = (ev.payload or {}).get("passed")
    return {
        "types": sorted(set(types)),
        "datasheet_docs": sorted(set(datasheet_docs)),
        "datasheet_chunks": datasheet_chunks,
        "concept_ids": sorted(set(concept_ids)),
        "scene_ids": sorted(set(scene_ids)),
        "verifier_passed": verifier,
    }


def main() -> None:
    classroom = deps.get_classroom()
    agent = deps.get_agent_service()

    for i, (q, topo, hope) in enumerate(CASES):
        sid = f"MP{i:02d}"
        station: dict[str, Any] = {"station_id": sid, "risk_level": "unknown"}
        if topo:
            station["topology_label"] = topo
        classroom.update_station(station)

        t = time.time()
        accepted = agent.submit(
            AngntAskRequest(station_id=sid, query=q, user_message=q, mode="agent_auto", top_k=5),
            classroom,
        )
        result = agent.get_status(accepted.job_id).result
        dt = time.time() - t

        intent = ""
        for ev in result.evidence:
            if ev.evidence_type == "intent":
                intent = (ev.payload or {}).get("intent") or (ev.payload or {}).get("label") or ""
        s = _evidence_summary(result)
        ans = (result.answer or "").replace("\n", " ")

        print("=" * 92)
        print(f"Q{i}: {q}")
        print(f"  期望: {hope}")
        print(f"  intent={intent!s:13s} verifier={s['verifier_passed']}  {dt:5.1f}s")
        print(f"  datasheet: docs={s['datasheet_docs']} chunks={s['datasheet_chunks']}  "
              f"concept={s['concept_ids']}  scene={s['scene_ids']}")
        print(f"  通道: {s['types']}")
        print(f"  答案: {ans[:240]}")
    print("=" * 92)


if __name__ == "__main__":
    main()
