"""
RAG 服务最小骨架

WP-0 (2026-05-24): legacy PDF KB (KbService / Chroma / OpenAI embeddings) has
been removed from the agent main path. The production retrieval contract is
documented in ``docs/retrieval-contract.md`` and only consumes:

  - teaching_scene + fault_case (via TeachingKbService, rule-based)
  - datasheet v2 (via DatasheetKbService, local OpenVINO embeddings)
  - circuit knowledge (via CircuitKbService)
  - structured station / error_tag facts

The legacy ``KbService`` class is retained for admin PDF upload tooling
(``app/api/v1/kb.py``) but is no longer reachable from the agent graph or
this service.
"""

from __future__ import annotations

from typing import Any

from app.schemas.angnt import AngntCitation, AngntEvidence
from app.services.classroom_state import ClassroomState
from app.services.error_tag_service import ErrorTagService
from app.services.mrag_service import MragService
from app.services.scene_resolver import resolve_scene_id, scene_display_name
from app.services.teaching_kb_service import TeachingKbService


class RagService:
    def __init__(
        self,
        teaching_kb_service: TeachingKbService | None = None,
        error_tag_service: ErrorTagService | None = None,
        mrag_service: MragService | None = None,
    ) -> None:
        self._teaching_kb_service = teaching_kb_service
        self._error_tag_service = error_tag_service
        self._mrag_service = mrag_service

    def build_context(
        self,
        *,
        classroom: ClassroomState,
        station_id: str,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        stations = classroom.get_all_stations()
        station = stations.get(station_id, {})
        reference = classroom.get_reference()

        citations: list[AngntCitation] = []
        evidence: list[AngntEvidence] = []

        if station:
            diagnostics = station.get("diagnostics", [])
            comparison_report = station.get("comparison_report", {}) or {}
            error_codes = self._extract_error_codes(comparison_report)
            error_tags = (
                self._error_tag_service.extract_tags(comparison_report)
                if self._error_tag_service
                else []
            )
            # WP-1 (2026-05-24): resolve scene_id from topology context.
            # None → no fault_case_pack will be added below; we MUST NOT
            # default to "exp_first_order_rc" (that bug is what WP-1 fixes).
            scene_id = resolve_scene_id(
                station=station,
                comparison_report=comparison_report,
            )
            scene_label = scene_display_name(scene_id) or "教学场景"
            summary = (
                f"risk={station.get('risk_level', 'safe')}, "
                f"progress={station.get('progress', 0.0):.2f}, "
                f"diagnostics={len(diagnostics)}"
            )
            citations.append(
                AngntCitation(
                    source_type="station_state",
                    source_id=station_id,
                    title="实时工位状态",
                    snippet=summary,
                )
            )
            evidence.append(
                AngntEvidence(
                    evidence_type="station_state",
                    source_id=station_id,
                    summary=summary,
                    payload={
                        "risk_level": station.get("risk_level", "safe"),
                        "risk_reasons": station.get("risk_reasons", []),
                        "diagnostics": diagnostics[:top_k],
                        "error_codes": error_codes[:top_k],
                        "error_tags": error_tags[:top_k],
                        "scene_id": scene_id or "",
                    },
                )
            )
            if error_tags:
                # WP-1: title/summary are now scene-aware (or generic when
                # topology unknown), not hardcoded to "一阶 RC".
                error_tag_scope = scene_id or "unknown_topology"
                citations.append(
                    AngntCitation(
                        source_type="error_tags",
                        source_id=f"{station_id}:{error_tag_scope}:error_tags",
                        title=f"{scene_label} 结构化错误标签",
                        snippet="；".join(tag["error_tag"] for tag in error_tags[:top_k]),
                    )
                )
                evidence.append(
                    AngntEvidence(
                        evidence_type="error_tags",
                        source_id=f"{station_id}:{error_tag_scope}:error_tags",
                        summary=f"从 validator_report_v2 映射得到的 {scene_label} 教学错误标签",
                        payload={"error_tags": error_tags[:top_k], "scene_id": scene_id or ""},
                    )
                )

            snapshot = station.get("circuit_snapshot", "")
            if snapshot:
                citations.append(
                    AngntCitation(
                        source_type="pipeline_snapshot",
                        source_id=f"{station_id}:snapshot",
                        title="电路快照",
                        snippet=snapshot[:240],
                    )
                )
                evidence.append(
                    AngntEvidence(
                        evidence_type="circuit_snapshot",
                        source_id=f"{station_id}:snapshot",
                        summary="来自 pipeline 的电路描述",
                        payload={"circuit_snapshot": snapshot},
                    )
                )

            if self._teaching_kb_service:
                error_tag_values = [tag["error_tag"] for tag in error_tags]
                # Cap teaching_scene hits to leave room for fault_case_pack
                # and other downstream evidence items. With 6 demo scenes in
                # the KB, an unbounded top_k=5 would consume all slots and
                # truncate the more-specific fault_case_pack at the end.
                scene_cap = max(1, top_k - 2)
                if scene_id:
                    # WP-1 (2026-05-24): when topology is resolved, fetch
                    # the resolved scene DIRECTLY rather than running the
                    # cross-scene query ranker. The ranker can truncate the
                    # correct scene below scene_cap and leave the topology
                    # entirely uncovered in evidence (silent recall miss).
                    teaching_hits = self._build_scene_hit_for_resolved_topology(
                        scene_id=scene_id,
                        query=query,
                        error_tag_values=error_tag_values,
                        error_codes=error_codes,
                        top_k=scene_cap,
                    )
                else:
                    # No topology context → keep the cross-scene query ranker
                    # for concept-style questions.
                    teaching_hits = self._teaching_kb_service.search(
                        query=query,
                        error_codes=error_codes,
                        error_tags=error_tag_values,
                        top_k=scene_cap,
                    )
                # WP-1: build_pack only runs when we have a resolved scene_id.
                # When None, knowledge_pack stays empty → no fault_case_pack
                # is added below. Non-RC topologies (or topology-unknown
                # turns) no longer pull RC fault cases.
                knowledge_pack = self._build_mrag_pack(
                    query=query,
                    scene_id=scene_id,
                    error_tag_values=error_tag_values,
                    error_codes=error_codes,
                    diagnostics=diagnostics,
                    station=station,
                    top_k=top_k,
                )
                for hit in teaching_hits:
                    matching_faults = hit.get("matching_faults", [])
                    snippets: list[str] = []
                    if matching_faults:
                        first_fault = matching_faults[0]
                        snippets.append(str(first_fault.get("symptom", "")))
                        snippets.append(str(first_fault.get("likely_reason", "")))
                    else:
                        goals = hit.get("learning_goals", [])
                        if goals:
                            snippets.append(str(goals[0]))
                    snippet = "；".join(part for part in snippets if part)[:260]
                    citations.append(
                        AngntCitation(
                            source_type="teaching_scene",
                            source_id=str(hit.get("scene_id", "")),
                            title=str(hit.get("scene_name") or "教学场景"),
                            snippet=snippet,
                        )
                    )
                    evidence.append(
                        AngntEvidence(
                            evidence_type="teaching_scene",
                            source_id=str(hit.get("scene_id", "")),
                            summary=str(hit.get("scene_name") or "教学场景"),
                            payload={
                                "course": hit.get("course", ""),
                                "learning_goals": hit.get("learning_goals", [])[:3],
                                "matching_faults": matching_faults[:3],
                                "expected_measurements": hit.get("expected_measurements", [])[:3],
                                "source_materials": hit.get("source_materials", []),
                            },
                        )
                    )
                # WP-1: fault_case_pack source_id is now scene-keyed. Block
                # is reached only when knowledge_pack has cases AND scene_id
                # was resolved (the _build_mrag_pack guard ensures both).
                if knowledge_pack.get("fault_cases") and scene_id:
                    pack_source_id = f"{station_id}:{scene_id}_fault_cases"
                    citations.append(
                        AngntCitation(
                            source_type="fault_case_pack",
                            source_id=pack_source_id,
                            title=f"{scene_label} 图文纠错知识包",
                            snippet="；".join(
                                str(case.get("title", ""))
                                for case in knowledge_pack["fault_cases"][:top_k]
                            )[:260],
                        )
                    )
                    evidence.append(
                        AngntEvidence(
                            evidence_type="fault_case_pack",
                            source_id=pack_source_id,
                            summary=f"{scene_label} 本地图文知识单元",
                            payload=knowledge_pack,
                        )
                    )

        if reference:
            citations.append(
                AngntCitation(
                    source_type="classroom_reference",
                    source_id="classroom_reference",
                    title="课堂参考电路",
                    snippet="当前课堂已设置参考电路",
                )
            )
            evidence.append(
                AngntEvidence(
                    evidence_type="reference_circuit",
                    source_id="classroom_reference",
                    summary="课堂参考电路已存在，可用于对照",
                    payload={"reference_keys": sorted(reference.keys())[:top_k]},
                )
            )

        # WP-0: legacy PDF KB fusion (KbService.retrieve → Chroma → OpenAI)
        # was removed here. Datasheet evidence now flows exclusively through
        # the ``datasheet_lookup_tool`` in the agent graph, backed by the
        # local DatasheetKbService (OpenVINO embeddings).

        used_retrieval = bool(citations) and bool(query.strip())
        # Cap citations at top_k for UI density.
        # Evidence is bumped to top_k + 3 because the fixed items (station_state,
        # error_tags, fault_case_pack) must survive even when many teaching
        # scenes hit; pure top_k truncation would otherwise silently drop the
        # higher-value fault_case_pack at the tail.
        return {
            "station": station,
            "reference": reference,
            "citations": citations[:top_k],
            "evidence": evidence[: top_k + 3],
            "used_retrieval": used_retrieval,
        }

    def _extract_error_codes(self, comparison_report: dict[str, Any]) -> list[str]:
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

    def _build_scene_hit_for_resolved_topology(
        self,
        *,
        scene_id: str,
        query: str,
        error_tag_values: list[str],
        error_codes: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Build a single teaching_scene hit for the resolved topology.

        WP-1: when topology is known we bypass the cross-scene ranker and
        return exactly one hit for the resolved scene_id. This guarantees:
          (a) the resolved scene is always present in evidence (no
              ranker-truncation recall misses), and
          (b) no other scene can leak into teaching_scene evidence.

        Returns ``[]`` if the scene_id is not loadable (caller treats this
        as "no teaching_scene evidence").
        """
        if not self._teaching_kb_service:
            return []
        scene = self._teaching_kb_service.get_scene(scene_id)
        if not scene:
            return []
        matching_faults = self._teaching_kb_service.search_fault_cases(
            query=query,
            scene_id=scene_id,
            error_tags=error_tag_values,
            error_codes=error_codes,
            top_k=max(1, top_k),
        )
        return [
            {
                "scene_id": scene.get("scene_id", scene_id),
                "scene_name": scene.get("scene_name", ""),
                "course": scene.get("course", ""),
                "learning_goals": scene.get("learning_goals", []),
                "circuit_principles": scene.get("circuit_principles", []),
                "expected_measurements": scene.get("expected_measurements", []),
                "matching_faults": matching_faults,
                "source_materials": scene.get("source_materials", []),
                "score": 100,  # synthetic; we know this is the correct scene
            }
        ]

    def _build_mrag_pack(
        self,
        *,
        query: str,
        scene_id: str | None,
        error_tag_values: list[str],
        error_codes: list[str],
        diagnostics: list[Any],
        station: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        # WP-1 (2026-05-24): when topology is unknown, return empty pack.
        # MUST NOT fall back to "exp_first_order_rc"; that's the bug WP-1
        # fixes. Caller treats empty pack as "no fault_case_pack evidence".
        if not scene_id:
            return {}
        if self._mrag_service:
            return self._mrag_service.build_pack(
                query=query,
                scene_id=scene_id,
                error_tags=error_tag_values,
                structured_context={
                    "error_codes": error_codes[:top_k],
                    "diagnostics": diagnostics[:top_k],
                    "risk_level": station.get("risk_level", "safe"),
                    "circuit_snapshot": station.get("circuit_snapshot", ""),
                },
                top_k=top_k,
            )
        if self._teaching_kb_service:
            # WP-1 v6 (2026-05-24): the no-MragService fallback path used
            # to skip ``error_codes`` — meaning any caller injected without
            # a MragService (legacy tests, downgraded runs) would get
            # ``fault_cases=[]`` even with valid FLOATING_PIN / NODE_MISMATCH
            # in evidence. Production DI always provides MragService so this
            # branch is rarely hit, but it MUST also honor the canonical
            # validator↔KB bridge to keep the contract uniform.
            return self._teaching_kb_service.build_knowledge_pack(
                query=query,
                scene_id=scene_id,
                error_tags=error_tag_values,
                error_codes=error_codes[:top_k],
                top_k=top_k,
            )
        return {}
