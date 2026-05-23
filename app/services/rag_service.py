"""
RAG 服务最小骨架

当前仅基于课堂态与 pipeline 结果构造结构化上下文与引用，
后续可替换为真实向量检索与知识库编排。
"""

from __future__ import annotations

from typing import Any

from app.schemas.angnt import AngntCitation, AngntEvidence
from app.services.classroom_state import ClassroomState
from app.services.error_tag_service import ErrorTagService
from app.services.kb_service import KbService
from app.services.mrag_service import MragService
from app.services.teaching_kb_service import TeachingKbService


class RagService:
    def __init__(
        self,
        kb_service: KbService | None = None,
        teaching_kb_service: TeachingKbService | None = None,
        error_tag_service: ErrorTagService | None = None,
        mrag_service: MragService | None = None,
    ) -> None:
        self._kb_service = kb_service
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
                    },
                )
            )
            if error_tags:
                citations.append(
                    AngntCitation(
                        source_type="error_tags",
                        source_id=f"{station_id}:rc_error_tags",
                        title="一阶 RC 结构化错误标签",
                        snippet="；".join(tag["error_tag"] for tag in error_tags[:top_k]),
                    )
                )
                evidence.append(
                    AngntEvidence(
                        evidence_type="error_tags",
                        source_id=f"{station_id}:rc_error_tags",
                        summary="从 validator_report_v2 映射得到的一阶 RC 教学错误标签",
                        payload={"error_tags": error_tags[:top_k]},
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
                teaching_hits = self._teaching_kb_service.search(
                    query=query,
                    error_codes=error_codes,
                    error_tags=error_tag_values,
                    top_k=scene_cap,
                )
                knowledge_pack = self._build_mrag_pack(
                    query=query,
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
                if knowledge_pack.get("fault_cases"):
                    citations.append(
                        AngntCitation(
                            source_type="fault_case_pack",
                            source_id=f"{station_id}:exp_first_order_rc_fault_cases",
                            title="一阶 RC 图文纠错知识包",
                            snippet="；".join(
                                str(case.get("title", ""))
                                for case in knowledge_pack["fault_cases"][:top_k]
                            )[:260],
                        )
                    )
                    evidence.append(
                        AngntEvidence(
                            evidence_type="fault_case_pack",
                            source_id=f"{station_id}:exp_first_order_rc_fault_cases",
                            summary="一阶 RC 本地图文知识单元",
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

        if self._kb_service and query.strip():
            kb_hits = self._kb_service.retrieve(query=query, top_k=top_k)
            for hit, _ in kb_hits:
                meta = hit.get("metadata", {}) or {}
                source_id = f'{meta.get("doc_id", "")}:{meta.get("chunk_index", "")}'
                citations.append(
                    AngntCitation(
                        source_type="datasheet_pdf",
                        source_id=source_id,
                        title=str(hit.get("title") or "datasheet"),
                        snippet=str(hit.get("snippet") or "")[:260],
                    )
                )
                evidence.append(
                    AngntEvidence(
                        evidence_type="datasheet_chunk",
                        source_id=source_id,
                        summary=str(hit.get("title") or "datasheet"),
                        payload={
                            "filename": meta.get("filename") or meta.get("source") or "datasheet",
                            "page": meta.get("page"),
                            "text": str(hit.get("text") or "")[:2400],
                        },
                    )
                )

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

    def answer_with_kb(
        self,
        *,
        query: str,
        top_k: int,
    ) -> tuple[str, list[AngntCitation], list[AngntEvidence], bool]:
        if not self._kb_service:
            return "未启用知识库。", [], [], False
        return self._kb_service.answer(query=query, top_k=top_k)

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

    def _build_mrag_pack(
        self,
        *,
        query: str,
        error_tag_values: list[str],
        error_codes: list[str],
        diagnostics: list[Any],
        station: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        if self._mrag_service:
            return self._mrag_service.build_pack(
                query=query,
                scene_id="exp_first_order_rc",
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
            return self._teaching_kb_service.build_knowledge_pack(
                query=query,
                scene_id="exp_first_order_rc",
                error_tags=error_tag_values,
                top_k=top_k,
            )
        return {}
