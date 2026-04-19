"""
RAG 服务最小骨架

当前仅基于课堂态与 pipeline 结果构造结构化上下文与引用，
后续可替换为真实向量检索与知识库编排。
"""

from __future__ import annotations

from typing import Any

from app.schemas.angnt import AngntCitation, AngntEvidence
from app.services.classroom_state import ClassroomState
from app.services.kb_service import KbService


class RagService:
    def __init__(self, kb_service: KbService | None = None) -> None:
        self._kb_service = kb_service

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
                    },
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
        return {
            "station": station,
            "reference": reference,
            "citations": citations[:top_k],
            "evidence": evidence[:top_k],
            "used_retrieval": used_retrieval,
        }

    def answer_with_kb(self, *, query: str, top_k: int) -> tuple[str, list[AngntCitation], list[AngntEvidence], bool]:
        if not self._kb_service:
            return "未启用知识库。", [], [], False
        return self._kb_service.answer(query=query, top_k=top_k)
