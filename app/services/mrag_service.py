from __future__ import annotations

from typing import Any

from app.services.teaching_kb_service import TeachingKbService


class MragService:
    """Builds local multimodal RAG packs for the first-order RC experiment."""

    DEFAULT_SCENE_ID = "exp_first_order_rc"

    def __init__(self, teaching_kb_service: TeachingKbService) -> None:
        self._teaching_kb_service = teaching_kb_service

    def build_pack(
        self,
        *,
        query: str = "",
        scene_id: str = DEFAULT_SCENE_ID,
        error_tags: list[str] | None = None,
        circuit_snapshot: str = "",
        structured_context: dict[str, Any] | None = None,
        top_k: int = 5,
        retrieved: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        scene = self._teaching_kb_service.get_scene(scene_id) or {}
        context = dict(structured_context or {})
        snapshot = str(circuit_snapshot or "").strip()
        if snapshot and not context.get("circuit_snapshot"):
            context["circuit_snapshot"] = snapshot
        fault_cases = self._teaching_kb_service.search_fault_cases(
            query=query,
            scene_id=scene_id,
            error_tags=error_tags,
            top_k=top_k,
        )
        compact_cases = [self._compact_fault_case(case) for case in fault_cases]

        normalized_retrieved = self._normalize_retrieved(retrieved)
        pack_version = "mrag_pack_v2" if normalized_retrieved else "mrag_pack_v1"

        pack: dict[str, Any] = {
            "pack_version": pack_version,
            "scene": {
                "scene_id": scene_id,
                "scene_name": scene.get("scene_name", ""),
                "course": scene.get("course", ""),
                "learning_goals": scene.get("learning_goals", [])[:4],
                "expected_measurements": scene.get("expected_measurements", [])[:4],
            },
            "query": query,
            "error_tags": error_tags or [],
            "structured_context": context,
            "fault_cases": compact_cases,
            "references": self._collect_references(compact_cases),
            "fix_steps": self._collect_fix_steps(compact_cases),
        }
        if normalized_retrieved:
            pack["retrieved"] = normalized_retrieved
        return pack

    @staticmethod
    def _normalize_retrieved(retrieved: dict[str, Any] | None) -> dict[str, Any] | None:
        if not retrieved:
            return None
        buckets = ("datasheet_chunks", "figures", "tables")
        cleaned: dict[str, list[Any]] = {}
        for key in buckets:
            values = retrieved.get(key) or []
            if not isinstance(values, list):
                continue
            cleaned_values = [v for v in values if v]
            if cleaned_values:
                cleaned[key] = cleaned_values
        return cleaned or None

    def _compact_fault_case(self, fault_case: dict[str, Any]) -> dict[str, Any]:
        return {
            "knowledge_id": fault_case.get("knowledge_id", ""),
            "fault_id": fault_case.get("fault_id", ""),
            "title": fault_case.get("title", ""),
            "error_tags": fault_case.get("error_tags", []),
            "related_error_codes": fault_case.get("related_error_codes", []),
            "trigger_conditions": fault_case.get("trigger_conditions", []),
            "reference_text": fault_case.get("reference_text", ""),
            "references": {
                "images": fault_case.get("reference_images", []),
                "waveforms": fault_case.get("reference_waveforms", []),
                "schematics": fault_case.get("reference_schematics", []),
            },
            "fix_steps": fault_case.get("fix_steps", []),
            "student_answer_template": fault_case.get("student_answer_template", ""),
            "source_materials": fault_case.get("source_materials", []),
        }

    def _collect_references(self, fault_cases: list[dict[str, Any]]) -> dict[str, list[str]]:
        references = {
            "texts": [],
            "images": [],
            "waveforms": [],
            "schematics": [],
        }
        for fault_case in fault_cases:
            text = str(fault_case.get("reference_text") or "")
            if text:
                references["texts"].append(text)
            case_refs = fault_case.get("references", {})
            references["images"].extend(case_refs.get("images", []))
            references["waveforms"].extend(case_refs.get("waveforms", []))
            references["schematics"].extend(case_refs.get("schematics", []))
        return {key: self._dedupe(values) for key, values in references.items()}

    def _collect_fix_steps(self, fault_cases: list[dict[str, Any]]) -> list[str]:
        steps: list[str] = []
        for fault_case in fault_cases:
            steps.extend(str(step) for step in fault_case.get("fix_steps", []) if step)
        return self._dedupe(steps)

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            if value not in result:
                result.append(value)
        return result
