from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT, settings


class TeachingKbService:
    """Loads scene-first teaching knowledge for circuit-lab RAG."""

    def __init__(
        self,
        base_dir: str | Path | None = None,
        fault_case_dir: str | Path | None = None,
    ) -> None:
        self._base_dir = Path(base_dir or settings.TEACHING_KB_DIR)
        if not self._base_dir.is_absolute():
            self._base_dir = PROJECT_ROOT / self._base_dir
        self._fault_case_dir = Path(fault_case_dir or settings.FAULT_CASE_KB_DIR)
        if not self._fault_case_dir.is_absolute():
            self._fault_case_dir = PROJECT_ROOT / self._fault_case_dir

    def list_scenes(self) -> list[dict[str, Any]]:
        if not self._base_dir.exists():
            return []
        scenes: list[dict[str, Any]] = []
        for path in sorted(self._base_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            payload.setdefault("_source_path", str(path))
            scenes.append(payload)
        return scenes

    def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        for scene in self.list_scenes():
            if scene.get("scene_id") == scene_id:
                return scene
        return None

    def list_fault_cases(self, *, scene_id: str | None = None) -> list[dict[str, Any]]:
        if not self._fault_case_dir.exists():
            return []
        cases: list[dict[str, Any]] = []
        for path in sorted(self._fault_case_dir.rglob("*.json")):
            payload = self._load_json(path)
            if not payload:
                continue
            if scene_id and payload.get("scene_id") != scene_id:
                continue
            payload.setdefault("_source_path", str(path))
            cases.append(payload)
        return cases

    def search(
        self,
        *,
        query: str = "",
        error_codes: list[str] | None = None,
        error_tags: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return scene hits with matching fault cases.

        This is intentionally simple and deterministic. Vector retrieval can be
        layered after this, but structured scene/error matches should remain the
        first routing signal for the teaching assistant.
        """
        normalized_query = query.lower().strip()
        query_terms = self._query_terms(normalized_query)
        requested_codes = {code for code in (error_codes or []) if code}
        requested_tags = {tag for tag in (error_tags or []) if tag}

        scored: list[tuple[int, dict[str, Any]]] = []
        for scene in self.list_scenes():
            score = self._score_scene(scene, query_terms, requested_codes, requested_tags)
            matching_faults = self._matching_faults(
                scene,
                query_terms,
                requested_codes,
                requested_tags,
            )
            if matching_faults:
                score += 8 * len(matching_faults)
            if score <= 0:
                continue
            hit = {
                "scene_id": scene.get("scene_id", ""),
                "scene_name": scene.get("scene_name", ""),
                "course": scene.get("course", ""),
                "learning_goals": scene.get("learning_goals", []),
                "circuit_principles": scene.get("circuit_principles", []),
                "expected_measurements": scene.get("expected_measurements", []),
                "matching_faults": matching_faults,
                "source_materials": scene.get("source_materials", []),
                "score": score,
            }
            scored.append((score, hit))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in scored[:top_k]]

    def search_fault_cases(
        self,
        *,
        query: str = "",
        scene_id: str = "",
        error_tags: list[str] | None = None,
        error_codes: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        # WP-1 (2026-05-24): default was ``"exp_first_order_rc"`` — that
        # silently fell back to RC for any non-RC caller. Now required.
        # Empty scene_id → no results (caller MUST NOT treat this as RC).
        #
        # WP-1 v3 (2026-05-24): ``error_codes`` parameter added. The
        # renamed scene-agnostic ``error_tags`` (missing_required_component
        # etc.) do not appear in fault_case JSONs (which use domain-specific
        # vocabularies like ``missing_power_connection``). The reliable
        # bridge between validator and KB is the ``related_error_codes``
        # field on each fault_case. We now score that intersection too.
        scene_id = (scene_id or "").strip()
        if not scene_id:
            return []
        query_terms = self._query_terms(query.lower().strip())
        requested_tags = {tag for tag in (error_tags or []) if tag}
        requested_codes = {code for code in (error_codes or []) if code}
        scored: list[tuple[int, dict[str, Any]]] = []

        for fault_case in self.list_fault_cases(scene_id=scene_id):
            score = self._score_fault_case(
                fault_case, query_terms, requested_tags, requested_codes
            )
            if score <= 0:
                continue
            scored.append((score, fault_case))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [case for _, case in scored[:top_k]]

    def build_knowledge_pack(
        self,
        *,
        query: str = "",
        scene_id: str = "",
        error_tags: list[str] | None = None,
        error_codes: list[str] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        # WP-1: same hardening as search_fault_cases.
        scene_id = (scene_id or "").strip()
        if not scene_id:
            return {}
        scene = self.get_scene(scene_id) or {}
        fault_cases = self.search_fault_cases(
            query=query,
            scene_id=scene_id,
            error_tags=error_tags,
            error_codes=error_codes,
            top_k=top_k,
        )
        return {
            "scene_id": scene_id,
            "scene_name": scene.get("scene_name", ""),
            "fault_cases": fault_cases,
            "references": self._collect_references(fault_cases),
            "fix_steps": self._collect_fix_steps(fault_cases),
        }

    def _score_scene(
        self,
        scene: dict[str, Any],
        query_terms: set[str],
        requested_codes: set[str],
        requested_tags: set[str],
    ) -> int:
        text = json.dumps(scene, ensure_ascii=False).lower()
        score = 0
        for term in query_terms:
            if term and term in text:
                score += 1
        scene_codes = {
            code
            for fault in scene.get("common_faults", [])
            for code in fault.get("related_error_codes", [])
        }
        scene_tags = {
            tag
            for fault in scene.get("common_faults", [])
            for tag in fault.get("error_tags", [])
        }
        score += 10 * len(scene_codes & requested_codes)
        score += 12 * len(scene_tags & requested_tags)
        return score

    def _matching_faults(
        self,
        scene: dict[str, Any],
        query_terms: set[str],
        requested_codes: set[str],
        requested_tags: set[str],
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for fault in scene.get("common_faults", []):
            fault_codes = set(fault.get("related_error_codes", []))
            fault_tags = set(fault.get("error_tags", []))
            fault_text = json.dumps(fault, ensure_ascii=False).lower()
            code_match = bool(fault_codes & requested_codes)
            tag_match = bool(fault_tags & requested_tags)
            query_match = any(term in fault_text for term in query_terms)
            if not code_match and not tag_match and not query_match:
                continue
            matches.append(
                {
                    "fault_id": fault.get("fault_id", ""),
                    "error_tags": fault.get("error_tags", []),
                    "related_error_codes": fault.get("related_error_codes", []),
                    "symptom": fault.get("symptom", ""),
                    "likely_reason": fault.get("likely_reason", ""),
                    "fix_steps": fault.get("fix_steps", []),
                    "teaching_hint": fault.get("teaching_hint", ""),
                }
            )
        return matches

    def _score_fault_case(
        self,
        fault_case: dict[str, Any],
        query_terms: set[str],
        requested_tags: set[str],
        requested_codes: set[str] | None = None,
    ) -> int:
        text = json.dumps(fault_case, ensure_ascii=False).lower()
        score = sum(1 for term in query_terms if term and term in text)
        fault_tags = set(fault_case.get("error_tags", []))
        score += 12 * len(fault_tags & requested_tags)
        # WP-1 v3 (2026-05-24): error_code intersection is the **primary**
        # validator↔KB bridge. validator emits codes like NODE_MISMATCH /
        # FLOATING_PIN; each fault_case JSON declares which it handles via
        # ``related_error_codes``. Weight slightly higher than tag matches
        # because codes are canonical, tags are vocabulary-drifty.
        if requested_codes:
            fault_codes = set(fault_case.get("related_error_codes", []))
            score += 15 * len(fault_codes & requested_codes)
        return score

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
            references["images"].extend(fault_case.get("reference_images", []))
            references["waveforms"].extend(fault_case.get("reference_waveforms", []))
            references["schematics"].extend(fault_case.get("reference_schematics", []))
        return {key: self._dedupe(values) for key, values in references.items()}

    def _collect_fix_steps(self, fault_cases: list[dict[str, Any]]) -> list[str]:
        steps: list[str] = []
        for fault_case in fault_cases:
            steps.extend(str(step) for step in fault_case.get("fix_steps", []) if step)
        return self._dedupe(steps)

    def _load_json(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            if value not in result:
                result.append(value)
        return result

    def _query_terms(self, query: str) -> set[str]:
        terms = {
            part
            for part in re.split(r"[\s,，。？?；;：:、/()（）]+", query)
            if len(part) >= 2
        }
        domain_terms = {
            "rc",
            "时间常数",
            "微分",
            "积分",
            "示波器",
            "探头",
            "黑夹子",
            "参考地",
            "面包板",
            "波形",
            "方波",
            "电容",
            "电阻",
        }
        return terms | {term for term in domain_terms if term.lower() in query}
