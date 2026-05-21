from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT, settings

# Minimum keyword-match score before a circuit is considered "hit".
# This prevents weak/accidental matches from returning irrelevant circuits.
_MIN_SCORE_THRESHOLD = 2.0

_CIRCUIT_DOMAIN_KEYWORDS: set[str] = {
    "放大电路", "放大", "放大器", "amplifier",
    "共射", "共集", "共基", "common emitter", "common collector",
    "反相", "同相", "inverting", "non-inverting",
    "加法器", "减法器", "summing", "adder", "subtractor",
    "差分", "differential",
    "比较器", "comparator", "滞回", "迟滞", "hysteresis", "schmitt",
    "积分", "微分", "integrator", "differentiator",
    "三极管", "transistor", "bjt", "npn", "pnp",
    "运放", "运算放大器", "opamp", "op-amp", "运筹",
    "偏置", "bias", "耦合", "coupling", "负反馈", "feedback",
    "跟随器", "follower", "缓冲", "buffer",
    "整流", "rectifier", "滤波", "filter",
    "振荡", "oscillator", "多谐振荡", "astable", "monostable",
    "门电路", "gate", "触发器", "flip-flop", "latch",
    "电路", "circuit", "原理图", "schematic",
    "拓扑", "topology", "连接", "connection",
    "rc", "rl", "rlc",
}


class CircuitKbService:
    """Loads circuit knowledge JSON files for schematic-level Q&A.

    Each JSON under ``knowledge/circuits/`` describes one typical circuit:
    its components, connections, analysis formulas, common faults, and
    teaching points.  This service performs deterministic keyword scoring;
    the structured payload is returned verbatim so the Agent LLM can reason
    about individual components, their roles, and fault scenarios.

    Gating:
    * ``search()`` returns an empty list when no keyword exceeds the
      relevance threshold — it never fabricates a "best guess".
    * Callers should additionally gate on ``_looks_like_circuit_query()``
      before wiring circuit results into a context pack or answer.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir or settings.CIRCUIT_KB_DIR)
        if not self._base_dir.is_absolute():
            self._base_dir = PROJECT_ROOT / self._base_dir

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def list_circuits(self) -> list[dict[str, Any]]:
        if not self._base_dir.exists():
            return []
        circuits: list[dict[str, Any]] = []
        for path in sorted(self._base_dir.glob("*.json")):
            payload = self._load_json(path)
            if payload is None:
                continue
            payload.setdefault("_source_path", str(path))
            circuits.append(payload)
        return circuits

    def get_circuit(self, circuit_id: str) -> dict[str, Any] | None:
        for circuit in self.list_circuits():
            if circuit.get("circuit_id") == circuit_id:
                return circuit
        return None

    def search(self, *, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Keyword-scored circuit search.

        Returns circuits whose keyword overlap exceeds
        ``_MIN_SCORE_THRESHOLD``, sorted by descending score.
        When nothing matches the query well enough the list is empty.
        """
        if not query or not self._base_dir.exists():
            return []

        query_lower = query.lower().strip()
        query_terms = self._tokenize(query_lower)

        scored: list[tuple[float, dict[str, Any]]] = []
        for circuit in self.list_circuits():
            score = self._score_circuit(circuit, query_terms, query_lower)
            if score < _MIN_SCORE_THRESHOLD:
                continue
            scored.append((score, circuit))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"circuit": circuit, "score": round(score, 2)}
            for score, circuit in scored[:max(1, top_k)]
        ]

    # ------------------------------------------------------------------
    # scoring helpers
    # ------------------------------------------------------------------

    def _score_circuit(
        self,
        circuit: dict[str, Any],
        query_terms: set[str],
        query_lower: str,
    ) -> float:
        score = 0.0

        # keywords field — strongest signal (weight 3.0 per hit)
        keywords: list[str] = circuit.get("keywords", [])
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in query_lower or any(
                term in kw_lower for term in query_terms
            ):
                score += 3.0
            elif any(term in query_lower for term in self._tokenize(kw_lower)):
                score += 1.5

        # name / summary — medium signal (weight 1.0 per term hit)
        name = (circuit.get("name") or "").lower()
        summary = (circuit.get("summary") or "").lower()
        category = (circuit.get("category") or "").lower()
        subcategory = (circuit.get("subcategory") or "").lower()
        for field in (name, category, subcategory):
            for term in query_terms:
                if term and term in field:
                    score += 1.0
        for term in query_terms:
            if term and term in summary:
                score += 1.0

        # components — high precision signal (weight 2.0 per ref/role/type hit)
        for comp in circuit.get("components", []):
            ref = (comp.get("ref") or "").lower()
            role = (comp.get("role") or "").lower()
            comp_type = (comp.get("type") or "").lower()
            for term in query_terms:
                if term and term in (ref, role, comp_type):
                    score += 2.0

        # teaching_points — weak bonus (0.5 per hit)
        for point in circuit.get("teaching_points", []):
            point_lower = (point or "").lower()
            for term in query_terms:
                if term and term in point_lower:
                    score += 0.5

        # common_faults — bonus for fault-related queries
        for fault in circuit.get("common_faults", []):
            fault_text = json.dumps(fault, ensure_ascii=False).lower()
            for term in query_terms:
                if term and term in fault_text:
                    score += 1.0

        return score

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = {
            part
            for part in re.split(r"[\s,，。？?；;：:、/()（）\[\]]+", text)
            if len(part) >= 2
        }
        tokens.update(
            term
            for term in _CIRCUIT_DOMAIN_KEYWORDS
            if term.lower() in text
        )
        return tokens


def looks_like_circuit_query(query: str) -> bool:
    """Lightweight gate: return True when *query* plausibly asks about circuits.

    This is intentionally loose — false positives are harmless because
    ``CircuitKbService.search()`` will still return an empty list when
    nothing matches above the threshold.  The gate's job is to skip the
    (cheap) search call entirely when the question is clearly off-topic.
    """
    if not query:
        return False
    msg = query.lower().strip()
    for keyword in _CIRCUIT_DOMAIN_KEYWORDS:
        if keyword.lower() in msg:
            return True
    return False
