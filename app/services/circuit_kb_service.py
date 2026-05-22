from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT, settings

# Minimum keyword-match score before a circuit is considered "hit".
# This prevents weak/accidental matches from returning irrelevant circuits.
_MIN_SCORE_THRESHOLD = 2.0
_DIRECT_HINT_SCORE = 4.0

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

_GENERIC_TERMS: set[str] = {
    "电路",
    "连接",
    "拓扑",
    "原理图",
    "运放",
    "运算放大器",
    "opamp",
    "op-amp",
    "放大",
    "放大器",
    "circuit",
    "connection",
    "schematic",
}

# Query expansion is deliberately local and deterministic.  The board can run
# with no embedding model, so common student paraphrases need to be mapped onto
# the same vocabulary used by the circuit JSON files.
_QUERY_EXPANSION_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("方波变三角波", "三角波", "斜坡", "爬坡", "积分", "累积", "面积", "低通", "防饱和"),
        ("积分电路", "积分器", "有源积分", "miller 积分器", "反馈电容", "时间常数", "rc 积分"),
        ("integrator_circuit",),
    ),
    (
        ("多个输入", "多路输入", "两路输入", "信号相加", "信号叠加", "求和", "加权", "混音", "加法"),
        ("反相加法器", "加法器", "求和电路", "summing amplifier", "加权求和", "虚地求和"),
        ("inverting_summing_amplifier",),
    ),
    (
        ("反相", "输出反着", "相位相反", "负增益", "增益为负", "rf/r1", "rf/rin", "虚地", "比例运算"),
        ("反相放大器", "反相比例", "inverting amplifier", "负反馈", "闭环增益"),
        ("inverting_amplifier",),
    ),
    (
        ("抖动", "来回跳", "毛刺", "抗噪", "抗干扰", "防抖", "双阈值", "上下阈值", "两个门限", "施密特"),
        ("滞回比较器", "迟滞比较器", "施密特触发器", "hysteresis comparator", "正反馈比较器"),
        ("comparator_hysteresis",),
    ),
    (
        ("过零", "单阈值", "阈值检测", "电平检测", "正弦波变方波", "比较电压", "没有滞回", "无滞回"),
        ("无滞回比较器", "电压比较器", "过零比较器", "开环比较", "饱和输出"),
        ("comparator_no_hysteresis",),
    ),
    (
        ("共射", "射极接地", "集电极输出", "8050", "三极管小信号", "基极偏置", "耦合电容", "音频放大"),
        ("共射放大", "共射极", "common emitter", "分压偏置", "三极管放大", "8050"),
        ("common_emitter_amplifier",),
    ),
    (
        ("差分", "差动", "两个输入相减", "相减", "减法", "共模", "cmrr", "长尾", "尾电流", "双端输出"),
        ("差分放大器", "差动放大", "bjt 差分对", "共模抑制", "尾电流源", "双端输出"),
        ("differential_amplifier",),
    ),
)

_CIRCUIT_LIKE_PATTERNS: tuple[str, ...] = (
    r"方波.*(三角波|斜坡)",
    r"(正弦波|模拟信号).*(方波|整形)",
    r"(输入|输出).*(反相|相位|增益|饱和|抖动|毛刺)",
    r"(两个|多路).*(输入|信号).*(相加|叠加|相减|差)",
    r"(阈值|门限).*(比较|检测|翻转)",
)


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
        query_terms, circuit_hints = self._expand_query(query_lower)

        scored: list[tuple[float, dict[str, Any], list[str]]] = []
        for circuit in self.list_circuits():
            score, matched = self._score_circuit(
                circuit,
                query_terms,
                query_lower,
                circuit_hints,
            )
            if score < _MIN_SCORE_THRESHOLD:
                continue
            scored.append((score, circuit, matched))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "circuit": circuit,
                "score": round(score, 2),
                "matched_features": matched[:8],
            }
            for score, circuit, matched in scored[:max(1, top_k)]
        ]

    # ------------------------------------------------------------------
    # scoring helpers
    # ------------------------------------------------------------------

    def _score_circuit(
        self,
        circuit: dict[str, Any],
        query_terms: set[str],
        query_lower: str,
        circuit_hints: dict[str, list[str]],
    ) -> tuple[float, list[str]]:
        score = 0.0
        matched: list[str] = []
        circuit_id = str(circuit.get("circuit_id") or "")
        specific_terms = {term for term in query_terms if term not in _GENERIC_TERMS}

        if circuit_id in circuit_hints:
            reasons = circuit_hints[circuit_id]
            bonus = _DIRECT_HINT_SCORE + min(2.0, len(reasons) * 0.4)
            score += bonus
            matched.append(f"intent:{'/'.join(reasons[:3])}")

        # keywords / aliases / retrieval_queries — strongest signals.
        keyword_fields: list[str] = []
        for key in ("keywords", "aliases", "retrieval_queries"):
            values = circuit.get(key, [])
            if isinstance(values, list):
                keyword_fields.extend(str(value) for value in values)

        for kw in keyword_fields:
            kw_lower = kw.lower()
            if not kw_lower:
                continue
            weight = 1.2 if kw_lower in _GENERIC_TERMS else 3.0
            if kw_lower in query_lower:
                score += weight
                matched.append(f"keyword:{kw}")
            elif any(term in kw_lower for term in specific_terms):
                score += weight
                matched.append(f"keyword:{kw}")
            elif any(
                term in query_lower
                for term in self._tokenize(kw_lower)
                if term not in _GENERIC_TERMS
            ):
                score += weight * 0.5
                matched.append(f"keyword_part:{kw}")

        # name / summary — medium signal (weight 1.0 per term hit)
        name = (circuit.get("name") or "").lower()
        summary = (circuit.get("summary") or "").lower()
        category = (circuit.get("category") or "").lower()
        subcategory = (circuit.get("subcategory") or "").lower()
        for field in (name, category, subcategory):
            for term in specific_terms:
                if term and term in field:
                    score += 1.0
                    matched.append(f"title:{term}")
        for term in specific_terms:
            if term and term in summary:
                score += 0.8
                matched.append(f"summary:{term}")

        # components — high precision signal (weight 2.0 per ref/role/type hit)
        for comp in circuit.get("components", []):
            ref = (comp.get("ref") or "").lower()
            role = (comp.get("role") or "").lower()
            comp_type = (comp.get("type") or "").lower()
            purpose = (comp.get("purpose") or "").lower()
            for term in specific_terms:
                if term and term in (ref, role, comp_type):
                    score += 2.0
                    matched.append(f"component:{term}")
                elif term and len(term) >= 3 and term in purpose:
                    score += 0.8
                    matched.append(f"component_purpose:{term}")

        # image_annotations / schematic metadata — useful when the user names
        # a label visible in the circuit picture (R0, R11, VT1, 8050, etc.).
        image_text = self._structured_text(
            circuit.get("image_annotations")
            or circuit.get("image_metadata")
            or circuit.get("schematic")
            or {}
        ).lower()
        for term in specific_terms:
            if term and term in image_text:
                score += 1.0
                matched.append(f"image:{term}")

        # teaching_points — weak bonus (0.5 per hit)
        for point in circuit.get("teaching_points", []):
            point_lower = (point or "").lower()
            for term in specific_terms:
                if term and term in point_lower:
                    score += 0.5
                    matched.append(f"teaching:{term}")

        # common_faults — bonus for fault-related queries
        for fault in circuit.get("common_faults", []):
            fault_text = json.dumps(fault, ensure_ascii=False).lower()
            for term in specific_terms:
                if term and term in fault_text:
                    score += 1.0
                    matched.append(f"fault:{term}")

        return score, self._dedupe(matched)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @classmethod
    def _expand_query(cls, text: str) -> tuple[set[str], dict[str, list[str]]]:
        terms = cls._tokenize(text)
        circuit_hints: dict[str, list[str]] = {}
        for triggers, expansions, circuit_ids in _QUERY_EXPANSION_RULES:
            hits = [trigger for trigger in triggers if trigger.lower() in text]
            if not hits:
                continue
            terms.update(term.lower() for term in expansions)
            for circuit_id in circuit_ids:
                circuit_hints.setdefault(circuit_id, []).extend(hits)

        return terms, circuit_hints

    @staticmethod
    def _structured_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = {
            part
            for part in re.split(r"[\s,，。？?；;：:、/()（）\[\]]+", text)
            if len(part) >= 2
        }
        tokens.update(
            term.lower()
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
    terms, circuit_hints = CircuitKbService._expand_query(msg)
    if circuit_hints:
        return True
    if terms & {keyword.lower() for keyword in _CIRCUIT_DOMAIN_KEYWORDS}:
        return True
    for pattern in _CIRCUIT_LIKE_PATTERNS:
        if re.search(pattern, msg):
            return True
    for keyword in _CIRCUIT_DOMAIN_KEYWORDS:
        if keyword.lower() in msg:
            return True
    return False
