"""LLM-backed intent classifier with deterministic keyword fallback.

Replaces the four overlapping keyword tables (``intent.py`` +
``agent_service._is_circuit_related_question`` +
``_AGENT_IDENTITY_PHRASES`` + ``_looks_like_current_context_follow_up``)
with **one** lightweight LLM call that emits structured JSON.

Design contract
---------------
* Single Ollama call with ``format="json"`` (Ollama 0.1.30+ enforces JSON).
* Short timeout (~5s) — intent classification must not block the agent.
* **Any failure** (no provider, timeout, parse error, unknown label) falls
  back deterministically to the existing keyword classifier — so this
  module is strictly a *quality enhancement*, never a regression.
* Returns ``IntentDecision`` so the caller can log telemetry
  (``source="llm" | "keyword"``, ``confidence``, ``reason``).

The keyword classifier in ``app/agent/intent.py`` stays as the
deterministic fallback. Tests that pin specific phrasings continue to
import ``classify_intent`` directly.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.agent.contracts import AgentIntent, RuntimeEvidence
from app.agent.intent import classify_intent as classify_intent_keyword
from app.core.config import settings

logger = logging.getLogger(__name__)

_VALID_INTENTS: tuple[AgentIntent, ...] = (
    "diagnostic",
    "concept_tutor",
    "lab_guidance",
    "mixed",
)
_DEFAULT_TIMEOUT_S = 5.0  # short — fall back fast on any hiccup
_MAX_USER_MSG_CHARS = 280


@dataclass(frozen=True)
class IntentDecision:
    """Outcome of intent classification.

    ``source`` is one of ``llm`` or ``keyword`` and records which path
    produced the label — useful for offline evaluation and for the
    AngntEvidence trace.
    """

    intent: AgentIntent
    source: str
    confidence: float
    reason: str = ""


def classify_intent_smart(
    user_message: str,
    evidence: RuntimeEvidence | None = None,
    *,
    timeout_s: float | None = None,
) -> IntentDecision:
    """Try LLM classification; deterministically fall back to keywords.

    The fallback path returns the exact same label as
    :func:`app.agent.intent.classify_intent` would have returned, so this
    is a drop-in replacement.
    """

    keyword_intent = classify_intent_keyword(user_message, evidence)
    msg = (user_message or "").strip()
    if not msg:
        return IntentDecision(
            intent=keyword_intent,
            source="keyword",
            confidence=0.50,
            reason="empty_message",
        )

    provider_name = (
        getattr(settings, "AGENT_LLM_PROVIDER", "template") or "template"
    ).strip().lower()
    # LLM-backed classification is enabled ONLY for ``ollama`` (JSON-mode
    # HTTP API). The on-board distilled student (``openvino_genai_text``)
    # deliberately uses the deterministic keyword classifier instead — this
    # is a measured choice, not an oversight. A board probe
    # (``scripts/board/probe_intent_student.py``) showed the 1.5B student
    # adds ~1.5–2.5 s/question on the shared iGPU yet scores *below* the
    # keyword tables (3/6 vs 4/6 on the probe set): it recognises "mixed"
    # questions but still mislabels them, and it regressed a context-driven
    # diagnostic case that keyword routing gets right via the evidence
    # tiebreak. Keyword routing is faster AND more accurate here, so we keep
    # it. (``template`` has no LLM either → same keyword path.)
    if provider_name != "ollama":
        return IntentDecision(
            intent=keyword_intent,
            source="keyword",
            confidence=0.55,
            reason=f"provider={provider_name}",
        )

    try:
        decision = _ask_ollama(
            msg=msg,
            evidence=evidence,
            timeout_s=timeout_s if timeout_s is not None else _DEFAULT_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 — broad fallback is intentional
        logger.info("intent_llm fallback to keyword: %s", exc)
        return IntentDecision(
            intent=keyword_intent,
            source="keyword",
            confidence=0.45,
            reason=f"llm_error:{type(exc).__name__}",
        )

    return decision


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "你是 LabGuardian 的意图分类器。把学生在电路实验助教里的提问归到 4 类之一：\n"
    "- diagnostic：明确询问当前/我的电路哪里错了、为什么短路/不通、参考差异、怎么改；\n"
    "- concept_tutor：在问电路/电子学的原理、公式、定义、知识点；\n"
    "- lab_guidance：在问用万用表/示波器怎么测，操作步骤，下一步该做什么；\n"
    "- mixed：既要诊断当前电路又要解释相关原理（两类都明显命中）。\n"
    "如果问题与电路实验完全无关（天气/闲聊/自我介绍），归为 concept_tutor。"
    "严禁输出多余字段，只输出符合 schema 的 JSON 对象。"
)


def _build_prompt(*, msg: str, evidence: RuntimeEvidence | None) -> str:
    has_ctx = bool(
        evidence
        and (
            evidence.error_codes
            or evidence.findings
            or evidence.risk_level in {"warning", "danger"}
        )
    )
    ctx_line = (
        "当前电路存在诊断上下文（有错误码或风险），用户可能在追问诊断。"
        if has_ctx
        else "当前没有诊断上下文，用户更可能在问原理或操作。"
    )
    truncated = msg[:_MAX_USER_MSG_CHARS]
    schema = (
        '{"intent":"diagnostic|concept_tutor|lab_guidance|mixed",'
        '"confidence":0.0-1.0,"reason":"<≤30字>"}'
    )
    return (
        f"{ctx_line}\n"
        f"用户问题：{truncated}\n"
        f"按 schema 输出 JSON：{schema}\n"
        "只输出 JSON，不要解释。"
    )


def _ask_ollama(
    *,
    msg: str,
    evidence: RuntimeEvidence | None,
    timeout_s: float,
) -> IntentDecision:
    base_url = str(
        getattr(settings, "AGENT_LLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        or "http://127.0.0.1:11434"
    ).rstrip("/")
    model = str(
        getattr(settings, "AGENT_LLM_OLLAMA_MODEL", "qwen3:4b") or "qwen3:4b"
    )
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "format": "json",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_prompt(msg=msg, evidence=evidence),
            },
        ],
        "options": {
            "temperature": 0.0,
            "num_predict": 80,
        },
    }
    endpoint = f"{base_url}/api/chat"
    with httpx.Client(timeout=timeout_s, trust_env=False) as client:
        response = client.post(endpoint, json=payload)
        response.raise_for_status()
        body = response.json()

    text = str(((body or {}).get("message") or {}).get("content") or "").strip()
    if not text:
        raise RuntimeError("empty llm response")

    parsed = _parse_intent_json(text)
    candidate = str(parsed.get("intent") or "").strip().lower()
    if candidate not in _VALID_INTENTS:
        raise ValueError(f"invalid intent label: {candidate!r}")

    try:
        confidence = float(parsed.get("confidence", 0.7))
    except (TypeError, ValueError):
        confidence = 0.7
    confidence = max(0.0, min(1.0, confidence))
    reason = str(parsed.get("reason") or "")[:60]

    return IntentDecision(
        intent=candidate,  # type: ignore[arg-type]
        source="llm",
        confidence=confidence,
        reason=reason or f"llm:{model}",
    )


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*?\}")


def _parse_intent_json(text: str) -> dict[str, Any]:
    """Robustly extract the first JSON object from a model response.

    Handles common deviations like leading whitespace, code fences, or a
    short explanation before/after the JSON body. ``format=json`` makes
    these rare but not impossible (older Ollama versions, fallback models).
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"no json object found in response: {text[:120]!r}")
    return json.loads(match.group())
