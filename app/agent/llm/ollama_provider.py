"""Ollama-backed text LLM provider for the diagnostic ReAct loop.

This provider keeps the same contract as `TemplateLLMProvider`:
- `plan()` returns a whitelist-safe `ToolCall | None`
- `reflect()` uses verifier as hard gate, with optional LLM rewrite hint
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from app.agent.contracts import ReflectionResult, ToolCall
from app.agent.llm.base import LLMProvider, PlanRequest, ReflectRequest
from app.agent.verification import verify_draft_answer

logger = logging.getLogger(__name__)

_TOOL_PRIORITY: list[str] = [
    "netlist_trace_tool",
    "board_schema_lookup_tool",
    "fault_case_lookup_tool",
    "datasheet_lookup_tool",
    "safety_rule_lookup_tool",
    "heatmap_overlay_tool",
    "teaching_concept_lookup_tool",
]


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = max(1.0, float(timeout_s))

    def warmup(self) -> None:
        endpoint = f"{self._base_url}/api/tags"
        timeout = max(20.0, self._timeout_s)
        self._request_json("GET", endpoint, timeout_s=timeout)

    def plan(self, request: PlanRequest) -> ToolCall | None:
        allowed = [tool.name for tool in request.context_pack.allowed_tools]
        if not allowed:
            return None

        already_called = {
            step.tool_call.tool_name
            for step in request.prior_steps
            if step.tool_call is not None
        }
        suggested = self._llm_plan_tool_name(
            query=request.query or request.user_message,
            error_family=request.context_pack.error_family,
            risk_level=request.context_pack.risk_level,
            allowed_tools=allowed,
            already_called=sorted(already_called),
        )
        tool_name = self._sanitize_tool_name(
            tool_name=suggested,
            allowed=allowed,
            already_called=already_called,
        )
        if tool_name is None:
            return None

        arguments: dict[str, Any] = {}
        first_finding = request.evidence.findings[0] if request.evidence.findings else None
        if tool_name == "netlist_trace_tool" and first_finding is not None:
            arguments = {
                "component_id": first_finding.component_id,
                "pin_name": first_finding.pin_name,
            }
        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            rationale=f"ollama({self._model}) planned next tool",
        )

    def reflect(self, request: ReflectRequest) -> ReflectionResult:
        report = verify_draft_answer(
            evidence=request.evidence,
            context_pack=request.context_pack,
            draft_answer=request.draft_answer,
        )
        if report.passed:
            return ReflectionResult(
                passed=True,
                reason="verifier_passed",
                next_hint="",
            )

        llm_hint = self._llm_reflect_hint(
            draft_answer=request.draft_answer,
            issues=report.issues,
            required_rewrite_hint=report.required_rewrite_hint,
        )
        return ReflectionResult(
            passed=False,
            reason="; ".join(report.issues) or "verification_failed",
            next_hint=llm_hint or report.required_rewrite_hint,
        )

    def _llm_plan_tool_name(
        self,
        *,
        query: str,
        error_family: str,
        risk_level: str,
        allowed_tools: list[str],
        already_called: list[str],
    ) -> str | None:
        prompt = "\n".join(
            [
                "你是电路故障诊断代理的规划器，只输出 JSON。",
                f"error_family={error_family}",
                f"risk_level={risk_level}",
                f"user_query={query}",
                f"allowed_tools={allowed_tools}",
                f"already_called={already_called}",
                '输出格式: {"tool_name":"<allowed_tools之一或null>","reason":"<简短原因>"}',
                "如果没有可调用工具，请输出 tool_name 为 null。",
            ]
        )
        raw = self._chat(prompt)
        if not raw:
            return None
        parsed = self._extract_json(raw)
        if not isinstance(parsed, dict):
            return None
        tool_name = parsed.get("tool_name")
        if tool_name is None:
            return None
        return str(tool_name).strip()

    def _llm_reflect_hint(
        self,
        *,
        draft_answer: str,
        issues: list[str],
        required_rewrite_hint: str,
    ) -> str:
        # Keep reflect path fast: verifier hint is already deterministic and
        # sufficient for rewrite; avoid another expensive LLM round trip.
        _ = draft_answer
        _ = issues
        return required_rewrite_hint

    def _sanitize_tool_name(
        self,
        *,
        tool_name: str | None,
        allowed: list[str],
        already_called: set[str],
    ) -> str | None:
        if tool_name and tool_name in allowed and tool_name not in already_called:
            return tool_name
        for candidate in _TOOL_PRIORITY:
            if candidate in allowed and candidate not in already_called:
                return candidate
        return None

    def _chat(self, prompt: str) -> str:
        endpoint = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "你是一个严谨的电路诊断助手。"},
                {"role": "user", "content": prompt},
            ],
            "keep_alive": "30m",
            "options": {
                "temperature": 0.1,
                "num_predict": 96,
            },
        }
        try:
            body = self._request_json(
                "POST",
                endpoint,
                payload=payload,
                timeout_s=self._timeout_s,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("Ollama request failed, fallback to deterministic planner: %s", exc)
            return ""
        message = body.get("message") if isinstance(body, dict) else None
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return ""

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_s: float,
    ) -> dict[str, Any]:
        # First-token latency can be high when model is cold-loaded.
        attempts = [
            min(max(6.0, timeout_s), 12.0),
            min(max(10.0, timeout_s * 1.2), 25.0),
        ]
        last_exc: Exception | None = None
        for idx, timeout in enumerate(attempts):
            try:
                # Force direct localhost access; avoid accidental proxy interception.
                with httpx.Client(timeout=timeout, trust_env=False) as client:
                    if method == "GET":
                        response = client.get(url)
                    else:
                        response = client.post(url, json=payload)
                    response.raise_for_status()
                    body = response.json()
                    return body if isinstance(body, dict) else {}
            except Exception as exc:
                last_exc = exc
                if idx < len(attempts) - 1:
                    time.sleep(0.25)
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        return {}

    @staticmethod
    def _extract_json(text: str) -> Any:
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return None
        return None
