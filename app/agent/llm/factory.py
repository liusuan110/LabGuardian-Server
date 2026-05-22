"""LLM provider factory for the ReAct loop.

Always falls back to the deterministic `template` provider on misconfiguration
so the diagnostic graph never crashes due to model/runtime issues.

Caching strategy: successfully created providers are cached to avoid repeated
warmup calls.  Fallback ``TemplateLLMProvider`` is **never** cached — this
allows recovery when Ollama becomes available after server start.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from app.agent.llm.base import LLMProvider
from app.agent.llm.template_provider import TemplateLLMProvider
from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manual cache: only cache successful non-template providers.
# A threading.Lock guards against races during warmup (which can take seconds).
# ---------------------------------------------------------------------------
_cache: dict[str, LLMProvider] = {}
_lock = threading.Lock()


def _cached_provider(key: str) -> LLMProvider | None:
    with _lock:
        return _cache.get(key)


def _cache_provider(key: str, provider: LLMProvider) -> None:
    with _lock:
        _cache.setdefault(key, provider)


def clear_llm_provider_cache() -> None:
    """Clear cached LLM providers (useful for tests or forced refresh)."""
    with _lock:
        _cache.clear()


def get_llm_provider(name: str | None = None) -> LLMProvider:
    """Return a (possibly cached) LLM provider instance.

    Supported providers:
    - ``template`` (deterministic baseline)
    - ``ollama`` (local LLM)
    - ``openvino_genai_text`` (reserved for Phase 7+)
    """
    requested = (name or getattr(settings, "AGENT_LLM_PROVIDER", "template") or "template").strip().lower()
    if requested in {"template", ""}:
        return TemplateLLMProvider()

    # Fast path: return previously cached successful provider.
    cached = _cached_provider(requested)
    if cached is not None:
        return cached

    if requested == "ollama":
        try:
            from app.agent.llm.ollama_provider import OllamaProvider  # noqa: PLC0415

            provider = OllamaProvider(
                base_url=getattr(settings, "AGENT_LLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                or "http://127.0.0.1:11434",
                model=getattr(settings, "AGENT_LLM_OLLAMA_MODEL", "qwen3:4b") or "qwen3:4b",
                timeout_s=float(getattr(settings, "AGENT_LLM_OLLAMA_TIMEOUT_S", 30.0) or 30.0),
            )
            provider.warmup()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "Failed to instantiate ollama provider: %s; "
                "falling back to template provider (will retry next call)",
                exc,
            )
            return TemplateLLMProvider()

        # Only cache on success — failures above just return template without caching.
        _cache_provider(requested, provider)
        return provider

    if requested == "openvino_genai_text":
        try:
            from app.agent.llm.openvino_genai_text import (  # noqa: PLC0415
                OpenVinoGenAITextProvider,
            )

            provider: Any = OpenVinoGenAITextProvider(
                model_dir=getattr(settings, "AGENT_LLM_OPENVINO_MODEL_DIR", "") or "",
                device=getattr(settings, "AGENT_LLM_OPENVINO_DEVICE", "GPU") or "GPU",
            )
            provider.warmup()
        except NotImplementedError:
            logger.warning(
                "AGENT_LLM_PROVIDER=openvino_genai_text is reserved for Phase 7+; "
                "falling back to template provider"
            )
            return TemplateLLMProvider()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "Failed to instantiate openvino_genai_text provider: %s; "
                "falling back to template provider (will retry next call)",
                exc,
            )
            return TemplateLLMProvider()

        _cache_provider(requested, provider)
        return provider

    logger.warning(
        "Unknown AGENT_LLM_PROVIDER=%r; falling back to template provider",
        requested,
    )
    return TemplateLLMProvider()
