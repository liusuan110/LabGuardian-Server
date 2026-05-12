"""LLM provider factory for the ReAct loop.

Always falls back to the deterministic `template` provider on misconfiguration
so the diagnostic graph never crashes due to model/runtime issues.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.agent.llm.base import LLMProvider
from app.agent.llm.template_provider import TemplateLLMProvider
from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_llm_provider(name: str | None = None) -> LLMProvider:
    """Return a cached LLM provider instance.

    Supported providers:
    - `template` (deterministic baseline)
    - `ollama` (local LLM, e.g. qwen3:4b)
    - `openvino_genai_text` (reserved for Phase 7+)
    """
    requested = (name or getattr(settings, "AGENT_LLM_PROVIDER", "template") or "template").strip().lower()
    if requested in {"template", ""}:
        return TemplateLLMProvider()

    if requested == "ollama":
        try:
            from app.agent.llm.ollama_provider import OllamaProvider

            provider = OllamaProvider(
                base_url=getattr(settings, "AGENT_LLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                or "http://127.0.0.1:11434",
                model=getattr(settings, "AGENT_LLM_OLLAMA_MODEL", "qwen3:4b") or "qwen3:4b",
                timeout_s=float(getattr(settings, "AGENT_LLM_OLLAMA_TIMEOUT_S", 30.0) or 30.0),
            )
            provider.warmup()
            return provider
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "Failed to instantiate ollama provider: %s; "
                "falling back to template provider",
                exc,
            )
        return TemplateLLMProvider()

    if requested == "openvino_genai_text":
        try:
            from app.agent.llm.openvino_genai_text import OpenVinoGenAITextProvider
            provider = OpenVinoGenAITextProvider(
                model_dir=getattr(settings, "AGENT_LLM_OPENVINO_MODEL_DIR", "") or "",
                device=getattr(settings, "AGENT_LLM_OPENVINO_DEVICE", "GPU") or "GPU",
            )
            provider.warmup()
            return provider
        except NotImplementedError:
            logger.warning(
                "AGENT_LLM_PROVIDER=openvino_genai_text is reserved for Phase 7+; "
                "falling back to template provider"
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "Failed to instantiate openvino_genai_text provider: %s; "
                "falling back to template provider",
                exc,
            )
        return TemplateLLMProvider()

    logger.warning(
        "Unknown AGENT_LLM_PROVIDER=%r; falling back to template provider",
        requested,
    )
    return TemplateLLMProvider()
