"""LLM provider factory for the ReAct loop.

Always falls back to the deterministic `template` provider on misconfiguration
so the diagnostic graph never crashes due to a missing model directory.
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

    Phase 4 only ships `template`; any other name (including the reserved
    `openvino_genai_text`) logs a warning and falls back to `template`.
    """
    requested = (name or getattr(settings, "AGENT_LLM_PROVIDER", "template") or "template").strip().lower()
    if requested in {"template", ""}:
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
