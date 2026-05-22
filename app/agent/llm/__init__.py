"""LLM provider package for the diagnostic ReAct loop.

Current providers:
- `template` deterministic baseline (CI-safe)
- `ollama` local model provider (offline edge inference)
- `openvino_genai_text` stub reserved for Phase 7+ validation
"""

from app.agent.llm.base import LLMProvider, PlanRequest, ReflectRequest
from app.agent.llm.factory import clear_llm_provider_cache, get_llm_provider

__all__ = [
    "LLMProvider",
    "PlanRequest",
    "ReflectRequest",
    "clear_llm_provider_cache",
    "get_llm_provider",
]
