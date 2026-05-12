"""LLM provider package for the diagnostic ReAct loop.

Current providers:
- `template` deterministic baseline (CI-safe)
- `ollama` local model provider (offline edge inference)
- `openvino_genai_text` stub reserved for Phase 7+ validation
"""

from app.agent.llm.base import LLMProvider, PlanRequest, ReflectRequest
from app.agent.llm.factory import get_llm_provider

__all__ = [
    "LLMProvider",
    "PlanRequest",
    "ReflectRequest",
    "get_llm_provider",
]
