"""LLM provider package for the diagnostic ReAct loop.

Phase 4 only ships the deterministic `template` provider; the
`openvino_genai_text` provider is a stub reserved for Phase 7+ when DK-2500
NPU is validated. The factory always falls back to `template` so existing
tests stay deterministic.
"""

from app.agent.llm.base import LLMProvider, PlanRequest, ReflectRequest
from app.agent.llm.factory import get_llm_provider

__all__ = [
    "LLMProvider",
    "PlanRequest",
    "ReflectRequest",
    "get_llm_provider",
]
